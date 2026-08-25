#!/usr/bin/env python
"""Offline GEMM quant-error budget for the W8A8 int8 recipe (gfx908, CPU-only).

Replays the exact production numeric chain per recorded GEMM instance
(bootA rank0) and decomposes the output error into legs:

  A  golden            y = x @ W_bf16.T                      (bf16 ref weights)
  B  fp16 weight cast  y = x @ W_bf16.half().float().T       (fp16cast leg)
  C  GPTQ dequant      y = x @ W_deq.T                       (gptq leg = C-B)
  D  CK weight requant y = x @ (ck_q*ck_s).T                 (ck_requant leg = D-C)
  E  production GEMM   y = (xq*s_x) @ ck_q.T * ck_s          (act_quant leg = E-D)

Weights are TP=4 rank0 shards of the merged vLLM modules; recorded prefix
`language_model.model.` maps to checkpoint prefix `model.language_model.`.
"""

import glob
import json
import math
import os
import re
import time

import numpy as np
import torch
from safetensors import safe_open

QDIR = os.path.expanduser("~/models/Qwen3.8-27B-GPTQ-8bit-gs128")
BDIR = os.path.expanduser("~/models/Qwen3.8-27B-bf16-ref")
BOOT = os.path.expanduser("~/models/kld/quant_audit/bootA/rank0")
OUT = os.path.expanduser("~/models/kld/quant_audit/replay/gemm_budget.json")

NUM_LAYERS = 64
TP = 4
MAX_INSTANCES = 2  # recorded instances per module
TIME_BUDGET_S = 13 * 60  # soft cap; stop starting new layers past this

SELF_ATTN_LAYERS = [i for i in range(NUM_LAYERS) if (i + 1) % 4 == 0]  # 3,7,...,63
GDN_LAYERS = [i for i in range(NUM_LAYERS) if (i + 1) % 4 != 0]

CKPT = "model.language_model.layers.{L}.{name}"
REC = "language_model.model.layers.{L}.{name}"


class ShardCache:
    """Lazy per-shard safetensors handle cache with index-based tensor lookup."""

    def __init__(self, root):
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.index = json.load(f)["weight_map"]
        self.root = root
        self._handles = {}

    def get(self, key):
        shard = self.index[key]
        if shard not in self._handles:
            self._handles[shard] = safe_open(
                os.path.join(self.root, shard), framework="pt"
            )
        return self._handles[shard].get_tensor(key)


def gptq_dequant_shard(qw, sc, n_rows=None, k_groups=None):
    """Unpack+dequant a GPTQ component to fp32 [N,K] with optional rank0 slicing.

    n_rows: row (output-channel) slice for column-parallel merge — [0:n_rows].
    k_groups: group slice for row-parallel K-sharding — groups [0:k_groups].
    """
    if k_groups is not None:
        qw = qw[: (k_groups * 128) // 4]  # qweight rows: k_groups*128/4
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32).view(1, 4, 1)
    w = ((qw.unsqueeze(1).to(torch.int32) >> shifts) & 0xFF).to(torch.int16)
    w = (w.reshape(-1, qw.shape[1]) - 128).to(torch.int8)  # [K,N] LSB-first along K
    if k_groups is not None:
        w = w[: k_groups * 128]
        sc = sc[:k_groups]
    W = w.t().contiguous().float()  # [N,K]
    s = sc.t().contiguous().float()  # [N,K//128]
    W_deq = (W.view(W.shape[0], s.shape[1], 128) * s.unsqueeze(-1)).reshape(
        W.shape[0], -1
    )
    if n_rows is not None:
        W_deq = W_deq[:n_rows]
    return W_deq


def ck_requant(W_deq):
    """Production CK per-output-channel int8 requant of the fp16-rounded weight."""
    W16 = W_deq.to(torch.float16).float()
    wmax = W16.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    ck_s = (wmax / 127.0).float()
    ck_q = torch.round(W16 / ck_s).clamp(-127, 127).to(torch.int8)
    return ck_q, ck_s


def module_spec(layer, family):
    """Rank0 shard spec: (component bf16 weight keys, shard kind).

    col_per_component: each component independently keeps its first N/TP rows
      (vLLM MergedColumnParallelLinear shards partitions separately; qkv_proj
      shards q/k/v per contiguous head-blocks which coincides with N/TP rows).
    row: rank0 keeps the first K/TP columns (down_proj/o_proj/out_proj).
    """
    L = layer
    if family == "mlp.gate_up_proj":
        return [
            CKPT.format(L=L, name="mlp.gate_proj.weight"),
            CKPT.format(L=L, name="mlp.up_proj.weight"),
        ], "col_per_component"
    if family == "mlp.down_proj":
        return [CKPT.format(L=L, name="mlp.down_proj.weight")], "row"
    if family == "linear_attn.in_proj_qkvz":
        return [
            CKPT.format(L=L, name="linear_attn.in_proj_qkv.weight"),
            CKPT.format(L=L, name="linear_attn.in_proj_z.weight"),
        ], "col_per_component"
    if family == "linear_attn.out_proj":
        return [CKPT.format(L=L, name="linear_attn.out_proj.weight")], "row"
    if family == "self_attn.qkv_proj":
        return [
            CKPT.format(L=L, name="self_attn.q_proj.weight"),
            CKPT.format(L=L, name="self_attn.k_proj.weight"),
            CKPT.format(L=L, name="self_attn.v_proj.weight"),
        ], "col_per_component"
    if family == "self_attn.o_proj":
        return [CKPT.format(L=L, name="self_attn.o_proj.weight")], "row"
    raise ValueError(family)


def build_family(layer, family, qs, bs):
    """Assemble rank0 W_bf16, W_deq, ck_q, ck_s for a merged module."""
    comps, kind = module_spec(layer, family)
    Wb_parts, Wd_parts, ckq_parts, cks_parts = [], [], [], []
    for key_bf16 in comps:
        stem = key_bf16[: -len(".weight")]
        qw = qs.get(stem + ".qweight")
        sc = qs.get(stem + ".scales")
        Wb = bs.get(key_bf16)  # [N_full, K_full] bf16
        N_full, K_full = Wb.shape
        n_rows = None
        k_groups = None
        if kind == "col_per_component":
            n_rows = N_full // TP
        else:  # row
            k_groups = (K_full // TP) // 128
        W_deq = gptq_dequant_shard(qw, sc, n_rows=n_rows, k_groups=k_groups)
        ck_q, ck_s = ck_requant(W_deq)
        Wb_part = Wb.float()
        if n_rows is not None:
            Wb_part = Wb_part[:n_rows]
        else:
            Wb_part = Wb_part[:, : k_groups * 128]
        assert Wb_part.shape == W_deq.shape, (Wb_part.shape, W_deq.shape)
        Wb_parts.append(Wb_part)
        Wd_parts.append(W_deq)
        ckq_parts.append(ck_q)
        cks_parts.append(ck_s)
    return (
        torch.cat(Wb_parts, 0),
        torch.cat(Wd_parts, 0),
        torch.cat(ckq_parts, 0),
        torch.cat(cks_parts, 0),
    )


def rel(u, v):
    n = v.norm().item()
    if n == 0.0:
        return float("nan")
    return ((u - v).norm().item()) / n


def main():
    torch.set_grad_enabled(False)
    t_start = time.time()
    qs = ShardCache(QDIR)
    bs = ShardCache(BDIR)

    # discover recorded instances: gemm_<mod>_N{n}_K{k}_{i}__x.pt
    inst_re = re.compile(
        r"gemm_language_model\.model\.layers\.(\d+)\."
        r"(mlp\.gate_up_proj|mlp\.down_proj|linear_attn\.in_proj_qkvz|"
        r"linear_attn\.out_proj|self_attn\.qkv_proj|self_attn\.o_proj)"
        r"_N(\d+)_K(\d+)_(\d)__x\.pt"
    )
    found = {}
    for path in glob.glob(os.path.join(BOOT, "gemm_*__x.pt")):
        m = inst_re.match(os.path.basename(path))
        if not m:
            continue
        layer, family, n_tag, k_tag, inst = (
            int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
        )
        found.setdefault((layer, family), {})[inst] = (n_tag, k_tag)

    layers = sorted({k[0] for k in found})
    print(f"discovered {len(found)} modules over {len(layers)} layers; "
          f"self_attn layers present: "
          f"{sorted({l for (l, f) in found if f.startswith('self_attn')})}")

    records = []
    excluded = []
    skipped_time = False
    for layer in layers:
        if time.time() - t_start > TIME_BUDGET_S:
            print(f"time budget hit before layer {layer}; sampling stops")
            skipped_time = True
            break
        families = sorted(f for (l, f) in found if l == layer)
        for family in families:
            try:
                Wb, W_deq, ck_q, ck_s = build_family(layer, family, qs, bs)
            except KeyError as e:
                print(f"skip layer {layer} {family}: missing tensor {e}")
                continue
            Wck = ck_q.float() * ck_s  # [N,K] fp32
            werr_gptq = rel(W_deq, Wb)
            werr_full = rel(Wck, Wb)
            werr_ck = rel(Wck, W_deq)

            base = REC.format(L=layer, name=family)
            for inst in sorted(found[(layer, family)])[:MAX_INSTANCES]:
                n_tag, k_tag = found[(layer, family)][inst]
                fx = os.path.join(BOOT, f"gemm_{base}_N{n_tag}_K{k_tag}_{inst}__x.pt")
                fxq = fx.replace("__x.pt", "__xq.pt")
                fxs = fx.replace("__x.pt", "__xs.pt")
                if not (os.path.exists(fxq) and os.path.exists(fxs)):
                    continue
                x = torch.load(fx, map_location="cpu", weights_only=True)   # [M,K] fp16
                xq = torch.load(fxq, map_location="cpu", weights_only=True)  # [M,K] i8
                xs = torch.load(fxs, map_location="cpu", weights_only=True)  # [M,1] fp16
                assert x.shape[1] == W_deq.shape[1], (x.shape, W_deq.shape)
                M = x.shape[0]
                xf = x.float()
                rmax = x.abs().amax(dim=1, keepdim=True)
                s_x = rmax / 127.0
                s_x = torch.where(s_x == 0, torch.ones_like(s_x), s_x)
                x_deq = xq.float() * s_x

                # recording-integrity gate: recorded per-token scale must match
                # the scale implied by the recorded x (rows where rmax==0 or
                # xq==0, or xs/s ratios off >5%, mean the x and xq/xs tensors
                # were captured from different forwards / padded dummy rows).
                ok_rows = (rmax > 0) & (xq.float().abs().amax(dim=1, keepdim=True) > 0)
                ratios = (xs.float() / s_x).where(ok_rows, torch.ones_like(s_x))
                max_ratio_dev = (ratios - 1).abs().max().item()
                trusted = bool(
                    ok_rows.all().item()
                    and max_ratio_dev < 0.05
                    and x.float().abs().sum() > 0
                )
                if not trusted:
                    excluded.append({
                        "layer": layer, "family": family, "instance": inst, "m": M,
                        "reason": ("x_all_zero" if not bool((rmax > 0).any())
                                   else "xq_zero_or_scale_mismatch"),
                        "max_xs_scale_ratio_dev": max_ratio_dev,
                    })

                A = xf @ Wb.t()
                B = xf @ Wb.to(torch.float16).float().t()
                C = xf @ W_deq.t()
                D = xf @ Wck.t()
                E = x_deq @ Wck.t()

                rec = {
                    "layer": layer,
                    "family": family,
                    "instance": inst,
                    "m": M,
                    "trusted": trusted,
                    "k": int(W_deq.shape[1]),
                    "n": int(W_deq.shape[0]),
                    "err_fp16cast": rel(B, A),   # leg: weight fp16 rounding
                    "err_gptq_total": rel(C, A),
                    "err_ck_total": rel(D, A),
                    "err_prod_total": rel(E, A),
                    "leg_gptq": rel(C, B),
                    "leg_ck_requant": rel(D, C),
                    "leg_act_quant": rel(E, D),
                    "werr_gptq": werr_gptq,
                    "werr_full_chain": werr_full,
                    "werr_ck_requant": werr_ck,
                    # sanity: recorded xq*xs (on-disk fp16 scale) vs recorded x
                    "xqxs_resid_disk_scale": rel(xq.float() * xs.float(), xf),
                    "xqxs_resid_recomputed_scale": rel(x_deq, xf),
                }
                records.append(rec)
            del Wb, W_deq, ck_q, ck_s, Wck
        print(f"layer {layer:2d} done  ({time.time() - t_start:6.1f}s, "
              f"{len(records)} instances)")

    # ---------------- aggregation ----------------
    def agg(vals):
        v = np.asarray(vals, dtype=np.float64)
        return {
            "n": int(v.size),
            "mean": float(v.mean()),
            "p95": float(np.percentile(v, 95)),
            "max": float(v.max()),
        }

    METRICS = [
        "err_fp16cast", "err_gptq_total", "err_ck_total", "err_prod_total",
        "leg_gptq", "leg_ck_requant", "leg_act_quant",
        "werr_gptq", "werr_full_chain", "werr_ck_requant",
        "xqxs_resid_disk_scale", "xqxs_resid_recomputed_scale",
    ]

    per_family = {}
    fams = sorted({r["family"] for r in records})
    good = [r for r in records if r["trusted"]]
    for fam in fams:
        sub = [r for r in good if r["family"] == fam]
        per_family[fam] = {m: agg([r[m] for r in sub]) for m in METRICS}

    def third(layer):
        return ("early", "mid", "late")[min(2, layer * 3 // NUM_LAYERS)]

    per_family_third = {}
    for fam in fams:
        per_family_third[fam] = {}
        for th in ("early", "mid", "late"):
            sub = [r for r in good if r["family"] == fam and third(r["layer"]) == th]
            if sub:
                per_family_third[fam][th] = {m: agg([r[m] for r in sub]) for m in METRICS}

    worst = sorted(good, key=lambda r: -r["err_prod_total"])[:15]

    result = {
        "meta": {
            "script": "scripts/quant_replay_gemms.py",
            "quant_checkpoint": QDIR,
            "bf16_ref": BDIR,
            "recorded": BOOT,
            "tp": TP,
            "instances_per_module": MAX_INSTANCES,
            "num_records": len(records),
            "num_trusted": len(good),
            "num_excluded": len(excluded),
            "trusted_layers": sorted({r["layer"] for r in good}),
            "layers_covered": sorted({r["layer"] for r in records}),
            "time_budget_truncated": skipped_time,
            "elapsed_s": time.time() - t_start,
            "note": (
                "Legs: A=bf16 golden; B=weight fp16 cast; C=GPTQ dequant(fp32); "
                "D=CK per-channel requant of fp16(dequant); E=production "
                "((xq*s_x)@(ck_q.T))*ck_s with recorded xq and s_x recomputed "
                "from recorded x. rel-L2 vs A. Instances failing the recording-"
                "integrity gate (recorded xs inconsistent with x by >5%, xq or "
                "x all-zero — x and xq/xs captured from different forwards) are "
                "kept in records[] with trusted=false and excluded from "
                "aggregates."
            ),
        },
        "sanity_xq_xs": {
            "disk_scale": agg([r["xqxs_resid_disk_scale"] for r in good]),
            "recomputed_scale": agg([r["xqxs_resid_recomputed_scale"] for r in good]),
            "excluded_disk_scale": agg(
                [r["xqxs_resid_disk_scale"] for r in records if not r["trusted"]]),
        },
        "recording_integrity_exclusions": excluded,
        "per_family": per_family,
        "per_family_third": per_family_third,
        "worst_instances": worst,
        "records": records,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nwrote {OUT} ({len(records)} instances, "
          f"{time.time() - t_start:.1f}s)")

    # ---------------- compact tables ----------------
    def fm(x):
        return "nan" if (x is None or math.isnan(x)) else f"{x * 100:.3f}%"

    print("\n=== per-family mean / p95 rel-L2 output error vs bf16 golden ===")
    hdr = (f"{'family':28s} {'n':>4s} {'fp16cast':>16s} {'gptq_leg':>16s} "
           f"{'ck_req_leg':>16s} {'act_q_leg':>16s} {'TOTAL(E)':>16s}")
    print(hdr)
    for fam in fams:
        a = per_family[fam]
        print(f"{fam:28s} {a['err_prod_total']['n']:4d} "
              f"{fm(a['err_fp16cast']['mean']):>7s}/{fm(a['err_fp16cast']['p95']):>7s} "
              f"{fm(a['leg_gptq']['mean']):>7s}/{fm(a['leg_gptq']['p95']):>7s} "
              f"{fm(a['leg_ck_requant']['mean']):>7s}/{fm(a['leg_ck_requant']['p95']):>7s} "
              f"{fm(a['leg_act_quant']['mean']):>7s}/{fm(a['leg_act_quant']['p95']):>7s} "
              f"{fm(a['err_prod_total']['mean']):>7s}/{fm(a['err_prod_total']['p95']):>7s}")

    print("\n=== per-family TOTAL(E) mean% by depth third ===")
    print(f"{'family':28s} {'early':>9s} {'mid':>9s} {'late':>9s}")
    for fam in fams:
        row = per_family_third[fam]
        print(f"{fam:28s} "
              + "".join(
                  f"{fm(row[th]['err_prod_total']['mean']):>9s}" if th in row else f"{'-':>9s}"
                  for th in ("early", "mid", "late")))

    print("\n=== worst instances by production total error (E vs A) ===")
    print(f"{'layer':>5s} {'family':28s} {'inst':>4s} {'M':>3s} "
          f"{'fp16':>8s} {'gptq':>8s} {'ck_req':>8s} {'act_q':>8s} {'TOTAL':>8s}")
    for r in worst:
        print(f"{r['layer']:5d} {r['family']:28s} {r['instance']:4d} {r['m']:3d} "
              f"{fm(r['err_fp16cast']):>8s} {fm(r['leg_gptq']):>8s} "
              f"{fm(r['leg_ck_requant']):>8s} {fm(r['leg_act_quant']):>8s} "
              f"{fm(r['err_prod_total']):>8s}")

    print("\n=== worst instances by weight-domain full-chain error ===")
    wworst = sorted(good, key=lambda r: -r["werr_full_chain"])[:10]
    print(f"{'layer':>5s} {'family':28s} {'gptq_W':>8s} {'ck_W':>8s} {'full_W':>8s}")
    for r in wworst:
        print(f"{r['layer']:5d} {r['family']:28s} {fm(r['werr_gptq']):>8s} "
              f"{fm(r['werr_ck_requant']):>8s} {fm(r['werr_full_chain']):>8s}")

    s = result["sanity_xq_xs"]
    print("\n=== sanity: recorded xq*scale vs recorded x (rel-L2, trusted only) ===")
    print(f"disk fp16 xs:      mean {fm(s['disk_scale']['mean'])}  "
          f"p95 {fm(s['disk_scale']['p95'])}  max {fm(s['disk_scale']['max'])}")
    print(f"recomputed s_x:    mean {fm(s['recomputed_scale']['mean'])}  "
          f"p95 {fm(s['recomputed_scale']['p95'])}  max {fm(s['recomputed_scale']['max'])}")
    ex = result["sanity_xq_xs"]["excluded_disk_scale"]
    print(f"excluded instances disk-xs residual: mean {fm(ex['mean'])} "
          f"p95 {fm(ex['p95'])} max {fm(ex['max'])}  "
          f"(recorder captured x and xq/xs from different forwards)")
    from collections import Counter
    print("exclusion reasons:", dict(Counter(e["reason"] for e in excluded)))


if __name__ == "__main__":
    main()
