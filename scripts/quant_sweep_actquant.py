#!/usr/bin/env python3
"""Offline sweep of per-token int8 activation-quant variants on recorded GEMM inputs.

Reads bootA rank0 recorded x (trusted instances only, same gating as
quant_replay_gemms.py), replays each quantizer variant against the recorded
CK weights (rebuilt from the gs128 checkpoint), and reports rel-L2 error of
the GEMM output vs the BF16 golden x @ W_bf16.T.

Variants:
  trunc        production aiter pertoken_quant (absmax/127, trunc toward zero)
  round        same scale, round-half-to-even
  clip99.0     scale from 99.0th-pct absmax (x1.0), clamp to [-127,127], round
  clip99.5     99.5th pct
  clip99.9     99.9th pct
  smoothA0.5   SmoothQuant-style per-input-channel migration alpha=0.5 folded
               into the weight (x/s quantized per-token absmax+round; weights
               rebuilt with s folded then CK-requantized)
  smoothA0.7   alpha=0.7
  smoothA0.85  alpha=0.85

Outputs $QUANT_AUDIT_OUT (default
~/.cache/int8-vllm/kld/quant_audit/replay/actquant_sweep.json) + ranked table.
CPU-only.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
CKPT = Path(
    os.environ.get(
        "MODEL_DIR",
        Path.home() / ".cache" / "int8-vllm" / "models" / "Qwen3.8-27B-GPTQ-8bit-gs128",
    )
)
REF = Path(
    os.environ.get(
        "BF16_REF_DIR",
        Path.home() / ".cache" / "int8-vllm" / "models" / "Qwen3.8-27B-bf16-ref",
    )
)
BOOT = Path(
    os.environ.get(
        "QUANT_AUDIT_BOOT",
        Path.home() / ".cache" / "int8-vllm" / "kld" / "quant_audit" / "bootA" / "rank0",
    )
)
OUT = Path(
    os.environ.get(
        "QUANT_AUDIT_OUT",
        Path.home() / ".cache" / "int8-vllm" / "kld" / "quant_audit" / "replay" / "actquant_sweep.json",
    )
)

FAMILIES = (
    "mlp.gate_up_proj",
    "mlp.down_proj",
    "linear_attn.in_proj_qkvz",
    "linear_attn.out_proj",
    "self_attn.qkv_proj",
    "self_attn.o_proj",
)
MAX_INST = 2
MAX_LAYERS_PER_FAMILY = 24  # even sampling


def load_shards(root: Path):
    files = sorted(root.glob("*.safetensors"))
    from safetensors import safe_open

    readers = [safe_open(str(f), framework="pt", device="cpu") for f in files]
    keys = []
    for r in readers:
        keys.extend(r.keys())
    return readers, set(keys)


def get_tensor(readers, keys, name):
    for r in readers:
        if name in r.keys():
            return r.get_tensor(name)
    raise KeyError(name)


def unpack_w(qweight, scales):
    K4, N = qweight.shape
    K = K4 * 4
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32).view(1, 4, 1)
    w = (qweight.unsqueeze(1).to(torch.int32) >> shifts) & 0xFF
    w = w.reshape(K, N).to(torch.int16) - 128
    w = w.t().contiguous()  # [N,K] int8 range
    s = scales.t().contiguous()  # [N, K//gs]
    return w, s


def dequant(w, s):
    N, K = w.shape
    gs = K // s.shape[1]
    return (w.float().view(N, s.shape[1], gs) * s.float().unsqueeze(-1)).reshape(N, K)


def ck_requant(W16):
    dense = W16.float()
    wmax = dense.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    s = (wmax / 127.0).float()
    q = (dense / s).round().clamp(-127, 127).to(torch.int8)
    return q, s


def pertoken(x, mode, pct=100.0):
    """x [M,K] fp16 -> q int8, s fp32 [M,1]."""
    xf = x.float()
    a = xf.abs()
    if pct >= 100.0:
        amax = a.amax(dim=1, keepdim=True)
    else:
        k = a.shape[1]
        # torch.quantile caps input size at 16M; rows here are <= 64 x <= 17408
        thr = torch.quantile(a, pct / 100.0, dim=1, keepdim=True)
        amax = thr.clamp(min=1e-12)
    s = (amax / 127.0).clamp(min=1e-12)
    v = xf / s
    if mode == "trunc":
        q = v.trunc()
    elif mode == "round":
        q = v.round()
    else:
        raise ValueError(mode)
    return q.clamp(-127, 127).to(torch.int8), s


def relerr(y, yref):
    return ((y - yref).norm() / yref.norm().clamp(min=1e-12)).item()


def smooth_scales(x, W_absmax_k, alpha):
    """s_j = (max_t |x_tj|)^alpha / (max_n |W_nj|)^(1-alpha), normalized."""
    xa = x.float().abs().amax(dim=0).clamp(min=1e-8)  # [K]
    wa = W_absmax_k.clamp(min=1e-8)  # [K]
    s = (xa**alpha) * (wa ** (alpha - 1.0))
    s = s / s.mean()  # keep scale neutral
    return s


def main():
    torch.manual_seed(0)
    ck_readers, ck_keys = load_shards(CKPT)
    ref_readers, ref_keys = load_shards(REF)

    files = sorted(BOOT.glob("gemm_*__x.pt"))
    pat = re.compile(r"gemm_(.+)\.layers\.(\d+)\.(.+)_N(\d+)_K(\d+)_(\d+)__(x|xq|xs)\.pt")
    instances = defaultdict(list)
    for f in files:
        m = pat.match(f.name)
        if not m:
            continue
        prefix, layer, fam, N, K, idx = m.group(1), int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5)), int(m.group(6))
        instances[(f"{prefix}.layers.{layer}.{fam}", fam, layer, N, K)].append(idx)

    # even layer sampling per family
    by_fam = defaultdict(list)
    for (mod, fam, layer, N, K), idxs in instances.items():
        by_fam[fam].append((fam, layer, mod, N, K, sorted(idxs)[:MAX_INST]))
    sel = []
    for fam, lst in by_fam.items():
        lst.sort()
        step = max(1, len(lst) // MAX_LAYERS_PER_FAMILY)
        sel.extend(lst[::step])

    results = defaultdict(lambda: defaultdict(list))
    n_used = 0
    for fam, layer, mod, N, K, idxs in sel:
        # recorded tensors
        xs_ok = []
        for idx in idxs:
            fx = BOOT / f"gemm_{mod}_N{N}_K{K}_{idx}__x.pt"
            fq = BOOT / f"gemm_{mod}_N{N}_K{K}_{idx}__xq.pt"
            fs = BOOT / f"gemm_{mod}_N{N}_K{K}_{idx}__xs.pt"
            if not (fx.exists() and fq.exists() and fs.exists()):
                continue
            x = torch.load(fx, map_location="cpu")
            if x.dim() != 2 or x.shape[1] != K or not torch.isfinite(x.float()).all():
                continue
            if float(x.float().abs().max()) == 0.0:
                continue
            # gate: recorded xs (fp16-rounded) vs recomputed absmax scale
            s_rec = torch.load(fs, map_location="cpu").float().flatten()
            s_rec = s_rec[: x.shape[0]]
            s_cmp = x.float().abs().amax(dim=1) / 127.0
            ok = s_cmp.clamp(min=1e-9)
            ratio = (s_rec.clamp(min=1e-9) / ok)
            if ((ratio > 0.95) & (ratio < 1.05)).float().mean() < 0.9:
                continue
            xs_ok.append(x)
        if not xs_ok:
            continue

        # weights: gs128 checkpoint + bf16 ref, rank0 slice
        short = mod.replace("language_model.model.", "model.language_model.")
        # figure out checkpoint suffix segments for merged projections
        fam_map = {
            "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
            "mlp.down_proj": ("mlp.down_proj",),
            "linear_attn.in_proj_qkvz": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
            "linear_attn.out_proj": ("linear_attn.out_proj",),
            "self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
            "self_attn.o_proj": ("self_attn.o_proj",),
        }
        Wq_parts, Ws_parts, Wref_parts = [], [], []
        colpar = fam in ("mlp.gate_up_proj", "linear_attn.in_proj_qkvz", "self_attn.qkv_proj")
        ok = True
        for sub in fam_map[fam]:
            base = short.rsplit(".", 2)[0]  # strip fam tail (2 components)
            name = f"{base}.{sub}"
            if not any(f"{name}.qweight" in k for k in ck_keys):
                base2 = short.rsplit(".", 1)[0]
                name = f"{base2}.{sub}"
            try:
                qw = get_tensor(ck_readers, ck_keys, f"{name}.qweight")
                sc = get_tensor(ck_readers, ck_keys, f"{name}.scales")
                rw = get_tensor(ref_readers, ref_keys, f"{name}.weight")
            except KeyError:
                ok = False
                break
            w, s = unpack_w(qw, sc)  # [Nfull_part, K]
            if colpar:
                n_loc = w.shape[0] // 4
                Wq_parts.append(w[:n_loc])
                Ws_parts.append(s[:n_loc])
                Wref_parts.append(rw.float()[:n_loc])
            else:
                k_loc = (w.shape[1] // 4 // 128) * 128
                Wq_parts.append(w[:, :k_loc])
                Ws_parts.append(s[:, : k_loc // 128])
                Wref_parts.append(rw.float()[:, :k_loc])
        if not ok or not Wq_parts:
            continue

        Wq = torch.cat(Wq_parts, dim=0)  # [N_loc, K]
        Ws = torch.cat(Ws_parts, dim=0)
        W_ref = torch.cat(Wref_parts, dim=0)
        built = tuple(Wq.shape)
        if (N not in (0, built[0]) or K != built[1]):
            print(f"SKIP {mod}: built {built} vs recorded N{N} K{K}; parts="
                  + ",".join(str(tuple(p.shape)) for p in Wq_parts))
            continue
        if N == 0:
            print(f"NOTE {mod}: recorded N0, using built {built}")

        W_deq = dequant(Wq, Ws)
        W16 = W_deq.to(torch.float16)
        ck_q, ck_s = ck_requant(W16)

        for x in xs_ok:
            x = x[:, : Wq.shape[1]].contiguous()
            y_ref = x.float() @ W_ref.t()
            n_used += 1
            for variant, mode, pct in (
                ("trunc", "trunc", 100.0),
                ("round", "round", 100.0),
                ("clip99.9", "round", 99.9),
                ("clip99.5", "round", 99.5),
                ("clip99.0", "round", 99.0),
            ):
                q, s = pertoken(x, mode, pct)
                y = ((q.float() * s) @ ck_q.float().t()) * ck_s.view(1, -1)
                results[variant][fam].append(relerr(y, y_ref))
            # smooth variants: fold s into weight, rebuild CK requant
            W_absmax_k = W_deq.abs().amax(dim=0)
            for alpha in (0.5, 0.7, 0.85):
                sv = smooth_scales(x, W_absmax_k, alpha)
                xs = x.float() / sv
                xs16 = xs.to(torch.float16)
                Ws_fold = (W_deq / sv.unsqueeze(0)).to(torch.float16)
                ck_q2, ck_s2 = ck_requant(Ws_fold)
                q2, s2 = pertoken(xs16, "round", 100.0)
                y2 = ((q2.float() * s2) @ ck_q2.float().t()) * ck_s2.view(1, -1)
                results[f"smoothA{alpha}"][fam].append(relerr(y2, y_ref))

    summary = {}
    for variant, fams in results.items():
        allv = [v for lst in fams.values() for v in lst]
        summary[variant] = {
            "mean": sum(allv) / len(allv),
            "p95": sorted(allv)[math.floor(0.95 * len(allv))],
            "by_family": {f: sum(v) / len(v) for f, v in sorted(fams.items())},
            "n": len(allv),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"instances used: {n_used}")
    print(f"{'variant':<12} {'mean':>8} {'p95':>8}   worst-family mean")
    for variant, s in sorted(summary.items(), key=lambda kv: kv[1]["mean"]):
        wf = max(s["by_family"].items(), key=lambda kv: kv[1])
        print(f"{variant:<12} {s['mean']*100:7.3f}% {s['p95']*100:7.3f}%   {wf[0]} {wf[1]*100:.2f}%")


if __name__ == "__main__":
    main()
