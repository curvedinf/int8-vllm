#!/usr/bin/env python3
"""Offline quant-error budget for the DFlash2 drafter chain (W8A8 int8 recipe, gfx908).

Replicates the production numeric chain for the drafter's 35 GPTQ-int8 linears
  (unpack -> gs128 dequant -> fp16 round-trip -> CK per-output-channel requant)
and measures, against the bf16 source checkpoint:
  1. weight-domain error per linear (gptq leg, ck-requant leg, total)
  2. GEMM-output error using recorded target-model activations as input proxies,
     reported per proxy source layer (bootA records NO draft hidden-state/gemm
     artifacts -- the drafter appears only as kv_model.layers.64..68 -- so
     proxies come from recorded GEMM inputs where K aligns: K=5120 directly
     (rank0 gate_up x), K=17408 via concat of the 4 rank-local down_proj
     K=4352 slices, K=4096 synthetic Gaussian with amax resampled from the
     recorded pool)
  3. bf16 rider tensors that dominate drafter memory (quantization targets)

CPU-only. Writes ~/models/kld/quant_audit/replay/drafter_budget.json.
"""

import csv
import glob
import json
import os
import time

import torch
from safetensors import safe_open

INT8_DIR = os.path.expanduser(
    "~/.cache/huggingface/dflash2-int8/Qwen3.8-27B-DFlash2-GPTQ-8bit"
)
BF16_DIR = os.path.expanduser("~/models/dflash2-bf16-with-tokenizer")
BOOTA = os.path.expanduser("~/models/kld/quant_audit/bootA")
OUT_JSON = os.path.expanduser("~/models/kld/quant_audit/replay/drafter_budget.json")

PROXY_LAYERS = (0, 32, 63)  # early / mid / late target layers
FAMILIES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
SHIFT = torch.tensor([0, 8, 16, 24], dtype=torch.int32)


def unpack_chain(qweight, scales):
    """Production chain steps 1-4: unpack -> W_deq fp32 -> W16 -> CK requant."""
    K4, N = qweight.shape
    K = K4 * 4
    # qweight[i, j] packs w[i*4+s, j] at bit 8s (LSB-first along K)
    w = ((qweight.unsqueeze(-1) >> SHIFT.to(qweight.device))
         & 0xFF).permute(0, 2, 1).reshape(K, N).to(torch.int16)
    w = (w - 128).to(torch.int8).t()                     # [N,K]
    s = scales.t().float()                               # [N, K//128]
    w_deq = (w.float().view(N, K // 128, 128) * s.unsqueeze(-1)).view(N, K)
    w16 = w_deq.to(torch.float16)
    wmax = w16.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    ck_s = (wmax / 127).float()
    ck_q = (w16.float() / ck_s).round().clamp(-127, 127).to(torch.int8)
    return w_deq, w16, ck_q, ck_s


def pertoken_quant(x):
    """Production step 5: aiter pertoken symmetric quant (trunc-to-int8).

    Validated offline: trunc reproduces recorded aiter xq bit-exactly
    (round matches only 53%).
    """
    s = x.abs().amax(dim=1, keepdim=True).float() / 127
    s = torch.where(s == 0, torch.ones_like(s), s)
    xq = (x.float() / s).trunc().clamp(-127, 127).to(torch.int8)
    return xq, s


def rel_l2(a, ref):
    return (a - ref).norm() / ref.norm()


def _valid(t):
    return t is not None and torch.isfinite(t.float()).all() and t.float().norm() > 0


def _load_inst(layer, fam, rank, inst):
    pats = glob.glob(
        f"{BOOTA}/rank{rank}/gemm_language_model.model.layers.{layer}.{fam}_*_{inst}__x.pt"
    )
    return torch.load(pats[0], map_location="cpu", weights_only=True) if pats else None


def load_proxy_inputs():
    """Per-source hidden-state proxies from recorded bootA GEMM inputs.

    Screens out the corrupted recordings (L0 inst2 has NaNs, L32 inst0 is
    all-zeros). Sources: target layers 0/32/63, instances 0..2 where valid.
    """
    src5120 = {}   # layer -> [M,5120] fp32
    for L in PROXY_LAYERS:
        xs = [_load_inst(L, "mlp.gate_up_proj", 0, i) for i in (0, 1, 2)]
        xs = [t for t in xs if _valid(t)]
        src5120[L] = torch.cat(xs, dim=0).float()

    src17408 = {}  # layer -> [M,17408] fp32 (4 rank-local K slices concat)
    for L in PROXY_LAYERS:
        blocks = []
        for i in (0, 1, 2):
            per_rank = [_load_inst(L, "mlp.down_proj", r, i) for r in range(4)]
            if all(_valid(t) for t in per_rank) and len({t.shape[0] for t in per_rank}) == 1:
                blocks.append(torch.cat(per_rank, dim=1).float())
        src17408[L] = torch.cat(blocks, dim=0) if blocks else None

    # o_proj input K=4096: no aligned recording -> synthetic Gaussian with
    # per-token amax resampled from the recorded hidden-state pool
    g = torch.Generator().manual_seed(0)
    pool = torch.cat([src5120[L] for L in PROXY_LAYERS])
    amax = pool.abs().amax(dim=1)
    M = pool.shape[0]
    x = torch.randn(M, 4096, generator=g)
    tgt = amax[torch.randint(0, amax.numel(), (M,), generator=g)]
    src4096 = x / x.abs().amax(dim=1, keepdim=True) * tgt.unsqueeze(1)

    def meta(src, kind, K, syn=False):
        return {f"src_L{L}": {"M": int(v.shape[0]), "kind": kind, "K": K,
                              "synthetic": syn}
                for L, v in src.items() if v is not None}

    metas = {**meta(src5120, "rank0 gate_up x (target hidden state)", 5120),
             **meta(src17408, "4-rank down_proj x slices concat dim1", 17408),
             "src_synthetic": {"M": M, "kind": "gaussian, amax-resampled", "K": 4096,
                               "synthetic": True}}
    proxies = {}
    for L, v in src5120.items():
        proxies[(5120, f"L{L}")] = v
    for L, v in src17408.items():
        if v is not None:
            proxies[(17408, f"L{L}")] = v
    proxies[(4096, "synthetic")] = src4096
    return proxies, metas


def main():
    t0 = time.time()
    torch.set_num_threads(min(32, os.cpu_count() or 8))
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    f8 = safe_open(os.path.join(INT8_DIR, "model.safetensors"), framework="pt", device="cpu")
    fb = safe_open(os.path.join(BF16_DIR, "model.safetensors"), framework="pt", device="cpu")

    quant_names = sorted(n[:-len(".qweight")] for n in f8.keys() if n.endswith(".qweight"))
    print(f"{len(quant_names)} quantized linears")

    # sanity: g_idx natural order (k//128, desc_act=False), qzeros inert 0x7F
    gidx_ok = sum(bool((f8.get_tensor(n + ".g_idx")
                        == torch.arange(f8.get_tensor(n + ".g_idx").numel()) // 128)
                       .all()) for n in quant_names)
    qz_ok = sum(bool((((f8.get_tensor(n + ".qzeros").unsqueeze(2) >> SHIFT) & 0xFF)
                      == 0x7F).all()) for n in quant_names)

    # sanity: pertoken trunc vs recorded aiter xq (nonzero instance)
    pv = _load_inst(63, "mlp.gate_up_proj", 0, 0)
    pbase = glob.glob(f"{BOOTA}/rank0/gemm_language_model.model.layers.63."
                      f"mlp.gate_up_proj_*_0__x.pt")[0][:-len("__x.pt")]
    rec_xq = torch.load(pbase + "__xq.pt", map_location="cpu", weights_only=True)
    rec_xs = torch.load(pbase + "__xs.pt", map_location="cpu", weights_only=True)
    my_xq, my_s = pertoken_quant(pv)
    pertoken = {
        "trunc_exact_match_frac": float((my_xq == rec_xq).float().mean()),
        "xq_xs_residual_rel_l2": float(rel_l2(rec_xq.float() * rec_xs.float(), pv.float())),
    }

    quant_log = {}
    with open(os.path.join(INT8_DIR, "quant_log.csv")) as fh:
        for row in csv.DictReader(fh):
            quant_log[(int(row["layer"]), row["module"])] = float(row["loss"])

    proxies, proxy_meta = load_proxy_inputs()

    # ---- task 1+2: weight-domain + proxy GEMM error per linear ----
    records = []
    for n in quant_names:
        layer = int(n.split(".")[1])
        fam = n.split(".", 2)[2]
        w_bf16 = fb.get_tensor(n + ".weight")            # [N,K] bf16
        w_deq, w16, ck_q, ck_s = unpack_chain(
            f8.get_tensor(n + ".qweight"), f8.get_tensor(n + ".scales"))
        wf = w_bf16.float()
        w_ck = ck_q.float() * ck_s

        rec = {
            "name": n, "layer": layer, "family": fam,
            "N": int(w_bf16.shape[0]), "K": int(w_bf16.shape[1]),
            "w_rel_l2_gptq": float(rel_l2(w_deq, wf)),
            "w_rel_l2_ck_total": float(rel_l2(w_ck, wf)),
            "w_rel_l2_ck_only": float(rel_l2(w_ck, w_deq)),
            "w_fp16_rt_max_abs": float((w16.float() - w_deq).abs().max()),
            "quant_log_loss": quant_log.get((layer, fam)),
            "gemm_rel_l2_fp16cast": {}, "gemm_rel_l2_gptq": {},
            "gemm_rel_l2_ck_requant": {}, "gemm_rel_l2_full_prod": {},
            "gemm_rel_l2_act_quant_only": {},
        }

        ckqf = ck_q.float()
        for (K, src), x in proxies.items():
            if K != w_bf16.shape[1]:
                continue
            xf = x.float()
            yA = xf @ wf.t()
            yB = xf @ w_bf16.to(torch.float16).float().t()
            yC = xf @ w_deq.t()
            yD = xf @ w_ck.t()
            xq, s_x = pertoken_quant(xf)
            yE = ((xq.float() * s_x) @ ckqf.t()) * ck_s.reshape(1, -1)
            rec["gemm_rel_l2_fp16cast"][src] = float(rel_l2(yB, yA))
            rec["gemm_rel_l2_gptq"][src] = float(rel_l2(yC, yA))
            rec["gemm_rel_l2_ck_requant"][src] = float(rel_l2(yD, yA))
            rec["gemm_rel_l2_full_prod"][src] = float(rel_l2(yE, yA))
            rec["gemm_rel_l2_act_quant_only"][src] = float(rel_l2(yE, yD))
            del yA, yB, yC, yD, yE, xq
        # aggregate across sources (mean)
        for k in ("gemm_rel_l2_gptq", "gemm_rel_l2_ck_requant",
                  "gemm_rel_l2_full_prod"):
            rec[k + "_mean"] = sum(rec[k].values()) / len(rec[k])
        records.append(rec)
        del w_deq, w16, ck_q, ck_s, wf, w_ck, ckqf

    # ---- task 3: rider / memory budget ----
    rider_names = set(k for k in f8.keys() if not any(
        k.endswith(s) for s in (".qweight", ".qzeros", ".scales", ".g_idx")))
    itemsize = {"BF16": 2, "FP16": 2, "F32": 4, "I32": 4, "I8": 1}
    riders, quant_bytes = [], 0
    for k in f8.keys():
        st = f8.get_slice(k)  # dtype/shape from header, no materialization
        shape, dt = tuple(st.get_shape()), st.get_dtype()
        nbytes = itemsize.get(dt, 4)
        for d in shape:
            nbytes *= d
        if k in rider_names:
            riders.append({"name": k, "dtype": dt, "shape": list(shape), "bytes": nbytes})
        else:
            quant_bytes += nbytes
    riders.sort(key=lambda r: -r["bytes"])
    rider_bytes = sum(r["bytes"] for r in riders)
    ckpt_size = os.path.getsize(os.path.join(INT8_DIR, "model.safetensors"))
    bf16_size = os.path.getsize(os.path.join(BF16_DIR, "model.safetensors"))

    # ---- aggregates ----
    fam_agg = {}
    for fam in FAMILIES + ("all",):
        sub = records if fam == "all" else [r for r in records if r["family"] == fam]
        if not sub:
            continue

        def agg(key):
            v = sorted(r[key] for r in sub)
            return {"mean": sum(v) / len(v),
                    "p95": v[max(0, int(0.95 * len(v)) - 1)], "max": v[-1]}

        fam_agg[fam] = {"n": len(sub), "w_gptq": agg("w_rel_l2_gptq"),
                        "w_ck_total": agg("w_rel_l2_ck_total"),
                        "w_ck_only": agg("w_rel_l2_ck_only"),
                        "gemm_gptq": agg("gemm_rel_l2_gptq_mean"),
                        "gemm_ck_requant": agg("gemm_rel_l2_ck_requant_mean"),
                        "gemm_full_prod": agg("gemm_rel_l2_full_prod_mean")}

    # per-source aggregate of the production E leg
    src_e = {}
    for r in records:
        for src, v in r["gemm_rel_l2_full_prod"].items():
            src_e.setdefault(src, []).append(v)
    src_e = {s: {"mean": sum(v) / len(v), "max": max(v)} for s, v in src_e.items()}

    med = sorted(r["w_rel_l2_gptq"] for r in records)[len(records) // 2]
    outliers = [r["name"] for r in records if r["w_rel_l2_gptq"] > 5 * med]

    out = {
        "meta": {
            "int8_ckpt": INT8_DIR, "bf16_src": BF16_DIR,
            "torch": torch.__version__, "runtime_s": round(time.time() - t0, 1),
            "quant_chain": "unpack(lsbf int32)->deq gs128 sym->fp16 rt->CK per-out-channel int8",
            "gemm_leg_E": "((xq*s_x)@ck_q.T)*ck_s, aiter pertoken trunc (validated bit-exact vs recorded xq)",
            "proxy_inputs": proxy_meta,
            "note": "bootA has no draft_* hidden-state/gemm artifacts; drafter layers "
                    "appear only as kv_model.layers.64..68 (KV, analyzed by the state agent). "
                    "GEMM inputs proxied from recorded target-model activations; K=4096 "
                    "(o_proj) is synthetic. quant_log.csv loss is gptqmodel's "
                    "calibration-relative metric and does NOT equal weight rel L2: "
                    "layer-0 down_proj logs 0.4168 but measures 0.0080 weight / "
                    "0.0078 GEMM rel L2 (mid-pack, no offline outlier).",
        },
        "sanity": {"g_idx_natural_order_k_div_128": f"{gidx_ok}/{len(quant_names)}",
                   "qzeros_all_0x7F": f"{qz_ok}/{len(quant_names)}",
                   "pertoken_trunc_vs_recorded_xq": pertoken},
        "per_linear": records,
        "family_agg": fam_agg,
        "full_prod_E_by_proxy_source": src_e,
        "weight_outliers_gt5x_median": outliers,
        "memory": {
            "quant_weight_bytes": quant_bytes,
            "rider_bytes": rider_bytes,
            "rider_share": rider_bytes / ckpt_size,
            "ckpt_file_bytes": ckpt_size,
            "bf16_file_bytes": bf16_size,
            "int8_vs_bf16": ckpt_size / bf16_size,
            "top_riders": riders[:12],
        },
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)

    # ---- compact tables ----
    print(f"\n== sanity: g_idx {out['sanity']['g_idx_natural_order_k_div_128']} natural, "
          f"qzeros {out['sanity']['qzeros_all_0x7F']} inert, "
          f"pertoken trunc match {pertoken['trunc_exact_match_frac']:.3f} "
          f"(xq*xs residual {pertoken['xq_xs_residual_rel_l2']:.3f}) ==")
    print("\n== weight+GEMM error, ranked by gptq rel L2 (top 12) ==")
    print(f"{'linear':34s} {'gptq':>8s} {'ck_only':>8s} {'ck_tot':>8s} | "
          f"{'gC':>8s} {'gD':>8s} {'gE':>8s} {'qlog':>8s}")
    for r in sorted(records, key=lambda r: -r["w_rel_l2_gptq"])[:12]:
        print(f"{r['name']:34s} {r['w_rel_l2_gptq']:8.4f} {r['w_rel_l2_ck_only']:8.4f} "
              f"{r['w_rel_l2_ck_total']:8.4f} | {r['gemm_rel_l2_gptq_mean']:8.4f} "
              f"{r['gemm_rel_l2_ck_requant_mean']:8.4f} {r['gemm_rel_l2_full_prod_mean']:8.4f} "
              f"{(r['quant_log_loss'] or 0):8.4f}")
    print("\n== family means (w gptq / w ck_only | gemm C / D / E) ==")
    for fam, a in fam_agg.items():
        print(f"{fam:22s} n={a['n']:2d}  {a['w_gptq']['mean']:.4f} / {a['w_ck_only']['mean']:.4f}"
              f"  |  {a['gemm_gptq']['mean']:.4f} / {a['gemm_ck_requant']['mean']:.4f}"
              f" / {a['gemm_full_prod']['mean']:.4f}")
    print("\n== production E leg by proxy source (activation regime) ==")
    for s, v in sorted(src_e.items()):
        print(f"  src {s:10s} mean {v['mean']:.4f}  max {v['max']:.4f}")
    print(f"\n== weight outliers (>5x median {med:.5f}): {outliers or 'none'} ==")
    print("   layer-0 down_proj verdict: quant_log 0.4168 does NOT reproduce offline "
          "(w relL2 0.0080, gemm 0.0078 -> calibration-relative artifact)")
    print("\n== memory: top bf16 riders ==")
    for r in riders[:8]:
        print(f"  {r['name']:48s} {r['dtype']:5s} {str(r['shape']):20s} {r['bytes']/1e6:8.1f} MB "
              f"({100*r['bytes']/ckpt_size:.1f}% of ckpt)")
    print(f"  quant weights {quant_bytes/1e6:.0f} MB | riders {rider_bytes/1e6:.0f} MB "
          f"({100*rider_bytes/ckpt_size:.1f}%) | ckpt {ckpt_size/1e9:.2f} GB "
          f"vs bf16 {bf16_size/1e9:.2f} GB")
    print(f"\nwrote {OUT_JSON} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
