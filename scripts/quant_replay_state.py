#!/usr/bin/env python3
"""State + custom-AR quant-error budget for the gfx908 W8A8 recipe (CPU-only).

Replays recorded boot artifacts from the quant-audit dir ($QUANT_AUDIT_DIR,
default ~/.cache/int8-vllm/kld/quant_audit):
  bootA  -- production boot (fp16 + int8 per-token-head KV cache)
  bootB  -- bf16 reference boot (same hooks; no kv-quant files, auto kv dtype)

Families analyzed:
  (a) kv_*__{k,ks,v,vs}.pt   int8 KV storage quant error (simulated exactly as
      triton_reshape_and_cache_flash_per_token_head_quant does:
      s = max(amax|k_head|/127, 1e-6) in fp32, round-half-away, clamp(-128,127)).
      Cross-instance scale-cache membership verifies the scale model.
  (b) gdn_*__h.pt            GDN recurrent state drift bootA vs bootB per layer
      (magnitude/spread ratios, per-head L2 drift, heavy-tail kurtosis flag).
  (c) ar_g4_*__{partial,out}.pt  custom all-reduce fp16 rounding damage:
      fp32 golden sum of the 4 recorded per-rank partials vs recorded out.
      Ranks record events independently, so candidate combos across instance
      indices are aligned by minimal residual (accepted below 1e-2).

Writes <audit-dir>/replay/state_ar_budget.json and prints
compact ranked tables. Run:
  CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/quant_replay_state.py
"""

import argparse
import itertools
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import torch

AUDIT_DIR = Path(
    os.environ.get(
        "QUANT_AUDIT_DIR", Path.home() / ".cache" / "int8-vllm" / "kld" / "quant_audit"
    )
)
RANKS = (0, 1, 2, 3)
AR_ALIGN_TOL = 1e-2
SCALE_MATCH_RTOL = 2e-3
HEAVY_TAIL_KURT = 100.0

KV_RE = re.compile(r"^kv_(?P<lay>.+)\.self_attn\.attn_(?P<inst>\d+)__(?P<suf>k|ks|v|vs)\.pt$")
GDN_RE = re.compile(r"^gdn_(?P<lay>.+)\.linear_attn_s(?P<step>\d+)_(?P<inst>\d+)__h\.pt$")
AR_RE = re.compile(r"^ar_g4_n(?P<n>\d+)_(?P<inst>\d+)__(?P<suf>partial|out)\.pt$")


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def stats(vals):
    if vals.numel() == 0:
        return {"mean": None, "p95": None, "max": None, "n": 0}
    return {
        "mean": float(vals.mean()),
        "p95": float(torch.quantile(vals, 0.95)) if vals.numel() > 1 else float(vals[0]),
        "max": float(vals.max()),
        "n": int(vals.numel()),
    }


def round_half_away(x):
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))


def kv_quant_err(x):
    """Per-(token, head) relative L2 error of the int8 per-token-head roundtrip.

    Replicates _reshape_cache_per_token_head: fp32 math on fp16 inputs,
    scale = max(amax/127, 1e-6), round-half-away-from-zero, clamp(-128, 127).
    """
    xf = x.float()
    s = torch.maximum(xf.abs().amax(dim=-1, keepdim=True) / 127.0, torch.tensor(1e-6))
    q = round_half_away(xf / s).clamp(-128, 127)
    deq = q * s
    denom = xf.norm(dim=-1)
    rel = (deq - xf).norm(dim=-1) / denom.clamp(min=1e-12)
    rel = torch.where(denom > 0, rel, torch.zeros_like(rel))
    return rel, s


def sim_scales(x):
    return torch.maximum(x.float().abs().amax(dim=-1) / 127.0, torch.tensor(1e-6)).flatten()


def scale_match(vals, cache):
    c = cache.float().flatten()
    c = c[torch.isfinite(c) & (c > 0)]
    if c.numel() == 0 or vals.numel() == 0:
        return None
    cs = torch.sort(c).values
    v = vals.double()
    idx = torch.clamp(torch.searchsorted(cs.double(), v), 1, cs.numel() - 1)
    d = torch.minimum((v - cs.double()[idx - 1]).abs(), (v - cs.double()[idx]).abs())
    d = d / v.clamp(min=1e-12)
    return float((d < SCALE_MATCH_RTOL).float().mean())


# ---------------------------------------------------------------- KV -------
def analyze_kv(boot_dir):
    per_layer = defaultdict(lambda: {"k": [], "v": []})
    files = {}
    for r in RANKS:
        d = boot_dir / f"rank{r}"
        if not d.is_dir():
            continue
        for f in os.listdir(d):
            m = KV_RE.match(f)
            if m:
                files[(r, m["lay"], int(m["inst"]), m["suf"])] = d / f
    layers = sorted({(lay, inst) for (_, lay, inst, _) in files})

    all_k, all_v = [], []
    for r in RANKS:
        for lay, inst in layers:
            kp = files.get((r, lay, inst, "k"))
            vp = files.get((r, lay, inst, "v"))
            if kp is None or vp is None:
                continue
            ek, _ = kv_quant_err(load(kp))
            ev, _ = kv_quant_err(load(vp))
            per_layer[lay]["k"].append(ek.flatten())
            per_layer[lay]["v"].append(ev.flatten())
            all_k.append(ek.flatten())
            all_v.append(ev.flatten())

    # Scale-model verification: instance i's simulated scales must appear in
    # instance i+1's scale-cache snapshot (the cache is captured pre-write, so
    # same-instance membership is impossible by construction).
    ver_k, ver_v = [], []
    main_layers = sorted({lay for lay in per_layer if lay.startswith("language_model")})
    aux_layers = sorted({lay for lay in per_layer if not lay.startswith("language_model")})
    sample = main_layers[::3] + aux_layers[::2]
    for lay in sample:
        for inst in (0, 1):
            kp = files.get((0, lay, inst, "k"))
            ksp = files.get((0, lay, inst + 1, "ks"))
            vp = files.get((0, lay, inst, "v"))
            vsp = files.get((0, lay, inst + 1, "vs"))
            if kp is None or ksp is None:
                continue
            mk = scale_match(sim_scales(load(kp)), load(ksp))
            if mk is not None:
                ver_k.append(mk)
            if vp is None or vsp is None:
                continue
            mv = scale_match(sim_scales(load(vp)), load(vsp))
            if mv is not None:
                ver_v.append(mv)

    layer_rows = []
    for lay in sorted(per_layer, key=lambda l: (not l.startswith("language_model"), l)):
        kcat = torch.cat(per_layer[lay]["k"])
        vcat = torch.cat(per_layer[lay]["v"])
        grp = "main_full_attn" if lay.startswith("language_model") else "aux_layers_64_68"
        layer_rows.append(
            {
                "layer": lay,
                "group": grp,
                "k_mean": float(kcat.mean()),
                "k_p95": float(torch.quantile(kcat, 0.95)),
                "v_mean": float(vcat.mean()),
                "v_p95": float(torch.quantile(vcat, 0.95)),
                "n_heads": int(kcat.numel()),
            }
        )
    result = {
        "overall": {"k": stats(torch.cat(all_k)), "v": stats(torch.cat(all_v))},
        "per_layer": layer_rows,
        "scale_verification": {
            "match_rate_k": sum(ver_k) / len(ver_k) if ver_k else None,
            "match_rate_v": sum(ver_v) / len(ver_v) if ver_v else None,
            "n_layer_instances_checked": len(ver_k),
            "rtol": SCALE_MATCH_RTOL,
            "note": (
                "ks/vs snapshots are taken before the kernel writes the current "
                "call's scales, so instance-i scales are searched in instance-i+1."
            ),
        },
    }
    return result


# -------------------------------------------------------------- GDN --------
def analyze_gdn(boot_a, boot_b):
    def collect(boot_dir):
        out = defaultdict(list)  # layer -> [(inst, rank, tensor)]
        for r in RANKS:
            d = boot_dir / f"rank{r}"
            if not d.is_dir():
                continue
            for f in os.listdir(d):
                m = GDN_RE.match(f)
                if m:
                    out[m["lay"]].append((int(m["inst"]), r, d / f))
        return out

    A, B = collect(boot_a), collect(boot_b)

    def layer_num(lay):
        m = re.search(r"layers\.(\d+)$", lay)
        return int(m.group(1)) if m else -1

    common = sorted(set(A) & set(B), key=layer_num)

    drifts_all, rows = [], []
    blowups = []
    for lay in common:
        a = {(i, r): load(p) for i, r, p in A[lay]}
        b = {(i, r): load(p) for i, r, p in B[lay]}
        keys = sorted(set(a) & set(b))
        if not keys:
            continue
        drifts, heads = [], []
        std_a, std_b, nrm_a, nrm_b, mx_a, mx_b, kurt_a = [], [], [], [], [], [], []
        n_zero, n_mismatch = 0, 0
        for k in keys:
            ha, hb = a[k].float(), b[k].float()
            na, nb = ha.norm(), hb.norm()
            if na == 0 or nb == 0:
                # Structurally empty state (sequence without prefill state /
                # dummy pass) recorded as all-zero in one of the boots.
                n_zero += 1
                continue
            nrm_a.append(na)
            nrm_b.append(nb)
            std_a.append(ha.std())
            std_b.append(hb.std())
            mx_a.append(ha.abs().max())
            mx_b.append(hb.abs().max())
            c = ha.flatten()
            std = c.std()
            m = c.mean()
            kurt_a.append(float((c - m).pow(4).mean() / std.pow(4) - 3.0) if std > 0 else 0.0)
            ratio = (na / nb).item()
            if ratio > 10 or ratio < 0.1:
                # Magnitude blow-up (or collapse) in one boot: not a comparable
                # sequence pair; flagged separately from the quant drift.
                blowups.append(
                    {
                        "layer": layer_num(lay),
                        "inst": k[0],
                        "rank": k[1],
                        "norm_A": float(na),
                        "norm_B": float(nb),
                        "ratio": ratio,
                    }
                )
                n_mismatch += 1
                continue
            drifts.append((ha - hb).norm() / nb)
            heads.append(((ha - hb).norm(dim=(-2, -1)) / hb.norm(dim=(-2, -1))).flatten())
        mag_ratio = float(torch.stack(nrm_a).mean() / torch.stack(nrm_b).mean()) if nrm_a else None
        spread_ratio = float(torch.stack(std_a).mean() / torch.stack(std_b).mean()) if std_a else None
        kurt = float(torch.tensor(kurt_a).mean()) if kurt_a else None
        row = {
            "layer": layer_num(lay),
            "layer_name": lay,
            "norm_ratio_A_over_B": mag_ratio,
            "std_ratio_A_over_B": spread_ratio,
            "absmax_A": float(torch.stack(mx_a).max()) if mx_a else None,
            "absmax_B": float(torch.stack(mx_b).max()) if mx_b else None,
            "kurtosis_excess_bootA": kurt,
            "heavy_tail": bool(kurt is not None and kurt > HEAVY_TAIL_KURT),
            "n_zero_excluded": n_zero,
            "n_magnitude_mismatch_excluded": n_mismatch,
        }
        if drifts:
            drift = float(torch.stack(drifts).mean())
            head_drift = torch.cat(heads)
            row.update(
                {
                    "elementwise_drift_norm_matched_pairs": drift,
                    "drift_p95_head": float(torch.quantile(head_drift, 0.95)),
                    "drift_max_head": float(head_drift.max()),
                    "n_norm_matched_pairs": len(drifts),
                }
            )
            drifts_all.append(drift)
        rows.append(row)

    rows.sort(key=lambda r: r["layer"])
    dt = torch.tensor(drifts_all) if drifts_all else torch.tensor([])
    thirds = {}
    for name, sl in (("early_1_21", slice(0, 16)), ("mid_22_42", slice(16, 32)), ("late_43_64", slice(32, 48))):
        if dt.numel() > sl.start:
            thirds[name] = float(dt[sl].mean())
    valid_rows = [r for r in rows if r.get("elementwise_drift_norm_matched_pairs") is not None]
    return {
        "overall": stats(dt),
        "per_layer": rows,
        "depth_thirds_mean_drift": thirds,
        "worst_layers": sorted(
            valid_rows, key=lambda r: -r["elementwise_drift_norm_matched_pairs"]
        )[:5],
        "magnitude_blowups": sorted(blowups, key=lambda b: -b["ratio"])[:10],
        "n_blowup_instances": len(blowups),
        "n_layers": len(rows),
        "n_layers_with_norm_matched_pairs": len(valid_rows),
        "note": (
            "Elementwise drift uses only pairs with norm ratio in [0.1, 10] "
            "(same-sequence proxy); even those retain trajectory divergence and "
            "cross-boot capture misalignment, so elementwise drift is an upper "
            "bound on quantization-induced state error, not a clean budget "
            "number. The distributional stats (norm_ratio/std_ratio/kurtosis) "
            "and the blow-up asymmetry are the robust bootA-vs-bootB signals."
        ),
    }


# --------------------------------------------------------------- AR --------
def analyze_ar(boot_dir):
    per_rank = defaultdict(dict)  # rank -> (n, inst) -> {"partial": path, "out": path}
    for r in RANKS:
        d = boot_dir / f"rank{r}"
        if not d.is_dir():
            continue
        for f in os.listdir(d):
            m = AR_RE.match(f)
            if m:
                per_rank[r].setdefault((int(m["n"]), int(m["inst"])), {})[m["suf"]] = d / f

    events = sorted(set(per_rank.get(0, {})) )
    rows, per_numel = [], defaultdict(list)
    n_unaligned, n_garbage = 0, 0
    nan_frac_parts_all, nan_out_all = [], []
    for ev in events:
        out_p = per_rank[0].get(ev, {}).get("out")
        if out_p is None:
            continue
        out = load(out_p).float()
        n = ev[0]
        # Candidate partials for the same message size on every rank.
        cands = {}
        for r in RANKS[1:]:
            cands[r] = [
                load(p).float()
                for (nn, _), suf in per_rank[r].items()
                if nn == n and "partial" in suf
                for p in [suf["partial"]]
            ]
        p0 = per_rank[0].get(ev, {}).get("partial")
        if p0 is None:
            continue
        p0 = load(p0).float()
        cand_lists = [cands.get(r, []) for r in RANKS[1:]]
        if any(len(cl) == 0 for cl in cand_lists):
            n_unaligned += 1
            continue

        best = None
        for combo in itertools.product(*cand_lists):
            parts = [p0, *combo]
            gold = parts[0]
            for p in parts[1:]:
                gold = gold + p
            mask = torch.isfinite(gold) & torch.isfinite(out)
            if mask.float().mean() < 0.5:
                continue
            e = ((out - gold)[mask].norm() / gold[mask].norm()).item()
            if best is None or e < best[0]:
                best = (e, parts, gold, mask)
        if best is None:
            n_garbage += 1
            continue
        e, parts, gold, mask = best
        nan_frac_parts_all.append(float(torch.stack([~torch.isfinite(p) for p in parts]).float().mean()))
        nan_out_all.append(float((~torch.isfinite(out)).float().mean()))
        if not (e < AR_ALIGN_TOL):  # also rejects NaN residuals
            n_unaligned += 1
            continue

        # fp16 sequential-sum comparator (what naive fp16 accumulation gives).
        seq = parts[0].half()
        for p in parts[1:]:
            seq = (seq.float() + p).half()
        e_seq = ((seq.float() - gold)[mask].norm() / gold[mask].norm()).item()
        e_seq = e_seq if math.isfinite(e_seq) else None
        bitwise = float((out == gold.half()).float().mean().item())
        rows.append(
            {
                "numel": n,
                "inst": ev[1],
                "relerr": e,
                "fp16_seq_relerr": e_seq,
                "bitwise_equal_fp32acc_fraction": bitwise,
                "finite_fraction": float(mask.float().mean()),
            }
        )
        per_numel[n].append(e)

    rel = torch.tensor([r["relerr"] for r in rows]) if rows else torch.tensor([])
    bucket_rows = [
        {
            "numel": n,
            "tokens": n // 5120,
            "n_aligned": len(v),
            "relerr_mean": sum(v) / len(v),
            "relerr_max": max(v),
        }
        for n, v in sorted(per_numel.items())
    ]
    return {
        "events_seen": len(events),
        "aligned": len(rows),
        "unaligned_or_shifted": n_unaligned,
        "garbage_nonfinite": n_garbage,
        "relerr": stats(rel),
        "fp16_seq_relerr_mean": (
            sum(r["fp16_seq_relerr"] for r in rows if r["fp16_seq_relerr"] is not None)
            / max(1, sum(1 for r in rows if r["fp16_seq_relerr"] is not None))
            if rows
            else None
        ),
        "bitwise_fp32acc_fraction_mean": (
            sum(r["bitwise_equal_fp32acc_fraction"] for r in rows) / len(rows) if rows else None
        ),
        "nonfinite_frac_partial_mean": (
            sum(nan_frac_parts_all) / len(nan_frac_parts_all) if nan_frac_parts_all else None
        ),
        "nonfinite_frac_out_mean": (
            sum(nan_out_all) / len(nan_out_all) if nan_out_all else None
        ),
        "per_numel": bucket_rows,
        "events": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-dir", default=str(AUDIT_DIR))
    ap.add_argument("--out", default=str(AUDIT_DIR / "replay" / "state_ar_budget.json"))
    args = ap.parse_args()
    audit = Path(args.audit_dir)
    boot_a, boot_b = audit / "bootA", audit / "bootB"

    print("== (a) KV int8 per-token-head storage quant (bootA) ==", flush=True)
    kv = analyze_kv(boot_a)
    print(
        f"overall relerr/head  k: mean {kv['overall']['k']['mean']:.4%}"
        f"  p95 {kv['overall']['k']['p95']:.4%}  max {kv['overall']['k']['max']:.4%}"
        f"  (n={kv['overall']['k']['n']})"
    )
    print(
        f"                     v: mean {kv['overall']['v']['mean']:.4%}"
        f"  p95 {kv['overall']['v']['p95']:.4%}  max {kv['overall']['v']['max']:.4%}"
        f"  (n={kv['overall']['v']['n']})"
    )
    v = kv["scale_verification"]
    print(
        f"scale-model verification (cross-instance cache membership):"
        f" k {v['match_rate_k']:.1%}  v {v['match_rate_v']:.1%}"
        f"  over {v['n_layer_instances_checked']} layer-instances @rtol {SCALE_MATCH_RTOL}"
    )
    worst = sorted(kv["per_layer"], key=lambda r: -(r["k_mean"] + r["v_mean"]))[:5]
    print("worst layers by mean(k+v):")
    for r in worst:
        print(
            f"  {r['layer']:<46} k {r['k_mean']:.4%}  v {r['v_mean']:.4%}"
            f"  [{r['group']}]"
        )

    print("\n== (b) GDN state drift bootA vs bootB ==", flush=True)
    gdn = analyze_gdn(boot_a, boot_b)
    o = gdn["overall"]
    if o["mean"] is not None:
        print(
            f"elementwise drift (norm-matched pairs, upper bound incl. trajectory"
            f" divergence): mean {o['mean']:.2%}  p95 {o['p95']:.2%}  max {o['max']:.2%}"
            f"  over {gdn['n_layers_with_norm_matched_pairs']}/{gdn['n_layers']} layers"
            f"  ({gdn['n_blowup_instances']} blow-up/mismatched instances excluded)"
        )
    else:
        print("elementwise drift: no valid (nonzero, magnitude-matched) pairs")
    for name, val in gdn["depth_thirds_mean_drift"].items():
        print(f"  depth {name}: mean drift {val:.2%}")
    print("worst drift layers (norm-matched pairs):")
    for r in gdn["worst_layers"]:
        print(
            f"  layer {r['layer']:>2}  drift {r['elementwise_drift_norm_matched_pairs']:.2%}"
            f"  max-head {r['drift_max_head']:.2%}"
            f"  normA/normB {r['norm_ratio_A_over_B']:.3f}"
            f"  stdA/stdB {r['std_ratio_A_over_B']:.3f}"
            f"  kurtA {r['kurtosis_excess_bootA']:.1f}"
        )
    n_ht = sum(1 for r in gdn["per_layer"] if r.get("heavy_tail"))
    print(f"heavy-tail layers (excess kurtosis > {HEAVY_TAIL_KURT:.0f}): {n_ht}/{gdn['n_layers']}")
    if gdn["magnitude_blowups"]:
        print("largest magnitude blow-ups (bootA vs bootB, mismatched pairs):")
        for b in gdn["magnitude_blowups"][:5]:
            print(
                f"  layer {b['layer']:>2} inst {b['inst']} rank {b['rank']}"
                f"  |hA| {b['norm_A']:.1f} vs |hB| {b['norm_B']:.1f}"
                f"  ratio {b['ratio']:.0f}x"
            )

    print("\n== (c) custom-AR fp16 rounding damage ==", flush=True)
    ar = {}
    for name in ("bootA", "bootB"):
        ar[name] = analyze_ar(audit / name)
        r = ar[name]
        line = (
            f"{name}: aligned {r['aligned']}/{r['events_seen']}"
            f" (shifted {r['unaligned_or_shifted']}, nonfinite {r['garbage_nonfinite']})"
        )
        if r["relerr"]["mean"] is not None:
            line += (
                f"  relerr mean {r['relerr']['mean']:.2e}"
                f"  p95 {r['relerr']['p95']:.2e}  max {r['relerr']['max']:.2e}"
                f"  fp16-seq {r['fp16_seq_relerr_mean']:.2e}"
                f"  bitwise-fp32acc {r['bitwise_fp32acc_fraction_mean']:.1%}"
            )
        nf = r["nonfinite_frac_partial_mean"]
        if nf is not None:
            line += f"  nonfinite partials {nf:.1%}"
        print(line)
        if name == "bootB":
            print(
                "  note: bootB AR buffers are bf16 (fp16-cast on disk), so its "
                "relerr is dominated by bf16 output rounding (~2^-9), not AR "
                "damage; bootA buffers were already fp16 (exact on disk)."
            )
        for b in ar[name]["per_numel"][:6]:
            print(
                f"    n={b['numel']:>6} ({b['tokens']:>2} tok x5120)"
                f"  events {b['n_aligned']}  relerr mean {b['relerr_mean']:.2e}"
                f"  max {b['relerr_max']:.2e}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "bootA": str(boot_a),
            "bootB": str(boot_b),
            "note": (
                "Recorder casts all float tensors to fp16 on disk. bootB ran "
                "--kv-cache-dtype auto, so no kv-quant artifacts exist for it. "
                "KV error replicates triton_reshape_and_cache_flash_per_token_head_quant "
                "(fp32 scale max(amax/127,1e-6), round-half-away, clamp -128..127). "
                "AR events are aligned across ranks by minimal residual over "
                f"instance-index combos; accepted below {AR_ALIGN_TOL}."
            ),
            "ar_align_tol": AR_ALIGN_TOL,
        },
        "kv_storage_quant": kv,
        "gdn_state_drift": gdn,
        "ar_fp16": ar,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
