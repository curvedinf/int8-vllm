#!/usr/bin/env python3
"""Layer-probe analysis: verify-path vs prefill-path hidden-state divergence.

Pairs (position, layer) records between the verify leg dump (spec row 0s)
and the clean-prefill replay dump (dense rows), same TP rank, and reports
per-layer 1-cosine of the 16-dim random projections. The first layer class
to diverge localizes the corruption's entry point.

Usage: python scripts/analyze_layerprobe.py <verify_pt> <prefill_pt> \
    [--min-pos 20040]
"""
import argparse

import torch
from collections import defaultdict
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verify_pt")
    ap.add_argument("prefill_pt")
    ap.add_argument("--min-pos", type=int, default=20040)
    args = ap.parse_args()

    ver = torch.load(args.verify_pt, weights_only=False)
    pre = torch.load(args.prefill_pt, weights_only=False)

    ver_by = defaultdict(list)
    for pos, layer, proj in ver:
        if pos >= args.min_pos:
            ver_by[(pos, layer)].append(proj)
    pre_by = defaultdict(list)
    for pos, layer, proj in pre:
        if pos >= args.min_pos:
            pre_by[(pos, layer)].append(proj)

    deltas = defaultdict(list)
    for key, vps in ver_by.items():
        pps = pre_by.get(key)
        if not pps:
            continue
        a = torch.tensor(vps[0])
        b = torch.tensor(pps[-1])
        cos = torch.dot(a, b) / (a.norm() * b.norm()).clamp(min=1e-9)
        deltas[key[1]].append(1 - cos.item())

    full_attn = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63}
    rows = [(l, statistics.mean(v), len(v), "ATTN" if l in full_attn else "GDN")
            for l, v in deltas.items()]
    rows.sort()
    total = sum(n for _, _, n, _ in rows) // max(len(rows), 1)
    print(f"paired positions per layer ≈ {total}")
    print("layer  class  mean(1-cos)")
    for l, m, n, c in rows:
        print(f"  {l:2d}   {c}   {m:.4f}  (n={n})")
    gdn = [m for _, m, _, c in rows if c == "GDN"]
    attn = [m for _, m, _, c in rows if c == "ATTN"]
    if gdn and attn:
        print(f"class means: GDN={statistics.mean(gdn):.4f} "
              f"ATTN={statistics.mean(attn):.4f}")


if __name__ == "__main__":
    main()
