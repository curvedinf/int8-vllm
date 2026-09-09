#!/usr/bin/env python3
"""Pair engine GPTQ LAYERPROBE records with the bf16 reference capture.

Rank0-only shards (engine ran without sequence parallel, so only rank0's
(pos, tok) keys align with its hidden rows). For every banked record
(pos, layer, tok, proj16) we take the bf16 mlp-stream projection at the same
position/layer and accumulate per-layer error metrics over the 16-dim
projection:

  energy = sqrt(sum||pg-pb||^2 / sum||pb||^2)   (robust relative error)
  meanrel = mean(||pg-pb|| / max(||pb||, floor))

The per-layer value is CUMULATIVE at that layer boundary (error from layers
0..L). The increment table (err_L - err_{L-1}) localizes contributions.
Prefill (pos < prompt_len) and decode (pos >= prompt_len) segments are
reported separately — the garble lives in decode.

Usage: .venv/bin/python scripts/garble_rank_layers.py \
         [--gptq logs/garble/rank_gptq] [--bf16 logs/garble/rank_bf16]
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gptq", default="logs/garble/rank_gptq")
    p.add_argument("--bf16", default="logs/garble/rank_bf16")
    p.add_argument("--top_positions", type=int, default=5)
    args = p.parse_args()

    cap = torch.load(os.path.join(args.bf16, "capture.pt"),
                     map_location="cpu", weights_only=False)
    ids = cap["ids"]
    prompt_len = len(torch.load(
        "/home/curved/vllm-gfx908/logs/garble/d30_origgptq_temp1_ids.pt",
        map_location="cpu", weights_only=False)["prompt_ids"])
    n_layers = len(cap["mlp"])
    print(f"bf16 capture: {ids.numel()} tokens, prompt_len {prompt_len}, "
          f"{n_layers} layers")

    # accumulate per layer, split prefill/decode
    stats = {}  # layer -> dict(seg -> [sum_e2, sum_n2, n, tokmm, worst[(e,pos)]])
    for seg in ("prefill", "decode"):
        for L in range(n_layers):
            stats[(L, seg)] = {"e2": 0.0, "n2": 0.0, "n": 0,
                               "worst": []}

    files = sorted(glob.glob(os.path.join(args.gptq, "lp_r0_*.pt")))
    print(f"rank0 shards: {len(files)}")
    n_rec = 0
    n_drop = 0
    for f in files:
        for (pos, L, tok, proj) in torch.load(f, map_location="cpu",
                                              weights_only=False):
            if L >= n_layers or pos >= ids.numel():
                n_drop += 1
                continue
            if int(ids[pos]) != int(tok):
                # engine dummy/profiling forwards (all-zero ids) share the
                # position range with the real replay — drop them
                n_drop += 1
                continue

            pb = cap["mlp"][L][pos]
            pg = torch.tensor(proj, dtype=torch.float32)
            seg = "prefill" if pos < prompt_len else "decode"
            s = stats[(L, seg)]
            d = pg - pb
            e2 = float(d.dot(d))
            n2 = float(pb.dot(pb))
            s["e2"] += e2
            s["n2"] += n2
            s["n"] += 1
            s["worst"].append((e2 / max(n2, 1e-9), pos))
            n_rec += 1

    print(f"paired records: {n_rec}  dropped (dummy/pos-overflow): {n_drop}")

    def table(seg: str, title: str):
        rows = []
        for L in range(n_layers):
            s = stats[(L, seg)]
            if s["n"] == 0:
                continue
            energy = (s["e2"] / s["n2"]) ** 0.5 if s["n2"] > 0 else float("nan")
            rows.append((L, energy, s["n"]))
        print(f"\n=== {title} (energy = sqrt(sum|d|^2 / sum|pb|^2)) ===")
        print(f"{'L':>3} {'energy':>9} {'records':>8}")
        for L, energy, n in rows:
            print(f"{L:>3} {energy:9.4f} {n:>8}")
        return rows

    pre_rows = table("prefill", "PREFILL positions (cumulative at boundary)")
    dec_rows = table("decode", "DECODE positions (cumulative at boundary)")

    # increments: contribution localized to layer L itself
    def increments(rows, title):
        print(f"\n=== {title} ===")
        print(f"{'L':>3} {'energy':>9} {'delta':>9}")
        prev = 0.0
        out = []
        for L, energy, n in rows:
            out.append((L, energy, energy - prev))
            prev = energy
        for L, energy, dl in out:
            print(f"{L:>3} {energy:9.4f} {dl:+9.4f}")
        return out

    dec_inc = increments(dec_rows, "DECODE increments (delta = err_L - err_{L-1})")

    worst_layers = sorted(dec_rows, key=lambda r: -r[1])[:10]
    print("\n=== top-10 layers by DECODE energy ===")
    for L, energy, n in worst_layers:
        s = stats[(L, "decode")]
        s["worst"].sort(reverse=True)
        tops = ", ".join(f"p{p}({r:.2f})" for r, p in s["worst"][:args.top_positions])
        print(f"L{L:>3} energy {energy:.4f}  worst: {tops}")

    # persist machine-readable summary
    import json
    summary = {
        "n_records": n_rec,
        "dropped": n_drop,
        "prefill": {str(L): e for L, e, _ in pre_rows},
        "decode": {str(L): e for L, e, _ in dec_rows},
    }
    with open(os.path.join(args.bf16, "ranking.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nwrote {os.path.join(args.bf16, 'ranking.json')}")


if __name__ == "__main__":
    main()
