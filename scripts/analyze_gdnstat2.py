#!/usr/bin/env python3
"""Decisive analysis for the extended GDNSTAT capture (kbase + blks records).

Answers, per worker stream (each tracked the leg request at its own rs):
1. delta = kbase - col per decode round: how often the seq_lens-derived
   kernel window base disagrees with the num_computed-derived align base
   (default/benign delta is 1). Non-1 deltas mean the SSM kernel wrote its
   14 checkpoints into blocks the align machinery does not track.
2. SSM (st1) NaN at preprocess time (i.e., after the previous round's
   kernel writes): rounds/rels/layers — these should never exist.
3. Resumed-slot health: st1/st0 norm of my rel (delta + ri), the block the
   next round's kernel reads as initial state.
4. Window aliasing: duplicate block ids inside one window row.

Triage only — text verdicts come from full manual reads.

Usage: python scripts/analyze_gdnstat2.py <dir> [--min-nct 19000]
"""
import argparse
import glob
import math
import os
from collections import Counter, defaultdict

import torch


def load_streams(d):
    by_pid = {}
    for p in glob.glob(os.path.join(d, "gdnstat_*.pt")):
        pid = int(os.path.basename(p).split("_")[1])
        by_pid.setdefault(pid, []).append(p)
    streams = {}
    for pid, paths in by_pid.items():
        paths.sort(key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
        rows = []
        for p in paths:
            rows.extend(torch.load(p, weights_only=False)["rounds"])
        streams[pid] = rows
    return streams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--min-nct", type=int, default=19000)
    args = ap.parse_args()

    streams = load_streams(args.dir)
    for pid, rows in sorted(streams.items()):
        # tracked request = rs with the most rounds past min-nct
        cnt = Counter(r["rs"] for r in rows if r["nct"] >= args.min_nct)
        if not cnt:
            print(f"pid {pid}: no rounds past nct {args.min_nct}")
            continue
        rs = cnt.most_common(1)[0][0]
        leg = sorted((r for r in rows if r["rs"] == rs
                      and r["nct"] >= args.min_nct), key=lambda r: r["n"])
        deltas = Counter(r.get("kbase", -99) - r["col"] for r in leg)
        print(f"\n=== pid {pid} rs={rs}: {len(leg)} rounds "
              f"(nct {leg[0]['nct']}..{leg[-1]['nct']}) ===")
        print(f"delta=kbase-col histogram: {dict(sorted(deltas.items()))}")
        bad = [r for r in leg if r.get("kbase", -99) - r["col"] != 1]
        for r in bad[:10]:
            print(f"  delta!=1: n={r['n']} nct={r['nct']} col={r['col']} "
                  f"kbase={r['kbase']} ri={r['ri']} T={r['T']} "
                  f"seq_len={r.get('seq_len')}")

        # SSM NaN at preprocess + resumed-slot health
        ssm_nan = []
        resumed_bad = []
        aliases = 0
        for r in leg:
            delta = r.get("kbase", -99) - r["col"]
            blks = r.get("blks", [])
            if len(set(blks)) != len(blks):
                aliases += 1
            for key, norms in r["norms"].items():
                if not key.endswith("#st1"):
                    continue
                for rel, v in enumerate(norms):
                    if math.isnan(v) or math.isinf(v):
                        ssm_nan.append((r["n"], r["nct"], key, rel, delta))
                slot = delta + r["ri"] if delta > 0 else r["ri"]
                if 0 <= slot < len(norms):
                    v = norms[slot]
                    if math.isnan(v) or math.isinf(v):
                        resumed_bad.append(
                            (r["n"], r["nct"], key, r["ri"], slot))
        print(f"SSM(st1) NaN at preprocess: {len(ssm_nan)} "
              f"(rounds: {sorted({x[0] for x in ssm_nan})[:12]})")
        for x in ssm_nan[:8]:
            print(f"   n={x[0]} nct={x[1]} {x[2]} rel={x[3]} delta={x[4]}")
        print(f"resumed-slot NaN/Inf reads: {len(resumed_bad)}")
        for x in resumed_bad[:8]:
            print(f"   n={x[0]} nct={x[1]} {x[2]} ri={x[3]} slot={x[4]}")
        print(f"window aliasing rows: {aliases}")


if __name__ == "__main__":
    main()
