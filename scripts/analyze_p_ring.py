#!/usr/bin/env python3
"""Analyze VLLM_P_RING dumps: per-round committed-token probabilities.

Each entry: {"rs": req_state, "pos0": anchor position, "n": committed count,
"tok": [tokens], "p": [p under the truncated verify distribution],
"top1": [row top-1 prob]}.

Finds repetition walls (>=15 consecutive rounds committing the same token at
n==1) and reports the p/top1 statistics inside them vs outside them:
  p ~ NaN        => all-NaN residual/logits tie-break (numeric defect)
  p ~ 1.0        => honest target attractor (state/value question)
  p << 1 (0.3-0.6) => resample returns low-p tokens (sampler defect)
"""
import glob
import math
import pickle
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "logs/garble/p_ring"
files = sorted(glob.glob(d + "/p_ring_*.dump"))
assert files, "no p_ring dumps"

# Merge workers; per req_state keep rounds ordered by pos0 (unique per req).
rounds = defaultdict(list)
for f in files:
    with open(f, "rb") as fh:
        while True:
            try:
                e = pickle.load(fh)
            except EOFError:
                break
            rounds[e["rs"]].append(e)

for rs, rs_rounds in sorted(rounds.items()):
    rs_rounds.sort(key=lambda e: e["pos0"])
    # drop warmup/synthetic request states: heuristics — real requests have
    # monotone pos0 with steps <= 14 and > 50 rounds
    if len(rs_rounds) < 50:
        continue
    wall_idx = [
        i
        for i, e in enumerate(rs_rounds)
        if e["n"] == 1 and len(e["tok"]) == 1
    ]
    # runs of identical committed tokens at n==1
    walls = []
    i = 0
    while i < len(rs_rounds):
        e = rs_rounds[i]
        if e["n"] == 1 and len(e["tok"]) == 1:
            t = e["tok"][0]
            j = i
            while (
                j + 1 < len(rs_rounds)
                and rs_rounds[j + 1]["n"] == 1
                and rs_rounds[j + 1]["tok"] == [t]
            ):
                j += 1
            if j - i + 1 >= 15:
                walls.append((i, j, t))
            i = j + 1
        else:
            i += 1
    print(f"req_state {rs}: {len(rs_rounds)} rounds, walls: "
          f"{[(a, b, b - a + 1, t) for a, b, t in walls]}")
    for a, b, t in walls:
        inside_p, inside_top1 = [], []
        for e in rs_rounds[a : b + 1]:
            inside_p.append(e["p"][0])
            inside_top1.append(e["top1"][0])
        nan_p = sum(1 for x in inside_p if math.isnan(x))
        nan_t = sum(1 for x in inside_top1 if math.isnan(x))
        fin_p = [x for x in inside_p if not math.isnan(x)]
        fin_t = [x for x in inside_top1 if not math.isnan(x)]
        n_in = len(inside_p)
        print(f"  wall rounds {a}..{b} token {t} ({n_in} rounds):")
        print(f"    p(committed):  NaN {nan_p}/{n_in}; finite: "
              f"min={min(fin_p) if fin_p else None} "
              f"med={sorted(fin_p)[len(fin_p)//2] if fin_p else None} "
              f"max={max(fin_p) if fin_p else None}")
        print(f"    top1:          NaN {nan_t}/{n_in}; finite: "
              f"min={min(fin_t) if fin_t else None} "
              f"med={sorted(fin_t)[len(fin_t)//2] if fin_t else None} "
              f"max={max(fin_t) if fin_t else None}")
        # entry phase: the 15 rounds before the wall
        pre = rs_rounds[max(0, a - 15) : a]
        print(f"    entry ({len(pre)} rounds before):")
        for e in pre[-8:]:
            print(f"      pos0={e['pos0']} n={e['n']} tok={e['tok'][:4]} "
                  f"p={[round(x,3) if not math.isnan(x) else float('nan') for x in e['p'][:4]]} "
                  f"top1={[round(x,3) if not math.isnan(x) else float('nan') for x in e['top1'][:4]]}")
