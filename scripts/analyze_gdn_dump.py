#!/usr/bin/env python3
"""Analyze VLLM_GDN_DUMP_DIR dumps: check the spec-rewind anchor invariant.

For each backend file (one GDN backend instance), consecutive spec steps
k -> k+1 (single request per step, cusl=[0,14]):
  - within-row distinctness of the 14 per-token state rows
  - anchor invariant: row_{k+1}[A_{k+1}-1] == row_k[A_{k+1}-1], where
    A_{k+1} = na of step k+1 (tokens accepted from step k's proposals).
    A violation means the next round resumes from a row that the previous
    round never wrote at that column -> stale/foreign recurrent state.
  - full-row stability across steps (expected under 'none'-mode reuse).
"""
import glob
import os
import sys
from collections import Counter

d = sys.argv[1] if len(sys.argv) > 1 else "logs/garble/gdn_dump"
files = sorted(glob.glob(os.path.join(d, "gdn_backend_*.log")))
tot_anchor_bad = tot_steps = 0
for f in files:
    rows = []
    with open(f) as fh:
        for line in fh:
            idx_part = line.split("idx=")[1].split(" na=")[0]
            na_part = line.split("na=")[1].split(" cusl=")[0]
            try:
                idx = [int(x) for x in idx_part.strip("[] ").split(",")]
                na = [int(x) for x in na_part.strip("[] ").split(",")]
            except ValueError:
                continue
            if len(idx) == 14:
                rows.append((idx, na))
    if len(rows) < 2:
        continue
    dup_steps = [(i, idx) for i, (idx, _) in enumerate(rows)
                 if len(set(idx)) != 14]
    anchor_bad = []
    unstable = []
    for k in range(len(rows) - 1):
        idx_k, _ = rows[k]
        idx_j, na_j = rows[k + 1]
        A = na_j[0]
        if not (1 <= A <= 14):
            anchor_bad.append((k, A, "A out of range"))
            continue
        if idx_j[A - 1] != idx_k[A - 1]:
            anchor_bad.append((k, A, idx_k[A - 1], idx_j[A - 1]))
        if idx_j != idx_k:
            unstable.append(k)
    tot_anchor_bad += len(anchor_bad)
    tot_steps += len(rows) - 1
    name = os.path.basename(f)
    flag = " <-- ANCHOR VIOLATIONS" if anchor_bad else ""
    print(f"{name}: steps={len(rows)} dup_rows={len(dup_steps)} "
          f"row_unstable={len(unstable)} anchor_bad={len(anchor_bad)}{flag}")
    # Aliased-column steps: which columns collide, and what A follows.
    for i, idx in dup_steps[:4]:
        from collections import Counter as C
        cc = {v: c for v, c in C(idx).items() if c > 1}
        nxt = rows[i + 1][1][0] if i + 1 < len(rows) else None
        print(f"    aliased step {i}: collisions={cc} next_A={nxt}")
    for v in anchor_bad[:5]:
        print(f"    step {v[0]}: A={v[1]} detail={v[2:]}")
print(f"\nTOTAL across {len(files)} backends: {tot_anchor_bad} anchor "
      f"violations in {tot_steps} transitions")
