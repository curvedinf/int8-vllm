#!/usr/bin/env python3
"""Parse p_ring_*.dump pickles; per request state index (rs), find the onset
round where the verify anchor row's top-1 collapses (garble entry signature)
and print the top-1[0] trace around it. Round ordinal should be 1:1 with the
kvline step counter n (one verify+propose per engine decode step).

Usage: pring_onset.py [p_ring_dir] [--rs RS] [--win N]
"""
import glob
import os
import pickle
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "logs/garble/p_ring"
rs_of = None
win = 8
if "--rs" in sys.argv:
    rs_of = int(sys.argv[sys.argv.index("--rs") + 1])
if "--win" in sys.argv:
    win = int(sys.argv[sys.argv.index("--win") + 1])

files = sorted(glob.glob(os.path.join(d, "p_ring_*.dump")))
if not files:
    print("no p_ring dumps")
    sys.exit(1)
for f in files:
    pid = os.path.basename(f).split("_")[-1].split(".")[0]
    per_rs = {}
    with open(f, "rb") as fh:
        i = 0
        while True:
            try:
                r = pickle.load(fh)
            except EOFError:
                break
            except Exception:
                i += 1
                continue
            per_rs.setdefault(r["rs"], []).append((i, r))
            i += 1
    print(f"== {os.path.basename(f)} (worker {pid}) ==")
    for rs, lst in sorted(per_rs.items()):
        if rs_of is not None and rs != rs_of:
            continue
        rounds = len(lst)
        t0 = [x[1]["top1"][0] if x[1]["top1"] else 1.0 for x in lst]
        low = [(i, t) for i, t in enumerate(t0) if t < 0.15]
        onset = low[0][0] if low else None
        frac = (sum(1 for t in t0 if t < 0.15) / rounds) if rounds else 0
        print(f"  rs={rs} rounds={rounds} onset={onset} "
              f"low_top1[0]_frac={frac:.3f}")
        if onset is not None:
            lo = max(0, onset - win)
            for i in range(lo, min(rounds, onset + win)):
                r = lst[i][1]
                print(f"    n={i} n_acc={r['n']} top1[0]={t0[i]:.3f} "
                      f"p[0]={r['p'][0] if r['p'] else -1:.3f} "
                      f"absmax={r['row_absmax']} nan={r['row_nan']} "
                      f"top5={r['top5']}")
