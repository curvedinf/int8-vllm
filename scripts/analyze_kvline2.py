#!/usr/bin/env python3
"""Analyze kvline v2 jsonl (attention anchors + mamba state surfaces).

Tests (pass 101 surface b):
  leak     pre[n] vs post[n] (only the draft forward runs between): any
           target-layer change = draft writing target state. Split by
           attention anchor rows vs mamba #st rows, target (idx<=63) vs
           drafter (>63) layers.
  colseq   state_idx column sequence per (pid, rs): col[n+1]-col[n]
           histogram; anything other than 0/+1 (or a reset to 0) is a
           bookkeeping anomaly.
  jump     running-block checksum deltas pre[n]->pre[n+1] per
           (pid, rs, layer): z-score outliers, NaN/inf/zero flags.
  align    (--onset STEP --rs RS): print the per-layer delta profile around
           a step of interest for cross-checking against p_ring onset.

Usage:
  analyze_kvline2.py [kvline_dir] [--onset N --rs RS]
"""
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "logs/garble/kvline"
onset = rs_of = None
if "--onset" in sys.argv:
    onset = int(sys.argv[sys.argv.index("--onset") + 1])
if "--rs" in sys.argv:
    rs_of = int(sys.argv[sys.argv.index("--rs") + 1])


def lidx(layer):
    m = re.search(r"(\d+)", layer)
    return int(m.group(1)) if m else -1


# recs[pid] = list of records in order
recs = {}
files = sorted(glob.glob(os.path.join(d, "kvline_*.jsonl")))
for f in files:
    pid = int(re.search(r"kvline_(\d+)", f).group(1))
    rs_ = []
    with open(f) as fh:
        for line in fh:
            try:
                rs_.append(json.loads(line))
            except Exception:
                pass
    recs[pid] = rs_
if not recs:
    print("no kvline files")
    sys.exit(1)
print(f"files: {[os.path.basename(x) for x in files]}")
for pid, rr in recs.items():
    print(f"  pid {pid}: {len(rr)} recs, n range "
          f"{min(x['n'] for x in rr)}..{max(x['n'] for x in rr)}")

# snap[pid][(phase, n)][(rs, layer, slot, v)] = k
snap = defaultdict(lambda: defaultdict(dict))
for pid, rr in recs.items():
    for r in rr:
        snap[pid][(r["phase"], r["n"])][
            (r["rs"], r["layer"], r["slot"], r["v"])] = r["k"]

# ---- test 1: leak (pre vs post within a step) ----
print("\n== LEAK (pre[n] vs post[n]; only draft propose runs between) ==")


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:
            return True  # nan == nan for change detection
    return a == b


leak = defaultdict(int)
leak_ex = []
ns = sorted({n for (ph, n) in snap[next(iter(snap))] if ph == "pre"})
for pid, s in snap.items():
    for n in ns:
        a, b = s.get(("pre", n)), s.get(("post", n))
        if not a or not b:
            continue
        for k4, kv in a.items():
            if k4 not in b:
                # row vanished (finished req / table shift) — count separately
                leak["VANISHED"] += 1
                continue
            if not _same(kv, b[k4]):
                rs, layer, slot, v = k4
                cat = ("mamba" if "#st" in layer else "attn")
                tgt = "target" if lidx(layer) <= 63 else "drafter"
                # NaN direction: garbage window blocks are nan in both phases;
                # only finite<->nan or finite->finite-different are real events
                if kv != kv and b[k4] != b[k4]:
                    sub = "nan->nan?"
                elif kv != kv:
                    sub = "nan->finite"
                elif b[k4] != b[k4]:
                    sub = "finite->nan"
                else:
                    sub = "finite->finite"
                leak[f"{cat}:{tgt}:{sub}"] += 1
                if (
                    sub == "finite->finite"
                    and len(leak_ex) < 12
                ):
                    leak_ex.append((pid, n, rs, layer, slot, v, kv, b[k4]))
for k in sorted(leak):
    print(f"  {k}: {leak[k]}")
for e in leak_ex:
    print(f"  EX pid={e[0]} n={e[1]} rs={e[2]} {e[3]} slot={e[4]} col={e[5]}"
          f" k {e[6]:.1f} -> {e[7]:.1f}")

# ---- test 2: state_idx column sequence ----
print("\n== COLSEQ (state_idx col per step, running rows) ==")
# running row per (pid, rs, layer, n): the min-v row among that layer's rows
run = defaultdict(dict)
for pid, s in snap.items():
    bykey = defaultdict(list)
    for (ph, n), rows in s.items():
        if ph != "pre":
            continue
        for (rs, layer, slot, v), k in rows.items():
            if "#st" in layer:
                bykey[(rs, layer, n)].append((v, slot, k))
    for (rs, layer, n), lst in bykey.items():
        v, slot, k = min(lst)
        run[(pid, rs, layer)][n] = (v, slot, k)
coldiff = defaultdict(int)
for key, m in run.items():
    nns = sorted(m)
    for i in range(1, len(nns)):
        coldiff[round(m[nns[i]][0] - m[nns[i - 1]][0])] += 1
for c in sorted(coldiff):
    print(f"  col delta {c:+}: {coldiff[c]}")

# ---- test 3: running-block checksum jumps ----
print("\n== JUMP (running-block |dk| outliers, pre[n]->pre[n+1]) ==")
# stats per (pid, rs, layer)
allj = []
nan_runs = 0
for key, m in run.items():
    nns = sorted(m)
    prev = None
    for n in nns:
        k = m[n][2]
        if k != k:
            nan_runs += 1
            prev = None
            continue
        if prev is not None:
            allj.append((abs(k - prev), key, n, None, prev, k))
        prev = k
if allj:
    js = sorted(j[0] for j in allj)
    med = js[len(js) // 2]
    p999 = js[int(len(js) * 0.999)]
    print(f"  |dk| median={med:.1f} p99.9={p999:.1f} max={js[-1]:.1f} "
          f"(nan running-block checksums: {nan_runs})")
    for j in sorted(allj, reverse=True)[:15]:
        if j[0] <= max(p999 * 3, med * 20):
            break
        print(f"  OUTLIER |dk|={j[0]:.1f} pid={j[1][0]} rs={j[1][1]} "
              f"{j[1][2]} n={j[2]} k {j[4]:.1f} -> {j[5]:.1f}")

# ---- optional: onset profile ----
if onset is not None and rs_of is not None:
    print(f"\n== ONSET profile rs={rs_of} n in [{onset - 3}, {onset + 3}] ==")
    for pid, s in snap.items():
        rows_here = [
            (n, layer, v, k)
            for (ph, n), rr in s.items()
            if ph == "pre" and abs(n - onset) <= 3
            for (rs, layer, slot, v), k in rr.items()
            if rs == rs_of and "#st" in layer
        ]
        for n, layer, v, k in sorted(rows_here)[:80]:
            print(f"  pid={pid} n={n} {layer} col={v} k={k:.1f}")
