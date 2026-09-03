#!/usr/bin/env python3
"""Restore-invariant test (pass 101 surface b, phase 2).

For each request (rs) and decode round N (p_ring ordinal == kvline step n):
the running mamba block content at pre[N+1] must equal the checkpoint-window
column written by round N's forward for the committed position k_N. If the
rollback/restore reads the wrong column, or the checkpoint content is
corrupt, the invariant breaks. Empirically identify the correct column
offset (k-1 / k / k+1) on healthy rounds, then flag exceptions and check
whether they cluster at the garble-onset round (p_ring top1[0] collapse).

Also: colseq deltas split at request boundaries (pos0 resets) — any negative
intra-request state_idx delta is a bookkeeping bug; across-boundary deltas
are rs-slot recycling (expected).

Usage: analyze_kvline2b.py [kvline_dir] [p_ring_dump]
"""
import glob
import json
import math
import os
import pickle
import re
import sys
from collections import defaultdict

kvdir = sys.argv[1] if len(sys.argv) > 1 else "logs/garble/kvline"
pring = sys.argv[2] if len(sys.argv) > 2 else None
if pring is None:
    c = sorted(glob.glob(os.path.join(kvdir, "..", "p_ring",
                                      "p_ring_*.dump")))
    pring = c[0] if c else None
print(f"kvline dir: {kvdir}\np_ring: {pring}")

# ---- load p_ring (rank 0): rounds[rs] = list of (n, pos0, k, top1_0) ----
rounds = defaultdict(list)
if pring and os.path.exists(pring):
    with open(pring, "rb") as fh:
        i = 0
        while True:
            try:
                r = pickle.load(fh)
            except EOFError:
                break
            except Exception:
                i += 1
                continue
            rounds[r["rs"]].append(
                (i, r["pos0"], r["n"], r["top1"][0] if r["top1"] else 1.0)
            )
            i += 1
onset = {}
for rs, lst in rounds.items():
    low = [x for x in lst if x[3] < 0.15]
    onset[rs] = low[0][0] if low else None
    print(f"  p_ring rs={rs} rounds={len(lst)} onset_n="
          f"{onset[rs]} last_pos0={lst[-1][1]}")

# k committed per round n (for invariant joins)
kof = {}   # (rs, n) -> k
pos0of = {}
for rs, lst in rounds.items():
    for n, pos0, k, t0 in lst:
        kof[(rs, n)] = k
        pos0of[(rs, n)] = pos0

# ---- load kvline ----
files = sorted(glob.glob(os.path.join(kvdir, "kvline_*.jsonl")))
pre_rows = defaultdict(dict)   # (pid, rs, layer, n) -> (v, ksum) running
win_post = defaultdict(dict)   # (pid, rs, layer, n) -> {col: ksum}
for f in files:
    pid = int(re.search(r"kvline_(\d+)", f).group(1))
    bykl = defaultdict(list)
    with open(f) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "#st" not in r["layer"]:
                continue
            key = (r["rs"], r["layer"])
            if r["phase"] == "pre":
                bykl[(key, r["n"])].append((r["v"], r["k"]))
                pre_rows.setdefault((pid, key[0], key[1]), {})
            else:
                win_post.setdefault((pid, key[0], key[1]), {})
    # fold: running = min-v at pre
    for pid_ in [pid]:
        pass
    for (key, n), lst in bykl.items():
        v, k = min(lst)
        pre_rows[(pid, key[0], key[1])][n] = (v, k)
    del bykl
    # re-parse post rows (second pass over same file, keep memory sane)
    with open(f) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "#st" not in r["layer"] or r["phase"] != "post":
                continue
            d = win_post[(pid, r["rs"], r["layer"])].setdefault(n if False else r["n"], {})
            d[int(r["v"])] = r["k"]

# layers that have window coverage (multiple cols at post)
win_layers = set()
for (pid, rs, layer), nd in win_post.items():
    for n, d in list(nd.items())[:5]:
        if len(d) > 2:
            win_layers.add(layer)
print(f"\nwindow-covered layers: {sorted(win_layers)}")

# ---- invariant ----
print("\n== RESTORE INVARIANT (run_pre[N+1] == win_post[N][col_N + off]) ==")
stats = defaultdict(int)
viol = []
examples = defaultdict(list)
for (pid, rs, layer), m in pre_rows.items():
    if layer not in win_layers:
        continue
    ns = sorted(m)
    for i in range(len(ns) - 1):
        n, n2 = ns[i], ns[i + 1]
        if n2 != n + 1:
            continue  # gap (flush truncation)
        k = kof.get((rs, n))
        k2 = kof.get((rs, n2))
        if not k or not k2:
            continue
        # request continuity: pos0 advances by committed
        if pos0of.get((rs, n2), -1) != pos0of.get((rs, n), -2) + k:
            stats["BOUNDARY"] += 1
            continue
        colN, ksumN = m[n]
        colN2, ksumN2 = m[n2]
        if colN2 != colN:
            stats["COLCROSS"] += 1
            continue
        wd = win_post.get((pid, rs, layer), {}).get(n, {})
        if not wd:
            stats["NOWIN"] += 1
            continue
        cands = {}
        for off in (k - 1, k, k + 1):
            c = int(colN) + off
            if c in wd:
                cands[off] = wd[c]
        if not cands:
            stats["NOCAND"] += 1
            continue
        def _eq(a, b):
            return (a != a and b != b) or a == b
        hit = [off for off, val in cands.items() if _eq(val, ksumN2)]
        if len(hit) == 1:
            stats[f"MATCH@{hit[0]-k:+d}"] += 1
        elif len(hit) > 1:
            stats["MATCH@ambig"] += 1
        else:
            # no candidate matches — violation
            stats["VIOLATION"] += 1
            viol.append((pid, rs, layer, n2, ksumN2, dict(cands)))
            if len(examples[(rs, n2)]) < 3:
                examples[(rs, n2)].append(
                    (pid, layer, ksumN2, cands))
for s in sorted(stats):
    print(f"  {s}: {stats[s]}")

# ---- onset correlation ----
print("\n== ONSET CORRELATION ==")
byround = defaultdict(list)
for pid, rs, layer, n2, ksumN2, cands in viol:
    byround[(rs, n2)].append((pid, layer))
for (rs, n2), lst in sorted(byround.items()):
    on = onset.get(rs)
    near = on is not None and abs(n2 - on) <= 5
    print(f"  VIOL rs={rs} n={n2} layers={len(lst)} onset={on} "
          f"{'<<< NEAR ONSET' if near else ''}")
    for pid, layer in lst[:5]:
        print(f"    pid={pid} {layer}")
