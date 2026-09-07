#!/usr/bin/env python3
"""kvline v3 lineage analysis (pass 102/103).

Per (pid, rs, layer): join pre[N] (pre-forward, post-precopy) with post[N-1]
(post-forward). Invariants for consecutive DECODE rounds (T<=14 both):
  A. slot binding: the physical block at each absolute column is unchanged
     between post[N-1] and pre[N] (no allocator re-binding mid-window).
  B. read lineage: pre[N] content at rel=read_idx (the block the GDN kernel
     reads as initial state) == post[N-1] content at the same absolute
     column (SSM copies byte-exact). At crossings (col_N == col_{N-1}+1,
     ri==0) the correct source is post[N-1] at rel = k_prev-1 (the precopy
     sources bt[src_col + token_bias]); report which rel actually matches
     and cross-check k_prev against p_ring.

Usage: analyze_kvline3.py [kvline3_dir] [p_ring_dump]
"""
import glob
import json
import os
import pickle
import re
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "logs/garble/kvline3"
pring = sys.argv[2] if len(sys.argv) > 2 else None
if pring is None:
    c = sorted(glob.glob("logs/garble/p_ring/p_ring_*.dump"))
    pring = c[0] if c else None

# ---- load p_ring: rounds[rs] = list of k (committed), ordinal order ----
prk = defaultdict(list)
if pring and os.path.exists(pring):
    with open(pring, "rb") as fh:
        while True:
            try:
                r = pickle.load(fh)
            except EOFError:
                break
            except Exception:
                continue
            prk[r["rs"]].append(r["n"])
    print(f"p_ring: {pring} rounds={sum(len(v) for v in prk.values())}")

# ---- load kvline3 ----
# pre[(pid,rs,layer)][n] = (col, ri, T, {abs_col: (slot,k)})
# post[(pid,rs,layer)][n] = (col, {abs_col: (slot,k)})
pre = defaultdict(dict)
post = defaultdict(dict)
files = sorted(glob.glob(os.path.join(d, "kvl3_*.jsonl")))
for f in files:
    pid = int(re.search(r"kvl3_(\d+)", f).group(1))
    with open(f) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            key = (pid, r["rs"], r["layer"])
            ac = r["col"] + r.get("rel", 0)
            if r["phase"] == "pre":
                e = pre[key].setdefault(
                    r["n"], [r["col"], r.get("ri", -1), r.get("T", 0), {}]
                )
                e[3][ac] = (r["slot"], r["k"])
            else:
                e = post[key].setdefault(r["n"], [r["col"], {}])
                e[1][ac] = (r["slot"], r["k"])
if not pre:
    print("no kvline3 files")
    sys.exit(1)
print(f"files: {[os.path.basename(x) for x in files]}")

# ---- per (rs) decode-round n list (from one layer to avoid dupes) ----
stats = defaultdict(int)
viol = []
cross_ok = 0
for key, pren in pre.items():
    pid, rs, layer = key
    postn = post.get(key, {})
    ns = sorted(n for n, e in pren.items() if e[2] <= 14)  # decode rounds
    for i in range(1, len(ns)):
        n0, n1 = ns[i - 1], ns[i]
        if n1 != n0 + 1:
            continue  # gap (flush/prefill interleave)
        e0, e1 = pren[n0], pren[n1]
        p0 = postn.get(n0)
        if not p0:
            stats["NOPOST"] += 1
            continue
        col0, col1, ri = e0[0], e1[0], e1[1]
        if ri < 0:
            stats["NORI"] += 1
            continue
        ac = col1 + ri
        if ac not in p0[1]:
            stats["READCOL-MISSING-POST"] += 1
            continue
        slot_post, k_post = p0[1][ac]
        if ac not in e1[3]:
            stats["READCOL-MISSING-PRE"] += 1
            continue
        slot_pre, k_pre = e1[3][ac]
        crossing = col1 == col0 + 1
        if slot_pre != slot_post:
            stats["SLOT-REBIND"] += 1
            viol.append(("SLOT", pid, rs, layer, n1, ac, slot_pre, slot_post))
            continue
        if k_pre == k_post or (k_pre != k_pre and k_post != k_post):
            stats["LINEAGE-OK" + ("+CROSS" if crossing else "")] += 1
            if crossing:
                cross_ok += 1
            continue
        # content mismatch: find which post rel actually matches
        match_rel = [
            acx - col0
            for acx, (s, kk) in p0[1].items()
            if (kk == k_pre or (kk != kk and k_pre != k_pre))
        ]
        stats["CONTENT-MISMATCH"] += 1
        viol.append(
            ("CONTENT", pid, rs, layer, n1, ac, k_pre, k_post, match_rel,
             col0, col1, ri)
        )

print("\n== LINEAGE ==")
for s in sorted(stats):
    print(f"  {s}: {stats[s]}")

print("\n== VIOLATIONS (first 25) ==")
for v in viol[:25]:
    if v[0] == "SLOT":
        print(f"  SLOT pid={v[1]} rs={v[2]} {v[3]} n={v[4]} abs_col={v[5]} "
              f"slot {v[7]} -> {v[6]}")
    else:
        print(f"  CONTENT pid={v[1]} rs={v[2]} {v[3]} n={v[4]} abs={v[5]} "
              f"col {v[9]}->{v[10]} ri={v[11]} k_pre={v[6]} k_post={v[7]} "
              f"match_rels={v[8]}")
if pring and prk:
    print("\n== p_ring k cross-check for crossing rounds ==")
    for key, pren in pre.items():
        pid, rs, layer = key
        ns = sorted(n for n, e in pren.items() if e[2] <= 14)
        for i in range(1, len(ns)):
            n0, n1 = ns[i - 1], ns[i]
            if n1 != n0 + 1:
                continue
            if pren[n1][0] == pren[n0][0] + 1:  # crossing
                # ordinal alignment: first decode n for this rs
                first = ns[0]
                print(f"  rs={rs} crossing n {n0}->{n1} "
                      f"(ord {n0-first} k_prev={prk[rs][n0-first] if n0-first < len(prk[rs]) else '?'})")
                break  # one per (rs,layer) is enough noise
        break  # single pid/layer for the check
