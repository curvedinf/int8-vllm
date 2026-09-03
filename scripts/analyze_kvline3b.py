#!/usr/bin/env python3
"""Pre-only attention-KV slice analysis (pass 103 surface a, phase 3).

Rows are fixed 8-token-aligned absolute slices (id s covers [s*8, s*8+8))
around each round's query boundary, recorded synchronously in
preprocess_state. Join pre[n] -> pre[n+1] by slice id (identical token
ranges — exact):

  COVERAGE   slice overlapping [nct, nct+T) MUST change (the verify's
             int8-PTH write for those query positions landed there).
  CTX        slice fully below nct must NOT change (no writes into
             history — wild writes, allocator recycling, connector).
  STALE      slices in [nct+T, nct1) (empty when committed <= T) — n/a.

Usage: analyze_kvline3b.py [kvline3_dir]
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "logs/garble/kvline3"
files = sorted(glob.glob(os.path.join(d, "kvl3_*.jsonl")))

pre = defaultdict(dict)  # (pid,rs,layer) -> n -> (nct, T, {s:(slot,k)})
for f in files:
    pid = int(re.search(r"kvl3_(\d+)", f).group(1))
    with open(f) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "#kv" not in r["layer"] or r["phase"] != "pre":
                continue
            if "s" not in r:
                continue
            key = (pid, r["rs"], r["layer"])
            e = pre[key].setdefault(r["n"], [r.get("col", 0), r.get("T", 0), {}])
            e[2][r["s"]] = (r["slot"], r["k"])
print(f"pre slice-rounds: {sum(len(v) for v in pre.values())} over {len(pre)} keys")

stats = defaultdict(int)
events = defaultdict(list)


def _eq(a, b):
    return a == b or (a != a and b != b)


for key, pren in pre.items():
    ns = sorted(n for n, e in pren.items() if e[1] <= 14)
    for i in range(1, len(ns)):
        n0, n1 = ns[i - 1], ns[i]
        if n1 != n0 + 1:
            continue
        nct, T, s0 = pren[n0]
        nct1, _, s1 = pren[n1]
        if nct1 < nct or nct1 - nct > 14:
            continue
        for s, (slot, k0) in s0.items():
            if s not in s1:
                continue
            slot1, k1 = s1[s]
            if slot1 != slot:
                stats["REBIND"] += 1
                if len(events["REBIND"]) < 10:
                    events["REBIND"].append((key, n0, s, slot, slot1))
                continue
            lo, hi = s * 8, s * 8 + 8
            if hi <= nct:
                # fully below query start: no writer may touch it
                if _eq(k0, k1):
                    stats["CTX-OK"] += 1
                else:
                    stats["CTX-VIOLATION"] += 1
                    events["CTX-VIOLATION"].append(
                        (key[0], key[1], key[2], n0, nct, T, s, slot, k0, k1))
            elif lo < nct + T:
                # overlaps the query span: the write must land
                if _eq(k0, k1):
                    stats["COVERAGE-MISS"] += 1
                    events["COVERAGE-MISS"].append(
                        (key[0], key[1], key[2], n0, nct, T, s, slot, k0, k1))
                else:
                    stats["COVERAGE-OK"] += 1
            else:
                # above the query span of n0 (unwritten tail)
                stats["TAIL"] += 1

print("\n== PRE-ONLY ATTN-KV SLICES ==")
for s_ in sorted(stats):
    print(f"  {s_}: {stats[s_]}")
for evk in ("COVERAGE-MISS", "CTX-VIOLATION", "REBIND"):
    lst = events.get(evk, [])
    print(f"\n== {evk} (first 20 of {len(lst)}) ==")
    for e in lst[:20]:
        print("   ", e)
    if lst:
        by_l = defaultdict(int)
        for e in lst:
            by_l[e[2]] += 1
        print(f"   by layer: {dict(sorted(by_l.items(), key=lambda x: -x[1])[:8])}")
        by_n = defaultdict(int)
        for e in lst:
            by_n[e[3]] += 1
        print(f"   by round (top): {dict(sorted(by_n.items(), key=lambda x: -x[1])[:8])}")
