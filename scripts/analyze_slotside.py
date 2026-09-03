#!/usr/bin/env python3
"""SLOTSIDE analysis (pass 105): does the attention slot mapping ever
disagree with the block table for REAL query tokens?

expected slot(token at position q) = bt[r, q//1728] * 1728 + q % 1728
The recorded sm[i] corresponds to query token i at position p+i (row 0 at
p). PAD (-1) or a mismatched value on a real token = the write goes to the
wrong place (or nowhere) while every written-slot audit stays byte-exact —
the mechanism that reconciles the slice-level misses with the clean
write-site readback.

Join: tokfeed rows (pid, rs, p, sm) with kvline3 slice rows (pid, rs, bc ->
slot) for the block table.

Usage: analyze_slotside.py [tokfeed_jsonl] [kvline3_dir]
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

tf = sys.argv[1] if len(sys.argv) > 1 else "logs/garble/kvline_tokfeed.jsonl"
d3 = sys.argv[2] if len(sys.argv) > 2 else "logs/garble/kvline3"

# block table per (pid, rs, leg): {bc: slot} — from slice rows (latest wins)
# legs segmented by p reset within (pid, rs).
rows_tf = []
with open(tf) as f:
    for line in f:
        rows_tf.append(json.loads(line))
rows_tf = [r for r in rows_tf if r["p"] >= 40000]

bt = defaultdict(lambda: defaultdict(dict))  # (pid,rs) -> leg_idx -> {bc: slot}
chains = defaultdict(list)
for r in rows_tf:
    chains[(r["pid"], r["rs"])].append(r)
leg_of = {}  # (pid, rs, n) -> leg idx
for key, lst in chains.items():
    lst.sort(key=lambda r: r["n"])
    leg = -1
    prev_p = None
    for r in lst:
        if prev_p is None or r["p"] < prev_p - 100:
            leg += 1
        leg_of[(key[0], key[1], r["n"])] = leg
        prev_p = r["p"]

for f in sorted(glob.glob(os.path.join(d3, "kvl3_*.jsonl"))):
    pid = int(re.search(r"kvl3_(\d+)", f).group(1))
    with open(f) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "#kv" not in r["layer"] or r["phase"] != "pre":
                continue
            li = leg_of.get((pid, r["rs"], r["n"]))
            if li is None:
                continue
            bt[(pid, r["rs"])][li][r["bc"]] = r["slot"]

stats = defaultdict(int)
ex = []
for r in rows_tf:
    key = (r["pid"], r["rs"])
    li = leg_of.get((key[0], key[1], r["n"]))
    table = bt[key].get(li) if li is not None else None
    T = r.get("T", 0)
    if T <= 0 or T > 32 or len(r.get("sm", [])) < T:
        continue
    for i in range(T):
        q = r["p"] + i
        got = r["sm"][i]
        if table is not None:
            exp_blk = table.get(q // 1728)
            exp = exp_blk * 1728 + q % 1728 if exp_blk is not None else None
        else:
            exp = None
        stats["tokens"] += 1
        if got < 0:
            stats["PAD"] += 1
            if len(ex) < 12:
                ex.append(("PAD", r["pid"], r["rs"], r["n"], q, got, exp))
        elif exp is not None and got != exp:
            stats["MISMATCH"] += 1
            if len(ex) < 12:
                ex.append(("MISM", r["pid"], r["rs"], r["n"], q, got, exp))
        elif exp is not None:
            stats["OK"] += 1
        else:
            stats["NOBT"] += 1

print(dict(stats))
for e in ex:
    print("  ", e)
