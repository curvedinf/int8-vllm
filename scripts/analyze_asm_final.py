#!/usr/bin/env python3
"""Definitive assembly audit: ring inputs vs rejection-sampler emissions.

Reconstruct each request's emitted stream from asm_out (concatenate the
per-round sampled rows, dropping -1 padding). Then, per round transition,
verify the assembly semantics:
  round k+1's row-0 token == the LAST accepted (emitted) token of round k.
Equivalently: the anchor token the target is fed for the next round must be
the token just committed. Violations at scale = the garble mechanism.
"""
import glob
import pickle
import sys
from collections import defaultdict

ring_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/garble/asm_ring"

# --- emissions: req -> [(step_seq, [tokens])]
emitted = defaultdict(list)
outs = sorted(glob.glob(ring_dir + "/asm_out_*.dump"))
assert outs, "no asm_out dumps"
n_out_steps = 0
for f in outs:
    with open(f, "rb") as fh:
        while True:
            try:
                e = pickle.load(fh)
            except EOFError:
                break
            for j, rid in enumerate(e["req_ids"]):
                row = [t for t in e["sampled"][j] if t != -1]
                emitted[rid].append(row)
            n_out_steps += 1
print(f"emissions: {n_out_steps} rounds over {len(emitted)} requests")

# --- inputs: req -> [(step, row0_token)]
rounds = defaultdict(list)
f = sorted(glob.glob(ring_dir + "/asm_ring_*.dump"))[0]
step = 0
with open(f, "rb") as fh:
    while True:
        try:
            e = pickle.load(fh)
        except EOFError:
            break
        ids, ns = e["ids"], e["nsched"]
        off = 0
        for j, rid in enumerate(e["req_ids"]):
            n = int(ns[j])
            rounds[rid].append((step, ids[off]))
            off += n
        step += 1
print(f"inputs: {step} steps over {len(rounds)} requests")

for rid, rows in rounds.items():
    em = emitted.get(rid)
    if not em:
        continue
    rows.sort()
    k = min(len(rows) - 1, len(em) - 1)
    bad = shown = 0
    first_bad = None
    for i in range(k):
        # round i+1's row0 must equal round i's last emitted token
        if not em[i]:
            continue
        expect = em[i][-1]
        got = rows[i + 1][1]
        if got != expect:
            bad += 1
            if first_bad is None:
                first_bad = (rows[i + 1][0], i, expect, got)
    if bad:
        print(f"{rid}: {bad}/{k} anchor violations; first at step "
              f"{first_bad[0]} (round {first_bad[1]}): fed={first_bad[3]} "
              f"emitted_last={first_bad[2]}")
    else:
        print(f"{rid}: CLEAN ({k} transitions)")
