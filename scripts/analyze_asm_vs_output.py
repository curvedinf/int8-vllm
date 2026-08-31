#!/usr/bin/env python3
"""Cross-check the assembly ring against the EMITTED token stream.

Ground truth: the API output text per request (saved by the probe harness).
For every spec round, the row-0 token fed to the target must equal the
emitted token at that round's row-0 position (positions are engine-truth
from the ring). A mismatch = the target was fed a wrong context token at a
known position — the garble mechanism itself.

Request<->output mapping: by prompt length (the corpus is identical, so
prompt_len is constant) and decode order — we simply try every saved output
against every ring request and report the best (matching-token) pairing.
"""
import glob
import pickle
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from garble_repro2 import get_tok  # noqa: E402

ring_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/garble/asm_ring"
out_glob = sys.argv[2] if len(sys.argv) > 2 else "logs/garble/ASM*.txt"
tok = get_tok()

rounds = defaultdict(list)
f = sorted(glob.glob(ring_dir + "/asm_ring_*.dump"))[0]
step = 0
with open(f, "rb") as fh:
    while True:
        try:
            e = pickle.load(fh)
        except EOFError:
            break
        ids, pos, ns = e["ids"], e["pos"], e["nsched"]
        off = 0
        for j, rid in enumerate(req_ids := e["req_ids"]):
            n = int(ns[j])
            rounds[rid].append((step, ids[off], pos[off]))
            off += n
        step += 1
print(f"{f}: {step} steps, {len(rounds)} requests")

outputs = {}
for path in sorted(glob.glob(out_glob)):
    outputs[path.split("/")[-1][:-4]] = tok.encode(open(path).read(),
                                                   add_special_tokens=False)

for rid, rs in rounds.items():
    rs.sort()
    p0 = rs[0][2]  # first decode position (≈ prompt_len)
    row0_tokens = [t for _, t, _ in rs]
    row0_pos = [p for _, _, p in rs]
    best = None
    for name, toks in outputs.items():
        m = sum(1 for t, p in zip(row0_tokens, row0_pos)
                if p - p0 < len(toks) and toks[p - p0] == t)
        if best is None or m > best[1]:
            best = (name, m)
    name, m = best
    n = min(len(row0_tokens), len(outputs[name]))
    print(f"{rid}: first_pos={p0} rounds={len(rs)} best_out={name} "
          f"row0-match {m}/{n}")
    # show first mismatches
    shown = 0
    for t, p in zip(row0_tokens, row0_pos):
        if p - p0 >= len(outputs[name]):
            continue
        if outputs[name][p - p0] != t:
            print(f"    pos={p} idx={p - p0}: fed={t} emitted={outputs[name][p - p0]}")
            shown += 1
            if shown >= 8:
                break
