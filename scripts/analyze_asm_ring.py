#!/usr/bin/env python3
"""Analyze VLLM_ASM_RING dumps: verify-input assembly invariants.

Per request, per consecutive round pair (k, k+1):
  pos:   round rows' positions must be contiguous within a round, and
         P_{k+1} = P_k + A_k where A_k = acceptance of round k, DERIVED
         from the position delta.
  token: round k+1's FIRST input token must equal round k's row (A_k - 1)
         token (the last committed token anchors the next round).
A token-continuity violation is the garble mechanism caught red-handed:
the target would compute logits against a wrong context token.
"""
import glob
import pickle
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "logs/garble/asm_ring"
files = sorted(glob.glob(d + "/asm_ring_*.dump"))
assert files, "no dumps"
f = files[0]
rounds = defaultdict(list)  # req_id -> [(step, ids_slice, pos_slice)]
step = 0
with open(f, "rb") as fh:
    while True:
        try:
            e = pickle.load(fh)
        except EOFError:
            break
        ids, pos, ns = e["ids"], e["pos"], e["nsched"]
        req_ids = e["req_ids"]
        off = 0
        for j, rid in enumerate(req_ids):
            n = int(ns[j])
            rounds[rid].append((step, ids[off:off + n], pos[off:off + n]))
            off += n
        step += 1
print(f"{f}: {step} steps, {len(rounds)} requests")

tot_pos_bad = tot_tok_bad = tot_trans = 0
for rid, rs in rounds.items():
    prev = None
    for (st, ids_s, pos_s) in rs:
        if pos_s and pos_s != list(range(pos_s[0], pos_s[0] + len(pos_s))):
            print(f"  {rid} step {st}: NON-CONTIGUOUS positions {pos_s[:16]}")
            tot_pos_bad += 1
        if prev is not None:
            pst, pids, ppos = prev
            A = pos_s[0] - ppos[0] if pos_s and ppos else None
            if A is not None and 1 <= A <= len(pids):
                tot_trans += 1
                if ids_s and pids[A - 1] != ids_s[0]:
                    print(f"  {rid} step {st}: TOKEN DISCONTINUITY "
                          f"next0={ids_s[0]} prev[A-1]={pids[A - 1]} "
                          f"A={A} pos_delta={A}")
                    tot_tok_bad += 1
            elif A is not None and not (1 <= A <= len(pids)):
                print(f"  {rid} step {st}: POSITION DELTA OUT OF RANGE "
                      f"A={A} prev_len={len(pids)}")
                tot_pos_bad += 1
        prev = (st, ids_s, pos_s)
print(f"\nTOTAL: {tot_pos_bad} position violations, {tot_tok_bad} token "
      f"discontinuities, over {tot_trans} transitions")

