#!/usr/bin/env python3
"""Raw-vs-processed-vs-echo split at corrections (pring_leg3 chain).

Reads /tmp/logs/garble/pring_ab dumps (records with raw_* fields), chains
against the leg's committed stream, joins with the echo comparison, and
splits the resample-row disagreement into FORWARD vs POST-PROCESSING:

  raw_p ~ echo_p, processed p != echo  -> sampler post-processing bug
  raw_p != echo_p                      -> verify forward inputs wrong
"""
import glob
import json
import math
import pickle

import torch

LEG = "pring_leg3"

com = torch.load(f"logs/garble/{LEG}_committed.pt", map_location="cpu",
                 weights_only=False)
committed = [int(x) for x in com["committed_ids"]]

recs = []
for path in sorted(glob.glob("/tmp/logs/garble/pring_ab/p_ring_*.dump")):
    with open(path, "rb") as f:
        while True:
            try:
                recs.append(pickle.load(f))
            except EOFError:
                break
uniq = {(r["pos0"], tuple(r["tok"])): r for r in recs
        if "raw_p" in r and r["pos0"] > 20000}
chain = sorted(uniq.values(), key=lambda r: r["pos0"])
cursor = 1  # skip the template header token (760)
walk = []
for r in chain:
    if cursor >= len(committed):
        break
    t = r["tok"]
    if committed[cursor:cursor + len(t)] == t:
        walk.append((cursor, r))
        cursor += len(t)
print(f"chained {len(walk)} rounds, {cursor}/{len(committed)} tokens; "
      f"raw records present: {'raw_p' in (walk[0][1] if walk else {})}")

echo = json.load(open(f"logs/garble/lp_echo_{LEG}.json"))
diff_at = {r[0]: (r[1], r[2]) for r in echo["rows"]}

print(f"\n{'pos':>5} {'n':>2} {'proc_p':>8} {'raw_p':>8} {'echo_p':>9} "
      f"{'|log raw-echo|':>14} verdict")
agg = {"forward": 0, "processing": 0, "both_ok": 0, "n": 0}
for start, r in walk:
    n = r["n"]
    i = start + n - 1
    if i not in diff_at:
        continue
    eng, echo_lp = diff_at[i]
    raw_p = r.get("raw_p")
    proc_p = r["p"][n - 1]
    if raw_p is None or raw_p <= 0:
        continue
    echo_p = math.exp(echo_lp)
    d_re = abs(math.log(raw_p) - echo_lp)
    d_pe = abs(math.log(max(proc_p, 1e-12)) - echo_lp)
    if d_re < 0.3 and d_pe >= 0.3:
        v = "processing"
    elif d_re >= 0.3:
        v = "forward"
    else:
        v = "both_ok"
    agg[v] += 1
    agg["n"] += 1
    if d_pe > 0.8 and len(str(agg["n"])) <= 3 or (d_pe > 2.0 and agg["n"] < 400):
        print(f"{i:>5} {n:>2} {proc_p:8.4f} {raw_p:8.4f} {echo_p:9.5f} "
              f"{d_re:14.3f} {v}  (proc-echo {d_pe:.2f})")

print("\nAGGREGATE over corrections:", agg)
if agg["n"]:
    print(f"forward-wrong share: {agg['forward']/agg['n']:.3f}  "
          f"processing-only share: {agg['processing']/agg['n']:.3f}")
