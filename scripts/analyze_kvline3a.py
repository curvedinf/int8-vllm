#!/usr/bin/env python3
"""Attention-KV tail analysis (pass 103 surface a) from kvline v3 rows.

Rows: {"phase","n","rs","layer"(...#kv0/#kv1),"col"=nct(pre only),"rel",
"slot","k","ri","T"(pre only)} — 32-token KV blocks around the request tail.

Tests per (pid, rs, layer-kv), decode rounds (T<=14):
  COVERAGE   blocks covering query span [nct, nct+T) MUST change pre->post
             (missed change = int8-PTH write missed the slot; next round
             reads stale/rejected-draft KV there).
  CTXINT     blocks fully below nct MUST NOT change pre->post (wild writes
             into history).
  XSTABLE    committed-position blocks: post[N] == pre[N+1] at same slot
             (something overwrote committed KV between rounds if not).

Usage: analyze_kvline3a.py [kvline3_dir]
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "logs/garble/kvline3"
files = sorted(glob.glob(os.path.join(d, "kvl3_*.jsonl")))
if not files:
    print("no files")
    sys.exit(1)

pre = defaultdict(dict)   # (pid,rs,layer) -> n -> (nct, T, {bc:(slot,k)})
post = defaultdict(dict)  # (pid,rs,layer) -> n -> {bc:(slot,k)}
for f in files:
    pid = int(re.search(r"kvl3_(\d+)", f).group(1))
    with open(f) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "#kv" not in r["layer"]:
                continue
            key = (pid, r["rs"], r["layer"])
            if r["phase"] == "pre":
                e = pre[key].setdefault(r["n"], [r.get("col", 0), r.get("T", 0), {}])
                e[2][r.get("bc", r["rel"])] = (r["slot"], r["k"])
            else:
                post[key].setdefault(r["n"], {})[r.get("bc", r["rel"])] = (
                    r["slot"], r["k"])
print(f"kv rows: {sum(len(v) for v in pre.values())} pre-rounds over "
      f"{len(pre)} (pid,rs,layer) keys")

stats = defaultdict(int)
events = defaultdict(list)


def _eq(a, b):
    return a == b or (a != a and b != b)


for key, pren in pre.items():
    postn = post.get(key, {})
    ns = sorted(n for n, e in pren.items() if e[1] <= 14)
    for n in ns:
        nct, T, rels = pren[n]
        po = postn.get(n)
        if not po or n not in po:
            stats["NOPOST"] += 1
            continue
        for bc, (slot, k_pre) in rels.items():
            if bc not in po[n]:
                stats["NOREL-POST"] += 1
                continue
            slot2, k_post = po[n][bc]
            if slot2 != slot:
                stats["SLOT-SWAP"] += 1
                continue
            base = bc * 1728
            covers_query = (base + 1728 > nct) and (base < nct + T)
            below_ctx = base + 1728 <= nct
            if covers_query:
                if _eq(k_pre, k_post):
                    stats["COVERAGE-MISS"] += 1
                    events["COVERAGE-MISS"].append(
                        (key[0], key[1], key[2], n, nct, T, bc, slot,
                         k_pre, k_post))
                else:
                    stats["COVERAGE-OK"] += 1
            elif below_ctx:
                if _eq(k_pre, k_post):
                    stats["CTX-OK"] += 1
                else:
                    stats["CTX-VIOLATION"] += 1
                    events["CTX-VIOLATION"].append(
                        (key[0], key[1], key[2], n, nct, T, bc, slot,
                         k_pre, k_post))
        # cross-round stability: committed span [nct, nct2)
        n2 = n + 1
        if n2 in pren and n2 in postn:
            nct2 = pren[n2][0]
            committed = nct2 - nct
            if 0 < committed <= 14:
                for bc, (slot, k_post) in po[n].items():
                    base = bc * 1728
                    if base + 1728 <= nct or base >= nct2:
                        continue  # not covering committed span
                    pre2 = pren[n2][2]
                    m = [x for x, (s2, _) in pre2.items() if s2 == slot]
                    if not m:
                        stats["XSLOT-GONE"] += 1
                        continue
                    k_pre2 = pre2[m[0]][1]
                    if not _eq(k_post, k_pre2):
                        stats["XS-VIOLATION"] += 1
                        events["XS-VIOLATION"].append(
                            (key[0], key[1], key[2], n, nct, nct2, bc,
                             slot, k_post, k_pre2))
                    else:
                        stats["XS-OK"] += 1

print("\n== ATTN-KV TAIL ==")
for s in sorted(stats):
    print(f"  {s}: {stats[s]}")
for evk in ("COVERAGE-MISS", "CTX-VIOLATION", "XS-VIOLATION", "SLOT-SWAP"):
    lst = events.get(evk, [])
    print(f"\n== {evk} (first 15 of {len(lst)}) ==")
    for e in lst[:15]:
        print("   ", e)
    # rounds histogram
    if lst:
        rounds = defaultdict(int)
        for e in lst:
            rounds[e[1]] += 1
        top = sorted(rounds.items(), key=lambda x: -x[1])[:8]
        print(f"   by rs: {dict(top)}")
