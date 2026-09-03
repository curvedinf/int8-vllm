#!/usr/bin/env python3
"""Disambiguate KV coverage misses (pass 103 surface a, phase 4).

A slice whose checksum doesn't change across a verify round is a true
missed write ONLY if the round's query tokens at those positions differ
from the previous round's tokens there (identical tokens -> identical
int8 K bytes -> benign no-change). Reconstruct per-position tokens from
p_ring (committed tok + drafts) and keep only unambiguous misses.

Query layout per round n: query 0 (anchor) at position nct-1, drafts at
[nct, nct+13]; the previous round's writes cover its own query span.

Usage: analyze_kvline3c.py [kvline3_dir] [p_ring_dump]
"""
import glob
import json
import os
import pickle
import re
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "logs/garble/kvline3"
c = sorted(glob.glob("logs/garble/p_ring/p_ring_*.dump"))
pring = sys.argv[2] if len(sys.argv) > 2 else (c[0] if c else None)

# p_ring: per rs, ordinal -> (pos0, k, committed tokens, drafts)
pr = defaultdict(list)
with open(pring, "rb") as fh:
    while True:
        try:
            r = pickle.load(fh)
        except EOFError:
            break
        except Exception:
            continue
        drafts = r.get("drafts") or []
        # queries at [pos0-1 .. pos0-1+ (1+len(drafts)+...)] — use tok+drafts:
        # committed tokens cover [pos0, pos0+k); drafts cover [pos0+? ...]
        pr[r["rs"]].append((r["pos0"], r["n"], r["tok"], drafts))
print(f"p_ring {pring}: {sum(len(v) for v in pr.values())} rounds")

pre = defaultdict(dict)
for f in sorted(glob.glob(os.path.join(d, "kvl3_*.jsonl"))):
    pid = int(re.search(r"kvl3_(\d+)", f).group(1))
    with open(f) as fh:
        for line in fh:
            r = json.loads(line)
            if "#kv" not in r["layer"] or r["phase"] != "pre":
                continue
            key = (pid, r["rs"], r["layer"])
            e = pre[key].setdefault(r["n"], [r.get("col", 0), r.get("T", 0), {}])
            e[2][r["s"]] = (r["slot"], r["k"])


def _eq(a, b):
    return a == b or (a != a and b != b)


# token_at(rs, pos) from p_ring: later ordinals overwrite earlier (each
# round re-proposes); returns None if unknown.
def build_token_map(rs):
    tok_at = {}

    def put(pos, tok):
        if tok is not None:
            tok_at[pos] = tok
    for o, (pos0, k, toks, drafts) in enumerate(pr[rs]):
        # committed tokens at [pos0, pos0+k)
        for j, t in enumerate(toks):
            put(pos0 + j, t)
        # this round's drafts: proposed for positions after the committed
        # prefix of the NEXT round's anchor; p_ring "drafts" are the rows
        # start+1..start+n-1 = the verified draft tokens at
        # [pos0+1, pos0+n) (row i scores position pos0+i).
        for j, t in enumerate(drafts):
            put(pos0 + 1 + j, t)
    return tok_at


# align p_ring ordinals to kvline n per rs: first decode round of a leg has
# nct == pos0; build n <-> ordinal from nct values.
stats = defaultdict(int)
real_miss = []
for rs_key in {k[1] for k in pre}:
    tok_at = build_token_map(rs_key)
    # nct by n for this rs (any pid/layer)
    nct_by_n = {}
    for (pid, rs, layer), pren in pre.items():
        if rs != rs_key:
            continue
        for n, (nct, T, _) in pren.items():
            if T <= 14:
                nct_by_n[n] = nct
    # ordinal lookup by pos0
    ord_at = {pos0: o for o, (pos0, _, _, _) in enumerate(pr[rs_key])}
    for (pid, rs, layer), pren in pre.items():
        if rs != rs_key:
            continue
        ns = sorted(n for n, e in pren.items() if e[1] <= 14)
        for i in range(1, len(ns)):
            n0, n1 = ns[i - 1], ns[i]
            if n1 != n0 + 1:
                continue
            nct, T, s0 = pren[n0]
            nct1, _, s1 = pren[n1]
            if nct1 < nct or nct1 - nct > 14:
                continue
            o0 = ord_at.get(nct)
            o1 = ord_at.get(nct1)
            if o0 is None or o1 is None:
                continue
            pos0, k, toks, drafts = pr[rs_key][o0]
            if o0 > 0:
                prev_pos0, prev_k, prev_toks, prev_drafts = pr[rs_key][o0 - 1]
                prev_n = len(prev_toks) if prev_toks else prev_k
            else:
                prev_pos0 = None
            for s, (slot, k0) in s0.items():
                if s not in s1:
                    continue
                slot1, k1 = s1[s]
                if slot1 != slot or not _eq(k0, k1):
                    continue  # only unchanged slices are miss candidates
                lo, hi = s * 8, s * 8 + 8
                # query positions of round n0: drafts at [nct, nct+T)
                q_lo, q_hi = max(lo, nct), min(hi, nct + T)
                if q_hi <= q_lo:
                    continue
                diff_tok = False
                overlap_c = 0
                for pos in range(q_lo, q_hi):
                    # cur: this round's committed token at pos
                    cur = toks[pos - nct] if pos - nct < len(toks) else None
                    # prev: previous round's token at pos
                    prev = None
                    if prev_pos0 is not None and prev_pos0 <= pos < prev_pos0 + prev_n:
                        off = pos - prev_pos0
                        prev = (
                            prev_toks[off]
                            if off < len(prev_toks)
                            else (
                                prev_drafts[off - 1]
                                if prev_drafts and off - 1 < len(prev_drafts)
                                else None
                            )
                        )
                    if cur is not None and cur != prev:
                        diff_tok = True
                    if pos < nct1:
                        overlap_c += 1
                if diff_tok:
                    stats["REAL-MISS" + ("+COMMITTED" if overlap_c else "")] += 1
                    real_miss.append((pid, rs, layer, n0, nct, nct1, s, slot,
                                      k0, overlap_c))
                else:
                    stats["SAME-TOKEN(benign)"] += 1

print("\n== DISAMBIGUATED ==")
for s_ in sorted(stats):
    print(f"  {s_}: {stats[s_]}")
print("\nREAL misses (first 25):")
for e in real_miss[:25]:
    print("   ", e)
