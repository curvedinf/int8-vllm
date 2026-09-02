#!/usr/bin/env python3
"""Stale-KV conviction test at failing n1 rounds.

Hypothesis: at failing n1 rounds the engine's anchor row (row 0) attended the
STALE KV at the anchor position — the previous round's REJECTED draft's KV
(the rewrite failed) — so the anchor-row distribution should match the
reference conditioned on the rejected draft, not on the anchor (committed).

Inputs: the p_ring (per-round verify data), the cand ring (per-step draft
proposals [num_reqs, NS]), a failing stream (rs, gen).

Steps:
1. Reconstruct committed + per-round (pos0, n, p, top1) from the p_ring.
2. Match each p-ring round to a cand-ring entry: the round's verified drafts
   (drafts[0..n-2]) must prefix-match one request-row of the entry.
3. Pick failing n1 rounds via the same-boot replay (reference at the committed
   position disagrees: |p_in - ref_p| > 0.5).
4. For each failing round: submit prompt + committed[0..j-2] + [stale] and
   read the reference at position P+j. Compare which conditioning explains
   the engine's anchor row.

Usage: test_stale_kv.py <pring_dump> <cand_dump> <rs> <gen> [max_failing]
"""
import json
import math
import os
import pickle
import sys
import urllib.request
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_docs_probe import build_corpus  # noqa
from garble_repro2 import get_tok  # noqa

API = "http://127.0.0.1:8020/v1/completions"


def load(p):
    out = []
    with open(p, "rb") as fh:
        while True:
            try:
                out.append(pickle.load(fh))
            except EOFError:
                break
    return out


def main():
    pring, candp, rs_q, gen_q = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    max_failing = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    tok = get_tok()
    corpus = build_corpus(tok)
    msg = ("Summarize the documentation below as exhaustive release notes with "
           "headers and numbered lists, quoting key config names inline. "
           "Do not stop early.\n\n" + corpus)
    tmpl = tok.apply_chat_template([{"role": "user", "content": msg}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    prompt_ids = tok.encode(tmpl, add_special_tokens=False)
    P = len(prompt_ids)

    # p-ring stream
    entries = load(pring)
    streams = defaultdict(list)
    gen = defaultdict(int)
    for e in entries:
        if e["pos0"] < P - 40:
            continue
        rs = e["rs"]
        key = (rs, gen[rs])
        if streams[key] and e["pos0"] < streams[key][-1]["pos0"]:
            gen[rs] += 1
            key = (rs, gen[rs])
        streams[key].append(e)
    rounds = streams[(rs_q, gen_q)]
    committed = [t for e in rounds for t in e["tok"]]
    print(f"stream rs{rs_q}g{gen_q}: {len(rounds)} rounds, {len(committed)} committed")

    # cand ring: per-step [num_reqs, NS] proposal lists
    cand = load(candp)
    print(f"cand entries: {len(cand)}")

    # baseline reference (committed sequence) — one call
    def ref_call(seq):
        body = {"model": "qwen3.8-27b-gptq8", "prompt": seq,
                "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 5}
        req = urllib.request.Request(
            API, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY','')}"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.loads(r.read())["choices"][0].get("prompt_logprobs") or []

    pl_base = ref_call(prompt_ids + committed)
    print(f"baseline pl: {len(pl_base)}")

    # locate failing n1 rounds: |p_in - ref_p| > 0.5 at committed[0] of the round
    loc = []
    for e in rounds:
        for c in range(e["n"]):
            loc.append(e)
    failing = []
    j = 0
    for e in rounds:
        for c in range(e["n"]):
            Q = P + j
            if Q < len(pl_base) and pl_base[Q]:
                m = pl_base[Q].get(str(committed[j]))
                ref_p = math.exp(m["logprob"]) if m else 0.0
                if e["n"] == 1 and c == 0 and abs(e["p"][0] - ref_p) > 0.5:
                    failing.append((j, e, ref_p))
            j += 1
    print(f"failing n1 rounds: {len(failing)} (cap {max_failing})")

    # match failing rounds to cand entries: find the cand row whose proposal
    # sequence contains this round's structure. For n1 rounds we don't have
    # the verified drafts (empty), so match by POSITION ORDER: the cand entry
    # at the same step. Approximate: k-th failing round -> scan cand entries
    # near the round index (cand entries are per step, rounds per request
    # interleave). Fall back: try every cand entry, find rows where
    # drafts[0] could precede; we cannot disambiguate — instead match on the
    # PREVIOUS round's drafts (which are recorded) to anchor the cand entry.
    matched = 0
    for j, e, ref_p in failing[:max_failing]:
        # previous round's drafts (from the p-ring) identify the cand entry
        # that produced them: its NEXT-step entry holds this round's proposals.
        ri = rounds.index(e)
        prev = rounds[ri - 1] if ri > 0 else None
        stale = None
        if prev is not None and prev["pos0"] + prev["n"] == e["pos0"]:
            prev_drafts = prev.get("drafts") or []
            n_acc = prev["n"] - 1  # accepted drafts
            for ci, ce in enumerate(cand):
                for row in ce["drafts"]:
                    # prev accepted drafts == row[0:n_acc]
                    if n_acc > 0 and row[:n_acc] == prev_drafts[:n_acc]:
                        # the rejected draft at the resample position:
                        stale = row[n_acc] if n_acc < len(row) else None
                        break
                if stale is not None:
                    break
        if stale is None:
            print(f"  j={j}: no cand match (prev drafts={prev_drafts[:4] if prev else None})")
            continue
        # stale-conditioned reference: committed[0..j-2] + [stale]; predict P+j
        seq = prompt_ids + committed[: j - 1] + [stale]
        pl_mod = ref_call(seq)
        Qm = len(seq) - 1
        entry = pl_mod[Qm] if Qm < len(pl_mod) else None
        p_stale = None
        if entry:
            m = entry.get(str(committed[j]))
            p_stale = math.exp(m["logprob"]) if m else 0.0
        p_in = e["p"][0]
        print(f"  j={j} pos={P+j}: committed={tok.decode([committed[j]])!r} "
              f"p_in={p_in:.3f} ref_p_anchor={ref_p:.3f} ref_p_stale="
              f"{p_stale if p_stale is None else round(p_stale,3)} "
              f"stale={tok.decode([stale], errors='replace')!r}")
        if p_stale is not None:
            d_anchor = abs(p_in - ref_p)
            d_stale = abs(p_in - p_stale)
            verdict = "STALE-EXPLAINED" if d_stale < d_anchor - 0.2 else (
                "ANCHOR-EXPLAINED" if d_anchor < d_stale - 0.2 else "neither")
            print(f"      -> {verdict} (d_anchor={d_anchor:.3f} d_stale={d_stale:.3f})")
            matched += 1
    print(f"matched/tested: {matched}")


if __name__ == "__main__":
    main()
