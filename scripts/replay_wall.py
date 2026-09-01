#!/usr/bin/env python3
"""Wall-loop replay differential: reference p (prefill-path prompt_logprobs)
vs in-engine spec-path p (VLLM_P_RING) at the same committed positions.

If in-engine p ~ 1.0 at loop positions while the reference p is materially
lower, the verify-path state diverges from the true state of the same
context (the garble carrier). If they agree, the attractor is genuine model
behavior and the entry-rate question moves to the sampler.

Usage: replay_wall.py <p_ring_dump> <rs> <gen_idx> <prompt_token_count>
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


def main():
    dump, rs_q, gen_q = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    tok = get_tok()
    corpus = build_corpus(tok)
    msg = ("Summarize the documentation below as exhaustive release notes with "
           "headers and numbered lists, quoting key config names inline. "
           "Do not stop early.\n\n" + corpus)
    tmpl = tok.apply_chat_template([{"role": "user", "content": msg}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    prompt_ids = tok.encode(tmpl, add_special_tokens=False)
    print(f"prompt ids: {len(prompt_ids)}")

    entries = []
    with open(dump, "rb") as fh:
        while True:
            try:
                entries.append(pickle.load(fh))
            except EOFError:
                break
    streams = defaultdict(list)
    gen = defaultdict(int)
    for e in entries:
        rs = e["rs"]
        if streams[(rs, gen[rs])] and e["pos0"] < streams[(rs, gen[rs])][-1]["pos0"]:
            gen[rs] += 1
        streams[(rs, gen[rs])].append(e)
    rounds = [e for e in streams[(rs_q, gen_q)] if e["pos0"] >= len(prompt_ids) - 2]
    print(f"rounds after warmup filter: {len(rounds)} "
          f"(pos0 {rounds[0]['pos0']} .. {rounds[-1]['pos0']})")
    committed = [t for e in rounds for t in e["tok"]]
    # in-engine p per absolute position: pos0 + 1 + c (row start+c predicts
    # committed[c]); entry pos0 = anchor position.
    in_eng = {}  # abs_pos -> (p, top1)
    for e in rounds:
        for c, (t, p, t1) in enumerate(zip(e["tok"], e["p"], e["top1"])):
            in_eng[e["pos0"] + 1 + c] = (p, t1)
    first_pos = rounds[0]["pos0"] + 1
    print(f"committed: {len(committed)} tokens, first abs pos {first_pos}, "
          f"prompt_len+1 = {len(prompt_ids)+1}")
    full_ids = prompt_ids + committed
    assert len(full_ids) > 0

    body = {
        "model": "qwen3.8-27b-gptq8",
        "prompt": full_ids,
        "max_tokens": 1, "temperature": 0.0,
        "prompt_logprobs": 1,
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY','')}"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    pl = out["choices"][0].get("prompt_logprobs") or []
    print(f"prompt_logprobs entries: {len(pl)}")

    # reference p of the committed token at abs position Q = pl[Q-1][token]
    rows = []
    for Q in range(first_pos, first_pos + len(committed)):
        if Q - 1 >= len(pl) or Q - 1 < 1:
            continue
        entry = pl[Q - 1]
        if not entry:
            continue
        m = entry.get(str(full_ids[Q]))
        if m is None:
            continue
        if Q in in_eng:
            p_in, t1_in = in_eng[Q]
            rows.append((Q, full_ids[Q], p_in, math.exp(m["logprob"])))
    # report: loop region = positions where in_eng p ~ 1.0 sustained
    divergent = [(Q, t, pi, pr) for Q, t, pi, pr in rows
                 if pi > 0.99 and pr < 0.9]
    agree = [(Q, t, pi, pr) for Q, t, pi, pr in rows if pi > 0.99 and pr >= 0.9]
    print(f"compared positions: {len(rows)}")
    print(f"in-engine p>0.99 AND reference p<0.9 (DIVERGENT): {len(divergent)}")
    print(f"in-engine p>0.99 AND reference p>=0.9 (agree lock): {len(agree)}")
    for Q, t, pi, pr in divergent[:15]:
        print(f"  DIVERGENT pos {Q}: tok={t} ({tok.decode([t])!r}) "
              f"p_in={pi:.3f} p_ref={pr:.3f}")
    # percentile band of agreement on a mid sample
    import statistics
    deltas = [abs(pi - pr) for _, _, pi, pr in rows]
    if deltas:
        print(f"|p_in - p_ref|: median={statistics.median(deltas):.4f} "
              f"p90={sorted(deltas)[int(0.9*len(deltas))]:.4f} max={max(deltas):.4f}")


if __name__ == "__main__":
    main()
