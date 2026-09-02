#!/usr/bin/env python3
"""Flat-regime replay: is the in-engine verify distribution's flatness
(top1 ~0.2 at clean-text positions) reproduced by the true model's prefill
path on the same committed prefix? If reference top1 >> in-engine top1 at
those positions, the verify-path state is corrupted (not model behavior).

Usage: replay_flat.py <p_ring_dump> <rs> <gen> [max_committed]
"""
import json
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
    max_committed = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
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
    committed = [t for e in rounds for t in e["tok"]][:max_committed]
    # in-engine per committed position: p(committed), top1 of its row
    in_eng = {}
    count = 0
    for e in rounds:
        for c, (t, p, t1) in enumerate(zip(e["tok"], e["p"], e["top1"])):
            if count >= max_committed:
                break
            in_eng[e["pos0"] + 1 + c] = (p, t1)
            count += 1
    first_pos = rounds[0]["pos0"] + 1
    print(f"committed: {len(committed)} first abs pos {first_pos} "
          f"prompt_len={len(prompt_ids)}")
    full_ids = prompt_ids + committed

    body = {
        "model": "qwen3.8-27b-gptq8",
        "prompt": full_ids,
        "max_tokens": 1, "temperature": 0.0,
        "prompt_logprobs": 20,
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY','')}"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    pl = out["choices"][0].get("prompt_logprobs") or []
    print(f"prompt_logprobs entries: {len(pl)}")

    # Per position Q (committed token full_ids[Q]): reference row = pl[Q-1].
    rows = []
    for Q in range(first_pos, first_pos + len(committed)):
        if Q - 1 >= len(pl) or Q - 1 < 1 or Q not in in_eng:
            continue
        entry = pl[Q - 1]
        if not entry:
            continue
        # reference top1 prob and top-5 tokens
        items = sorted(entry.items(), key=lambda kv: kv[1]["logprob"],
                       reverse=True)
        ref_top1 = items[0][1]["logprob"]
        ref_top5 = [int(k) for k, _ in items[:5]]
        m = entry.get(str(full_ids[Q]))
        ref_p = m["logprob"] if m else None
        p_in, t1_in = in_eng[Q]
        rows.append((Q, full_ids[Q], p_in, t1_in, ref_p, ref_top1, ref_top5))

    import math
    import statistics
    flat_rows = [r for r in rows if r[3] < 0.3]
    print(f"compared: {len(rows)}; in-engine flat (top1<0.3): {len(flat_rows)}")
    if flat_rows:
        ref_t1 = [math.exp(r[5]) for r in flat_rows]
        print(f"at in-engine-flat positions: reference top1 "
              f"median={statistics.median(ref_t1):.3f} "
              f"p10={sorted(ref_t1)[len(ref_t1)//10]:.3f} "
              f"p90={sorted(ref_t1)[9*len(ref_t1)//10]:.3f}")
        convict = [r for r in flat_rows if math.exp(r[5]) > 0.6]
        print(f"CONVICT positions (in-engine top1<0.3, reference top1>0.6): "
              f"{len(convict)}/{len(flat_rows)}")
        for Q, t, p_in, t1_in, ref_p, ref_t1, ref_top5 in convict[:20]:
            ref_top5s = " ".join(repr(tok.decode([x]))[:10] for x in ref_top5)
            print(f"  pos {Q}: committed={tok.decode([t])!r} "
                  f"p_in={p_in:.3f} top1_in={t1_in:.3f} "
                  f"ref_top1={math.exp(ref_t1):.3f} ref_top5=[{ref_top5s}]")
    # overall agreement for context
    both = [(p_in, math.exp(ref_p)) for _, _, p_in, _, ref_p, _, _ in rows
            if ref_p is not None]
    if both:
        d = [abs(a - b) for a, b in both]
        print(f"|p_in-p_ref| committed: median={statistics.median(d):.4f} "
              f"p90={sorted(d)[int(0.9*len(d))]:.4f} max={max(d):.4f}")


if __name__ == "__main__":
    main()
