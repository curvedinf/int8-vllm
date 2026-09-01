#!/usr/bin/env python3
"""Analyze fidelity_probe captures: decode-path vs prefill-reference
per-position distribution shape (top-1 prob and top-20 entropy).

For each FIDEL_r{k}.json: rebuild prompt+text, replay through
/v1/completions prompt_logprobs=20, and compare per position:
  top1prob_decode vs top1prob_prefill, and H20 (entropy of the top-20 mass).
A systematic signed gap = the decode path computes a different distribution
than prefill (the no-spec attractor-escape mechanism candidate).
"""
import json
import math
import os
import statistics
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_docs_probe import build_corpus
from garble_repro2 import get_tok

tok = get_tok()
corpus = build_corpus(tok)
msg = ("Summarize the documentation below as exhaustive release notes with "
       "headers and numbered lists, quoting key config names inline. "
       "Do not stop early.\n\n" + corpus)
tmpl = tok.apply_chat_template([{"role": "user", "content": msg}],
                               tokenize=False, add_generation_prompt=True,
                               enable_thinking=False)
prompt_ids = tok.encode(tmpl, add_special_tokens=False)
BASE = len(prompt_ids)
API = "http://127.0.0.1:8020/v1/completions"


def post(body, timeout=1800):
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY','')}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def probs(entries):
    """top-1 prob + normalized top-20 entropy from a top_logprobs entry."""
    if not entries:
        return None, None
    lps = [e["logprob"] for e in entries]
    m = max(lps)
    ps = [math.exp(x - m) for x in lps]
    Z = sum(ps)
    ps = [x / Z for x in ps]
    top1 = max(ps)
    ent = -sum(p * math.log(max(p, 1e-12)) for p in ps)
    return top1, ent


def main():
    files = sorted(f for f in os.listdir("../logs/garble")
                   if f.startswith("FIDEL_r") and f.endswith(".json"))
    for fn in files:
        d = json.load(open(f"../logs/garble/{fn}"))
        text = d["text"]
        lp = d["lp"]
        ids = tok.encode(text, add_special_tokens=False)
        full = prompt_ids + ids
        out = post({"model": "qwen3.8-27b-gptq8", "prompt": full,
                    "max_tokens": 1, "temperature": 0.0,
                    "prompt_logprobs": 20})
        pl = out["choices"][0].get("prompt_logprobs") or []
        dtop1, ptop1, dent, pent = [], [], [], []
        for i, e in enumerate(lp):
            Q = BASE + i  # position of token ids[i]
            if not (0 <= Q < len(pl)):
                continue
            pd = {t["token"]: t["logprob"] for t in e["top"]}
            pr = pl[Q]
            if not pr:
                continue
            d1, de = probs(e["top"])
            p1, pe = probs([{"logprob": v["logprob"]} for v in pr.values()])
            if d1 is None or p1 is None:
                continue
            dtop1.append(d1)
            ptop1.append(p1)
            dent.append(de)
            pent.append(pe)
        n = len(dtop1)
        if not n:
            print(f"{fn}: no comparable positions")
            continue
        diff = [a - b for a, b in zip(dtop1, ptop1)]
        ediff = [a - b for a, b in zip(dent, pent)]
        print(f"{fn}: n={n} | decode-top1 minus prefill-top1: "
              f"median={statistics.median(diff):+.4f} mean={statistics.mean(diff):+.4f} "
              f"| entropy diff: median={statistics.median(ediff):+.4f} "
              f"mean={statistics.mean(ediff):+.4f} "
              f"| decode top1 mean={statistics.mean(dtop1):.3f} vs prefill {statistics.mean(ptop1):.3f}")


if __name__ == "__main__":
    main()
