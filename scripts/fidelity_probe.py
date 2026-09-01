#!/usr/bin/env python3
"""Decode-path distribution fidelity: per-position top-1 prob + entropy of the
plain-decode path (its own reported logprobs) vs the prefill reference
(prompt_logprobs) at the same positions of the same generated stream.

Run on a SPECOFF boot. If the decode path's distributions are systematically
SHARPER (higher top-1 / lower entropy) than prefill at the same positions,
the no-spec path is numerically protected from attractors; if they match,
the no-spec immunity is unexplained by numerics.
"""
import json
import math
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_docs_probe import build_corpus
from garble_repro2 import get_tok, API

import threading

tok = get_tok()
corpus = build_corpus(tok)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
TAG = sys.argv[2] if len(sys.argv) > 2 else "FIDEL"
results = [None] * N


def post(url, body, timeout=1800):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY','')}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def probe(k):
    body = {
        "model": "qwen3.8-27b-gptq8",
        "messages": [{"role": "user", "content":
            "Summarize the documentation below as exhaustive release notes with "
            "headers and numbered lists, quoting key config names inline. "
            "Do not stop early.\n\n" + corpus}],
        "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "max_tokens": 2000, "stream": False,
        "logprobs": True, "top_logprobs": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        out = post(API, body)
        ch = out["choices"][0]
        content = (ch.get("message") or {}).get("content") or ""
        lp = (ch.get("logprobs") or {}).get("content") or []
        with open(f"../logs/garble/{TAG}_r{k}.json", "w") as f:
            json.dump({"text": content,
                       "lp": [{"token": e["token"], "logprob": e["logprob"],
                               "top": [{"token": t["token"], "logprob": t["logprob"]}
                                       for t in e.get("top_logprobs", [])]}
                              for e in lp]}, f)
        results[k] = len(content)
        print(f"[{TAG} r{k}] chars={len(content)} lp={len(lp)}")
    except Exception as e:
        print(f"[{TAG} r{k}] ERR {e}")
        results[k] = -1


threads = [threading.Thread(target=probe, args=(k,)) for k in range(N)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("DONE", results)
