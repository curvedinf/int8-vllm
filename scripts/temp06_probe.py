#!/usr/bin/env python3
"""Concurrent temp-1.0 garble probes (the production C8 pattern that garbles
reliably). Reuses the 40k docs corpus; N parallel streams, degen scoring
identical to garble_rate.sh."""
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_docs_probe import build_corpus
from garble_repro2 import get_tok, MODEL, API, save

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TAG = sys.argv[2] if len(sys.argv) > 2 else "CONC"
tok = get_tok()
corpus = build_corpus(tok)
results = [None] * N


def probe(k):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content":
            "Summarize the documentation below as exhaustive release notes with "
            "headers and numbered lists, quoting key config names inline. "
            "Do not stop early.\n\n" + corpus}],
        "temperature": 0.6, "top_p": 0.95, "top_k": 20,
        "max_tokens": 4096, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY','')}"})
    pieces = []
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                d0 = line[6:]
                if d0 == "[DONE]":
                    break
                try:
                    chunk = json.loads(d0)
                except Exception:
                    continue
                d = chunk.get("choices", [{}])[0].get("delta", {}) or {}
                if d.get("content"):
                    pieces.append(d["content"])
    except Exception as e:
        results[k] = (k, f"stream error {e}", 0, 0)
        return
    text = "".join(pieces)
    if len(text) < 3000:
        results[k] = (k, "short", len(text), 0)
        return
    save(f"{TAG}_r{k}", text)
    tail = text[-300:]
    deg = tail.count("**") + sum(
        1 for ln in tail.splitlines() if len(ln.strip()) < 6)
    # Repetition wall ("ductduct...") scores 0 above; count the most
    # repeated 4-gram in the tail as well.
    gram = max((tail.count(tail[i:i + 4]) for i in range(0, 60)), default=0)
    deg = max(deg, gram * 2)
    results[k] = (k, f"dur={time.time()-t0:.0f}s chars={len(text)} "
                     f"degen={deg} corrupt={'YES' if deg > 25 else 'no'}",
                  deg, len(text))


threads = [threading.Thread(target=probe, args=(k,)) for k in range(N)]
for t in threads:
    t.start()
for t in threads:
    t.join()
corrupt = sum(1 for r in results if r and r[2] > 25)
for r in results:
    if r:
        print(f"[{TAG} r{r[0]}] {r[1]}", flush=True)
print(f"CONC {TAG}: {corrupt} corrupt of {N}", flush=True)
