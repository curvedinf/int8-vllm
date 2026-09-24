#!/usr/bin/env python3
"""Concurrent temp-1.0 garble probes (the production C8 pattern that garbles
reliably). Reuses the 40k docs corpus; N parallel streams, degen scoring
identical to garble_rate.sh."""
import json
import hashlib
import os
import re
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_docs_probe import build_corpus
from garble_repro2 import get_tok, MODEL, API, save

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TAG = sys.argv[2] if len(sys.argv) > 2 else "CONC"
CTX = int(os.environ.get("CONC_CTX", "40000"))
OUT = int(os.environ.get("CONC_OUT", "4096"))
SEED = int(os.environ.get("CONC_SEED", "301"))
tok = get_tok()
corpus = build_corpus(tok, target=CTX)
print(f"probe ctx={CTX} out={OUT} streams={N} seed={SEED} "
      f"corpus_sha256={hashlib.sha256(corpus.encode()).hexdigest()[:16]}",
      flush=True)
results = [None] * N


def probe(k):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content":
            "Summarize the documentation below as exhaustive release notes with "
            "headers and numbered lists, quoting key config names inline. "
            "Do not stop early.\n\n" + corpus}],
        "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "seed": SEED + k,
        "max_tokens": OUT, "stream": True,
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
        results[k] = (k, f"stream error {e}", True, 0)
        return
    text = "".join(pieces)
    save(f"{TAG}_r{k}", text)
    if len(text) < 3000:
        results[k] = (k, f"short chars={len(text)}", False, len(text))
        return
    tail = text[-300:]
    deg = tail.count("**") + sum(
        1 for ln in tail.splitlines() if len(ln.strip()) < 6)
    # Repetition wall ("ductduct...") scores 0 above; count the most
    # repeated 4-gram in the tail as well.
    gram = max((tail.count(tail[i:i + 4]) for i in range(0, 60)), default=0)
    deg = max(deg, gram * 2)
    # This English documentation prompt can fail as dense multilingual token
    # soup without short lines or repeated Markdown. Scan the WHOLE output:
    # the 2026-09-23 C6 repro failed at chars 3584-4096 on 4/6 streams while
    # the old tail-only `deg` score reported 0/6.
    windows = [(i, text[i:i + 512]) for i in range(0, len(text), 256)]
    unicode_onset = next((i for i, w in windows if len(w) >= 128 and
                          sum(ord(c) > 127 for c in w) / len(w) > 0.05), None)
    newline_run = max((len(m.group()) for m in re.finditer(r"\n+", text)),
                      default=0)
    line_run = 0
    fragment_run = 0
    max_fragment_run = 0
    previous = None
    current = 0
    for line in (line.strip() for line in text.splitlines()):
        if len(line) < 4:
            continue
        current = current + 1 if line == previous else 1
        previous = line
        line_run = max(line_run, current)
        if re.fullmatch(r"[A-Za-z][A-Za-z ]{0,18}", line):
            fragment_run += 1
            max_fragment_run = max(max_fragment_run, fragment_run)
        else:
            fragment_run = 0
    corrupt = (deg > 25 or unicode_onset is not None or
               newline_run >= 12 or line_run >= 8 or
               max_fragment_run >= 8)
    results[k] = (k, f"dur={time.time()-t0:.0f}s chars={len(text)} "
                     f"degen={deg} unicode_onset={unicode_onset} "
                     f"newline_run={newline_run} line_run={line_run} "
                     f"fragment_run={max_fragment_run} "
                     f"corrupt={'YES' if corrupt else 'no'}",
                  corrupt, len(text))


threads = [threading.Thread(target=probe, args=(k,)) for k in range(N)]
for t in threads:
    t.start()
for t in threads:
    t.join()
corrupt = sum(1 for r in results if r and r[2])
for r in results:
    if r:
        print(f"[{TAG} r{r[0]}] {r[1]}", flush=True)
print(f"CONC {TAG}: {corrupt} corrupt of {N}", flush=True)
