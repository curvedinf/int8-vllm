#!/usr/bin/env python3
"""Real-content corruption probe: repo-docs 40k corpus, tools+streaming+temp1.

Reproduces the user's temp-1.0 spec-only corruption class on real technical
text (synthetic corpora often bail early). Usage:
  VLLM_API_KEY=... .venv/bin/python scripts/garble_docs_probe.py [N_RUNS]
Each run streams a 4096-token completion; the tail degeneration score flags
the bullet/fragment-collapse signature (degen > 25 = corrupt).
"""
import glob
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_repro2 import API, MODEL, get_tok, save

TOOLS = [{"type": "function", "function": {"name": "read_file",
          "description": "Read a project file",
          "parameters": {"type": "object",
                         "properties": {"path": {"type": "string"}},
                         "required": ["path"]}}}]


def build_corpus(tok, target=40000):
    chunks = []
    for f in sorted(glob.glob(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docs", "**", "*.md"), recursive=True)):
        try:
            chunks.append(open(f, errors="replace").read())
        except Exception:
            pass
    ids = tok("\n\n".join(chunks))["input_ids"]
    parts, n, i = [], 0, 0
    span = max(1, len(ids) - 2000)
    while n < target:
        lo = (i * 2000) % span
        parts.append(f"<!-- section {i} -->\n" + tok.decode(ids[lo:lo + 2000]))
        n += 2006
        i += 1
    return tok.decode(tok("\n".join(parts))["input_ids"][:target])


def run(tok, corpus, tag):
    body = {"model": MODEL, "messages": [{"role": "user", "content":
        "Write exhaustive structured technical release notes summarizing the "
        "system described below, with headers, numbered lists, and inline "
        "references. Be exhaustive.\n\n" + corpus}],
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 4096,
        "stream": True, "tools": TOOLS, "tool_choice": "auto",
        "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY','')}"})
    pieces, t0 = [], time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            d = chunk.get("choices", [{}])[0].get("delta", {}) or {}
            if d.get("content"):
                pieces.append(d["content"])
    text = "".join(pieces)
    save(tag, text)
    tail = text[-300:]
    deg = tail.count("**") + sum(1 for ln in tail.splitlines() if len(ln.strip()) < 6)
    print(f"[{tag}] T={time.strftime('%H:%M:%S')} {time.time()-t0:.1f}s "
          f"chars={len(text)} degen={deg} corrupt={'YES' if deg > 25 else 'no'}",
          flush=True)
    print(f"  tail: {tail[-90:]!r}", flush=True)
    return deg > 25


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    tok = get_tok()
    corpus = build_corpus(tok)
    corrupt = 0
    for k in range(n):
        corrupt += run(tok, corpus, f"DOCS_r{k}")
    print(f"corrupt runs: {corrupt}/{n}")


if __name__ == "__main__":
    main()
