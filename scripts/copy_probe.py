#!/usr/bin/env python3
"""Greedy copy-heavy 32k C6 probe (GOALOPT iter 10 lookup A/B).

Six concurrent streams over the same 32k corpus with a verbatim-repeat
instruction at temperature 0 - the traffic class where lookup drafting
should win. Reports aggregate tok/s and per-stream rates.
"""
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_repro import build_prompt

API = "http://127.0.0.1:" + os.environ.get("PORT", "8020") + "/v1/completions"
KEY = os.environ.get("VLLM_API_KEY", "")
PREFIX = build_prompt(32000, "lookup-copy-probe")
SALT = sys.argv[1] if len(sys.argv) > 1 else "a"


def one(i, out_tokens, results):
    body = {
        "model": "qwen3.8-27b-gptq8",
        "prompt": PREFIX,
        "max_tokens": out_tokens,
        "ignore_eos": True,
        "temperature": float(os.environ.get("COPY_TEMP", "0.0")),
        "seed": 5,
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    dt = time.time() - t0
    results[i] = (d["usage"]["completion_tokens"], dt)


def run(tag):
    # warm prefix
    warm = {}
    one(-1, 4, warm)
    results = {}
    ts = [threading.Thread(target=one, args=(i, 800, results))
          for i in range(6)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - t0
    tot = sum(r[0] for r in results.values())
    print(f"[{tag}-{SALT}] streams=6 out={tot} wall={wall:.1f}s "
          f"AGGREGATE={tot / wall:.2f} tok/s "
          f"per-stream={[f'{r[0] / r[1]:.1f}' for r in results.values()]}",
          flush=True)


if __name__ == "__main__":
    run("copy")
