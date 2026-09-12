#!/usr/bin/env python3
"""Steady-state C6 throughput: all streams share one prefix so prefill
is a cache hit and the measurement is pure concurrent decode."""
import sys, os, json, time, threading, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_repro import build_prompt

API = "http://127.0.0.1:" + os.environ.get("PORT", "8020") + "/v1/chat/completions"
KEY = os.environ.get("VLLM_API_KEY", "")

PREFIX = build_prompt(int(os.environ.get("STEADY_CTX", "20000")), "shared-prefix-steady")  # same corpus for all


def one(i, out_tokens, results):
    try:
        body = {
            "model": "qwen3.8-27b-gptq8",
            "messages": [{"role": "user", "content": (
                "You are given reference notes. Write a long, coherent "
                "chronological essay synthesizing them. Write as much as "
                "possible; do not stop early.\n\n" + PREFIX)}],
            "max_tokens": out_tokens, "ignore_eos": out_tokens > 1,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repetition_penalty": 1.0,
        }
        req = urllib.request.Request(
            API, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {KEY}"})
        t0 = time.time()
        resp = json.load(urllib.request.urlopen(req, timeout=3600))
        results[i] = (time.time() - t0, resp["usage"]["completion_tokens"])
    except Exception as e:
        results[i] = (-1.0, str(e)[:200])


def run(streams, out_tokens):
    # Warm the prefix into the cache with one solo request first.
    results = {}
    one(99, 4, results)
    print(f"prefix warm: {results.get(99)}", flush=True)
    results = {}
    threads = [threading.Thread(target=one, args=(i, out_tokens, results))
               for i in range(streams)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    errs = [(i, r[1]) for i, r in sorted(results.items()) if isinstance(r[1], str)]
    for i, e in errs:
        print(f"stream {i} ERROR: {e}", flush=True)
    total = sum(r[1] for r in results.values() if not isinstance(r[1], str))
    print(f"streams={streams} out_total={total} wall={wall:.1f}s "
          f"AGGREGATE={total/wall:.2f} tok/s "
          f"per-stream={[f'{r[1]/r[0]:.1f}' for r in results.values() if not isinstance(r[1], str)]}",
          flush=True)


if __name__ == "__main__":
    streams = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    run(streams, out_tokens)
