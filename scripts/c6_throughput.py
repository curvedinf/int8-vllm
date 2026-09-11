#!/usr/bin/env python3
"""C6 aggregate throughput: N concurrent legs, aggregate tok/s."""
import sys, os, json, time, threading, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_repro import build_prompt

API = "http://127.0.0.1:8020/v1/chat/completions"
KEY = os.environ.get("VLLM_API_KEY", "")


def one(i, in_tokens, out_tokens, results):
    try:
        corpus = build_prompt(in_tokens, f"c6-{i}-{int(time.time()*1000)%10**9}")
        body = {
            "model": "qwen3.8-27b-gptq8",
            "messages": [{"role": "user", "content": (
                "You are given reference notes. Write a long, coherent "
                "chronological essay synthesizing them. Write as much as "
                "possible; do not stop early.\n\n" + corpus)}],
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
        dt = time.time() - t0
        results[i] = (dt, resp["usage"]["completion_tokens"])
    except Exception as e:  # surface thread failures
        results[i] = (-1.0, str(e)[:200])


def c6(in_tokens, out_tokens, streams):
    results = {}
    threads = [threading.Thread(target=one, args=(i, in_tokens, out_tokens, results))
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
    print(f"in={in_tokens} streams={streams} out_total={total} wall={wall:.1f}s "
          f"AGGREGATE={total/wall:.2f} tok/s "
          f"per-stream={[f'{r[1]/r[0]:.1f}' for r in results.values()]}", flush=True)


if __name__ == "__main__":
    streams = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    # Warm the lazy tokenizer import in the main thread before any worker
    # threads race transformers' lazy module initialization.
    build_prompt(16, "warm")
    c6(2000, 800, streams)
    c6(20000, 800, streams)
