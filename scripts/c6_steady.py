#!/usr/bin/env python3
"""Measure C6 output throughput on repeated shared-prefix requests.

Set STEADY_CTX for prompt length, STEADY_STREAM=1 for concurrent HTTP response
streams, and STEADY_SEED_BASE for a reproducible sampled continuation. The
full-request rate includes any prefill work; streaming mode also reports the
interval when all streams have emitted at least one token.
"""
import sys, os, json, time, threading, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_repro import build_prompt

API = "http://127.0.0.1:" + os.environ.get("PORT", "8020") + "/v1/chat/completions"
KEY = os.environ.get("VLLM_API_KEY", "")

PREFIX = build_prompt(int(os.environ.get("STEADY_CTX", "20000")), "shared-prefix-steady")  # same corpus for all


def one(i, out_tokens, results):
    try:
        streaming = os.environ.get("STEADY_STREAM") == "1"
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
        if os.environ.get("STEADY_SEED_BASE"):
            body["seed"] = int(os.environ["STEADY_SEED_BASE"]) + i
        if streaming:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
            body["return_token_ids"] = True
        req = urllib.request.Request(
            API, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {KEY}"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as response:
            if streaming:
                tokens = None
                seen_ids = 0
                token_events = []
                for raw in response:
                    if not raw.startswith(b"data: "):
                        continue
                    data = raw[6:].strip()
                    if data == b"[DONE]":
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise RuntimeError(f"stream error: {chunk['error']}")
                    usage = chunk.get("usage")
                    if usage is not None:
                        tokens = usage["completion_tokens"]
                    for choice in chunk.get("choices") or []:
                        n = len(choice.get("token_ids") or [])
                        seen_ids += n
                        if n:
                            if not token_events:
                                print(f"first token stream {i}: "
                                      f"{time.monotonic():.3f}", flush=True)
                            token_events.append((time.monotonic(), n))
                if tokens is None:
                    tokens = seen_ids
            else:
                tokens = json.load(response)["usage"]["completion_tokens"]
        results[i] = (time.time() - t0, tokens, token_events if streaming else [])
    except Exception as e:
        results[i] = (-1.0, str(e)[:200])


def run(streams, out_tokens):
    # Warm the prefix into the cache with one solo request first.
    results = {}
    one(99, 4, results)
    warm = results.get(99)
    print(f"prefix warm: {warm[:2] if warm else None}", flush=True)
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
    if os.environ.get("STEADY_STREAM") == "1" and not errs:
        events = [r[2] for r in results.values()]
        if all(events):
            common_start = max(stream_events[0][0] for stream_events in events)
            common_end = min(stream_events[-1][0] for stream_events in events)
            if common_end > common_start:
                overlap_tokens = sum(
                    n for stream_events in events for at, n in stream_events
                    if common_start <= at <= common_end
                )
                print(f"all-stream decode overlap: {overlap_tokens} tokens / "
                      f"{common_end-common_start:.2f}s = "
                      f"{overlap_tokens/(common_end-common_start):.2f} tok/s",
                      flush=True)
            else:
                print("all-stream decode overlap: none", flush=True)
        else:
            print(f"all-stream decode overlap: missing token IDs in "
                  f"{sum(not stream_events for stream_events in events)} streams",
                  flush=True)


if __name__ == "__main__":
    streams = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    run(streams, out_tokens)
