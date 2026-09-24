#!/usr/bin/env python3
"""Measure cold-prefill TTFT for solo long-prompt completions.

Each repetition uses a distinct prompt salt so no prefix-cache hit shortens
the leg. Reports per-rep TTFT (first streamed token) and total wall.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_repro import build_prompt

API = "http://127.0.0.1:" + os.environ.get("PORT", "8020") + "/v1/completions"
KEY = os.environ.get("VLLM_API_KEY", "")


def one(salt: str, ctx: int, out_tokens: int) -> tuple[float, float]:
    prompt = build_prompt(ctx, salt)
    body = {
        "model": "qwen3.8-27b-gptq8",
        "prompt": prompt,
        "max_tokens": out_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": out_tokens > 1,
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    t0 = time.monotonic()
    ttft = None
    with urllib.request.urlopen(req, timeout=3600) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            for choice in chunk.get("choices") or []:
                text = choice.get("text") or choice.get("logprobs")
                if text:
                    ttft = time.monotonic() - t0
                    break
            if ttft is not None:
                # Drain the rest without measuring.
                response.read()
                break
    total = time.monotonic() - t0
    return ttft if ttft is not None else -1.0, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32000)
    ap.add_argument("--out-tokens", type=int, default=1)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--salt-prefix", default="ttft")
    args = ap.parse_args()

    ttfts = []
    for i in range(args.reps):
        ttft, total = one(f"{args.salt_prefix}-{i}-{int(time.time())}",
                          args.ctx, args.out_tokens)
        ttfts.append(ttft)
        print(f"rep{i}: ttft={ttft:.2f}s total={total:.2f}s "
              f"ctx={args.ctx}", flush=True)
    ok = [t for t in ttfts if t > 0]
    if ok:
        print(f"TTFT mean={sum(ok)/len(ok):.2f}s min={min(ok):.2f}s "
              f"max={max(ok):.2f}s over {len(ok)} reps", flush=True)


if __name__ == "__main__":
    main()
