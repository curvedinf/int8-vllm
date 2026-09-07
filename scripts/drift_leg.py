#!/usr/bin/env python3
"""Temp-1.0 drift legs on the garble-repro corpus (any spec mode).

Saves full text for line-by-line reading. Usage:
  VLLM_API_KEY=... python scripts/drift_leg.py --tag X [--temp 1.0]
  [--in-tokens 20000] [--out-tokens 4096] [--seed 5]
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_repro import build_prompt  # same corpus builder

API = "http://127.0.0.1:8020/v1/chat/completions"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--in-tokens", type=int, default=20000)
    p.add_argument("--out-tokens", type=int, default=4096)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args()

    nonce = f"{args.tag}-{int(time.time())}"
    corpus = build_prompt(args.in_tokens, nonce)
    body = {
        "model": "qwen3.8-27b-gptq8",
        "messages": [{
            "role": "user",
            "content": (
                "You are given reference notes. Write a long, coherent "
                "chronological essay synthesizing them. Write as much as "
                "possible; do not stop early.\n\n" + corpus),
        }],
        "max_tokens": args.out_tokens,
        "temperature": args.temp,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": args.seed,
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', '')}"})
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=3600))
    dt = time.time() - t0
    msg = resp["choices"][0]["message"]
    text = (msg.get("content") or "") + (msg.get("reasoning") or "")
    path = f"/home/curved/vllm-gfx908/logs/garble/{args.tag}.txt"
    with open(path, "w") as f:
        f.write(text)
    print(f"[{args.tag}] {dt:.1f}s {resp['usage']} -> {path} ({len(text)} chars)",
          flush=True)


if __name__ == "__main__":
    main()
