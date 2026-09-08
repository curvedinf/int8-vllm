#!/usr/bin/env python3
"""Spec-off (SPECOFF flag) greedy leg with per-token engine logprobs.

Records the engine's own logprob for every committed token (logprobs=0 on a
greedy request), so the accumulation curve can be scored against a clean
prefill of the same transcript without PRING.

Usage:
  VLLM_API_KEY=... python scripts/specoff_leg.py --tag X [--temp 0.0]
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_repro import build_prompt

API = "http://127.0.0.1:8020/v1/completions"
MODEL = "qwen3.8-27b-gptq8"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--in-tokens", type=int, default=20000)
    p.add_argument("--out-tokens", type=int, default=3072)
    p.add_argument("--temp", type=float, default=0.0)
    args = p.parse_args()

    nonce = f"{args.tag}-{int(time.time())}"
    corpus = build_prompt(args.in_tokens, nonce)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(
        "/home/curved/models/Qwen3.8-27B-GPTQ-8bit-gs128")
    prompt_ids = tk.apply_chat_template(
        [{"role": "user", "content": corpus}],
        tokenize=True, add_generation_prompt=True,
        enable_thinking=True, reasoning_effort="low")["input_ids"]

    body = {
        "model": MODEL,
        "prompt": prompt_ids,
        "max_tokens": args.out_tokens,
        "temperature": args.temp,
        "top_p": 1.0 if args.temp == 0.0 else 0.95,
        "top_k": -1 if args.temp == 0.0 else 20,
        "logprobs": 1,
        "return_token_ids": True,
        "seed": 5,
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', '')}"})
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=3600))
    dt = time.time() - t0
    ch = resp["choices"][0]
    text = ch.get("text", "")
    lpd = ch.get("logprobs") or {}
    lps = [x for x in (lpd.get("token_logprobs") or []) if x is not None]
    ids = ch.get("token_ids") or [t for t in (lpd.get("tokens") or [])]
    out = f"/home/curved/vllm-gfx908/logs/garble/{args.tag}"
    with open(out + ".txt", "w") as f:
        f.write(text)
    import torch
    torch.save({"prompt_ids": prompt_ids,
                "committed_ids": [i for i in ids if i is not None],
                "eng_logprobs": [l for l in lps if l is not None]},
               out + "_ids.pt")
    print(f"[{args.tag}] {dt:.1f}s {resp['usage']} -> {out}.txt "
          f"({len(text)} chars, {len(lps)} logprob rows)")


if __name__ == "__main__":
    main()
