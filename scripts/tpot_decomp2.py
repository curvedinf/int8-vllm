#!/usr/bin/env python3
"""Clean TPOT decomposition: non-streaming, two legs per context.

(a) max_tokens=1 leg -> prefill wall time
(b) max_tokens=N leg -> total wall time
decode_rate = N / (t_b - t_a)   [tokens/s, no parser artifacts]
"""
import sys, os, json, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_repro import build_prompt

API = "http://127.0.0.1:8020/v1/chat/completions"
KEY = os.environ.get("VLLM_API_KEY", "")


def call(in_tokens, out_tokens, tag):
    corpus = build_prompt(in_tokens, f"{tag}-{int(time.time()*1000)%10**9}")
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
    out = resp["usage"]["completion_tokens"]
    return dt, out


def decomp(in_tokens, out_tokens=800, tag="x"):
    t1, o1 = call(in_tokens, 1, tag + "p")     # prefill
    t2, o2 = call(in_tokens, out_tokens, tag + "g")  # prefill + decode
    decode = o2 / (t2 - t1) if t2 > t1 else float("nan")
    tpot_ms = 1000 * (t2 - t1) / o2 if o2 else float("nan")
    print(f"in={in_tokens:6d} prefill={t1:6.1f}s total={t2:6.1f}s out={o2} "
          f"decode={decode:7.2f} tok/s TPOT={tpot_ms:7.1f}ms", flush=True)


if __name__ == "__main__":
    for in_tok in (2000, 8000, 20000):
        decomp(in_tok, tag=str(in_tok))
