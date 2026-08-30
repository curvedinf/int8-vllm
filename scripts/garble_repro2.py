#!/usr/bin/env python3
"""Faithful-shape garble repro: default sampling, multi-turn agent-loop shape,
and concurrency variants."""
import argparse
import json
import os
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MODEL = "qwen3.8-27b-gptq8"
API = "http://127.0.0.1:8020/v1/chat/completions"
TOKENIZER_PATH = "/home/curved/models/Qwen3.8-27B-GPTQ-8bit-gs128"

from garble_repro import WORDS, TOPICS, scan  # reuse corpus + scanner


CODE_SNIPPETS = [
    "def process_{n}(items, config):\n    results = []\n    for idx, item in enumerate(items):\n        if item.status == 'active' and idx % 3 == 0:\n            scaled = item.value * config.scale_factor\n            results.append({{'id': item.id, 'scaled': scaled, 'pos': idx}})\n    return sorted(results, key=lambda r: -r['scaled'])\n",
    "class Node{nickname}:\n    def __init__(self, value, parent=None):\n        self.value = value\n        self.parent = parent\n        self.children = []\n    def walk(self, depth=0):\n        yield depth, self\n        for c in self.children:\n            yield from c.walk(depth + 1)\n",
    "async def fetch_batch_{n}(session, urls, retries=3):\n    out = []\n    for u in urls:\n        for attempt in range(retries):\n            try:\n                async with session.get(u, timeout=5) as resp:\n                    if resp.status == 200:\n                        out.append(await resp.json())\n                        break\n            except asyncio.TimeoutError:\n                continue\n    return out\n",
    "# Utility {n}: merge intervals with tolerance\ndef merge_{n}(intervals, tol=1e-9):\n    intervals.sort(key=lambda p: p[0])\n    merged = []\n    for lo, hi in intervals:\n        if merged and lo - merged[-1][1] <= tol:\n            merged[-1][1] = max(merged[-1][1], hi)\n        else:\n            merged.append([lo, hi])\n    return merged\n",
]


def build_code_corpus(tok, target_tokens, nonce):
    rng = random.Random(hash(nonce) & 0xFFFF)
    parts = [f"# module: synthlib/{nonce}", "# Agrab the full module below.", ""]
    n = 0
    while True:
        tpl = rng.choice(CODE_SNIPPETS)
        parts.append(tpl.format(n=n, nickname=f"C{n}"))
        n += 1
        if n % 20 == 0 and len(tok("\n".join(parts))["input_ids"]) >= target_tokens:
            break
    ids = tok("\n".join(parts))["input_ids"]
    return tok.decode(ids[:target_tokens])


def get_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)


def build_corpus(tok, target_tokens, nonce):
    rng = random.Random(hash(nonce) & 0xFFFF)
    parts = [f"Reference dossier {nonce}. Below are independent notes."]
    n = 60
    while True:
        topic = rng.choice(TOPICS)
        words = [rng.choice(WORDS) for _ in range(rng.randint(60, 110))]
        parts.append(f"Note {n}. On {topic}: " + " ".join(words) + ".")
        n += 1
        if n % 20 == 0 and len(tok("\n".join(parts))["input_ids"]) >= target_tokens:
            break
    ids = tok("\n".join(parts))["input_ids"]
    return tok.decode(ids[:target_tokens])


def chat(messages, max_tokens, tag, seed=1234, greedy=False):
    body = {
        "model": MODEL,
        "messages": messages,
        # Server defaults (override_generation_config): temp 1.0 top_p 0.95 top_k 20
        "temperature": 0.0 if greedy else 1.0,
        "top_p": 1.0 if greedy else 0.95,
        "top_k": -1 if greedy else 20,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', '')}",
        })
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    msg = out["choices"][0].get("message") or {}
    fin = out["choices"][0].get("finish_reason")
    text = msg.get("content")
    reasoning = msg.get("reasoning") or msg.get("reasoning_content")
    if reasoning:
        tag = tag if isinstance(tag, str) else tag
        with open(f"/home/curved/vllm-gfx908/logs/garble/{tag}.reasoning.txt", "w") as f:
            f.write(reasoning)
    if text is None:
        # All-reasoning degeneration at max_tokens yields null content.
        text = f"[DEGENERATE: content=None finish={fin} reasoning_consumed_all]"
    print(f"[{tag}] {time.time()-t0:.1f}s usage={out.get('usage', {})}")
    return text


def leg_single(args, tok):
    nonce = args.nonce or f"{args.tag}-{int(time.time())}"
    if args.code:
        corpus = build_code_corpus(tok, args.in_tokens, nonce)
        instr = ("Here is a large Python module. Extend it with new, coherent "
                 "functionality in the same style. Write as much code as "
                 "possible; do not stop early.\n\n")
    else:
        corpus = build_corpus(tok, args.in_tokens, nonce)
        instr = ("You are given reference notes. Write a long, coherent "
                 "chronological essay synthesizing them. Write as much as "
                 "possible; do not stop early.\n\n")
    msgs = [{"role": "user", "content": instr + corpus}]
    text = chat(msgs, args.out_tokens, args.tag, greedy=args.greedy)
    save(args.tag, text)
    onset = scan(text, label=args.tag)
    if onset is not None:
        print(f"[{args.tag}] GARBLE ONSET ~char {onset}")


def leg_multiturn(args, tok):
    nonce = f"{args.tag}-{int(time.time())}"
    corpus = build_corpus(tok, args.in_tokens, nonce)
    msgs = [{"role": "user", "content": (
        "You are given reference notes. Write a long, coherent chronological "
        "essay synthesizing them. Write as much as possible; do not stop early.\n\n" + corpus)}]
    for turn in range(args.turns):
        text = chat(msgs, args.out_tokens, f"{args.tag}.t{turn}", greedy=args.greedy)
        save(f"{args.tag}.t{turn}", text)
        onset = scan(text, label=f"{args.tag}.t{turn}")
        if onset is not None:
            print(f"[{args.tag}.t{turn}] GARBLE ONSET ~char {onset}")
        # Agent-loop shape: append the assistant turn, ask to continue.
        msgs.append({"role": "assistant", "content": text})
        msgs.append({"role": "user", "content": "Continue the chronicle in the same detail."})


def leg_concurrent(args, tok):
    nonce = f"{args.tag}-{int(time.time())}"
    def one(i):
        corpus = build_corpus(tok, args.in_tokens, f"{nonce}-{i}")
        msgs = [{"role": "user", "content": (
            "Write a long, coherent chronicle from these notes; do not stop early.\n\n" + corpus)}]
        text = chat(msgs, args.out_tokens, f"{args.tag}.s{i}")
        save(f"{args.tag}.s{i}", text)
        onset = scan(text, label=f"{args.tag}.s{i}")
        if onset is not None:
            print(f"[{args.tag}.s{i}] GARBLE ONSET ~char {onset}")
    with ThreadPoolExecutor(max_workers=args.streams) as ex:
        list(ex.map(one, range(args.streams)))


def save(tag, text):
    os.makedirs("/home/curved/vllm-gfx908/logs/garble", exist_ok=True)
    with open(f"/home/curved/vllm-gfx908/logs/garble/{tag}.txt", "w") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["single", "multiturn", "concurrent"])
    ap.add_argument("--in-tokens", type=int, default=20000)
    ap.add_argument("--out-tokens", type=int, default=2048)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--streams", type=int, default=2)
    ap.add_argument("--tag", default="leg2")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--code", action="store_true")
    ap.add_argument("--nonce", default=None,
                    help="fixed nonce for deterministic corpus replay")
    args = ap.parse_args()
    tok = get_tok()
    {"single": leg_single, "multiturn": leg_multiturn,
     "concurrent": leg_concurrent}[args.mode](args, tok)


if __name__ == "__main__":
    main()
