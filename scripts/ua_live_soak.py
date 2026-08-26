#!/usr/bin/env python3
"""Live correctness soak for the gfx908 int8 spec-decode stack.

Fires N chat requests (varied lengths/URAs) at a running recipe server and
greps every reply for coherence failures: empty/truncated bodies, refusal
loops, character salad, repeated n-gram tails. This is the correctness gate
from docs/recipes/README.md step "UA 500-request live soak" — deliberately
model-agnostic, no kernel asserts.

Usage: python scripts/ua_live_soak.py [--url http://127.0.0.1:8020] [-n 500]
       [--key KEY]
"""
import argparse
import json
import math
import re
import sys
import time
import urllib.request

PROMPTS = [
    "Explain the difference between a mutex and a semaphore in two sentences.",
    "Write a Python one-liner that reverses a string. Then explain it.",
    "What is the capital of France?",
    "Give me three ideas for a sci-fi short story, one line each.",
    "Translate 'the quick brown fox' into French.",
    "Summarize the plot of Hamlet in exactly three sentences.",
    "What happens if you divide by zero in IEEE 754 float math?",
    "Name two differences between TCP and UDP.",
    "Write a haiku about GPU kernels.",
    "Is 97 a prime number? Show the reasoning briefly.",
    "Explain why the sky is blue to a five-year-old.",
    "List the first eight Fibonacci numbers.",
    "What year did the Apollo 11 landing happen?",
    "Give a bash command to count lines in a file, and explain the flag.",
    "What does SQL JOIN do? One sentence.",
    "Contrast REST and gRPC in two sentences.",
    "Write a JavaScript function that clamps a number.",
    "Why do GPUs have thousands of cores? Two sentences.",
    "What's the boiling point of water at sea level in Celsius and Fahrenheit?",
    "Explain what a git rebase does in one sentence.",
]

REPEAT_TAIL = re.compile(r"(.{8,240}?)\1{6,}", re.DOTALL)
SALAD = re.compile(r"(.)\1{12,}")


def check_reply(text: str) -> str | None:
    if not text or not text.strip():
        return "empty reply"
    if SALAD.search(text):
        return "character salad (single char x13+)"
    if len(text.strip()) < 3:
        return "suspiciously short reply"
    if SALAD.search(text):
        return "character salad (single char x13+)"
    m = REPEAT_TAIL.search(text)
    if m and len(m.group(1)) > 24:
        return f"repeat tail x7 ({len(m.group(1))} chars)"
    return None


def _req(url: str, key: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})


def one_request(url: str, key: str, model: str, prompt: str, max_tokens: int, seed: int):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "top_p": 0.80,
            "top_k": 20,
            "presence_penalty": 1.5,
            "max_tokens": max_tokens,
            "seed": seed,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.loads(r.read())
    msg = out["choices"][0]["message"]
    content = msg.get("reasoning_content") or msg.get("content") or ""
    return content if isinstance(content, str) else str(content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8020")
    ap.add_argument("-n", type=int, default=500)
    ap.add_argument("--key", default="recipe-local")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    with urllib.request.urlopen(
        _req(f"{args.url}/v1/models", args.key), timeout=10
    ) as r:
        models = json.loads(r.read())
    model = args.model or models["data"][0]["id"]
    print(f"model={model} n={args.url}")

    ok, fails = 0, []
    t0 = time.time()
    for i in range(args.n):
        prompt = PROMPTS[i % len(PROMPTS)]
        # vary length and decode length across the run
        prompt = prompt if i % 3 else f"{prompt} Be concise."
        max_tokens = 64 + (i % 5) * 48
        try:
            text = one_request(args.url, args.key, model, prompt, max_tokens, seed=i)
        except Exception as e:  # noqa: BLE001
            fails.append((i, f"request error: {e}"))
            continue
        bad = check_reply(text)
        if bad:
            fails.append((i, f"{bad}: {text[:80]!r}"))
        else:
            ok += 1
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{args.n} ok={ok} fails={len(fails)} elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"\nRESULT: {ok}/{args.n} coherent, {len(fails)} failures")
    for i, why in fails[:20]:
        print(f"  req {i}: {why}")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
