#!/usr/bin/env python3
"""Live correctness soak for the gfx908 int8 spec-decode stack.

Fires N short-answer chat requests with reasoning disabled at a recipe server and
checks final answers for empty bodies, character salad and repeated tails.
Task accuracy and reasoning need separate checks. This is the coherence gate
from docs/recipes/README.md step "UA 500-request live soak" — deliberately
model-agnostic, no kernel asserts.

Usage: python scripts/ua_live_soak.py [--url http://127.0.0.1:8020] [-n 500]
       [--concurrency 8] [--api-key-file KEY_FILE]
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

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
SALAD = re.compile(r"(\S)\1{12,}")
MARKDOWN_RULE = re.compile(
    r"(?m)^[ \t]*(?:\|[ \t]*)?:?-{3,}:?[ \t]*"
    r"(?:\|[ \t]*:?-{3,}:?[ \t]*)*\|?[ \t]*$"
)


def check_reply(text: str) -> str | None:
    if not text or not text.strip():
        return "empty reply"
    visible_text = MARKDOWN_RULE.sub("", text)
    if not visible_text.strip():
        return "empty reply"
    if SALAD.search(visible_text):
        return "character salad (single char x13+)"
    if len(text.strip()) < 3:
        return "suspiciously short reply"
    m = REPEAT_TAIL.search(text)
    if m and len(m.group(1)) > 24:
        return f"repeat tail x7 ({len(m.group(1))} chars)"
    return None


def _req(url: str, key: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})


def one_request(
    url: str, key: str, model: str, prompt: str, max_tokens: int, seed: int
):
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
            # The 64..256-token budgets exercise final answers. Reasoning
            # has a separate functional gate with its own output budget.
            "chat_template_kwargs": {"enable_thinking": False},
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
    content = msg.get("content") or ""
    return content if isinstance(content, str) else str(content)


def check_one(i: int, *, url: str, key: str, model: str):
    prompt = PROMPTS[i % len(PROMPTS)]
    prompt = prompt if i % 3 else f"{prompt} Be concise."
    max_tokens = 64 + (i % 5) * 48
    try:
        text = one_request(url, key, model, prompt, max_tokens, seed=i)
    except Exception as e:  # noqa: BLE001
        return i, f"request error: {e}"
    bad = check_reply(text)
    return i, f"{bad}: {text[:80]!r}" if bad else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8020")
    ap.add_argument("-n", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--key", default=os.environ.get("VLLM_API_KEY", "recipe-local"))
    ap.add_argument("--api-key-file", type=Path)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    if args.n < 1 or args.concurrency < 1:
        ap.error("request count and concurrency must be positive")
    if args.api_key_file:
        args.key = args.api_key_file.read_text().strip()

    with urllib.request.urlopen(
        _req(f"{args.url}/v1/models", args.key), timeout=10
    ) as r:
        models = json.loads(r.read())
    model = args.model or models["data"][0]["id"]
    print(f"model={model} requests={args.n} concurrency={args.concurrency}")

    ok, fails = 0, []
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        checks = pool.map(
            lambda i: check_one(i, url=args.url, key=args.key, model=model),
            range(args.n),
        )
        for completed, (i, issue) in enumerate(checks, 1):
            if issue:
                fails.append((i, issue))
            else:
                ok += 1
            if completed % 50 == 0:
                print(
                    f"{completed}/{args.n} ok={ok} fails={len(fails)} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )

    print(f"\nRESULT: {ok}/{args.n} coherent, {len(fails)} failures")
    for i, why in fails[:20]:
        print(f"  req {i}: {why}")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
