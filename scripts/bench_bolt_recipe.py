#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Run against an otherwise idle TP4/C8 Bolt server; outputs retain all prompts.

Repeat once to distinguish first-request compilation from warm execution.
The result is a bounded smoke/acceptance measurement, not a quality benchmark.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests

parser = argparse.ArgumentParser(description="C8 greedy Bolt smoke and acceptance gate")
parser.add_argument("--base-url", default="http://127.0.0.1:8020")
parser.add_argument("--api-key-file", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max-tokens", type=int, default=96)
parser.add_argument("--prefix-file", type=Path)
args = parser.parse_args()
key = (
    args.api_key_file.read_text().strip()
    if args.api_key_file
    else os.environ.get("VLLM_API_KEY", "")
)
if not key:
    parser.error("Provide --api-key-file or VLLM_API_KEY")
url = args.base_url.rstrip("/")
hdr = {"Authorization": "Bearer " + key}
prompts = [
    "Explain the difference between a Python list and tuple in three short sentences.",
    (
        "Write a Python function that returns the first n Fibonacci numbers. "
        "Include only code."
    ),
    "Explain why Earth has seasons in three short sentences.",
    "Calculate 17 times 23, and explain your arithmetic briefly.",
    "List three ways to prevent SQL injection in a web application.",
    "Explain what a CPU cache does in three short sentences.",
    "Write a short paragraph explaining photosynthesis to a ten-year-old.",
    "List the first ten prime numbers, then explain what makes a number prime.",
]
prefix = args.prefix_file.read_text() if args.prefix_file else ""
prompts = [prefix + prompt for prompt in prompts]


def counters():
    response = requests.get(url + "/metrics", headers=hdr, timeout=30)
    response.raise_for_status()
    t = response.text
    return {
        n: sum(
            float(v)
            for v in re.findall(
                r"^vllm:spec_decode_" + n + r"_total\{[^\n]*\} ([\d.e+-]+)", t, re.M
            )
        )
        for n in ["num_drafts", "num_draft_tokens", "num_accepted_tokens"]
    }


def run(item):
    i, prompt = item
    start = time.monotonic()
    r = requests.post(
        url + "/v1/chat/completions",
        headers=hdr,
        json={
            "model": "nemotron-bolt",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=600,
    )
    r.raise_for_status()
    v = r.json()
    return {
        "i": i,
        "prompt": prompt,
        "elapsed": time.monotonic() - start,
        "text": v["choices"][0]["message"]["content"],
        "usage": v["usage"],
    }


for _ in range(60):
    try:
        if requests.get(url + "/health", timeout=1).ok:
            break
    except requests.RequestException:
        pass
    time.sleep(1)
else:
    raise RuntimeError("Not ready")
startc = counters()
start = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(run, enumerate(prompts)))
wall = time.monotonic() - start
# The API exports engine counters on its periodic metrics tick.
time.sleep(11)
endc = counters()
delta = {n: endc[n] - startc[n] for n in endc}
report = {
    "max_tokens": args.max_tokens,
    "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
    "wall": wall,
    "tokens": sum(r["usage"]["completion_tokens"] for r in results),
    "counters": delta,
    "results": results,
}
if delta["num_drafts"]:
    report["acceptance_length"] = 1 + delta["num_accepted_tokens"] / delta["num_drafts"]
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
