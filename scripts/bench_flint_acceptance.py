#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""C12 greedy acceptance/throughput witness for the Flint recipe.

Same counter method as bench_bolt_recipe.py: read vLLM's spec-decode
counters before/after, report acceptance length = 1 + accepted/drafts.
Works with any draft behind the speculator (DFlash2 default or
SPEC_METHOD=ksdraft).

  python scripts/bench_flint_acceptance.py --output runs/witness_ksdraft.json
"""

import argparse
import concurrent.futures
import json
import os
import re
import time
from pathlib import Path

import requests

PROMPTS = [
    "Write a Python function that merges two sorted lists. Include only code.",
    "Implement binary search in Python and explain the invariant briefly.",
    "What is the time complexity of a hash table lookup, and why?",
    "Write a SQL query that returns the top 5 customers by revenue.",
    "Explain the difference between TCP and UDP in three sentences.",
    "Compute 17 * 23 and show your arithmetic step by step.",
    "Write a short C function that reverses a linked list.",
    "Explain why floating point addition is not associative.",
    "Refactor this code to be cleaner: def f(l): return [x for x in l if x]",
    "Write a regex that validates an IPv4 address and explain it.",
    "Describe how a proof-of-work difficulty adjustment works.",
    "Summarize the CAP theorem in three sentences.",
]

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:8030")
ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
ap.add_argument("--model", default="flint")
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--concurrency", type=int, default=12)
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()

hdr = {"Authorization": f"Bearer {args.api_key}"}
url = args.base_url.rstrip("/")


def counters():
    text = requests.get(url + "/metrics", headers=hdr, timeout=30).text
    return {
        n: sum(float(v) for v in re.findall(
            r"^vllm:spec_decode_" + n + r"_total\{[^\n]*\} ([\d.e+-]+)",
            text, re.M))
        for n in ["num_drafts", "num_draft_tokens", "num_accepted_tokens"]
    }


def run(item):
    i, prompt = item
    t0 = time.monotonic()
    r = requests.post(
        url + "/v1/chat/completions",
        headers=hdr,
        json={"model": args.model,
              "messages": [{"role": "user", "content": prompt}],
              "temperature": 0, "max_tokens": args.max_tokens},
        timeout=900,
    )
    r.raise_for_status()
    v = r.json()
    return {"i": i, "elapsed": time.monotonic() - t0,
            "completion_tokens": v["usage"]["completion_tokens"],
            "text": (v["choices"][0]["message"].get("content")
                     or v["choices"][0]["message"].get("reasoning_content")
                     or "")[:200]}


for _ in range(120):
    try:
        if requests.get(url + "/health", timeout=1).ok:
            break
    except requests.RequestException:
        pass
    time.sleep(1)
else:
    raise RuntimeError("server not ready")

startc = counters()
t0 = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
    results = list(pool.map(run, enumerate(PROMPTS)))
wall = time.monotonic() - t0
time.sleep(11)                       # metrics export tick
endc = counters()
delta = {n: endc[n] - startc[n] for n in endc}
tokens = sum(r["completion_tokens"] for r in results)
report = {
    "max_tokens": args.max_tokens,
    "concurrency": args.concurrency,
    "wall_s": round(wall, 2),
    "tokens": tokens,
    "aggregate_tok_s": round(tokens / wall, 1) if wall else None,
    "per_stream_tok_s": round(tokens / wall / args.concurrency, 1) if wall else None,
    "mean_request_s": round(sum(r["elapsed"] for r in results) / len(results), 2),
    "counters": delta,
    "results": results,
}
if delta["num_drafts"]:
    report["acceptance_length"] = 1 + delta["num_accepted_tokens"] / delta["num_drafts"]
    report["draft_tokens_per_step"] = delta["num_draft_tokens"] / delta["num_drafts"]
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
