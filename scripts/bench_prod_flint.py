#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Production-shaped Flint measurement (W1 agent-steady, W2 agent-cold).

Follows docs/recipes/optimization_procedure.md: real text, recipe sampling
(T=1.0, top-p 0.95, top-k 20), C12, 256 output, tools present for W1.

  W1  one shared ~2k system prompt (tool schemas), prefix-warmed, 12 distinct
      user turns -> decode throughput / acceptance / total request time
  W2  12 distinct ~2k prompts, no prefix cache -> prefill/queueing (TTFT)

Each measurement must run once per server boot (prefix cache is part of the
workload definition). Outputs JSON with per-request TTFT, total, tokens and
aggregate/stream rates plus acceptance from the spec-decode counters.
"""

import argparse
import asyncio
import json
import random
import re
import statistics
import time
from pathlib import Path

import httpx

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run tests.",
        "parameters": {"type": "object",
                       "properties": {"cmd": {"type": "string"}},
                       "required": ["cmd"]}}},
]
USERS = [
    "Find where the batching logic is defined and summarize the call sites.",
    "The cache eviction test is failing after the refactor. Diagnose and fix it.",
    "Add validation to the config loader and run the tests.",
    "Refactor the duplicated retry branches in the client; keep behaviour.",
    "Write a regression test for the sorting bug and run it.",
    "Explain the trade-offs in the scheduler's admission policy.",
    "Where does the KV tier promotion happen? Trace the path.",
    "Add a metric for queue depth and wire it into the dashboard.",
    "Review the migration script for unsafe DDL.",
    "Summarize the README and list the missing setup steps.",
    "Why does the worker restart loop happen under load?",
    "Write a short design note for the new eval harness.",
]


async def counters(client, url, hdr):
    text = (await client.get(url + "/metrics", headers=hdr, timeout=30)).text
    return {n: sum(float(v) for v in re.findall(
        r"^vllm:spec_decode_" + n + r"_total\{[^\n]*\} ([\d.e+-]+)", text, re.M))
        for n in ["num_drafts", "num_draft_tokens", "num_accepted_tokens"]}


def real_text(tokens, n_chars):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(Path.home() / "models/Flint-GPTQ-INT8-W8A8-GS128-PTQR-r1s40-512k"))
    text = tok.decode(tokens[:20000].tolist())
    return text[:n_chars]


async def one(client, url, hdr, messages, tools, max_tokens, out):
    payload = {"model": "flint", "messages": messages, "max_tokens": max_tokens,
               "temperature": 1.0, "top_p": 0.95, "top_k": 20,
               "stream": True, "stream_options": {"include_usage": True}}
    if tools:
        payload["tools"] = tools
    t0 = time.monotonic()
    ttft = None
    chunks = 0
    usage_tokens = None
    async with client.stream("POST", url + "/v1/chat/completions",
                             headers=hdr, json=payload, timeout=900) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage_tokens = obj["usage"].get("completion_tokens")
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                if (d.get("content") or d.get("reasoning")
                        or d.get("reasoning_content") or d.get("tool_calls")):
                    chunks += 1
                    if ttft is None:
                        ttft = time.monotonic() - t0
    total = time.monotonic() - t0
    out.append({"ttft": ttft, "total": total,
                "tokens": usage_tokens if usage_tokens is not None else chunks})


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


async def run_workload(args, name, prompts, tools):
    url = args.base_url.rstrip("/")
    hdr = {"Authorization": f"Bearer {args.api_key}"}
    async with httpx.AsyncClient() as c:
        if name == "W1":
            # prefix warm: one request with the shared system prompt
            await one(c, url, hdr, prompts[0], tools, 8, [])
        before = await counters(c, url, hdr)
        t0 = time.monotonic()
        out = []
        await asyncio.gather(*[one(c, url, hdr, p, tools, args.output, out)
                               for p in prompts])
        wall = time.monotonic() - t0
        await asyncio.sleep(11)
        after = await counters(c, url, hdr)
    d = {k: after[k] - before[k] for k in after}
    tokens = sum(r["tokens"] for r in out)
    ttfts = [r["ttft"] for r in out if r["ttft"] is not None]
    totals = [r["total"] for r in out]
    tpot = [((r["total"] - r["ttft"]) / max(1, r["tokens"] - 1))
            for r in out if r["ttft"] is not None and r["tokens"] > 1]
    rep = {"workload": name, "concurrency": len(prompts),
           "output_cap": args.output, "wall_s": round(wall, 2),
           "tokens": tokens,
           "aggregate_tok_s": round(tokens / wall, 1),
           "per_stream_tok_s": round(tokens / wall / len(prompts), 1),
           "ttft_p50_ms": round(pct(ttfts, 0.5) * 1000, 1) if ttfts else None,
           "ttft_p95_ms": round(pct(ttfts, 0.95) * 1000, 1) if ttfts else None,
           "total_p50_s": round(pct(totals, 0.5), 2),
           "tpot_p50_ms": round(pct(tpot, 0.5) * 1000, 1) if tpot else None,
           "counters": d,
           "acceptance_length": (round(1 + d["num_accepted_tokens"] / d["num_drafts"], 3)
                                 if d["num_drafts"] else None),
           "requests": [{k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in r.items()} for r in out]}
    return rep


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8030")
    ap.add_argument("--api-key", default="recipe-local")
    ap.add_argument("--output", type=int, default=256)
    ap.add_argument("--corpus", default=str(
        Path.home() / "ksdraft/runs/onpolicy_ready/train_tokens.pt"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    tokens = torch.load(args.corpus, map_location="cpu", weights_only=True)["tokens"]
    text = real_text(tokens, 24000)
    rng = random.Random(0)
    # W1: shared system prompt + distinct user turns
    shared = {"role": "system",
              "content": "You are a precise coding agent with tools.\n\n" + text[:6000]}
    w1 = [[shared, {"role": "user", "content": u}] for u in USERS[:12]]
    # W2: distinct ~2k-token prompts, no shared prefix
    w2 = []
    for i in range(12):
        start = i * 2000
        body = text[start:start + 6000] or text[:6000]
        nonce = f"session {i}-{random.randrange(10**9)}: "
        w2.append([{"role": "system", "content": "You are a coding agent."},
                   {"role": "user", "content": nonce + body}])
    reports = []
    for name, prompts, tools in (("W1", w1, TOOLS), ("W2", w2, None)):
        rep = await run_workload(args, name, prompts, tools)
        reports.append(rep)
        print(json.dumps({k: v for k, v in rep.items() if k != "requests"},
                         indent=2), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
