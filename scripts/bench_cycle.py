#!/usr/bin/env python3
"""Workload harness implementing docs/recipes/optimization_procedure.md (W1-W5).

Modes
  agent    W1: C12 sharing a ~16k-token prefix (warmed, then measured with new
           suffixes), tools present, 256 output
  cold     W2: C12 with 12 distinct ~2k-token prompts, no shared prefix
  long     W3: W1 with --output-len 2048
  mixed    W4: 6 requests, then 6 more after --delay seconds
  witness  W5: one chat+tools request, ~3k prompt, 256 output

Metrics follow the procedure's contract: tokens from the API's usage block,
acceptance/steps from Prometheus counter deltas, per-stream and overlapped
aggregate decode rates, TTFT, total request time, and chunk-interval p95
(labelled as chunks, not tokens).

Usage:
  .venv/bin/python scripts/bench_cycle.py agent --n 12 --output-len 256
  .venv/bin/python scripts/bench_cycle.py cold
  .venv/bin/python scripts/bench_cycle.py witness
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import statistics
import time

import httpx

BASE = "http://127.0.0.1:8030"
KEY = "recipe-local"
TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from disk",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]},
    },
}]


def corpus() -> tuple[str, str]:
    text = "".join(open(p, errors="ignore").read()
                   for p in sorted(glob.glob("docs/recipes/*.md")))
    text = (text + "\n\n") * 40
    digest = hashlib.sha256(text[:400000].encode()).hexdigest()[:16]
    return text, digest


def scrape() -> dict[str, float]:
    out: dict[str, float] = {}
    for line in httpx.get(f"{BASE}/metrics", timeout=10).text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            val = float(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + val
    return out


async def one(client, prompt, n_out, tool_mode, temp, top_p, top_k, t0):
    if tool_mode:
        body = {"model": "flint",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": n_out, "ignore_eos": True, "stream": True,
                "stream_options": {"include_usage": True}, "tools": TOOLS,
                "tool_choice": "none",
                "temperature": temp, "top_p": top_p, "top_k": top_k}
        url = f"{BASE}/v1/chat/completions"
    else:
        body = {"model": "flint", "prompt": prompt, "max_tokens": n_out,
                "ignore_eos": True, "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": temp, "top_p": top_p, "top_k": top_k}
        url = f"{BASE}/v1/completions"
    first = None
    stamps: list[float] = []
    usage = None
    async with client.stream("POST", url, headers={"Authorization": f"Bearer {KEY}"},
                             json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning") or d.get("tool_calls") \
                        or ch.get("text"):
                    now = time.monotonic() - t0
                    if first is None:
                        first = now
                    stamps.append(now)
    return {"first": first, "end": time.monotonic() - t0, "stamps": stamps,
            "ct": (usage or {}).get("completion_tokens", 0),
            "pt": (usage or {}).get("prompt_tokens", 0)}


def summarize(rows, before, after, n_streams, label) -> dict:
    ok = [r for r in rows if r["ct"] > 1 and r["first"] is not None]
    ttfts = [r["first"] for r in ok]
    totals = [r["end"] for r in ok]
    per_stream = [(r["ct"] - 1) / (r["end"] - r["first"]) for r in ok]
    # Fully overlapped window: all streams are decoding between max(first) and
    # min(end). Aggregate under full overlap is the sum of per-stream rates.
    agg = sum(per_stream)
    d_acc = after.get("vllm:spec_decode_num_accepted_tokens_total", 0.0) - \
        before.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    d_drafts = after.get("vllm:spec_decode_num_drafts_total", 0.0) - \
        before.get("vllm:spec_decode_num_drafts_total", 0.0)
    d_gen = after.get("vllm:generation_tokens_total", 0.0) - \
        before.get("vllm:generation_tokens_total", 0.0)
    d_prompt = after.get("vllm:prompt_tokens_total", 0.0) - \
        before.get("vllm:prompt_tokens_total", 0.0)
    overlap = max(0.0, min(totals) - max(ttfts)) if ok else 0.0
    acc = 1.0 + d_acc / d_drafts if d_drafts else None
    tps = d_gen / d_drafts if d_drafts else None
    # Per-token time is unaffected by start/finish jitter; step time is that
    # times tokens emitted per step.
    per_tok = [(r["end"] - r["first"]) / (r["ct"] - 1) for r in ok]
    step_ms = (statistics.median(per_tok) * 1000.0 * (acc or 1.0)) if per_tok else None
    intervals = sorted(r["stamps"][i] - r["stamps"][i - 1]
                       for r in ok for i in range(1, len(r["stamps"])))
    p95 = intervals[int(0.95 * (len(intervals) - 1))] * 1000 if intervals else None
    return {
        "workload": label, "requests": len(rows),
        "prompt_tokens": int(d_prompt), "completion_tokens": int(d_gen),
        "ttft_ms_p50": round(statistics.median(ttfts) * 1000, 1) if ttfts else None,
        "ttft_ms_p95": round(sorted(ttfts)[int(0.95 * (len(ttfts) - 1))] * 1000, 1)
        if ttfts else None,
        "total_ms_p50": round(statistics.median(totals) * 1000, 1) if totals else None,
        "total_ms_p95": round(sorted(totals)[int(0.95 * (len(totals) - 1))] * 1000, 1)
        if totals else None,
        "decode_tok_s_per_stream_median": round(statistics.median(per_stream), 1)
        if per_stream else None,
        "decode_tok_s_overlap_aggregate": round(agg, 1),
        "acceptance_tokens_per_step": round(acc, 3) if acc else None,
        "tokens_per_step_prom": round(tps, 3) if tps else None,
        "steps": int(d_drafts), "step_ms": round(step_ms, 2) if step_ms else None,
        "overlap_s": round(overlap, 3),
        "chunk_interval_p95_ms": round(p95, 1) if p95 else None,
    }


async def run(args, prompts, tool_mode, t0, delay=0.0):
    limits = httpx.Limits(max_connections=64, max_keepalive_connections=64)
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        async def launch(p):
            return await one(client, p, args.output_len, tool_mode,
                             args.temperature, args.top_p, args.top_k, t0)
        if delay:
            half = len(prompts) // 2
            first = asyncio.gather(*[launch(p) for p in prompts[:half]])
            await asyncio.sleep(delay)
            second = asyncio.gather(*[launch(p) for p in prompts[half:]])
            return list(await first) + list(await second)
        return list(await asyncio.gather(*[launch(p) for p in prompts]))


def build(args) -> tuple[list[str], bool, str]:
    text, digest = corpus()
    if args.mode == "witness":
        return [text[:12000]], True, "W5-witness"
    if args.mode == "cold":
        step = 9000
        return [text[i * step:i * step + 8000] for i in range(args.n)], False, "W2-cold"
    # agent-steady / long / mixed share a prefix (default 64k chars ~= 16k tokens)
    prefix = text[:args.prefix_chars]
    base = args.prefix_chars
    suffixes = [text[base + i * 700:base + i * 700 + 600] for i in range(args.n)]
    prompts = [prefix + s for s in suffixes]
    if args.warm:
        # warm the shared prefix with different suffixes, then measure
        warm = [prefix + text[base + 90000 + i * 700:base + 90000 + i * 700 + 600]
                for i in range(args.n)]
        return prompts, True, "W1-agent", warm
    return prompts, True, "W1-agent"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["agent", "cold", "long", "mixed", "witness"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--prefix-chars", type=int, default=64000,
                    help="shared prefix size in characters (4 chars ~= 1 token)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--warm", action="store_true", default=True)
    ap.add_argument("--no-warm", dest="warm", action="store_false")
    args = ap.parse_args()
    if args.mode == "long":
        args.output_len = max(args.output_len, 2048)

    built = build(args)
    if len(built) == 4:
        prompts, tool_mode, label, warm_prompts = built
    else:
        prompts, tool_mode, label = built
        warm_prompts = None
    t0 = time.monotonic()
    if warm_prompts:
        await run(args, warm_prompts, tool_mode, t0)
    before = scrape()
    t0 = time.monotonic()
    rows = await run(args, prompts, tool_mode, t0,
                     delay=args.delay if args.mode == "mixed" else 0.0)
    after = scrape()
    out = summarize(rows, before, after, len(prompts), label)
    out["corpus_sha"] = corpus()[1]
    out["sampling"] = {"temperature": args.temperature, "top_p": args.top_p,
                       "top_k": args.top_k}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
