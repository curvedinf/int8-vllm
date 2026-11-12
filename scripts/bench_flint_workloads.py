#!/usr/bin/env python3
"""Steady and staggered C12 workload harness for the Flint recipe.

RETIRED AS A PERFORMANCE INSTRUMENT (2026-11-10). This harness feeds uniform
random token IDs, samples greedily, counts SSE chunks as tokens and divides by
a wall clock that includes prefill, so its rate is not the served path
(measured: 28.8-29.6 tok/s single stream at recipe sampling). Use it as a
synthetic diagnostic only. Performance work follows
docs/recipes/optimization_procedure.md (workloads W1-W5).

Workloads (token-id prompts, ignore_eos, greedy):
  steady      N concurrent 2048-in/256-out completions, all at t=0
  staggered   6 completions at t=0, DELAY later a second 6-completion wave
              (sized so the second prefill overlaps the first wave's decode)

Streams SSE so TTFT and per-token latency are measured per request.
Reports aggregate output/prompt throughput and per-wave TTFT/TPOT.

NOTE: prompts are deterministic per --seed, and the recipe enables prefix
caching. Run each measurement once per server boot (or vary --seed) when
comparing configurations; repeats with the same seed measure the cache.

Usage:
  .venv/bin/python scripts/bench_flint_workloads.py steady --n 12
  .venv/bin/python scripts/bench_flint_workloads.py staggered --delay 2.0
"""

import argparse
import asyncio
import json
import random
import statistics
import time

import httpx

VOCAB = 150144


def _prompt(rng: random.Random, n_in: int) -> list[int]:
    return [rng.randrange(VOCAB) for _ in range(n_in)]


async def _one(client: httpx.AsyncClient, url: str, key: str, model: str,
               ids: list[int], n_out: int, t0: float) -> dict:
    body = {
        "model": model,
        "prompt": ids,
        "max_tokens": n_out,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.monotonic() - t0
    first = None
    tok_times: list[float] = []
    usage = None
    async with client.stream("POST", url, json=body,
                             headers={"Authorization": f"Bearer {key}"}) as r:
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
                if ch.get("text"):
                    now = time.monotonic() - t0
                    if first is None:
                        first = now
                    tok_times.append(now)
    end = time.monotonic() - t0
    n_tok = len(tok_times)
    tpot = None
    if n_tok >= 2:
        tpot_ms = (tok_times[-1] - tok_times[0]) / (n_tok - 1) * 1000
    else:
        tpot_ms = None
    return {
        "start": start, "first": first, "end": end,
        "ttft_ms": None if first is None else (first - start) * 1000,
        "tpot_ms": tpot_ms,
        "tokens": n_tok,
        "prompt_tokens": (usage or {}).get("prompt_tokens", len(ids)),
        "completion_tokens": (usage or {}).get("completion_tokens", n_tok),
    }


def _summary(tag: str, rows: list[dict]) -> dict:
    ok = [r for r in rows if r["tokens"] > 0]
    wall = max(r["end"] for r in rows)
    prompt_tok = sum(r["prompt_tokens"] for r in rows)
    out_tok = sum(r["completion_tokens"] for r in rows)
    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"] is not None]
    out = {
        "workload": tag, "requests": len(rows), "wall_s": round(wall, 3),
        "prompt_tok_s": round(prompt_tok / wall, 1),
        "output_tok_s": round(out_tok / wall, 1),
        "ttft_ms_mean": round(statistics.mean(ttfts), 1) if ttfts else None,
        "ttft_ms_p95": round(sorted(ttfts)[int(0.95 * (len(ttfts) - 1))], 1)
        if ttfts else None,
        "tpot_ms_mean": round(statistics.mean(tpots), 2) if tpots else None,
        "tpot_ms_p95": round(sorted(tpots)[int(0.95 * (len(tpots) - 1))], 2)
        if tpots else None,
    }
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("workload", choices=["steady", "staggered"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8020")
    ap.add_argument("--api-key", default="recipe-local")
    ap.add_argument("--model", default="flint")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--wave", type=int, default=6)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--input-len", type=int, default=2048)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    url = args.base_url.rstrip("/") + "/v1/completions"
    t0 = time.monotonic()
    limits = httpx.Limits(max_connections=64, max_keepalive_connections=64)
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        if args.workload == "steady":
            tasks = [
                _one(client, url, args.api_key, args.model,
                     _prompt(rng, args.input_len), args.output_len, t0)
                for _ in range(args.n)
            ]
            rows = await asyncio.gather(*tasks)
            print(json.dumps(_summary("steady", list(rows)), indent=2))
            return

        w = args.wave
        tasks = [
            _one(client, url, args.api_key, args.model,
                 _prompt(rng, args.input_len), args.output_len, t0)
            for _ in range(2 * w)
        ]
        first = tasks[:w]
        second = tasks[w:]
        await asyncio.sleep(args.delay)
        wave2_start = time.monotonic() - t0
        rows = await asyncio.gather(*first, *second)
        rows = list(rows)
        print(json.dumps(
            {
                "workload": "staggered",
                "delay_s": args.delay,
                "wave2_start_s": round(wave2_start, 3),
                "all": _summary("staggered-all", rows),
                "wave1": _summary("wave1", rows[:w]),
                "wave2": _summary("wave2", rows[w:]),
            }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
