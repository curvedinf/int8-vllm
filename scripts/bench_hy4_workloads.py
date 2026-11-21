#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure HY4 with real code-review prompts and exact streamed token IDs."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import subprocess
import time
import uuid
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import aiohttp
from prometheus_client.parser import text_string_to_metric_families

from scripts.bench_agent_mixed import TOOLS, Run, digest, percentile

SOURCES = (
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/engine/core.py",
    "vllm/v1/worker/gpu/model_runner.py",
    "vllm/model_executor/layers/linear.py",
    "vllm/v1/core/kv_cache_manager.py",
    "vllm/v1/metrics/loggers.py",
    "vllm/distributed/parallel_state.py",
    "vllm/v1/attention/backend.py",
    "vllm/model_executor/model_loader/utils.py",
    "vllm/engine/arg_utils.py",
    "vllm/v1/request.py",
    "vllm/v1/worker/gpu_worker.py",
)
QUESTIONS = (
    "Walk through the main execution path and identify where it records progress.",
    "Explain how this implementation responds to cancellation and partial failure.",
    "Identify the invariants a maintainer must preserve when changing this code.",
    "Describe where this code allocates and releases memory, using concrete examples.",
    "Write a test plan for the most consequential edge cases in these excerpts.",
    "Trace the state changes for a typical inference request using these excerpts.",
    "Explain how this code handles concurrency, scheduling and backpressure.",
    "Review this code for an engineer debugging a sporadic latency regression.",
    "Describe the data structures and why each one is needed in this implementation.",
    "Identify useful instrumentation for understanding performance of this code.",
    "Explain how configuration choices affect the runtime behavior shown here.",
    "Describe what a new maintainer should understand before modifying this code.",
)
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "ignore_eos": True}


def prepare(args):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    base = subprocess.check_output(
        ["git", "rev-parse", args.revision], text=True
    ).strip()
    excerpts = []
    for name in SOURCES:
        source = subprocess.check_output(["git", "show", f"{base}:{name}"], text=True)
        excerpts.append((name, tokenizer.encode(source, add_special_tokens=False).ids))

    def excerpt(index, count, start):
        name, ids = excerpts[index]
        if len(ids) < count:
            raise ValueError(f"Source excerpt too short for {name}")
        offset = start % (len(ids) - count + 1)
        return (
            name
            + "\n"
            + tokenizer.decode(ids[offset : offset + count], skip_special_tokens=False)
        )

    shared = "Review these repository excerpts.\n\n" + "\n\n".join(
        excerpt(i, 1360, 730 * (i + 1)) for i in range(len(excerpts))
    )
    short = [excerpt(i, 1950, 974 * (i + 1)) for i in range(len(excerpts))]
    corpus = {
        "revision": base,
        "shared": shared,
        "short": short,
        "questions": QUESTIONS,
        "witness": excerpt(0, 2900, 849),
    }
    corpus["sha256"] = digest(corpus)
    args.manifest.write_text(json.dumps(corpus, ensure_ascii=False) + "\n")


def metrics(text):
    out = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            name = sample.name
            if not name.startswith("vllm:"):
                continue
            if name.endswith("_created"):
                continue
            if name.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
                name += "." + sample.labels.get(
                    "position", sample.labels.get("pos", "")
                )
            elif name == "vllm:inter_token_latency_seconds_bucket":
                name += "." + sample.labels["le"]
            elif name not in {
                "vllm:spec_decode_num_drafts_total",
                "vllm:spec_decode_num_draft_tokens_total",
                "vllm:spec_decode_num_accepted_tokens_total",
                "vllm:num_requests_running",
                "vllm:num_requests_waiting",
                "vllm:prompt_tokens_total",
                "vllm:generation_tokens_total",
                "vllm:prefix_cache_hits_total",
                "vllm:prefix_cache_queries_total",
            }:
                continue
            out[name] = out.get(name, 0) + sample.value
    return out


async def snapshot(session, url):
    async with session.get(url + "/metrics") as r:
        r.raise_for_status()
        values = metrics(await r.text())
    return (time.monotonic_ns(), values)


def counter(name, last, first):
    return last.get("vllm:" + name, 0.0) - first.get("vllm:" + name, 0.0)


def spec_stats(first, last):
    rounds = counter("spec_decode_num_drafts_total", last, first)
    drafts = counter("spec_decode_num_draft_tokens_total", last, first)
    accepted = counter("spec_decode_num_accepted_tokens_total", last, first)
    prefix = "vllm:spec_decode_num_accepted_tokens_per_pos_total."
    by_pos = {
        int(k[len(prefix) :]): v - first.get(k, 0.0)
        for k, v in last.items()
        if k.startswith(prefix) and k[len(prefix) :].isdigit()
    }
    return {
        "rounds": rounds,
        "drafted": drafts,
        "accepted": accepted,
        "acceptance_rate": accepted / drafts if drafts else None,
        "emitted_per_round": 1 + accepted / rounds if rounds else None,
        "per_position": [by_pos[k] / rounds for k in sorted(by_pos)] if rounds else [],
    }


def overlap(results, samples):
    count = len(results)
    first = max(r["tokens"][0][0] for r in results)
    last = min(r["tokens"][-1][0] for r in results)
    groups = []
    current = []
    for t, metrics in samples:
        if first <= t <= last and metrics.get("vllm:num_requests_running") == count:
            current.append((t, metrics))
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    valid = [g for g in groups if len(g) >= 2]
    if not valid:
        return None
    g = max(valid, key=lambda items: items[-1][0] - items[0][0])
    (start, begin), (end, finish) = g[0], g[-1]
    seconds = (end - start) / 1e9
    tokens = sum(
        len(ids) for r in results for t, ids in r["tokens"] if start < t <= end
    )
    spec = spec_stats(begin, finish)
    return {
        "seconds": seconds,
        "tokens": tokens,
        "decode_tok_s": tokens / seconds,
        "running_samples": len(g),
        "spec": spec,
        "ms_per_step": 1000 * seconds * count / spec["rounds"]
        if spec["rounds"]
        else None,
    }


def histogram_quantile(first, last, p):
    prefix = "vllm:inter_token_latency_seconds_bucket."
    buckets = sorted(
        (float(k[len(prefix) :]), v - first.get(k, 0.0))
        for k, v in last.items()
        if k.startswith(prefix)
    )
    rank = p * buckets[-1][1] if buckets else 0
    if not rank:
        return None
    low = previous = 0
    for high, value in buckets:
        if value >= rank:
            if not math.isfinite(high):
                return low
            return (
                low + (high - low) * (rank - previous) / (value - previous)
                if value > previous
                else high
            )
        low, previous = high, value
    return None


def scheduled_shapes(log, offset):
    if log is None:
        return [], []
    with log.open("rb") as f:
        f.seek(offset)
        lines = f.read().decode(errors="replace")
    counts = {}
    for total, prompt in re.findall(
        r"Prefill iteration total=(\d+) prompt=([0-9,]+)", lines
    ):
        key = (int(total), tuple(map(int, prompt.split(","))))
        counts[key] = counts.get(key, 0) + 1
    prefill = [
        {"scheduled_tokens": total, "prompt_chunks": sizes, "steps": n}
        for (total, sizes), n in sorted(counts.items())
    ]
    decode = sorted(
        {
            (int(total), int(requests))
            for total, requests in re.findall(
                r"Decode iteration total=(\d+) requests=(\d+)", lines
            )
        }
    )
    return prefill, [{"scheduled_tokens": m, "requests": c} for m, c in decode]


async def bench(args):
    corpus = json.loads(args.manifest.read_text())
    identity = corpus.pop("sha256")
    if digest(corpus) != identity:
        raise ValueError("Frozen workload manifest was modified")
    count = 1 if args.workload == "W5" else args.concurrency
    if not 1 <= count <= len(corpus["questions"]):
        raise ValueError("Unsupported concurrency")
    args.output.mkdir(parents=True, exist_ok=True)
    service = Run(
        SimpleNamespace(
            url=args.url, output=args.output / "streams", max_outstanding=count
        ),
        {},
    )
    headers = {}
    if token := os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + token
    salt = uuid.uuid4().hex
    warm = args.workload in ("W1", "W3")
    budget = (
        1
        if args.prefill_only
        else args.output_tokens or (2048 if args.workload == "W3" else 256)
    )

    def body(i, *, warmup=False):
        question = corpus["questions"][i]
        if warm:
            messages = [
                {
                    "role": "system",
                    "content": "\n\n".join([corpus["shared"]] * args.prefix_repeat),
                },
                {
                    "role": "user",
                    "content": "Acknowledge the excerpts." if warmup else question,
                },
            ]
        else:
            excerpt = corpus["witness"] if args.workload == "W5" else corpus["short"][i]
            messages = [{"role": "user", "content": excerpt + "\n\n" + question}]
        result = {
            **SAMPLING,
            "model": args.model,
            "messages": messages,
            "max_tokens": 1 if warmup else budget,
            "stream": True,
            "return_token_ids": True,
            "stream_options": {"include_usage": True},
            "seed": 7919 + i,
            "cache_salt": salt if warm else salt + "-" + str(i),
        }
        if args.workload in ("W1", "W3", "W5"):
            result.update(tools=TOOLS, tool_choice="auto")
        return result

    async with aiohttp.ClientSession(
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=args.timeout),
        read_bufsize=1024 * 1024,
        max_line_size=2**23,
        max_field_size=2**23,
    ) as session:
        if warm:
            w = await service.request(session, body(0, warmup=True), "warm")
            if w["status"] != "complete":
                raise RuntimeError(w.get("error"))
        log_offset = args.server_log.stat().st_size if args.server_log else 0
        measurements = [await snapshot(session, args.url)]
        stopped = asyncio.Event()

        async def poll():
            while not stopped.is_set():
                measurements.append(await snapshot(session, args.url))
                with suppress(TimeoutError):
                    await asyncio.wait_for(stopped.wait(), timeout=0.2)

        watcher = asyncio.create_task(poll())

        async def send(i):
            return await service.request(session, body(i), f"request-{i:02d}")

        try:
            if args.workload == "W4" and count >= 2:
                first_wave = [asyncio.create_task(send(i)) for i in range(count // 2)]
                if args.inject_during_decode:
                    async with asyncio.timeout(args.timeout):
                        while sum(bool(r["tokens"]) for r in service.results) < len(
                            first_wave
                        ):
                            if any(
                                t.done() and not t.result()["tokens"]
                                for t in first_wave
                            ):
                                raise RuntimeError(
                                    "The initial cohort failed during prefill"
                                )
                            await asyncio.sleep(0.02)
                else:
                    await asyncio.sleep(2)
                results = await asyncio.gather(
                    *first_wave, *(send(i) for i in range(count // 2, count))
                )
            else:
                results = await asyncio.gather(*(send(i) for i in range(count)))
        finally:
            stopped.set()
            await watcher
        measurements.append(await snapshot(session, args.url))
    service.events.close()
    (args.output / "metrics.json").write_text(json.dumps(measurements))
    for r in results:
        if r["status"] != "complete" or r["token_accounting_delta"] != 0:
            raise RuntimeError(
                str(
                    (
                        r["name"],
                        r["status"],
                        r.get("error"),
                        r["token_accounting_delta"],
                    )
                )
            )
    first, last = measurements[0][1], measurements[-1][1]
    prefill_seconds = (
        max(r["tokens"][0][0] for r in results) - min(r["sent_ns"] for r in results)
    ) / 1e9
    prompt = sum(r["usage"]["prompt_tokens"] for r in results)
    ttfts = [r["ttft_s"] for r in results]
    totals = [(r["finished_ns"] - r["sent_ns"]) / 1e9 for r in results]
    per_stream = [
        (r["usage"]["completion_tokens"] - 1)
        * 1e9
        / (r["tokens"][-1][0] - r["tokens"][0][0])
        for r in results
        if r["tokens"][-1][0] > r["tokens"][0][0]
    ]
    prefill, decode = scheduled_shapes(args.server_log, log_offset)
    summary = {
        "workload": args.workload,
        "concurrency": count,
        "corpus_sha256": identity,
        "cache": "warmed" if warm else "distinct cold",
        "sampling": SAMPLING,
        "output_budget": budget,
        "prefill_only": args.prefill_only,
        "prompt_tokens": prompt,
        "output_tokens": sum(r["usage"]["completion_tokens"] for r in results),
        "ttft_p50": percentile(ttfts, 0.5),
        "ttft_p95": percentile(ttfts, 0.95),
        "total_p50": percentile(totals, 0.5),
        "total_p95": percentile(totals, 0.95),
        "per_stream_decode_tok_s": per_stream,
        "per_stream_decode_p50": percentile(per_stream, 0.5) if per_stream else None,
        "itl_p95_s": histogram_quantile(first, last, 0.95),
        "spec": spec_stats(first, last),
        "decode_overlap": overlap(results, measurements),
        "prefill_phase_s": prefill_seconds,
        "prefill_tok_s": prompt / prefill_seconds
        if args.prefill_only and not warm
        else None,
        "prefix_hits": counter("prefix_cache_hits_total", last, first),
        "prefill_chunks": prefill,
        "observed_decode_shapes": decode,
        "shared_prefix_repetitions": args.prefix_repeat if warm else 0,
        "boot_fingerprint": args.fingerprint,
        "request_prompt_tokens": [r["usage"]["prompt_tokens"] for r in results],
        "prompt_sha256": [
            hashlib.sha256(
                json.dumps(r.get("prompt_token_ids", [])).encode()
            ).hexdigest()
            for r in results
        ],
    }
    if args.workload == "W4" and count >= 2:
        initial, injected = results[: count // 2], results[count // 2 :]
        arrival = min(r["sent_ns"] for r in injected)
        first_injected = max(r["tokens"][0][0] for r in injected)
        seconds = (first_injected - arrival) / 1e9
        prompt_tokens = sum(r["usage"]["prompt_tokens"] for r in injected)
        concurrent_output = sum(
            len(ids)
            for r in initial
            for timestamp, ids in r["tokens"]
            if arrival < timestamp <= first_injected
        )
        summary["injected_prefill"] = {
            "trigger": "initial cohort emitted first tokens"
            if args.inject_during_decode
            else "two seconds",
            "wall_s": seconds,
            "injected_prompt_tokens": prompt_tokens,
            "input_tok_s": prompt_tokens / seconds,
            "initial_streams_decoding_at_arrival": sum(
                r["tokens"][0][0] <= arrival < r["tokens"][-1][0] for r in initial
            ),
            "initial_output_during_injected_prefill": concurrent_output,
            "concurrent_output_tok_s": concurrent_output / seconds,
            "ttft_p50": percentile([r["ttft_s"] for r in injected], 0.5),
            "ttft_p95": percentile([r["ttft_s"] for r in injected], 0.95),
        }
    (args.output / "measurement.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--revision", default="HEAD")
    p.add_argument("--workload", choices=("W1", "W2", "W3", "W4", "W5"), default="W2")
    p.add_argument("--output", type=Path)
    p.add_argument("--url", default="http://127.0.0.1:8020")
    p.add_argument("--model", default="hy4-flash")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--output-tokens", type=int)
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--prefill-only", action="store_true")
    p.add_argument(
        "--inject-during-decode",
        action="store_true",
        help="In W4, inject the second cohort once the first cohort starts decoding.",
    )
    p.add_argument(
        "--prefix-repeat",
        type=int,
        default=1,
        help="Repeat the shared real-text excerpt (2 exercises the 32k context band).",
    )
    p.add_argument("--server-log", type=Path)
    p.add_argument(
        "--fingerprint",
        help="Identifier of the measured boot and numeric configuration",
    )
    args = p.parse_args()
    if args.inject_during_decode and (args.workload != "W4" or args.concurrency < 2):
        p.error(
            "Decode-triggered injection requires a W4 run with at least two requests"
        )
    if args.prefix_repeat < 1:
        p.error("--prefix-repeat must be positive")
    if args.prepare:
        if not args.tokenizer:
            p.error("--prepare needs the tokenizer directory")
        prepare(args)
    else:
        if not args.output:
            p.error("--output is required")
        asyncio.run(bench(args))


if __name__ == "__main__":
    main()
