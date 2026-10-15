#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Bounded Bolt streaming workloads for separate timing and profiler runs.

The default repeated-text completion favors speculation. --chat uses eight
technical tutorial prompts. Both force a fixed output budget with ignore_eos;
these are performance probes, not model-quality evaluations. Raw requests,
responses, usage and stream timestamps are retained for inspection.
"""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import requests
from tokenizers import Tokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--c", type=int, default=8)
p.add_argument("--tokens", type=int, default=256)
p.add_argument("--prompt-tokens", type=int, default=32)
p.add_argument("--out", required=True)
p.add_argument("--salt", default="baseline")
p.add_argument("--chat", action="store_true")
p.add_argument("--base-url", default="http://127.0.0.1:8021")
p.add_argument("--api-key-file", type=Path, required=True)
p.add_argument("--tokenizer", type=Path, required=True)
a = p.parse_args()
if min(a.c, a.tokens, a.prompt_tokens) < 1:
    p.error("Concurrency and token counts must be positive")
key = a.api_key_file.read_text().strip()
if not key:
    p.error("API key file is empty")
headers = {"Authorization": "Bearer " + key}
tok = Tokenizer.from_file(str(a.tokenizer))


def run(i):
    ids = tok.encode(
        f"{a.salt} Example {i}. Explain how computers execute programs, "
        "with detailed examples. "
        + "Memory processors instructions software. "
        * a.prompt_tokens
    ).ids[: a.prompt_tokens]
    body = {
        "model": "nemotron-bolt",
        "prompt": ids,
        "max_tokens": a.tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if a.chat:
        topics = [
            "Python hash tables",
            "CPU cache coherence",
            "database transactions",
            "a reliable job queue in Python",
            "compiler optimization passes",
            "TCP congestion control",
            "a Python graph search library",
            "distributed consensus",
        ]
        body.pop("prompt")
        body["messages"] = [
            {
                "role": "user",
                "content": "Write a detailed technical tutorial about "
                + topics[i % 8]
                + ". Include concrete examples and code where appropriate. "
                "Aim for at least 1500 words.",
            }
        ]
        body["chat_template_kwargs"] = {"enable_thinking": False}
    start = time.monotonic()
    events = []
    text = ""
    usage = None
    with requests.post(
        a.base_url.rstrip("/")
        + "/v1/"
        + ("chat/completions" if a.chat else "completions"),
        headers=headers,
        json=body,
        stream=True,
        timeout=600,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            v = json.loads(line[6:])
            usage = v.get("usage") or usage
            if v.get("choices"):
                s = (
                    (v["choices"][0].get("delta", {}).get("content") or "")
                    if a.chat
                    else v["choices"][0].get("text", "")
                )
                if s:
                    events.append(time.monotonic() - start)
                    text += s
    return {
        "id": i,
        "request": body,
        "elapsed": time.monotonic() - start,
        "first_token": events[0] if events else None,
        "last_token": events[-1] if events else None,
        "events": events,
        "usage": usage,
        "text": text,
    }


start = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=a.c) as pool:
    rows = list(pool.map(run, range(a.c)))
wall = time.monotonic() - start
report = {
    "concurrency": a.c,
    "prompt_tokens_requested": None if a.chat else a.prompt_tokens,
    "chat": a.chat,
    "max_tokens": a.tokens,
    "wall": wall,
    "output_tokens": sum(r["usage"]["completion_tokens"] for r in rows),
    "rows": rows,
}
report["tok_s"] = report["output_tokens"] / wall
Path(a.out).write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)
print("TTFT:", [round(r["first_token"], 3) for r in rows], flush=True)
print(
    "decode tok/s:",
    [
        round(
            (r["usage"]["completion_tokens"] - 1)
            / max(1e-9, r["last_token"] - r["first_token"]),
            2,
        )
        for r in rows
    ],
    flush=True,
)
