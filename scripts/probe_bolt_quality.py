#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Run the standard 52-prompt corpus through Bolt's chat template.

This checks final-answer coherence and finite token logprobs. It is a capture,
not an official-reference KLD comparison or a broad task-accuracy certificate.
"""

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path

import requests
from kld_probe_v2 import MAX_TOKENS, PROMPTS, SEED, TOP_K
from ua_live_soak import check_reply


def check_row(row):
    choice = row["response"]["choices"][0]
    text = choice["message"].get("content") or ""
    issue = check_reply(text)
    # The corpus explicitly requests this valid two-character answer.
    if row["index"] == 30 and text.strip() == "12":
        issue = None
    positions = (choice.get("logprobs") or {}).get("content") or []
    finite = bool(positions) and all(
        math.isfinite(position["logprob"])
        and position.get("top_logprobs")
        and all(math.isfinite(item["logprob"]) for item in position["top_logprobs"])
        for position in positions
    )
    row["coherence_issue"] = issue
    row["finite_logprobs"] = bool(finite)
    return issue is None and bool(finite)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8020")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    key = (
        args.api_key_file.read_text().strip()
        if args.api_key_file
        else os.environ.get("VLLM_API_KEY", "")
    )
    if not key:
        parser.error("Provide --api-key-file or VLLM_API_KEY")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def run(item):
        index, prompt = item
        response = requests.post(
            args.base_url.rstrip("/") + "/v1/chat/completions",
            headers={"Authorization": "Bearer " + key},
            json={
                "model": "nemotron-bolt",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "seed": SEED,
                "max_tokens": MAX_TOKENS,
                "chat_template_kwargs": {"enable_thinking": False},
                "logprobs": True,
                "top_logprobs": TOP_K,
            },
            timeout=600,
        )
        response.raise_for_status()
        return {"index": index, "prompt": prompt, "response": response.json()}

    rows, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for row in pool.map(run, enumerate(PROMPTS)):
            if not check_row(row):
                failures.append(row["index"])
            rows.append(row)
            args.output.write_text(
                json.dumps({"input_format": "chat_no_thinking", "rows": rows}, indent=2)
            )
            if len(rows) % 8 == 0:
                print(
                    f"{len(rows)}/{len(PROMPTS)} failures={len(failures)}", flush=True
                )
    assert len(rows) == len(PROMPTS) == 52
    assert not failures, failures
    print("PASS 52/52 chat corpus coherence and finite logprobs")


if __name__ == "__main__":
    main()
