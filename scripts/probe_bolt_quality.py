#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Run the standard 52-prompt corpus through the served model's chat template.

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


def check_row(row, *, require_complete=False):
    choice = row["response"]["choices"][0]
    text = choice["message"].get("content") or ""
    issue = check_reply(text)
    if text.lstrip().startswith("<think>"):
        issue = "reasoning marker in final answer"
    # The short-reply heuristic also catches valid standalone math answers.
    short_answer = {19: "11", 30: "12"}.get(row["index"])
    if short_answer is not None and text.strip() == short_answer:
        issue = None
    if issue is None and require_complete and choice["finish_reason"] == "length":
        issue = "answer cut off at the output limit"
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
    parser.add_argument("--model", default="nemotron-bolt")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--thinking-token-budget", type=int)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail when an answer hits its output limit (always enabled for Flint)",
    )
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

    if args.thinking_token_budget is not None:
        thinking = {"thinking_token_budget": args.thinking_token_budget}
    elif args.model == "flint":
        thinking = {}
    else:
        thinking = {"chat_template_kwargs": {"enable_thinking": False}}

    def run(item):
        index, prompt = item
        response = requests.post(
            args.base_url.rstrip("/") + "/v1/chat/completions",
            headers={"Authorization": "Bearer " + key},
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "seed": SEED,
                "max_tokens": args.max_tokens,
                **thinking,
                "logprobs": True,
                "top_logprobs": TOP_K,
            },
            timeout=600,
        )
        response.raise_for_status()
        return {"index": index, "prompt": prompt, "response": response.json()}

    rows, failures = [], []
    require_complete = args.require_complete or args.model == "flint"
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for row in pool.map(run, enumerate(PROMPTS)):
            if not check_row(row, require_complete=require_complete):
                failures.append(row["index"])
            rows.append(row)
            args.output.write_text(
                json.dumps(
                    {
                        "input_format": "chat_no_thinking"
                        if "chat_template_kwargs" in thinking
                        else "chat",
                        "model": args.model,
                        "sampling": thinking,
                        "rows": rows,
                    },
                    indent=2,
                )
            )
            if len(rows) % 8 == 0:
                print(
                    f"{len(rows)}/{len(PROMPTS)} failures={len(failures)}", flush=True
                )
    if failures:
        raise SystemExit(f"Failed prompt indices: {failures}")
    print(f"PASS {len(rows)}/{len(PROMPTS)} chat corpus coherence and finite logprobs")


if __name__ == "__main__":
    main()
