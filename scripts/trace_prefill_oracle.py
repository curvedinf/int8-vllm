#!/usr/bin/env python3
"""Score exact output IDs as prompt tokens through the prefill path."""

import argparse
import json
import os
import urllib.request
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--onset-token", type=int)
    args = ap.parse_args()
    if args.max_tokens <= 0:
        ap.error("--max-tokens must be positive")
    key = os.environ.get("VLLM_API_KEY")
    if not key:
        ap.error("VLLM_API_KEY is required")

    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    prompt_ids = trace["prompt_token_ids"]
    output_ids = trace["output_token_ids"][:args.max_tokens]
    ids = prompt_ids + output_ids
    body = {
        "model": "qwen3.8-27b-gptq8",
        "prompt": ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": 0,
    }
    request = urllib.request.Request(
        "http://127.0.0.1:8020/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    entries = result["choices"][0]["prompt_logprobs"]
    if len(entries) != len(ids):
        raise RuntimeError(f"expected {len(ids)} logprob rows; got {len(entries)}")

    rows = []
    for i, token_id in enumerate(output_ids):
        item = entries[len(prompt_ids) + i][str(token_id)]
        rows.append({"i": i, "id": token_id,
                     "lp": item["logprob"], "rank": item["rank"]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"trace": str(args.trace),
                                    "prompt_len": len(prompt_ids),
                                    "output_len": len(output_ids),
                                    "rows": rows}), encoding="utf-8")
    for label, subset in (
        ("all", rows),
        ("before", rows[:args.onset_token] if args.onset_token else []),
        ("after", rows[args.onset_token:] if args.onset_token else []),
    ):
        if not subset:
            continue
        print(json.dumps({
            "phase": label,
            "tokens": len(subset),
            "mean_lp": sum(row["lp"] for row in subset) / len(subset),
            "min_lp": min(row["lp"] for row in subset),
            "rank_gt20": sum(row["rank"] > 20 for row in subset),
            "max_rank": max(row["rank"] for row in subset),
        }))


if __name__ == "__main__":
    main()
