#!/usr/bin/env python3
"""Score an exact generated-token trace with target-only decode.

Run against a server started with --enable-trace-replay and speculation off.
The input is a *.token_ids.json file from mixed_phase_probe.py.
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--onset-token", type=int)
    ap.add_argument("--max-tokens", type=int,
                    help="score only the first N committed tokens")
    args = ap.parse_args()

    key = os.environ.get("VLLM_API_KEY")
    if not key:
        ap.error("VLLM_API_KEY is required")
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    prompt_ids = trace.get("prompt_token_ids")
    output_ids = trace.get("output_token_ids")
    if not isinstance(prompt_ids, list) or not isinstance(output_ids, list):
        ap.error("trace requires prompt_token_ids and output_token_ids lists")
    if not prompt_ids or not output_ids:
        ap.error("trace token lists must be nonempty")
    if args.max_tokens is not None:
        if args.max_tokens <= 0:
            ap.error("--max-tokens must be positive")
        output_ids = output_ids[:args.max_tokens]

    body = {
        "model": "qwen3.8-27b-gptq8",
        "token_ids": prompt_ids,
        "sampling_params": {
            "max_tokens": len(output_ids),
            "min_tokens": len(output_ids),
            "ignore_eos": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.0,
            "seed": args.seed,
            "logprobs": 20,
            "trace_decode_token_ids": output_ids,
        },
        "stream": False,
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8020/inference/v1/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=1800) as response:
        result = json.load(response)
    choices = result.get("choices") or []
    if len(choices) != 1:
        raise RuntimeError(f"expected one choice, got {len(choices)}")
    choice = choices[0]
    emitted = choice.get("token_ids") or []
    if emitted != output_ids:
        raise RuntimeError(
            f"trace replay diverged: expected {len(output_ids)} ids, "
            f"received {len(emitted)}"
        )
    content = (choice.get("logprobs") or {}).get("content") or []
    if len(content) != len(output_ids):
        raise RuntimeError(
            f"expected {len(output_ids)} logprobs, received {len(content)}"
        )
    rows = []
    for i, item in enumerate(content):
        lp = item.get("logprob")
        top = item.get("top_logprobs") or []
        rows.append({
            "i": i,
            "id": output_ids[i],
            "token": item.get("token"),
            "lp": lp,
            "best_listed_lp": max(
                (x.get("logprob", float("-inf")) for x in top),
                default=None,
            ),
            "better_than_chosen": sum(
                x.get("logprob", float("-inf")) > lp + 1e-4 for x in top
            ) if lp is not None else None,
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "trace": str(args.trace),
        "prompt_len": len(prompt_ids),
        "output_len": len(output_ids),
        "onset_token": args.onset_token,
        "rows": rows,
    }), encoding="utf-8")
    for label, subset in (
        ("all", rows),
        ("before", rows[:args.onset_token] if args.onset_token else []),
        ("after", rows[args.onset_token:] if args.onset_token else []),
    ):
        if not subset:
            continue
        lps = [x["lp"] for x in subset if x["lp"] is not None]
        print(json.dumps({
            "phase": label,
            "tokens": len(subset),
            "mean_lp": sum(lps) / len(lps),
            "below_minus_12": sum(x < -12 for x in lps),
            "min_lp": min(lps),
            "rank_at_least20": sum(
                x["better_than_chosen"] is not None
                and x["better_than_chosen"] >= 19 for x in subset
            ),
        }))


if __name__ == "__main__":
    main()
