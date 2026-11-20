#!/usr/bin/env python3
"""Phase-4 battery: 8 legs, 20k+ prompts, 4096 out, t1.0/p0.95/k20, up to C6.

Legs 1-2 sequential, legs 3-8 concurrent (C6). Every leg's full output saved
to logs/qat_validation/<tag>_leg<N>.txt for whole-file reading by the agent.

Usage: python scripts/qat_battery.py --tag ptqr_r10s60 [--api-key K]
"""
import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import time
import urllib.request

import torch

URL = "http://127.0.0.1:8020/v1/chat/completions"
VAL = "/home/curved/SDGraft/data/qwen38_longctx/val_tokens.pt"

TASKS = [
    "Summarize the document above in detail, covering every major section.",
    ("List every distinct API, class, and function named in the document above, "
    "with one line each."),
    ("Write a technical critique of the document above: "
    "structure, correctness, clarity."),
    ("Rewrite the key algorithm from the document above in idiomatic Python "
    "with comments."),
    "Extract a structured outline of the document above (headings, sub-points).",
    "What questions does the document above leave unanswered? Enumerate them.",
    ("Translate the main argument of the document above into plain English "
    "for a non-programmer."),
    "Identify inconsistencies, bugs, or risks in the document above and explain each.",
]

# 20k-token windows inside the 250k-token val corpus (8 non-overlapping)
WINDOWS = [
    (2_000, 24_000), (32_000, 54_000), (62_000, 84_000),
    (92_000, 114_000), (122_000, 144_000), (152_000, 174_000),
    (182_000, 204_000), (212_000, 234_000),
]


def run_leg(idx, tokens, tok, args):
    prompt = tok.decode(tokens.tolist())
    body = {
        "model": args.model,
        "messages": [{"role": "user",
                      "content": TASKS[idx] + "\n\nDOCUMENT:\n" + prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "seed": 1000 + idx,
    }
    if args.thinking_token_budget is not None:
        body["thinking_token_budget"] = args.thinking_token_budget
    req = urllib.request.Request(
        args.url, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {args.api_key}",
                 "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as response:
        resp = json.load(response)
    dt = time.time() - t0
    choice = resp["choices"][0]
    msg = choice["message"]
    answer = msg.get("content") or ""
    path = pathlib.Path(args.out_dir) / f"{args.tag}_leg{idx + 1}.txt"
    path.write_text(answer)
    path.with_suffix(".reasoning.txt").write_text(msg.get("reasoning") or "")
    usage = resp["usage"]
    issue = None
    if not answer.strip():
        issue = "no final answer"
    elif answer.lstrip().startswith("<think>"):
        issue = "reasoning marker in final answer"
    elif args.require_complete and choice["finish_reason"] == "length":
        issue = "answer cut off at the output limit"
    print(
        f"leg{idx + 1}: prompt {usage['prompt_tokens']} "
        f"out {usage['completion_tokens']} "
        f"in {dt:.0f}s; finish={choice['finish_reason']}; {issue or 'answer saved'}",
        flush=True,
    )
    return issue


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--api-key",
                   default=os.environ.get("KLD_KEY", "test-key-local-only"))
    p.add_argument("--out-dir", default="logs/qat_validation")
    p.add_argument("--model", default="qwen3.8-27b-gptq8")
    p.add_argument("--url", default=URL)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--require-complete", action="store_true",
                   help="Fail at the output limit (always enabled for Flint)")
    models = pathlib.Path.home() / "models"
    original = models / "Qwen3.8-27B-GPTQ-8bit-gs128"
    p.add_argument(
        "--corpus-tokenizer",
        default=original if original.exists() else models / "Qwen3.8-27B-PTQR-R10S60",
        help="Tokenizer used to encode the original Qwen validation windows",
    )
    p.add_argument("--thinking-token-budget", type=int, default=None)
    args = p.parse_args()
    args.require_complete = args.require_complete or args.model == "flint"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.corpus_tokenizer)
    v = torch.load(VAL, weights_only=True)
    outdir = pathlib.Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    def run(i):
        a, b = WINDOWS[i]
        return run_leg(i, v[a:b], tok, args)

    issues = [run(i) for i in (0, 1)]
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        issues.extend(ex.map(run, range(2, 8)))
    if any(issues):
        raise SystemExit("Failed legs: " + ", ".join(
            f"{i}: {issue}" for i, issue in enumerate(issues, 1) if issue
        ))
    print("BATTERY DONE:", args.tag, flush=True)


if __name__ == "__main__":
    main()
