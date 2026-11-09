"""Build the Flint PTQR long-context training corpus (token stream).

Writes /home/curved/SDGraft/data/flint_longctx/{train,val}_tokens.pt —
flat int64 token streams consumed by scripts/ptqr_train_flint.py via
build_dataloader (same format as qwen38_longctx).

Corpus tiers (mirrors the Qwen3.8 PTQR long-context recipe, adapted to
Flint's coding-agent profile and its chat template):
  1. code (evol-codealpaca-v1) — instruction/response pairs
  2. general text (C4 en sample)
  3. chat-templated agentic legs — Flint's chat template applied to
     tool-use / reasoning-shaped turns (the deployment manifold: the
     model serves as a coding agent with tool calls + reasoning)
  4. long-context legs — chained documents to 4k-16k tokens wrapped in a
     "synthesize these reference notes" chat turn (the long-context
     tier; targets the drift regime the way the Qwen r2roll legs did)

Usage (int8-vllm venv, CPU):
  .venv/bin/python scripts/ptqr_flint_corpus.py [--target-train 5000000]
"""

from __future__ import annotations

import argparse
import os
import random

import torch

MODEL_DIR = "/home/curved/models/flint-bf16"
OUT_DIR = "/home/curved/SDGraft/data/flint_longctx"


def build(tokenizer, args, rng):
    from datasets import load_dataset

    ids: list[int] = []

    def emit_text(text: str, cap: int | None = None):
        toks = tokenizer.encode(text, add_special_tokens=False)
        ids.extend(toks[: cap or args.max_sample])

    def _as_ids(out):
        if isinstance(out, dict):
            out = out.get("input_ids", [])
        if hasattr(out, "ids"):
            out = out.ids
        elif hasattr(out, "input_ids"):
            out = out.input_ids
        if len(out) and hasattr(out[0], "ids"):
            out = out[0].ids
        elif len(out) and isinstance(out[0], (list, tuple)):
            out = out[0]
        return [int(t) for t in out]

    def emit_chat(messages, tools=None, cap: int | None = None):
        try:
            if tools is not None:
                out = tokenizer.apply_chat_template(
                    messages, tools=tools, tokenize=True, add_generation_prompt=True
                )
            else:
                out = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False
                )
            ids.extend(_as_ids(out)[: cap or args.max_sample])
        except Exception:
            pass  # template-mismatched leg; skip rather than poison the corpus

    # tier 1: code
    code = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
    idxs = list(range(len(code)))
    rng.shuffle(idxs)
    n_code = 0
    for i in idxs:
        if n_code >= args.n_code:
            break
        s = code[i]
        text = f"### Instruction:\n{s.get('instruction', '')}\n\n### Response:\n{s.get('output', '')}"
        if len(text) < 50:
            continue
        emit_text(text)
        n_code += 1
    print(f"tier1 code samples: {n_code} tokens so far {len(ids)}")

    # tier 2: general text
    c4 = load_dataset(
        "allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train"
    )
    n_c4 = 0
    for i, s in enumerate(c4):
        if i >= 50000 or n_c4 >= args.n_c4:
            break
        text = s.get("text", "")
        if len(text) < 200:
            continue
        emit_text(text)
        n_c4 += 1
    print(f"tier2 c4 samples: {n_c4} tokens so far {len(ids)}")

    # tier 3: chat-templated agentic legs (tool use + reasoning shapes)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_tests",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                },
            },
        },
    ]
    code_tasks = [
        "Fix the failing test in the repository checkout and explain the root cause.",
        "Refactor the parser module to remove the duplicated token loop.",
        "Add type hints to the caching layer and fix the mypy errors.",
        "Write a reproducer for the race in the background worker pool.",
        "Implement the missing retry logic in the HTTP client wrapper.",
    ]
    for k in range(args.n_agent):
        task = code_tasks[k % len(code_tasks)]
        s = code[idxs[k % len(idxs)]]
        ctx = (s.get("instruction", "") or "")[:1200]
        msgs = [
            {
                "role": "user",
                "content": f"{task}\n\nRepository context:\n{ctx}",
            },
            {
                "role": "assistant",
                "content": (
                    f"I'll start by inspecting the relevant module. "
                    f"Plan: locate the {task.split()[1].lower() if len(task.split()) > 1 else 'code'}, "
                    f"read the surrounding tests, then make the minimal change."
                ),
            },
        ]
        emit_chat(msgs, tools=tools if k % 2 == 0 else None)
    print(f"tier3 agentic legs: {args.n_agent} tokens so far {len(ids)}")

    # tier 4: long-context synthesis legs (4k-16k chained documents)
    docs = []
    for i, s in enumerate(c4):
        if i >= 20000:
            break
        t = s.get("text", "")
        if len(t) > 400:
            docs.append(t)
        if len(docs) >= 4000:
            break
    for k in range(args.n_long):
        target = rng.choice([4096, 8192, 12288, 16384])
        parts, cur = [], 0
        while cur < target * 3:  # chars ~3x tokens; let the token cap trim
            parts.append(docs[rng.randrange(len(docs))])
            cur += len(parts[-1])
        corpus = "\n\n---\n\n".join(parts)
        emit_chat(
            [
                {
                    "role": "user",
                    "content": (
                        "You are given reference notes. Write a long, coherent "
                        "chronological synthesis of them. Write as much as "
                        "possible; do not stop early.\n\n" + corpus
                    ),
                }
            ]
        )
    print(f"tier4 long legs: {args.n_long} tokens so far {len(ids)}")

    # tier 5: 512k needle-in-haystack documents. Each document is a 512k-token
    # haystack of C4 text with a planted fact; half the needles sit within the
    # last 16k tokens (trainable long-range retrieval at the feasible window),
    # half at a uniformly random depth (512k-document exposure).
    if args.n_xlong:
        cap = 512 * 1024
        for k in range(args.n_xlong):
            target = 460 * 1024 * 3  # chars ~3x tokens; leaves headroom
            parts, cur = [], 0
            while cur < target:
                parts.append(docs[rng.randrange(len(docs))])
                cur += len(parts[-1])
            hay = "\n\n".join(parts)
            if k % 2:
                depth = rng.randrange(max(1, len(hay) - 16 * 1024 * 3))
            else:
                depth = rng.randrange(max(1, len(hay) - 16 * 1024 * 3),
                                      max(1, len(hay) - 64 * 3))
            code = f"ZK{rng.randrange(10**9):09d}"
            project = f"PROJECT-{rng.randrange(10**6):06d}"
            fact = (f"\n\nThe access code for {project} is {code}.\n\n")
            doc = hay[:depth] + fact + hay[depth:]
            emit_chat(
                [
                    {
                        "role": "user",
                        "content": (
                            "The following reference notes contain one access "
                            "code. Read them carefully.\n\n" + doc +
                            f"\n\nQuestion: what is the access code for "
                            f"{project}? Answer with the code only."
                        ),
                    },
                    {"role": "assistant", "content": code},
                ],
                cap=cap,
            )
            print(f"tier5 xlong doc {k + 1}/{args.n_xlong} tokens so far {len(ids)}")
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-code", type=int, default=5000)
    p.add_argument("--n-c4", type=int, default=3500)
    p.add_argument("--n-agent", type=int, default=800)
    p.add_argument("--n-long", type=int, default=48)
    p.add_argument("--n-xlong", type=int, default=8,
                   help="512k-token needle-in-haystack documents (tier 5)")
    p.add_argument("--max-sample", type=int, default=16384)
    p.add_argument("--val-frac", type=float, default=0.01)
    args = p.parse_args()

    from transformers import AutoTokenizer

    rng = random.Random(0)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    ids = build(tok, args, rng)

    n_val = max(4096, int(len(ids) * args.val_frac))
    # round the val boundary to a document-ish boundary (any cut is fine for
    # a token-stream LM; keep val contiguous from the end)
    train, val = ids[: len(ids) - n_val], ids[len(ids) - n_val :]
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(torch.tensor(train, dtype=torch.int64), f"{OUT_DIR}/train_tokens.pt")
    torch.save(torch.tensor(val, dtype=torch.int64), f"{OUT_DIR}/val_tokens.pt")
    print(
        f"saved {OUT_DIR}: train {len(train)} tokens, val {len(val)} tokens"
    )


if __name__ == "__main__":
    main()
