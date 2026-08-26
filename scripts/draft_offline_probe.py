#!/usr/bin/env python3
"""Offline DFlash2 drafter-correctness harness (single GPU).

Boots the draft model alone (TP1, no server), feeds it the aux hidden-state
sequence recorded from the target checkpoint's own forward on a short text,
and measures how often the drafter's first-step candidate (argmax of the
selector walk) equals the target's next token. Isolates drafter quality from
all serving plumbing (CUDA graphs, schedulers, TP4).

Rate >> 0% => drafter fine, bug is in serving plumbing.
Rate ~ 0%  => drafter/inputs are wrong (checkpoint, aux states, or scale).

Usage:
  HIP_VISIBLE_DEVICES=0 .venv/bin/python scripts/draft_offline_probe.py \
      [--draft <drafter-checkpoint-dir>] \
      [--text "The capital of France is"]
"""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DFLASH2_INT8_DIR = os.environ.get(
    "DFLASH2_INT8_DIR",
    os.path.expanduser(
        "~/.cache/int8-vllm/dflash2-int8/Qwen3.8-27B-DFlash2-GPTQ-8bit"
    ),
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default=DFLASH2_INT8_DIR,
                    help="draft checkpoint dir (env override: DFLASH2_INT8_DIR)")
    ap.add_argument("--text", default="The capital of France is")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.draft)
    ids = tok(args.text, return_tensors="pt").input_ids[0].tolist()
    print(f"prompt tokens ({len(ids)}): {ids[:12]}")

    # Part 1: target aux states. We don't have the target here — approximate
    # the aux concat with N(0,1) noise of the right shape to sanity-check the
    # drafter runs, but the REAL test below uses the drafter's own embedding
    # of the true prefix as context (precompute path takes hidden states).
    # Instead of faking: run the DRAFT model as its own "target" is invalid.
    # So: this harness measures DRAFT NEXT-TOKEN QUALITY against the actual
    # next tokens of a reference continuation recorded in --ref (jsonl of
    # {prefix_ids, next_id, aux_states}). For now, without target states we
    # can only smoke-test shapes and non-degeneracy of candidates.
    print("NOTE: full acceptance test needs recorded target aux states;")
    print("      this run performs the non-degeneracy smoke test only.")

    llm = LLM(
        model=args.draft,
        tensor_parallel_size=1,
        dtype="float16",
        max_model_len=512,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        disable_log_stats=True,
    )

    # The draft model as a causal LM proposes from its lm_head; run a normal
    # generation and check the outputs are not constant-token garbage.
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    outs = llm.generate([args.text, "List three colors:", "def fibonacci(n):"],
                        sp)
    degenerate = 0
    for o in outs:
        text = o.outputs[0].text
        toks = list(o.outputs[0].token_ids)
        uniq = len(set(toks))
        print(f"  out={text[:60]!r} tokens={toks[:12]} uniq={uniq}")
        if uniq <= 1:
            degenerate += 1
    print(f"degenerate outputs: {degenerate}/{len(outs)}")
    print("VERDICT:", "DRAFTER DEGENERATE (constant tokens) — checkpoint/weights issue"
          if degenerate else "drafter produces varied tokens in isolation")


if __name__ == "__main__":
    main()
