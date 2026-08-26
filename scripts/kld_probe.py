#!/usr/bin/env python3
"""KLD probe: distribution-shift gate for int8-native changes on gfx908.

Baseline mode captures the reference model's per-position top-20 logprob
distribution over a fixed 64-prompt corpus (seed pinned). Variant mode
loads the dump and reports mean KL(D_base || D_var) plus greedy-prefix
agreement. Exit code 0 = pass, 2 = marginal (see thresholds), 3 = fail.

Usage:
  # capture baseline (run against the current prod config once)
  HIP_VISIBLE_DEVICES=0,1,2,3 python scripts/kld_probe.py capture \
      --model <models>/Qwen3.8-27B-GPTQ-8bit-gs128 --out <kld-dir>/q38_gs128.npz

  # evaluate a variant
  HIP_VISIBLE_DEVICES=0,1,2,3 python scripts/kld_probe.py compare \
      --model <models>/Qwen3.8-27B-GPTQ-8bit-gs128 --base <kld-dir>/q38_gs128.npz

Artifacts default to KLD_DIR (env override), defaulting to
~/.cache/int8-vllm/kld.

Gate (from the int8-native program plan): KLD <= 0.02 and agreement >= 85%
passes; KLD in (0.02, 0.05] passes only with a measured perf gain >= 5%
(passed via --perf-gain); above fails.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

TOP_K = 20
MAX_TOKENS = 256
SEED = 1234

KLD_DIR = Path(
    os.environ.get("KLD_DIR", Path.home() / ".cache" / "int8-vllm" / "kld")
)

# Fixed corpus: 32 mixed code/general (mirrors quant calibration mix),
# 16 code, 16 long-context chains. Same prompts for capture and compare.
PROMPTS = [
    "### Instruction:\nExplain recursion to a junior developer.\n\n### Response:\n",
    "### Instruction:\nWrite a SQL query joining three tables on user id.\n\n### Response:\n",
    "### Instruction:\nWhat is the capital of Australia?\n\n### Response:\n",
] * 10 + [  # 30 mixed (dedup by use of varied content follows)
    "### Instruction:\nSummarize the water cycle.\n\n### Response:\n",
    "### Instruction:\nDebug: why does my Python list append return None?\n\n### Response:\n",
]
CODE_PROMPTS = [
    "Write a Python function that checks if a string is a palindrome, ignoring punctuation and case:",
    "Implement a thread-safe LRU cache in Python with O(1) get/put.",
    "Refactor this C function to remove the off-by-one error in the loop bound.",
    "Write a Rust function that parses an ISO-8601 timestamp without external crates.",
    "Explain what this regex does: ^(?=.*[A-Z])(?=.*\\d)[A-Za-z\\d]{8,}$",
    "Generate a Makefile target that builds a shared library from src/*.c.",
    "Write a bash one-liner to find the 10 largest files under a directory.",
    "Translate this Python class into idiomatic Go, preserving behavior.",
    "Add type hints and docstrings to this function: def process(items, flag=False): ...",
    "Write a unit test for a function that retries on ConnectionError with backoff.",
    "Convert this recursive fibonacci to an iterative version with memoization.",
    "Write SQL to compute a 7-day rolling average per user from an events table.",
    "Sketch a Dockerfile for a FastAPI app with a healthcheck and non-root user.",
    "Write a git pre-commit hook that blocks commits touching generated files.",
    "Optimize this numpy loop using vectorized operations.",
    "Write a regular expression to extract the hostname from varied URL formats.",
]
LONG_PROMPTS = [
    "\n\n---\n\n".join(
        ["The history of computing begins with mechanical calculators.",
         "Babbage designed the Difference Engine to tabulate polynomials.",
         "The Analytical Engine anticipated modern architecture with its mill and store.",
         "Ada Lovelace wrote what is often called the first program for it.",
         "A century later, electromechanical relays gave way to vacuum tubes.",
         "The ENIAC weighed 27 tons and performed ballistics calculations.",
         "Transistors replaced tubes, then integrated circuits replaced transistors.",
         "Moore's law described the doubling of density roughly every two years.",
         "Microprocessors put a CPU on a single chip by the early 1970s.",
         "The personal computer democratized access to computation."] * i
    )
    + "\n\nSummarize the key inflection points above:"
    for i in (2, 3, 4)
] * 5 + [
    ("Context: The ROCm software stack provides HIP as a CUDA-like portable API. "
     "gfx908 (MI100) is the first CDNA GPU with Matrix Cores targeting matrix "
     "operations. MFMA instructions operate on matrix fragments held in "
     "registers, with accumulator fragments in fp32 regardless of input dtype. "
     "The int8 path packs four int8 elements per 32-bit lane and doubles "
     "throughput relative to fp16, which packs two. " * 6)
    + "\n\nBased on this context, explain why int8 is preferred on MI100:"
]

PROMPTS = PROMPTS[:32] + CODE_PROMPTS[:16] + LONG_PROMPTS[:16]
assert len(PROMPTS) == 64


def run_model(model_dir: str, need_logprobs: bool):
    """Generate with fixed seed, return (greedy_texts, logprob dumps)."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_dir,
        tensor_parallel_size=2,
        dtype="half",
        max_model_len=8192,
        gpu_memory_utilization=0.92,
        kv_cache_dtype="int8_per_token_head",
        attention_backend="TRITON_ATTN",
        seed=SEED,
        enforce_eager=True,  # determinism over speed for the gate
        disable_log_stats=True,
    )
    sp_greedy = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    greedy_out = llm.generate(PROMPTS, sp_greedy)

    lp_dump = None
    if need_logprobs:
        sp_lp = SamplingParams(
            temperature=1.0, max_tokens=MAX_TOKENS,
            logprobs=TOP_K, seed=SEED,
        )
        lp_out = llm.generate(PROMPTS, sp_lp)
        lp_dump = [
            [
                {
                    int(tid): float(lp.logprob)
                    for tid, lp in (pos or {}).items()
                    if lp is not None
                }
                for pos in o.outputs[0].logprobs  # per generated position (None for first)
                if pos is not None
            ]
            for o in lp_out
        ]
    texts = [o.outputs[0].text for o in greedy_out]
    del llm
    import torch
    torch.cuda.empty_cache()
    return texts, lp_dump


def greedy_agreement(base_texts, var_texts):
    """Mean fraction of tokens before first divergence (capped at MAX_TOKENS)."""
    ratios = []
    for b, v in zip(base_texts, var_texts):
        bt, vt = b.split(), v.split()
        n = 0
        for x, y in zip(bt, vt):
            if x != y:
                break
            n += 1
        ratios.append(n / max(len(bt), 1))
    return float(np.mean(ratios))


def kl_divergence(base_dump, var_dump):
    """Mean KL(base || var) over all positions with overlapping support."""
    kls = []
    missing_mass = 0.0
    for base_sent, var_sent in zip(base_dump, var_dump):
        for base_pos, var_pos in zip(base_sent, var_sent):
            if not base_pos:
                continue
            # Renormalize over the union of top-k supports.
            vb = np.array(list(base_pos.values()))
            keys = set(base_pos) & set(var_pos)
            miss = 1.0 - sum(np.exp(base_pos[k]) for k in keys)
            missing_mass = max(missing_mass, miss)
            if not keys:
                continue
            p = {k: np.exp(base_pos[k]) for k in keys}
            z = sum(p.values()) + (1.0 - sum(np.exp(list(base_pos.values()))))
            p = {k: v / z for k, v in p.items()}
            q = {k: np.exp(var_pos[k]) for k in keys}
            zq = sum(q.values()) + (1.0 - sum(np.exp(list(var_pos.values()))))
            q = {k: v / zq for k, v in q.items()}
            # Smoothing for keys missing on either side.
            eps = 1e-9
            kl = sum(
                p[k] * np.log(p[k] / (q.get(k, eps) + eps)) for k in p
            )
            kls.append(kl)
    return (float(np.mean(kls)) if kls else float("inf")), missing_mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "compare"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--perf-gain", type=float, default=None,
                    help="measured perf gain %% to justify a marginal KLD")
    ap.add_argument("--max-kld", type=float, default=0.02)
    ap.add_argument("--min-agree", type=float, default=0.85)
    args = ap.parse_args()

    if args.mode == "capture":
        texts, dump = run_model(args.model, need_logprobs=True)
        out = args.out or str(KLD_DIR / (Path(args.model).name + ".npz"))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            prompts=np.array(PROMPTS),
            texts=np.array(texts),
            dumps=np.array(json.dumps(dump)),
        )
        print(f"BASELINE SAVED: {out} ({len(texts)} prompts)")
        return 0

    # compare
    base = np.load(args.base, allow_pickle=True)
    base_texts = list(base["texts"])
    base_dump = json.loads(str(base["dumps"]))
    var_texts, var_dump = run_model(args.model, need_logprobs=True)

    kld, miss = kl_divergence(base_dump, var_dump)
    agree = greedy_agreement(base_texts, var_texts)
    print(f"KLD(base||var): {kld:.5f}  (worst top-k missing mass {miss:.3f})")
    print(f"Greedy prefix agreement: {agree:.1%}")
    verdict = "PASS"
    if kld > args.max_kld * 2.5 or agree < args.min_agree:
        verdict = "FAIL"
    elif kld > args.max_kld:
        verdict = "MARGINAL"
        if args.perf_gain is None or args.perf_gain < 5.0:
            verdict = "FAIL (marginal without >=5% perf gain)"
    print(f"GATE: {verdict}")
    row = {
        "model": args.model, "kld": round(kld, 6), "agreement": round(agree, 4),
        "verdict": verdict, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    KLD_DIR.mkdir(parents=True, exist_ok=True)
    KLD_DIR.joinpath("results.jsonl").open("a").write(
        json.dumps(row) + "\n"
    )
    return 0 if verdict.startswith("PASS") else (2 if verdict == "MARGINAL" else 3)


if __name__ == "__main__":
    sys.exit(main())
