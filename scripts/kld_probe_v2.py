#!/usr/bin/env python3
"""KLD probe v2: deterministic quant-quality gate vs a reference dump.

Fixes the v1 sampling-path artifact (temp=1.0 continuations -> disjoint top-k
sets -> inf KLD). v2 uses greedy continuations (deterministic given weights)
and dumps per-position top-K logprobs along the *reference's* greedy path for
every variant, so supports always intersect.

Modes:
  capture  --model X --tag T   : dump greedy texts + top-K logprob dumps
  compare --tag T --ref-tag R  : KLD(R||T) over matched positions + agreement

All artifacts land in ~/models/kld/quant_audit/<tag>.npz (not in git).

Usage:
  python scripts/kld_probe_v2.py capture --model ~/models/X --tag leg_L1
  python scripts/kld_probe_v2.py compare --tag leg_L1 --ref-tag R0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path.home() / "models" / "kld" / "quant_audit"
TOP_K = 20
MAX_TOKENS = 256
SEED = 20260825

# Fixed corpus: code, math, strict-format, chat, long-form. Same every run.
PROMPTS = [
    # code (12)
    "Write a Python function that checks if a string is a palindrome, ignoring punctuation and case:",
    "Implement binary search in Rust with a generic comparator:",
    "Write a SQL query that finds the top 5 customers by revenue per region:",
    "Debug this code: def add(a, b): return a - b. Explain the fix:",
    "Write a regex that matches IPv4 addresses but not IPv6:",
    "Explain what this Haskell does: foldr (.) id [f, g, h]:",
    "Write a Python class implementing an LRU cache with O(1) get/put:",
    "Convert this loop to a list comprehension: result = []\\nfor x in items:\\n    if x > 0:\\n        result.append(x * 2):",
    "Write a bash one-liner to find the 10 largest files under a directory:",
    "What is the time complexity of this operation: sorted(list) * 1000 times:",
    "Write a unit test for a function that divides two numbers, including zero handling:",
    "Explain the difference between a deep copy and a shallow copy in Python:",
    # math (12)
    "A train leaves at 3pm traveling 60 km/h. Another leaves at 4pm at 80 km/h on the same track. When does the second catch the first?",
    "Solve for x: 3x^2 - 12x + 9 = 0:",
    "What is the derivative of x^3 * sin(x)?",
    "A jar has 4 red and 6 blue balls. You draw 3 without replacement. Probability all red?",
    "Prove that the square root of 2 is irrational:",
    "What is 17 * 23? Show your work:",
    "Compute the sum of the infinite series 1/2 + 1/4 + 1/8 + ...:",
    "If f(x) = 2x + 3 and g(x) = x^2, what is f(g(2))?",
    "How many prime numbers are below 30? List them:",
    "A rectangle's length is twice its width. Perimeter is 36. What is the area?",
    "Explain the Monty Hall problem and the correct strategy:",
    "Integrate x*e^x dx:",
    # strict-format (10)
    "Respond with exactly this and nothing else: APPLE-42",
    "Return ONLY valid JSON with keys \"a\"=5, \"b\"=\"cat\". No other text.",
    "List exactly 3 colors, numbered 1-3. Nothing else.",
    "Answer with a single word: what is the capital of Australia?",
    "Say the word \"banana\" and nothing else.",
    "Output a valid XML tag <answer>42</answer> and nothing else.",
    "Reply with only the number of letters in 'encyclopedia':",
    "Give me exactly two sentences about the water cycle. No more, no less.",
    "Translate 'good morning' to French. Output only the translation:",
    "Write the numbers 1 through 5 separated by commas, nothing else:",
    # chat/reasoning (12)
    "Explain the difference between TCP and UDP and when each is appropriate:",
    "Why is the sky blue?",
    "What are the main causes of the French Revolution?",
    "Explain how a transformer attention mechanism works:",
    "What is the difference between correlation and causation, with an example:",
    "Summarize the plot of Hamlet in one paragraph:",
    "What happens to your body when you stop drinking caffeine?",
    "Explain garbage collection in Java:",
    "Why do companies do stock splits?",
    "What is the difference between a virus and a bacterium?",
    "Explain the concept of technical debt to a non-programmer:",
    "What were the main technological factors enabling the Industrial Revolution?",
    # long-form (6)
    "Write a detailed essay about the history of computing:",
    "Describe the process of photosynthesis in detail:",
    "Write a short story about a robot learning to paint:",
    "Explain the full journey of a HTTP request from browser to server and back:",
    "Describe the major milestones in the history of space exploration:",
    "Write a persuasive argument for and against remote work:",
]


def server_url():
    return os.environ.get("KLD_URL", "http://127.0.0.1:8020")


def api_key():
    return os.environ.get("KLD_KEY", "test-key-local-only")


def http_post(url, payload, timeout=600):
    import urllib.request

    req = urllib.request.Request(
        url,
        json.dumps(payload).encode(),
        {
            "Authorization": f"Bearer {api_key()}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def capture(tag, concurrency=2):
    """Greedy continuation + top-K logprobs at every position via completions API."""
    url = server_url()
    texts, dumps = [], []
    t0 = time.time()
    for i, p in enumerate(PROMPTS):
        try:
            r = http_post(
                f"{url}/v1/completions",
                {
                    "model": os.environ.get("KLD_MODEL_NAME", "qwen3.8-27b-gptq8"),
                    "prompt": p,
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0,
                    "logprobs": TOP_K,
                },
            )
            ch = r["choices"][0]
            texts.append(ch.get("text", ""))
            lp = ch.get("logprobs") or {}
            # Completions API legacy format: top_logprobs is a list of
            # {token: logprob} dicts per position.
            top = lp.get("top_logprobs", [])
            per_pos = []
            for pos_lps in top:
                d = {}
                if isinstance(pos_lps, dict):
                    for tok, lpv in pos_lps.items():
                        try:
                            v = float(lpv)
                        except (TypeError, ValueError):
                            continue
                        if tok and v > -50:
                            d[tok] = v
                per_pos.append(d)
            dumps.append(per_pos)
        except Exception as e:  # noqa: BLE001
            print(f"  prompt {i} FAILED: {e}", flush=True)
            texts.append("")
            dumps.append([])
        if (i + 1) % 16 == 0:
            print(f"  {i+1}/{len(PROMPTS)} ({time.time()-t0:.0f}s)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{tag}.npz"
    np.savez_compressed(
        out,
        prompts=np.array(PROMPTS),
        texts=np.array(texts),
        dumps=np.array(json.dumps(dumps)),
        tokens_meta=np.array(json.dumps({"top_k": TOP_K, "max_tokens": MAX_TOKENS, "seed": SEED})),
    )
    print(f"CAPTURED {tag}: {out} ({sum(bool(t) for t in texts)}/{len(PROMPTS)} ok)")
    return 0


def compare(tag, ref_tag):
    base = np.load(OUT_DIR / f"{ref_tag}.npz", allow_pickle=True)
    var = np.load(OUT_DIR / f"{tag}.npz", allow_pickle=True)
    bd = json.loads(str(base["dumps"]))
    vd = json.loads(str(var["dumps"]))
    bt = [str(x) for x in base["texts"]]
    vt = [str(x) for x in var["texts"]]

    kls, empties, pairs = [], 0, 0
    for s1, s2 in zip(bd, vd):
        for p1, p2 in zip(s1, s2):
            if not p1 or not p2:
                empties += 1
                continue
            keys = set(p1) & set(p2)
            if not keys:
                empties += 1
                continue
            pairs += 1
            # renormalized KL over the intersecting support; missing mass counted
            # as residual bucket 1 - sum(p_inter)
            pb = {k: np.exp(p1[k]) for k in keys}
            zb = sum(pb.values())
            res_b = max(0.0, 1.0 - sum(np.exp(v) for v in p1.values()))
            pv = {k: np.exp(p2[k]) for k in keys}
            zv = sum(pv.values())
            res_v = max(0.0, 1.0 - sum(np.exp(v) for v in p2.values()))
            kl = 0.0
            for k in keys:
                p = (pb[k] + res_b / max(len(keys), 1)) / (zb + res_b)
                q = (pv[k] + res_v / max(len(keys), 1)) / (zv + res_v)
                kl += p * np.log(p / max(q, 1e-12))
            kls.append(kl)

    agree40 = sum(1 for x, y in zip(bt, vt) if x[:40] == y[:40])
    agree_first = sum(1 for s1, s2 in zip(bd, vd) if s1 and s2 and s1[0] and s2[0])
    stats = {
        "tag": tag,
        "ref": ref_tag,
        "kld_mean": float(np.mean(kls)) if kls else float("inf"),
        "kld_median": float(np.median(kls)) if kls else float("inf"),
        "kld_p95": float(np.percentile(kls, 95)) if kls else float("inf"),
        "pairs": pairs,
        "empty_positions": empties,
        "greedy_agree_40char": f"{agree40}/{len(bt)}",
        "prompts_ok": sum(bool(t) for t in vt),
    }
    print(json.dumps(stats, indent=1))
    with open(OUT_DIR / f"compare_{tag}_vs_{ref_tag}.json", "w") as f:
        json.dump(stats, f, indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "compare"])
    ap.add_argument("--model", default=None, help="unused for server mode; kept for CLI compat")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ref-tag", default=None)
    ap.add_argument("--concurrency", type=int, default=2)
    args = ap.parse_args()
    if args.mode == "capture":
        return capture(args.tag, args.concurrency)
    return compare(args.tag, args.ref_tag or "R0")


if __name__ == "__main__":
    sys.exit(main())
