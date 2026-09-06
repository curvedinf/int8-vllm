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
    "List every distinct API, class, and function named in the document above, with one line each.",
    "Write a technical critique of the document above: structure, correctness, clarity.",
    "Rewrite the key algorithm from the document above in idiomatic Python with comments.",
    "Extract a structured outline of the document above (headings, sub-points).",
    "What questions does the document above leave unanswered? Enumerate them.",
    "Translate the main argument of the document above into plain English for a non-programmer.",
    "Identify inconsistencies, bugs, or risks in the document above and explain each.",
]

# 20k-token windows from different parts of the corpus (code/en/non-en mix)
WINDOWS = [
    (100_000, 122_000), (400_000, 422_000), (700_000, 722_000),
    (1_000_000, 1_022_000), (1_300_000, 1_322_000), (1_600_000, 1_622_000),
    (1_900_000, 1_922_000), (2_200_000, 2_222_000),
]


def run_leg(idx, tokens, tok, api_key):
    prompt = tok.decode(tokens.tolist())
    body = {
        "model": "qwen3.8-27b-gptq8",
        "messages": [{"role": "user",
                      "content": TASKS[idx] + "\n\nDOCUMENT:\n" + prompt}],
        "max_tokens": 4096,
        "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "seed": 1000 + idx,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"})
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=3600))
    dt = time.time() - t0
    msg = resp["choices"][0]["message"]
    out = msg.get("content") or msg.get("reasoning") or ""
    u = resp["usage"]
    return idx, out, u["prompt_tokens"], u["completion_tokens"], dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--api-key", default=os.environ.get("KLD_KEY", "test-key-local-only"))
    p.add_argument("--out-dir", default="logs/qat_validation")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "/home/curved/models/Qwen3.8-27B-GPTQ-8bit-gs128")
    v = torch.load(VAL, weights_only=True)
    outdir = pathlib.Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for i in (0, 1):  # sequential legs
        a, b = WINDOWS[i]
        idx, out, pt, ct, dt = run_leg(i, v[a:b], tok, args.api_key)
        (outdir / f"{args.tag}_leg{idx+1}.txt").write_text(out)
        print(f"leg{idx+1}: prompt {pt} out {ct} in {dt:.0f}s", flush=True)

    with cf.ThreadPoolExecutor(max_workers=6) as ex:  # C6 legs
        futs = [ex.submit(run_leg, i, v[WINDOWS[i][0]:WINDOWS[i][1]], tok,
                          args.api_key) for i in range(2, 8)]
        for f in cf.as_completed(futs):
            idx, out, pt, ct, dt = f.result()
            (outdir / f"{args.tag}_leg{idx+1}.txt").write_text(out)
            print(f"leg{idx+1}: prompt {pt} out {ct} in {dt:.0f}s", flush=True)
    print("BATTERY DONE:", args.tag, flush=True)


if __name__ == "__main__":
    main()
