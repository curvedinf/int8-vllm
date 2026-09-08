#!/usr/bin/env python3
"""Curve scorer for specoff_leg.py outputs (per-token engine logprobs saved
client-side). Compares the engine's greedy logprob per committed token
against a clean prefill of the same transcript.

Usage:
  VLLM_API_KEY=... python scripts/score_specoff_curve.py <tag_ids.pt>
"""
import argparse
import json
import math
import os
import urllib.request

import torch

API = "http://127.0.0.1:8020/v1/completions"
MODEL = "qwen3.8-27b-gptq8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids_pt")
    ap.add_argument("--max-out", type=int, default=3072)
    args = ap.parse_args()

    d = torch.load(args.ids_pt, weights_only=False)
    prompt_ids = d["prompt_ids"]
    committed = d["committed_ids"][: args.max_out]
    eng_lp = d["eng_logprobs"][: args.max_out]
    full = list(prompt_ids) + committed
    body = {"model": MODEL, "prompt": full, "max_tokens": 1,
            "temperature": 0.0, "prompt_logprobs": 1}
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', '')}"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    pl = out["choices"][0].get("prompt_logprobs") or []
    np_ = len(prompt_ids)
    diffs = []
    for j in range(len(committed)):
        pos = np_ + j
        if pos >= len(pl):
            break
        e = pl[pos]
        if not e:
            continue
        lp_clean = next(iter(e.values()))["logprob"]
        diffs.append((j, abs(eng_lp[j] - lp_clean)))
    print(f"{len(diffs)} scored")
    for b in range(8):
        seg = [x for j, x in diffs if b * 384 <= j < (b + 1) * 384]
        if not seg:
            continue
        segs = sorted(seg)
        print(f"  out[{b*384:5d}:{(b+1)*384:5d}] mean={sum(seg)/len(seg):7.4f} "
              f"med={segs[len(seg)//2]:7.4f} p90={segs[int(len(seg)*.9)]:7.4f} "
              f"frac>0.5nat={sum(1.0 for x in seg if x>0.5)/len(seg):6.4f}")


if __name__ == "__main__":
    main()
