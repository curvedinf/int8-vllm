#!/usr/bin/env python3
"""Accumulation-curve scorer for drift legs (any temperature).

Inputs: a PRING dump (per-round natural committed rows: tok ids + target p)
and the leg's *_ids.pt (exact prompt token ids). Builds the full committed
sequence, asks the server for clean-prefill prompt_logprobs over it, and
buckets |log p_engine - log p_clean| by output position.

The PRING rows are pre-ACCEPT1-clamp (natural rows); for natural legs the
committed tokens are exactly the row's valid entries.

Usage:
  VLLM_API_KEY=... python scripts/score_leg_curve.py <pring_dump> \
      <ids_pt> [--buckets 8] [--max-out 3072]
"""
import argparse
import io
import math
import pickle
import urllib.request

import torch

API = "http://127.0.0.1:8020/v1/completions"
MODEL = "qwen3.8-27b-gptq8"


def load_stream(path):
    recs = []
    with open(path, "rb") as f:
        bio = io.BytesIO(f.read())
    while True:
        try:
            r = pickle.load(bio)
        except Exception:
            break
        if isinstance(r, dict):
            recs.append(r)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pring")
    ap.add_argument("ids_pt")
    ap.add_argument("--buckets", type=int, default=8)
    ap.add_argument("--max-out", type=int, default=3072)
    args = ap.parse_args()

    recs = load_stream(args.pring)
    # keep the longest-rs stream (the leg request); order by record order
    from collections import Counter
    cnt = Counter(r.get("rs") for r in recs)
    rs = cnt.most_common(1)[0][0]
    rows = [r for r in recs if r.get("rs") == rs]
    print(f"{len(rows)} PRING rounds (rs={rs})")

    committed, eng_p = [], []
    for r in rows:
        toks = [t for t in (r.get("tok") or []) if t is not None and t >= 0]
        ps = r.get("p") or []
        for j, t in enumerate(toks):
            committed.append(int(t))
            eng_p.append(float(ps[j]) if j < len(ps) and ps[j] is not None
                         else float("nan"))
    committed = committed[: args.max_out]
    eng_p = eng_p[: args.max_out]
    print(f"{len(committed)} committed tokens")

    prompt_ids = torch.load(args.ids_pt, weights_only=False)["prompt_ids"]
    full = list(prompt_ids) + committed
    n_prompt = len(prompt_ids)

    body = {
        "model": MODEL,
        "prompt": full,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": 1,
    }
    req = urllib.request.Request(
        API, data=__import__("json").dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {__import__('os').environ.get('VLLM_API_KEY', '')}"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = __import__("json").loads(r.read())
    pl = out["choices"][0].get("prompt_logprobs") or []
    print(f"clean prompt_logprobs rows: {len(pl)}")

    # align: prompt_logprobs[0] is None; row i covers token i.
    diffs, flips = [], []
    for j in range(len(committed)):
        pos = n_prompt + j
        if pos >= len(pl):
            break
        d = pl[pos]
        if not d:
            continue
        e = next(iter(d.values()))
        lp_clean = e["logprob"]
        lp_eng = math.log(eng_p[j]) if eng_p[j] and eng_p[j] > 0 else None
        if lp_eng is None:
            continue
        diffs.append(abs(lp_eng - lp_clean))
        # top-1 flip: engine argmax vs clean argmax at this position
        top_clean = e.get("bytes")  # not reliable; skip flip detection here

    B = args.buckets
    n = len(diffs)
    step = max(n // B, 1)
    print("\n|logP(eng) - logP(clean)| by output-position bucket:")
    for b in range(B):
        seg = diffs[b * step: (b + 1) * step]
        if not seg:
            continue
        seg_sorted = sorted(seg)
        med = seg_sorted[len(seg) // 2]
        p90 = seg_sorted[int(len(seg) * 0.9)]
        big = sum(1.0 for x in seg if x > 1.0) / len(seg)
        print(f"  out[{b * step:5d}:{(b + 1) * step:5d}] "
              f"mean={sum(seg)/len(seg):7.4f} med={med:7.4f} "
              f"p90={p90:7.4f} frac>1nat={big:6.4f}")


if __name__ == "__main__":
    main()
