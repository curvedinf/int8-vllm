#!/usr/bin/env python3
"""Noise-floor control + spec-vs-replay divergence, one script.

1. Spec run (temp 1.0, logprobs per committed token).
2. Plain replay A of the same token stream (prompt_logprobs).
3. Plain replay B (identical call) — determinism floor.
4. Compare: A-vs-B (floor), spec-vs-A (divergence).
"""
import json
import os
import statistics
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_repro2 import get_tok, MODEL, API  # noqa

tok = get_tok()
PROMPT = ("Write a highly technical changelog with numbered entries about "
          "GPU kernel optimization. Go deep for many entries.")


def post(url, body, timeout=900):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY','')}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def replay_lps(full_text):
    out = post("http://127.0.0.1:8020/v1/completions", {
        "model": MODEL, "prompt": full_text, "max_tokens": 1,
        "temperature": 0.0, "logprobs": 0, "prompt_logprobs": 1})
    return out["choices"][0].get("prompt_logprobs") or []


def compare(spec_lps, gen_ids, pl, base, label):
    diffs = []
    n = min(len(spec_lps), len(gen_ids), len(pl) - base - 1)
    for k in range(n):
        lp = spec_lps[k].get("logprob")
        entry = pl[base + k]
        if not entry or lp is None:
            continue
        m = entry.get(str(gen_ids[k]))
        if m is None:
            continue
        diffs.append((abs(lp - m["logprob"]), k))
    ds = sorted(d[0] for d in diffs)
    big = [k for d, k in diffs if d > 0.5]
    print(f"{label}: {len(diffs)} pos, median={statistics.median(ds):.4f} "
          f"p90={ds[int(.9*len(ds))]:.4f} max={ds[-1]:.4f} "
          f"n>0.3={sum(1 for d in ds if d>0.3)} n>0.5={len(big)} "
          f"first>0.5={big[0] if big else None}")
    return {k: d for d, k in diffs}


def main():
    out = post(API, {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 500,
        "stream": False, "logprobs": True, "top_logprobs": 1, "seed": 31337,
        "chat_template_kwargs": {"enable_thinking": False}})
    ch = out["choices"][0]
    spec_lps = ch.get("logprobs", {}).get("content") or []
    spec_text = (ch.get("message") or {}).get("content") or ""
    gen_ids = tok.encode(spec_text, add_special_tokens=False)
    tmpl_str = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    full = tmpl_str + spec_text
    base = len(tok.encode(tmpl_str, add_special_tokens=False))
    plA = replay_lps(full)
    plB = replay_lps(full)
    # floor: A vs B (both keyed by gen id at base+k)
    floor = {}
    for k in range(min(len(gen_ids), len(plA) - base - 1)):
        ea, eb = plA[base + k], plB[base + k]
        if not ea or not eb:
            continue
        ma, mb = ea.get(str(gen_ids[k])), eb.get(str(gen_ids[k]))
        if ma and mb:
            floor[k] = abs(ma["logprob"] - mb["logprob"])
    fvals = sorted(floor.values())
    print(f"floor (A vs B): {len(fvals)} pos, median={statistics.median(fvals):.4f} "
          f"max={fvals[-1]:.4f} n>0.3={sum(1 for d in fvals if d>0.3)}")
    spec_d = compare(spec_lps, gen_ids, plA, base, "spec-vs-replay")
    # overlap: big spec divergences where floor is small = real corruption
    real = [(k, round(spec_d[k], 2)) for k, d in spec_d.items()
            if d > 0.5 and floor.get(k, 0) < 0.1]
    print(f"REAL divergence (spec>0.5 & floor<0.1): {len(real)} positions:")
    print(" ", sorted(real)[:20])


if __name__ == "__main__":
    main()
