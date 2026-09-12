#!/usr/bin/env python3
"""Noise-floor controls for logprob_drift_ab.py's methodology.

The drift curve compares LIVE spec-round logprobs against a prefill-echo
replay of the same token stream. Before convicting the verify path, the
methodology itself needs a floor:

  C1 (determinism): the SAME full stream echoed twice — replays must be
      bit-identical; any difference = run-to-run prefill nondeterminism
      (chunk scheduling/batching), which alone pollutes the A/B.
  C2 (chunk-boundary shift): echo the same stream with a different
      prompt/output split (shift the junction by S tokens) — the junction
      sits at a different chunk boundary, so fp reduction-order in the
      chunked GDN/attention path differs. The measured divergence over the
      output region = the fp-noise floor of "replay vs replay".

Usage:
  VLLM_API_KEY=... python scripts/logprob_ab_controls.py --ab-tag ab1
"""
import argparse
import json
import os
import sys
import urllib.request

API = "http://127.0.0.1:8020/v1/completions"
TOK_PATH = "/home/curved/models/Qwen3.8-27B-PTQR-R10S60"
MODEL = "qwen3.8-27b-gptq8"


def post(body, timeout=7200):
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', '')}"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def echo(full_text):
    b = post({"model": MODEL, "prompt": full_text, "max_tokens": 1,
              "temperature": 0.0, "prompt_logprobs": 1})
    ch = b["choices"][0]
    return list(ch["prompt_token_ids"]), ch["prompt_logprobs"]


def actual_lp(plp, full_ids, tok):
    strs = tok.convert_ids_to_tokens(full_ids)
    out = []
    for i, entry in enumerate(plp):
        if entry is None:
            out.append(None)
            continue
        want = strs[i]
        val = None
        for s, info in entry.items():
            lp = info["logprob"] if isinstance(info, dict) else info
            if s == want or (isinstance(info, dict)
                             and info.get("decoded_token") == want):
                val = lp
                break
        out.append(val)
    return out


def curve(a, b, base, n, tag):
    diffs = []
    for i in range(n):
        av, bv = a[base + i], b[base + i]
        if av is None or bv is None:
            continue
        diffs.append((i, av - bv))
    W = 256
    print(f"\n{tag}")
    print(f"{'window':>8} {'mean':>10} {'mean|d|':>10} {'max|d|':>9} {'n':>5}")
    for w0 in range(0, n, W):
        seg = [d for i, d in diffs if w0 <= i < w0 + W]
        if not seg:
            continue
        print(f"{w0:>8} {sum(seg)/len(seg):10.5f} "
              f"{sum(abs(x) for x in seg)/len(seg):10.5f} "
              f"{max(abs(x) for x in seg):9.4f} {len(seg):>5}")
    return diffs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ab-tag", default="ab1")
    p.add_argument("--shift", type=int, default=512)
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)

    with open(f"logs/garble/lp_ab_{args.ab_tag}.json") as f:
        ab = json.load(f)
    # Rebuild the full stream text from the A/B record.
    prompt_ids, gen_ids = ab.get("prompt_ids"), ab.get("gen_ids")
    # prompt_ids_tail only stored; full prompt text is not — regenerate via
    # the saved nonce if present, else fail with instructions.
    if prompt_ids is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from garble_repro import build_prompt
        nonce = ab["nonce"]
        corpus = build_prompt(20000, nonce)
        prompt_text = ("You are given reference notes. Write a long, coherent "
                       "chronological essay synthesizing them. Write as much "
                       "as possible; do not stop early.\n\n" + corpus)
        full_text = prompt_text + tok.decode(ab["gen_ids"])
    else:
        full_text = tok.decode(prompt_ids + ab["gen_ids"])

    n = len(ab["gen_ids"])

    ids1, plp1 = echo(full_text)
    lp1 = actual_lp(plp1, ids1, tok)
    ids2, plp2 = echo(full_text)
    lp2 = actual_lp(plp2, ids2, tok)
    base = len(ids1) - n
    curve(lp1, lp2, base, n, "C1 determinism: replay vs identical replay")

    # C2: junction shifted — move the last `shift` prompt tokens into the
    # "output" side measurement region... physically the same text; instead
    # shift by requesting a shorter prompt and longer tail is impossible
    # with echo. Use the A/B's own seam difference: compare replay1 vs the
    # A/B's stored leg-B values (different run, same stream).
    b_lp = ab.get("b_lp", [])
    if len(b_lp) >= base + n:
        curve(lp1, b_lp, base, n, "C2 cross-run: this replay vs A/B replay")
    else:
        print("C2 skipped: A/B b_lp missing/short")

    out = {"ids_len": len(ids1), "base": base, "n": n,
           "lp1": lp1, "lp2": lp2}
    with open("logs/garble/lp_ab_controls.json", "w") as f:
        json.dump(out, f)
    print("\nsaved logs/garble/lp_ab_controls.json")


if __name__ == "__main__":
    main()
