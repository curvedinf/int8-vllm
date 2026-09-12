#!/usr/bin/env python3
"""Logits-level drift A/B: spec-live generation vs prefill-echo ground truth.

Motivation: every in-vitro spec compute surface is exonerated (conv rolling
buffer, GDN in-chunk recurrence, KV read, GEMM shapes, sampler, resume
index), yet the live engine drifts at temp 1.0 after ~2-3k output tokens.
This measures the LIVE model's distributional health directly:

  Leg A (spec-live): generate 4k tokens on a 20k-token prompt with the prod
      sampler (temp 1.0 / top_p .95 / top_k 20), recording each generated
      token's chosen-token logprob as computed by the verify forward.
  Leg B (ground truth): feed the EXACT same token stream (prompt +
      generated ids) back through the engine as a PREFILL with
      prompt_logprobs=1 — states built by the clean prefill path.

At every generated position the context is token-identical between legs, so
chosen-token logprobs must agree to numerical noise. A divergence that
grows toward the garble onset = per-round state drift convicted (and the
curve localizes the onset). No divergence = the model's distribution is
healthy end-to-end and the fault lives in token selection/acceptance.

Usage:
  VLLM_API_KEY=... python scripts/logprob_drift_ab.py --tag ab1 \
      [--in-tokens 20000] [--out-tokens 4096] [--seed 5]
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_repro import build_prompt  # noqa: E402

API = "http://127.0.0.1:8020/v1/completions"
TOK_PATH = "/home/curved/models/Qwen3.8-27B-PTQR-R10S60"
MODEL = "qwen3.8-27b-gptq8"


def post(body, timeout=7200):
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', '')}"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--in-tokens", type=int, default=20000)
    p.add_argument("--out-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--temp", type=float, default=1.0)
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)

    # Raw completions (no chat template): BOTH legs share the identical
    # raw sequence; token ids come back from the server.
    nonce = f"{args.tag}-{int(time.time())}"
    corpus = build_prompt(args.in_tokens, nonce)
    prompt_text = (
        "You are given reference notes. Write a long, coherent "
        "chronological essay synthesizing them. Write as much as "
        "possible; do not stop early.\n\n" + corpus)

    # ---- Leg A: spec-live generation with chosen-token logprobs ----
    t0 = time.time()
    a = post({
        "model": MODEL,
        "prompt": prompt_text,
        "max_tokens": args.out_tokens,
        "temperature": args.temp,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": args.seed,
        "logprobs": 1,
        "return_token_ids": True,
    })
    ch = a["choices"][0]
    prompt_ids = list(ch["prompt_token_ids"])
    gen_ids = list(ch["token_ids"])
    a_lp = list(ch["logprobs"]["token_logprobs"])
    assert len(gen_ids) == len(a_lp)
    print(f"leg A: prompt {len(prompt_ids)} + {len(gen_ids)} gen tokens "
          f"in {time.time()-t0:.0f}s")

    # ---- Leg B: prefill-echo ground truth over the identical stream ----
    full_text = prompt_text + tok.decode(gen_ids)
    t0 = time.time()
    b = post({
        "model": MODEL,
        "prompt": full_text,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": 1,
    })
    bch = b["choices"][0]
    b_plp = bch.get("prompt_logprobs")
    full_ids = list(bch["prompt_token_ids"])
    print(f"leg B: echo of {len(full_ids)} tokens in {time.time()-t0:.0f}s")
    assert b_plp is not None, "no prompt_logprobs in response"
    # Seam check: retokenized tail must equal the generated ids exactly,
    # otherwise positions shift and the comparison is invalid.
    seam_ok = full_ids[-len(gen_ids):] == gen_ids
    print(f"seam retokenization exact: {seam_ok}")

    # Extract actual-token logprob per prompt position. Entries are dicts
    # {token_str: {"logprob":..., "rank":...}} (or None for pos 0); the
    # actual token's entry is matched by token string.
    tok_strs = tok.convert_ids_to_tokens(full_ids)
    b_lp = []
    for i, entry in enumerate(b_plp):
        if entry is None:
            b_lp.append(None)
            continue
        want = tok_strs[i]
        val = None
        for tid_str, info in entry.items():
            lp = info["logprob"] if isinstance(info, dict) else info
            if tid_str == want or (isinstance(info, dict)
                                   and info.get("decoded_token") == want):
                val = lp
                break
        b_lp.append(val)

    # ---- Compare over the generated region ----
    n = len(gen_ids)
    diffs = []
    missing = 0
    base = len(full_ids) - n     # position of gen_ids[0] inside full_ids
    for i in range(n):
        pos = base + i
        bv = b_lp[pos] if pos < len(b_lp) else None
        if bv is None:
            missing += 1
            continue
        diffs.append((i, a_lp[i], bv, a_lp[i] - bv))
    print(f"compared {len(diffs)} positions ({missing} missing actual-token "
          "logprobs in leg B)")

    # Divergence curve in 256-token windows
    W = 256
    print(f"\n{'window':>10} {'mean(a-b)':>12} {'mean|a-b|':>12} "
          f"{'max|a-b|':>10} {'n':>5}")
    for w0 in range(0, n, W):
        seg = [d for d in diffs if w0 <= d[0] < w0 + W]
        if not seg:
            continue
        m = sum(d[3] for d in seg) / len(seg)
        ma = sum(abs(d[3]) for d in seg) / len(seg)
        mx = max(abs(d[3]) for d in seg)
        print(f"{w0:>10} {m:12.5f} {ma:12.5f} {mx:10.5f} {len(seg):>5}")

    out = {
        "tag": args.tag, "nonce": nonce, "seed": args.seed,
        "prompt_len": len(prompt_ids), "gen_len": n,
        "gen_ids": gen_ids, "a_lp": a_lp, "b_lp": b_lp,
        "prompt_ids_tail": prompt_ids[-8:],
        "diffs": diffs,
    }
    path = f"logs/garble/lp_ab_{args.tag}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    text = tok.decode(gen_ids)
    with open(f"logs/garble/lp_ab_{args.tag}.txt", "w") as f:
        f.write(text)
    print(f"\nsaved {path} (+.txt)")


if __name__ == "__main__":
    main()
