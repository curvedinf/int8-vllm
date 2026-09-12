#!/usr/bin/env python3
"""Echo leg for a saved drifted spec leg (committed_ids + eng_logprobs).

Feeds the exact token stream [prompt_ids + committed_ids] back through the
engine as a PREFILL with prompt_logprobs=1 (clean-path ground truth) and
compares the actual-token logprob per generated position against the
engine-recorded live logprob (eng_logprobs). Contexts are token-identical,
so any consistent difference = live spec-round distribution error.

Usage:
  VLLM_API_KEY=... python scripts/logprob_echo_leg.py --leg d11_lprobe2_leg1
"""
import argparse
import json
import os
import urllib.request

import torch

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
    p.add_argument("--leg", required=True)
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)

    ids = torch.load(f"logs/garble/{args.leg}_ids.pt", map_location="cpu",
                     weights_only=False)
    com = torch.load(f"logs/garble/{args.leg}_committed.pt", map_location="cpu",
                     weights_only=False)
    prompt_ids = [int(x) for x in ids["prompt_ids"]]
    committed = [int(x) for x in com["committed_ids"]]
    eng_lp = [float(x) for x in com["eng_logprobs"]]
    n = len(committed)
    print(f"leg {args.leg}: prompt {len(prompt_ids)} + committed {n}")

    # echo text: decode jointly to avoid a seam at the junction
    full_text = tok.decode(prompt_ids + committed)
    b = post({"model": MODEL, "prompt": full_text, "max_tokens": 1,
              "temperature": 0.0, "prompt_logprobs": 1,
              "return_token_ids": True})
    bch = b["choices"][0]
    full_ids = list(bch.get("prompt_token_ids") or [])
    plp = bch.get("prompt_logprobs")
    if not full_ids:
        # fall back to offline tokenization
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    print(f"echo: {len(full_ids)} tokens; seam-exact: "
          f"{full_ids[-n:] == committed}")

    strs = tok.convert_ids_to_tokens(full_ids)
    base = len(full_ids) - n
    rows = []
    miss = 0
    for i in range(n):
        pos = base + i
        entry = plp[pos] if plp and pos < len(plp) else None
        val = None
        if entry:
            want_id = str(full_ids[pos])
            want = strs[pos]
            for s, info in entry.items():
                lp = info["logprob"] if isinstance(info, dict) else info
                if s == want_id or s == want or (
                    isinstance(info, dict)
                    and info.get("decoded_token") in (want, want_id)
                ):
                    val = lp
                    break
        if val is None:
            miss += 1
            continue
        rows.append((i, eng_lp[i], val, eng_lp[i] - val))
    print(f"compared {len(rows)} positions ({miss} missing)")

    W = 100
    print(f"\n{'window':>8} {'mean(d)':>10} {'mean|d|':>10} {'max|d|':>9} "
          f"{'>0.3':>5} {'n':>5}")
    for w0 in range(0, n, W):
        seg = [r for r in rows if w0 <= r[0] < w0 + W]
        if not seg:
            continue
        big = sum(1 for r in seg if abs(r[3]) > 0.3)
        print(f"{w0:>8} {sum(r[3] for r in seg)/len(seg):10.5f} "
              f"{sum(abs(r[3]) for r in seg)/len(seg):10.5f} "
              f"{max(abs(r[3]) for r in seg):9.4f} {big:>5} {len(seg):>5}")

    with open(f"logs/garble/lp_echo_{args.leg}.json", "w") as f:
        json.dump({"rows": rows, "full_ids_len": len(full_ids), "miss": miss}, f)
    print(f"\nsaved logs/garble/lp_echo_{args.leg}.json")


if __name__ == "__main__":
    main()
