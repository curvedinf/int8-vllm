#!/usr/bin/env python3
"""Capture/compare served top-20 logprobs for the lm_head int8 quality test.

capture: greedy 64-token continuations for fixed random token-id prompts,
         saving per-position top-20 (token, logprob) from the API.
compare: top-1 agreement, top-20 set overlap, and KLD over matched sets.

Usage:
  probe_lm_head_quality.py capture --tag head_bf16
  probe_lm_head_quality.py compare --ref head_bf16 --cand head_i8
"""
import argparse
import json
import math
import os
import random
import urllib.request

OUT = os.environ.get("LMHEAD_OUT", "/home/curved/.cache/int8-vllm/lm_head_probe")
VOCAB = 150144


def _post(url, key, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def capture(args):
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(1234)
    prompts = [[rng.randrange(VOCAB) for _ in range(256)] for _ in range(6)]
    out = []
    url = args.base_url.rstrip("/") + "/v1/completions"
    for i, ids in enumerate(prompts):
        # Teacher forcing: echo the fixed prompt and read its per-position
        # top-20 logprobs. Greedy continuations diverge between heads and
        # would compare different contexts.
        body = {"model": args.model, "prompt": ids, "max_tokens": 0,
                "echo": True, "temperature": 0.0, "logprobs": 20}
        d = _post(url, args.api_key, body)
        ch = d["choices"][0]
        toks = ch.get("token_ids")
        lp = ch.get("logprobs") or {}
        out.append({"prompt": i, "token_ids": toks,
                    "top": lp.get("top_logprobs") or lp.get("top") or []})
        print(f"captured prompt {i}: {len(out[-1]['top'])} positions",
              flush=True)
    path = os.path.join(OUT, f"{args.tag}.json")
    with open(path, "w") as fh:
        json.dump(out, fh)
    print("wrote", path)


def _pos_table(entry):
    """Per position: {token_text: logprob}. The API returns top_logprobs as a
    list of {token_str: logprob} dicts (no token ids)."""
    table = {}
    for pos, cands in enumerate(entry["top"]):
        if not cands:
            continue
        if isinstance(cands, dict):
            h = {str(k): float(v) for k, v in cands.items()}
        else:
            h = {str(c): float(v) for c, v in cands.items()}
        table[pos] = h
    return table


def compare(args):
    ref = json.load(open(os.path.join(OUT, f"{args.ref}.json")))
    cand = json.load(open(os.path.join(OUT, f"{args.cand}.json")))
    n = min(len(ref), len(cand))
    top1 = agree = 0
    kl_sum = 0.0
    kl_n = 0
    overlap_sum = 0.0
    for i in range(n):
        a, b = _pos_table(ref[i]), _pos_table(cand[i])
        for pos in sorted(set(a) & set(b)):
            ha, hb = a[pos], b[pos]
            ta = max(ha, key=ha.get)
            tb = max(hb, key=hb.get)
            top1 += 1
            agree += int(ta == tb)
            shared = set(ha) & set(hb)
            overlap_sum += len(shared) / max(1, len(ha))
            # KLD over the shared support, renormalized per side
            sa = sum(math.exp(ha[t]) for t in shared)
            sb = sum(math.exp(hb[t]) for t in shared)
            if sa <= 0 or sb <= 0:
                continue
            kl = 0.0
            for t in shared:
                pa = math.exp(ha[t]) / sa
                pb = math.exp(hb[t]) / sb
                if pa > 0 and pb > 0:
                    kl += pa * math.log(pa / pb)
            kl_sum += kl
            kl_n += 1
    print(json.dumps({
        "positions": top1,
        "top1_agreement": agree,
        "top1_agree_pct": round(100.0 * agree / max(1, top1), 3),
        "top20_overlap_pct": round(100.0 * overlap_sum / max(1, top1), 3),
        "kld_mean": round(kl_sum / max(1, kl_n), 6),
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "compare"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8030")
    ap.add_argument("--api-key", default="recipe-local")
    ap.add_argument("--model", default="flint")
    ap.add_argument("--tag")
    ap.add_argument("--ref")
    ap.add_argument("--cand")
    args = ap.parse_args()
    if args.mode == "capture":
        capture(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
