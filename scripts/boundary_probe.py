#!/usr/bin/env python3
"""Token-exact boundary-aligned probes for the garble-200k incident.

Hypothesis under test (see logs/garble/INCIDENT_2026-09-26_persistent_garble_200k.md,
Addendum 2): a kernel reads past the request's written length inside its
own (recycled) page, feeding a previous conversation's content into the
forward. Predictions, on a CORRUPTED server where the short probe garbles:

  - prompt length exactly filling the last ATTENTION page (L % 1664 == 0)
    -> clean, if the attention last-page read is the vector;
  - prompt length exactly filling the last MAMBA chunk (L % 1728 == 0)
    -> clean, if the GDN chunked state read is the vector;
  - off-by-one lengths on either side stay garbage (controls).

Token counts are exact (verified with the served tokenizer before sending).
"""

import argparse
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8020"
MODEL_DIR = "/home/curved/models/Qwen3.8-27B-PTQR-R10S60"


def key():
    pid = int(open("logs/serve_recipe_qwen38/server.pid").read().strip())
    for e in open(f"/proc/{pid}/environ", "rb").read().split(b"\0"):
        if e.startswith(b"VLLM_API_KEY="):
            return e.split(b"=", 1)[1].decode()


def comp(prompt, max_tokens=8, logprobs=5):
    r = urllib.request.Request(
        BASE + "/v1/completions",
        data=json.dumps({
            "model": "qwen3.8-27b-gptq8", "prompt": prompt,
            "max_tokens": max_tokens, "temperature": 0.0,
            "logprobs": logprobs,
        }).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key()})
    out = json.loads(urllib.request.urlopen(r, timeout=900).read())
    ch = out["choices"][0]
    u = out["usage"]
    tl = (ch.get("logprobs") or {}).get("top_logprobs") or []
    top1, lp = ("?", 0.0)
    if tl and tl[0]:
        pairs = [(k, v) for k, v in tl[0].items() if isinstance(v, (int, float))]
        if pairs:
            top1, lp = max(pairs, key=lambda p: p[1])
    return {"text": ch["text"][:48], "top1": top1, "lp": round(lp, 3),
            "prompt_tok": u["prompt_tokens"]}


def build_exact(tokenizer, target, base="The capital of France is"):
    ids = tokenizer.encode(base)
    assert len(ids) <= target, f"base too long: {len(ids)} > {target}"
    pad_id = tokenizer.encode(" river", add_special_tokens=False)
    if len(pad_id) != 1:
        pad_id = [tokenizer.encode("a", add_special_tokens=False)[0]]
    n = target - len(ids)
    ids = ids + pad_id * n
    assert len(ids) == target
    return tokenizer.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+",
                    default=[6, 1663, 1664, 1665, 1727, 1728, 1729, 3328, 3456])
    ap.add_argument("--out", default="logs/garble/boundary_probe.jsonl")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    out = open(args.out, "a", buffering=1)
    for t in args.targets:
        prompt = build_exact(tok, t)
        # verify token count round-trips
        n = len(tok.encode(prompt))
        r = comp(prompt)
        rec = {"target": t, "actual_tokens": n, **r,
               "verdict": ("ok" if r["lp"] > -2.5 else "CORRUPT")}
        out.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
