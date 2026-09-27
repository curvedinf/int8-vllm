#!/usr/bin/env python3
"""Compact bisect driver for the 2026-09-26 persistent-garble incident.

Sequence (the twice-reproduced collapse shape):
  1. fill one conversation to --target-tokens (16k-token turns),
  2. fixed-prompt logprobs sanity check (sharp = healthy, flat = corrupt),
  3. POST /reset_prefix_cache,
  4. companion decode stream + continuation of the conversation
     (external prefix hits if the offload tier is on),
  5. verdicts: logprobs probes + coherence probes.

Exit code 0 = clean, 2 = corrupted, 1 = client error.
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8020"
MODEL = "qwen3.8-27b-gptq8"
PID_FILE = "logs/serve_recipe_qwen38/server.pid"

FILLER_WORDS = (
    "river system delta sediment gauge measurement archive pipeline corridor "
    "harvest festival lantern orchestra telescope beacon cartography ledger "
    "furnace alloy tribunal meadow anchor kiln viaduct granary compass "
)


def read_api_key() -> str:
    for var in ("VLLM_API_KEY", "LLAMA_API_KEY"):
        if os.environ.get(var):
            return os.environ[var].strip()
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        with open(f"/proc/{pid}/environ", "rb") as f:
            for entry in f.read().split(b"\0"):
                if entry.startswith(b"VLLM_API_KEY="):
                    return entry.split(b"=", 1)[1].decode().strip()
    except (OSError, ValueError):
        pass
    with open("/etc/llama/llama-api.key") as f:
        return f.read().strip()


def post(path, payload=None, timeout=3600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode() if payload is not None else b"",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {read_api_key()}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode(errors="replace")


def chat(messages, temperature, max_tokens, timeout=3600):
    body = {"model": MODEL, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "stream": False}
    status, body = post("/v1/chat/completions", body, timeout)
    return json.loads(body)


def logprobs_probe():
    """Fixed 6-token greedy completion. Returns (top1_token, top1_lp, text)."""
    status, raw = post("/v1/completions", {
        "model": MODEL, "prompt": "The capital of France is",
        "max_tokens": 8, "temperature": 0.0, "logprobs": 5})
    out = json.loads(raw)
    ch = out["choices"][0]
    lp = ch.get("logprobs") or {}
    tl = lp.get("top_logprobs") or []
    top1 = ("?", 0.0)
    if tl and tl[0]:
        first = tl[0]
        if isinstance(first, dict):
            if "token" in first and isinstance(first.get("logprob"), (int, float)):
                best = max(first, key=lambda d: d["logprob"])
                top1 = (best["token"], best["logprob"])
            else:
                pairs = [(k, v) for k, v in first.items()
                         if isinstance(v, (int, float))]
                if pairs:
                    best = max(pairs, key=lambda p: p[1])
                    top1 = (best[0], best[1])
    return top1[0], top1[1], ch["text"]


def make_filler(target_tokens, seed):
    rng = random.Random(seed)
    pool = FILLER_WORDS.split()
    words = [rng.choice(pool) for _ in range(int(target_tokens * 1.35) + 64)]
    return " ".join(words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=200000)
    ap.add_argument("--turn-tokens", type=int, default=16384)
    ap.add_argument("--skip-reset", action="store_true")
    ap.add_argument("--tag", default="leg")
    ap.add_argument("--out", default="logs/garble/repro_bisect.jsonl")
    args = ap.parse_args()

    outfile = open(args.out, "a", buffering=1)

    def record(**kw):
        kw["t"] = time.strftime("%H:%M:%S")
        outfile.write(json.dumps(kw) + "\n")
        print(json.dumps(kw), file=sys.stderr, flush=True)

    tok, lp, text = logprobs_probe()
    record(phase="baseline_lp", tag=args.tag, top1=tok, lp=round(lp, 3),
           text=text[:40], verdict=("CORRUPT" if lp < -2.5 else "ok"))

    convo = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",
         "content": "I will paste a long document in pieces. Reply with the "
                    "single word OK to each piece."},
    ]
    out = chat(convo, 0.0, 8)
    convo.append({"role": "assistant",
                  "content": out["choices"][0]["message"]["content"] or "OK"})
    prompt_tokens = out["usage"]["prompt_tokens"]
    turn = 0
    while prompt_tokens < args.target_tokens:
        turn += 1
        chunk = make_filler(args.turn_tokens, seed=4000 + turn)
        convo.append({"role": "user",
                      "content": f"[piece {turn}]\n{chunk}\n(reply OK only)"})
        out = chat(convo, 0.0, 8)
        msg = out["choices"][0]["message"]
        reply = (msg.get("content") or msg.get("reasoning") or "OK")
        convo.append({"role": "assistant", "content": reply[:64]})
        prompt_tokens = out["usage"]["prompt_tokens"]
    record(phase="filled", tag=args.tag, prompt_tokens=prompt_tokens)

    tok, lp, text = logprobs_probe()
    record(phase="prefill_done_lp", tag=args.tag, top1=tok, lp=round(lp, 3),
           text=text[:40], verdict=("CORRUPT" if lp < -2.5 else "ok"))

    if not args.skip_reset:
        status, body = post("/reset_prefix_cache")
        record(phase="reset", tag=args.tag, status=status, body=body[:80])
        time.sleep(2.0)

    companion_done = threading.Event()

    def companion():
        try:
            out = chat([{"role": "user",
                         "content": "Write a very long, detailed story about a "
                                    "lighthouse keeper on a remote rocky island."}],
                       1.0, 3000)
            m = out["choices"][0]["message"]
            record(phase="companion_done", tag=args.tag,
                   sample=(m.get("content") or "")[:100])
        except Exception as e:  # noqa: BLE001
            record(phase="companion_error", tag=args.tag, error=repr(e))
        companion_done.set()

    threading.Thread(target=companion, daemon=True).start()
    time.sleep(8.0)

    convo.append({"role": "user",
                  "content": "Summarize the document above in one sentence, "
                             "then stop."})
    out = chat(convo, 0.0, 256)
    msg = out["choices"][0]["message"]
    text = (msg.get("reasoning") or "") + "\n" + (msg.get("content") or "")
    record(phase="continuation", tag=args.tag,
           verdict=("GARBAGE" if "\ufffd" in text else "ok"),
           sample=text.strip()[:200])

    corrupt = False
    for i in range(4):
        tok, lp, text = logprobs_probe()
        bad = lp < -2.5
        corrupt |= bad
        record(phase="lp_probe", tag=args.tag, i=i, top1=tok, lp=round(lp, 3),
               text=text[:40], verdict=("CORRUPT" if bad else "ok"))
        time.sleep(2.0)
    companion_done.wait(timeout=900)
    tok, lp, text = logprobs_probe()
    bad = lp < -2.5
    corrupt |= bad
    record(phase="final_lp", tag=args.tag, top1=tok, lp=round(lp, 3),
           text=text[:40], verdict=("CORRUPT" if bad else "ok"))
    record(phase="verdict", tag=args.tag, corrupted=bool(corrupt))
    sys.exit(2 if corrupt else 0)


if __name__ == "__main__":
    main()
