#!/usr/bin/env python3
"""Round-3 reproducer: LONG decode at ~200k context.

Rounds 1-2 cleared: serial fill to 219k is healthy; a continuation that
external-loads ~60% of a 200k prefix during concurrent decode is healthy.
The incident's first garbage appeared in the long conversation's own output
while it DECODED at ~200k context (user report + acceptance collapse in the
prefill+decode window). This round decodes thousands of tokens at that
depth, then probes fresh streams.

Usage:
  .venv/bin/python scripts/repro_garble_round3.py [--target-tokens 195000]
      [--decode-tokens 3000] [--decode-temp 1.0] [--out logs/garble/repro_round3.jsonl]
"""

import argparse
import json
import os
import random
import sys
import time
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


def chat(messages, temperature, max_tokens, timeout=3600):
    body = {"model": MODEL, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {read_api_key()}"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    return out, time.time() - t0


def make_filler(target_tokens, seed):
    rng = random.Random(seed)
    n_words = int(target_tokens * 1.35) + 64
    pool = FILLER_WORDS.split()
    words = [rng.choice(pool) for _ in range(n_words)]
    return " ".join(words)


def coherence_probe():
    salt = os.urandom(8).hex()
    msgs = [
        {"role": "system", "content": f"You are a helpful assistant. [p:{salt}]"},
        {"role": "user",
         "content": "Name the capital of France, then count 1 to 5. "
                    "Two short lines only."},
    ]
    out, dt = chat(msgs, 0.0, 96)
    text = (out["choices"][0]["message"].get("reasoning") or "") + "\n" + \
        (out["choices"][0]["message"].get("content") or "")
    bad = ("\ufffd" in text) or ("paris" not in text.lower())
    return ("GARBAGE" if bad else "ok"), text.strip(), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=195000)
    ap.add_argument("--turn-tokens", type=int, default=16384)
    ap.add_argument("--decode-tokens", type=int, default=3000)
    ap.add_argument("--decode-temp", type=float, default=1.0)
    ap.add_argument("--out", default="logs/garble/repro_round3.jsonl")
    args = ap.parse_args()

    outfile = open(args.out, "a", buffering=1)

    def record(**kw):
        kw["t"] = time.strftime("%H:%M:%S")
        outfile.write(json.dumps(kw) + "\n")
        print(kw, file=sys.stderr, flush=True)

    verdict, sample, dt = coherence_probe()
    record(phase="baseline", verdict=verdict, dt=round(dt, 1),
           sample=sample[:160])

    convo = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",
         "content": "I will paste a long document in pieces. Reply with the "
                    "single word OK to each piece."},
    ]
    out, dt = chat(convo, 0.0, 8)
    convo.append({"role": "assistant",
                  "content": out["choices"][0]["message"]["content"] or "OK"})
    prompt_tokens = out["usage"]["prompt_tokens"]
    turn = 0
    while prompt_tokens < args.target_tokens:
        turn += 1
        chunk = make_filler(args.turn_tokens, seed=3000 + turn)
        convo.append({"role": "user",
                      "content": f"[piece {turn}]\n{chunk}\n(reply OK only)"})
        out, dt = chat(convo, 0.0, 8)
        msg = out["choices"][0]["message"]
        reply = (msg.get("content") or msg.get("reasoning") or "OK")
        convo.append({"role": "assistant", "content": reply[:64]})
        prompt_tokens = out["usage"]["prompt_tokens"]
        record(phase="fill", turn=turn, prompt_tokens=prompt_tokens,
               dt=round(dt, 1))
    record(phase="filled", prompt_tokens=prompt_tokens)

    # LONG decode at depth, matching the incident's sampling defaults.
    convo.append({"role": "user",
                  "content": "Now write a very long, richly detailed essay "
                             "about the history of lighthouses. Do not stop "
                             "early."})
    out, dt = chat(convo, args.decode_temp, args.decode_tokens)
    msg = out["choices"][0]["message"]
    text = (msg.get("reasoning") or "") + "\n" + (msg.get("content") or "")
    u = out["usage"]
    record(phase="long_decode", dt=round(dt, 1),
           completion_tokens=u.get("completion_tokens"),
           verdict=("GARBAGE" if "\ufffd" in text else "ok"),
           head=text.strip()[:200], tail=text.strip()[-200:])
    convo.append({"role": "assistant", "content": (msg.get("content") or "")[-200:]})

    # Mid- and post-decode fresh-stream probes.
    for i in range(6):
        verdict, sample, dt = coherence_probe()
        record(phase="probe", i=i, verdict=verdict, dt=round(dt, 1),
               sample=sample[:160])
        # keep a short continuation decoding between probes
        convo.append({"role": "user", "content": f"Continue ({i})."})
        out2, _ = chat(convo, args.decode_temp, 512)
        m2 = out2["choices"][0]["message"]
        t2 = (m2.get("reasoning") or "") + (m2.get("content") or "")
        record(phase="cont", i=i, verdict=("GARBAGE" if "\ufffd" in t2 else "ok"),
               tail=t2[-160:])
        convo.append({"role": "assistant", "content": (m2.get("content") or "")[-120:]})


if __name__ == "__main__":
    main()
