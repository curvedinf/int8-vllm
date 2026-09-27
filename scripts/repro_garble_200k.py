#!/usr/bin/env python3
"""Reproduce the 2026-09-26 persistent-garble incident (see
logs/garble/INCIDENT_2026-09-26_persistent_garble_200k.md).

Drives one long multi-turn conversation past a target context size while
periodically firing fresh unique-salt greedy probes in a separate
conversation. Records, per checkpoint, the long conversation's prompt token
count and whether a FRESH stream is coherent — the incident signature is
fresh streams turning garbage and staying garbage.

Auth mirrors scripts/serve_recipe_qwen38.sh read_api_key(): VLLM_API_KEY /
LLAMA_API_KEY env, else the live server's environ via the pid file, else
/etc/llama/llama-api.key. The key is used in-memory only.

Usage:
  .venv/bin/python scripts/repro_garble_200k.py [--target-tokens 210000] \
      [--turn-tokens 8192] [--probe-every 16384] [--out logs/garble/repro.jsonl]
"""

import argparse
import json
import os
import random
import sys
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


HEADERS = {}


def chat(messages, temperature, max_tokens, timeout=1800):
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
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
    # ~1.35 tokens per word for this vocab; overshoot slightly.
    n_words = int(target_tokens * 1.35) + 64
    words = []
    pool = FILLER_WORDS.split()
    for _ in range(n_words):
        words.append(rng.choice(pool))
        if rng.random() < 0.08:
            words.append("\n")
    return " ".join(words)


def coherence_probe():
    """Fresh conversation, unique salt, greedy. Returns (verdict, sample)."""
    salt = os.urandom(8).hex()
    msgs = [
        {"role": "system",
         "content": f"You are a helpful assistant. [probe:{salt}]"},
        {"role": "user",
         "content": "Name the capital of France, then count 1 to 5. "
                    "Two short lines only."},
    ]
    out, dt = chat(msgs, 0.0, 96)
    text = (out["choices"][0]["message"].get("reasoning") or "") + "\n" + \
        (out["choices"][0]["message"].get("content") or "")
    bad = ("\ufffd" in text) or ("paris" not in text.lower())
    return ("GARBAGE" if bad else "ok"), text.strip(), dt, salt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=210000)
    ap.add_argument("--turn-tokens", type=int, default=8192)
    ap.add_argument("--probe-every", type=int, default=16384)
    ap.add_argument("--out", default="logs/garble/repro_garble.jsonl")
    args = ap.parse_args()

    outfile = open(args.out, "a", buffering=1)

    def record(**kw):
        kw["t"] = time.strftime("%H:%M:%S")
        outfile.write(json.dumps(kw) + "\n")
        print(kw, file=sys.stderr)

    # Baseline probe before any long-context traffic.
    verdict, sample, dt, salt = coherence_probe()
    record(phase="baseline", verdict=verdict, dt=round(dt, 1),
           sample=sample[:200])

    convo = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",
         "content": "I am going to paste a long document in pieces. Just "
                    "reply with the single word OK for each piece."},
    ]
    out, dt = chat(convo, 0.0, 8)
    convo.append({"role": "assistant",
                  "content": out["choices"][0]["message"]["content"] or "OK"})

    prompt_tokens = out["usage"]["prompt_tokens"]
    next_probe = args.probe_every
    turn = 0
    while prompt_tokens < args.target_tokens:
        turn += 1
        chunk = make_filler(args.turn_tokens, seed=1000 + turn)
        convo.append({"role": "user", "content": f"[piece {turn}]\n{chunk}\n"
                        "(reply OK only)"})
        out, dt = chat(convo, 0.0, 8)
        msg = out["choices"][0]["message"]
        reply = (msg.get("content") or msg.get("reasoning") or "OK")
        convo.append({"role": "assistant", "content": reply[:64]})
        prompt_tokens = out["usage"]["prompt_tokens"]
        record(phase="fill", turn=turn, prompt_tokens=prompt_tokens,
               dt=round(dt, 1), reply=reply[:80])
        if prompt_tokens >= next_probe:
            next_probe += args.probe_every
            verdict, sample, pdt, salt = coherence_probe()
            record(phase="probe", at_tokens=prompt_tokens, verdict=verdict,
                   dt=round(pdt, 1), sample=sample[:200])

    # Final probes: fresh stream after the long conversation finished.
    verdict, sample, dt, salt = coherence_probe()
    record(phase="final", at_tokens=prompt_tokens, verdict=verdict,
           dt=round(dt, 1), sample=sample[:200])
    verdict, sample, dt, salt = coherence_probe()
    record(phase="final2", at_tokens=prompt_tokens, verdict=verdict,
           dt=round(dt, 1), sample=sample[:200])


if __name__ == "__main__":
    main()
