#!/usr/bin/env python3
"""Round-2 reproducer for the 2026-09-26 persistent-garble incident.

Round 1 (single conversation to 219k, no other traffic) stayed healthy and
issued ZERO CPU-tier loads. The incident boot showed ~23-27% external prefix
hit rate, i.e. H2D loads from the OffloadingConnector's CPU tier — the path
audited as the prime suspect (hipMemcpyBatchAsync on a side transfer stream
racing CUDA-graph replay; stores got the classic stream-0 fix, loads did not).

This round forces that path:
  1. fill conversation A to ~target tokens (turns are separate requests;
     local prefix cache serves them),
  2. POST /reset_prefix_cache (dev endpoint; local GPU cache only — the
     connector's CPU tier keeps its stored blocks),
  3. start a long companion decode stream (CUDA-graph replays),
  4. send A's continuation turn -> external prefix hits -> big H2D batch
     load while the companion decodes,
  5. probe fresh unique-salt greedy streams for the persistent-garble
     signature.

Usage:
  .venv/bin/python scripts/repro_garble_round2.py [--target-tokens 200000]
      [--out logs/garble/repro_round2.jsonl]
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


def post(path, payload=None, timeout=1800):
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
    words = []
    pool = FILLER_WORDS.split()
    for _ in range(n_words):
        words.append(rng.choice(pool))
        if rng.random() < 0.08:
            words.append("\n")
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
    ap.add_argument("--target-tokens", type=int, default=200000)
    ap.add_argument("--turn-tokens", type=int, default=16384)
    ap.add_argument("--out", default="logs/garble/repro_round2.jsonl")
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
        chunk = make_filler(args.turn_tokens, seed=2000 + turn)
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

    # 1) baseline probe right after fill
    verdict, sample, dt = coherence_probe()
    record(phase="probe_prefill_done", verdict=verdict, dt=round(dt, 1),
           sample=sample[:160])

    # 2) drop the local GPU prefix cache; CPU tier keeps its stores
    try:
        status, body = post("/reset_prefix_cache")
        record(phase="reset", status=status, body=body[:200])
    except urllib.error.HTTPError as e:
        record(phase="reset", error=str(e), body=e.read().decode()[:200])
        return
    time.sleep(2.0)

    # 3) companion decode stream (keeps CUDA-graph replay busy)
    companion_done = threading.Event()

    def companion():
        msgs = [{"role": "user",
                 "content": "Write a very long, detailed story about a "
                            "lighthouse keeper on a remote rocky island. "
                            "Keep going for as long as you can."}]
        try:
            out, dt = chat(msgs, 1.0, 3000)
            text = (out["choices"][0]["message"].get("content") or "")[:120]
            record(phase="companion_done", dt=round(dt, 1), sample=text)
        except Exception as e:  # noqa: BLE001
            record(phase="companion_error", error=repr(e))
        companion_done.set()

    threading.Thread(target=companion, daemon=True).start()
    time.sleep(8.0)  # let the companion enter steady decode

    # 4) continuation of A: external prefix hits -> H2D loads during decode
    convo.append({"role": "user",
                  "content": "Summarize the document above in one sentence, "
                             "then stop."})
    out, dt = chat(convo, 0.0, 256)
    msg = out["choices"][0]["message"]
    text = (msg.get("reasoning") or "") + "\n" + (msg.get("content") or "")
    bad = "\ufffd" in text
    record(phase="continuation", dt=round(dt, 1),
           verdict="GARBAGE" if bad else "ok", sample=text.strip()[:300])

    # 5) fresh-stream probes while/after the companion decodes
    for i in range(3):
        verdict, sample, dt = coherence_probe()
        record(phase="probe", i=i, companion_alive=not companion_done.is_set(),
               verdict=verdict, dt=round(dt, 1), sample=sample[:160])
        time.sleep(2.0)

    companion_done.wait(timeout=600)
    verdict, sample, dt = coherence_probe()
    record(phase="final", verdict=verdict, dt=round(dt, 1),
           sample=sample[:160])


if __name__ == "__main__":
    main()
