#!/usr/bin/env python3
"""Two-conversation discrimination leg for the 2026-09-26 garble incident.

Finding to discriminate: every CLEAN leg so far was the FIRST long
conversation on a fresh boot; every CORRUPT leg was second-conversation or
post-reset traffic. This driver fills TWO independent deep conversations
back-to-back on one boot and checks the fixed-prompt greedy logprobs after
each, plus a final continuation of conversation A (recycled-prefix shape).

Exit 0 = clean, 2 = corrupted.
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_garble_bisect import FILLER_WORDS, chat, logprobs_probe  # noqa: E402

OUT = None


def record(**kw):
    kw["t"] = time.strftime("%H:%M:%S")
    line = json.dumps(kw)
    OUT.write(line + "\n")
    print(line, file=sys.stderr, flush=True)


def make_filler(target_tokens, seed):
    rng = random.Random(seed)
    pool = FILLER_WORDS.split()
    words = [rng.choice(pool) for _ in range(int(target_tokens * 1.35) + 64)]
    return " ".join(words)


def fill_conversation(seed, target_tokens, turn_tokens, tag):
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
    while prompt_tokens < target_tokens:
        turn += 1
        chunk = make_filler(turn_tokens, seed=seed + turn)
        convo.append({"role": "user",
                      "content": f"[piece {turn}]\n{chunk}\n(reply OK only)"})
        out = chat(convo, 0.0, 8)
        msg = out["choices"][0]["message"]
        reply = (msg.get("content") or msg.get("reasoning") or "OK")
        convo.append({"role": "assistant", "content": reply[:64]})
        prompt_tokens = out["usage"]["prompt_tokens"]
    record(phase="filled", tag=tag, prompt_tokens=prompt_tokens)
    return convo, prompt_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=200000)
    ap.add_argument("--turn-tokens", type=int, default=16384)
    ap.add_argument("--tag", default="twoconv")
    ap.add_argument("--out", default="logs/garble/repro_bisect.jsonl")
    args = ap.parse_args()

    global OUT
    OUT = open(args.out, "a", buffering=1)

    def lp(tag):
        tok, lpv, text = logprobs_probe()
        bad = lpv < -2.5
        record(phase="lp", tag=tag, top1=tok, lp=round(lpv, 3),
            text=text[:40], verdict=("CORRUPT" if bad else "ok"))
        return bad

    corrupt = False
    corrupt |= lp(args.tag + "/baseline")

    convo_a, ptoks = fill_conversation(
        seed=7000, target_tokens=args.target_tokens,
        turn_tokens=args.turn_tokens, tag=args.tag + "/A")
    corrupt |= lp(args.tag + "/afterA")

    convo_b, _ = fill_conversation(
        seed=9000, target_tokens=args.target_tokens,
        turn_tokens=args.turn_tokens, tag=args.tag + "/B")
    corrupt |= lp(args.tag + "/afterB")

    # Continue A once more: deep prefix whose blocks were stored/evicted/
    # recycled during B's fill.
    convo_a.append({"role": "user", "content": "Summarize everything in one "
                                             "sentence, then stop."})
    try:
        out = chat(convo_a, 0.0, 128)
    except Exception as e:  # noqa: BLE001
        record(phase="contA", tag=args.tag, error=repr(e))
        corrupt = True
    else:
        msg = out["choices"][0]["message"]
        text = (msg.get("reasoning") or "") + "\n" + (msg.get("content") or "")
        record(phase="contA", tag=args.tag,
           verdict=("GARBAGE" if "\ufffd" in text else "ok"),
           sample=text.strip()[:160])

    for i in range(3):
        corrupt |= lp(f"{args.tag}/final{i}")
        time.sleep(1.0)

    record(phase="verdict", tag=args.tag, corrupted=bool(corrupt))
    sys.exit(2 if corrupt else 0)


if __name__ == "__main__":
    main()
