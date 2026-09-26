#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the same server boot with several independent 200k conversations."""

import argparse
import json
import os
import random
import time
import urllib.request
from pathlib import Path

FILLER_WORDS = [
    "river",
    "system",
    "delta",
    "sediment",
    "gauge",
    "measurement",
    "archive",
    "pipeline",
    "corridor",
    "harvest",
    "festival",
    "lantern",
    "orchestra",
    "telescope",
    "beacon",
    "cartography",
    "ledger",
    "furnace",
    "alloy",
    "tribunal",
    "meadow",
    "anchor",
    "kiln",
    "viaduct",
    "granary",
    "compass",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8020")
    parser.add_argument("--model", default="qwen3.8-27b-gptq8")
    parser.add_argument("--target-tokens", type=int, default=210000)
    parser.add_argument("--turn-tokens", type=int, default=16384)
    parser.add_argument("--conversations", type=int, default=3)
    parser.add_argument("--final-probe-rounds", type=int, default=8)
    parser.add_argument("--out", default="logs/garble/repro_long_context.jsonl")
    args = parser.parse_args()
    if min(args.target_tokens, args.turn_tokens, args.conversations) <= 0:
        parser.error("token counts and conversation count must be positive")
    api_key = os.environ.get("VLLM_API_KEY") or os.environ.get("LLAMA_API_KEY")
    if not api_key:
        api_key = Path("/etc/llama/llama-api.key").read_text().strip()
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bad = False

    with output_path.open("a", buffering=1) as output:

        def record(**fields):
            fields["t"] = time.strftime("%H:%M:%S", time.gmtime())
            line = json.dumps(fields)
            output.write(line + "\n")
            print(line, flush=True)

        def post(endpoint, payload):
            request = urllib.request.Request(
                args.base_url.rstrip("/") + endpoint,
                data=json.dumps({"model": args.model, **payload}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=3600) as response:
                return json.load(response)

        def chat(messages, max_tokens):
            return post(
                "/v1/chat/completions",
                {"messages": messages, "max_tokens": max_tokens, "temperature": 0},
            )

        def message_text(response):
            message = response["choices"][0]["message"]
            return (
                (message.get("reasoning") or "") + "\n" + (message.get("content") or "")
            )

        def probe(tag):
            nonlocal bad
            for repeat in range(3):
                response = post(
                    "/v1/completions",
                    {
                        "prompt": "The capital of France is",
                        "max_tokens": 8,
                        "temperature": 0,
                        "logprobs": 5,
                    },
                )
                choice = response["choices"][0]
                logprobs = choice["logprobs"]
                token = logprobs["tokens"][0]
                lp = logprobs["token_logprobs"][0]
                failed = token != " Paris" or lp < -1.0
                bad |= failed
                record(
                    phase="lp",
                    tag=tag,
                    repeat=repeat,
                    top1=token,
                    lp=lp,
                    text=choice["text"],
                    verdict="CORRUPT" if failed else "ok",
                )

        probe("baseline")
        conversations = []
        for index in range(args.conversations):
            tag = str(index + 1)
            seed = 7000 + 2000 * index
            started = time.monotonic()
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "I will paste a long document in pieces. Reply with "
                    "the single word OK to each piece.",
                },
            ]
            response = chat(conversation, 8)
            conversation.append(
                {
                    "role": "assistant",
                    "content": response["choices"][0]["message"]["content"] or "OK",
                }
            )
            prompt_tokens = response["usage"]["prompt_tokens"]
            turn = 0
            while prompt_tokens < args.target_tokens:
                turn += 1
                rng = random.Random(seed + turn)
                filler = " ".join(
                    rng.choice(FILLER_WORDS)
                    for _ in range(int(args.turn_tokens * 1.35) + 64)
                )
                conversation.append(
                    {
                        "role": "user",
                        "content": f"[piece {turn}]\n{filler}\n(reply OK only)",
                    }
                )
                response = chat(conversation, 8)
                message = response["choices"][0]["message"]
                reply = message.get("content") or message.get("reasoning") or "OK"
                conversation.append({"role": "assistant", "content": reply[:64]})
                prompt_tokens = response["usage"]["prompt_tokens"]
                record(phase="turn", tag=tag, turn=turn, tokens=prompt_tokens)
            conversations.append(conversation)
            record(
                phase="filled",
                tag=tag,
                prompt_tokens=prompt_tokens,
                seconds=time.monotonic() - started,
            )
            probe("after" + tag)

        conversations[0].append(
            {"role": "user", "content": "Summarize this document in one sentence."}
        )
        response = chat(conversations[0], 256)
        text = message_text(response)
        record(phase="contA", text=text, usage=response["usage"])
        bad |= "\ufffd" in text
        for repeat in range(args.final_probe_rounds):
            probe("final" + str(repeat))
        response = chat(
            [
                {
                    "role": "user",
                    "content": "Name the capital of France and count from 1 to 5.",
                }
            ],
            256,
        )
        text = message_text(response)
        record(phase="fresh_chat", text=text, usage=response["usage"])
        bad |= "\ufffd" in text or "Paris" not in text
        record(phase="verdict", corrupted=bad)
    return 2 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
