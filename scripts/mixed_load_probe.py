#!/usr/bin/env python3
"""Measure prefix reuse and exact token cadence during concurrent prefill.

Uses the production model, three established decoders and three fresh
prefills. Saves token timestamps so mixed-phase rates exclude decode-only
time and TTFT. Run on an isolated recipe listener with PORT and VLLM_API_KEY.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

from garble_docs_probe import build_corpus
from garble_repro2 import get_tok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--prompt-tag",
        help="Reuse an A/B prompt marker; requires a fresh cache namespace",
    )
    parser.add_argument("--context", type=int, default=32000)
    parser.add_argument("--output", type=int, default=2048)
    parser.add_argument("--prefix-only", action="store_true")
    args = parser.parse_args()
    prompt_tag = args.prompt_tag or args.tag
    tok = get_tok()
    corpus = build_corpus(tok, target=args.context)
    url = f"http://127.0.0.1:{os.environ.get('PORT', '8020')}/v1/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["VLLM_API_KEY"]}
    origin = time.monotonic()
    ready = [threading.Event() for _ in range(3)]

    def stream(name, prompt, count, signal=None):
        body = {"model": "qwen3.8-27b-gptq8", "prompt": prompt,
                "max_tokens": count, "ignore_eos": True, "temperature": 0,
                "seed": 123, "stream": True, "return_token_ids": True,
                "stream_options": {"include_usage": True}}
        result = {"name": name, "sent": time.monotonic() - origin,
                  "events": [], "text": "", "usage": None}
        req = urllib.request.Request(url, json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=1800) as response:
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                data = raw[6:].strip()
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                if chunk.get("usage"):
                    result["usage"] = chunk["usage"]
                for choice in chunk.get("choices", []):
                    if choice.get("prompt_token_ids") is not None:
                        result["prompt_token_ids"] = choice["prompt_token_ids"]
                    ids = choice.get("token_ids") or []
                    if ids:
                        result["events"].append([time.monotonic() - origin, ids])
                        if signal and sum(len(e[1]) for e in result["events"]) >= 128:
                            signal.set()
                    result["text"] += choice.get("text") or ""
        result["finished"] = time.monotonic() - origin
        if not result["events"]:
            raise RuntimeError(f"No token IDs returned for {name}")
        result["ttft"] = result["events"][0][0] - result["sent"]
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ("events", "text", "prompt_token_ids")}),
              flush=True)
        return result

    prompt = (
        f"Probe {prompt_tag}. Reference documentation:\n"
        f"{corpus}\nSummarize in detail:\n"
    )
    results = [stream("prefix_cold", prompt, 16),
               stream("prefix_repeat", prompt, 16),
               stream("prefix_append", prompt + "Focus on deployment.\n", 16)]
    summary = {"tag": args.tag, "prompt_tag": prompt_tag, "context": args.context,
               "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
               "results": results}
    if not args.prefix_only:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            decoders = [pool.submit(stream, f"decode{i}", prompt, args.output, ready[i])
                        for i in range(3)]
            for signal in ready:
                if not signal.wait(600):
                    raise RuntimeError("Decoder failed to reach prefill trigger")
            begin = time.monotonic() - origin
            prefills = [pool.submit(stream, f"prefill{i}",
                        f"Fresh {prompt_tag} {i}.\n{corpus}\nSummarize:\n", 16)
                        for i in range(3)]
            prefills = [f.result() for f in prefills]
            decoders = [f.result() for f in decoders]
        prefill_end = max(r["events"][0][0] for r in prefills)
        end = min(prefill_end, *(r["events"][-1][0] for r in decoders))
        if end <= begin:
            raise RuntimeError("No common decode/prefill overlap")
        rates = [sum(len(ids) for at, ids in r["events"] if begin <= at <= end)
                 / (end - begin) for r in decoders]
        summary["mixed"] = {"begin": begin, "end": end,
                            "prefill_end": prefill_end,
                            "decoder_tok_s": rates, "aggregate_tok_s": sum(rates),
                            "prefill_ttft": [r["ttft"] for r in prefills]}
        results.extend(decoders + prefills)
        print(json.dumps(summary["mixed"]), flush=True)
    output = Path("logs/mixed_load")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.tag}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
