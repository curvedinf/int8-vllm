#!/usr/bin/env python3
"""Reproduce long decode while other C6 slots prefill 32k prompts.

Run once with --prefillers 0 for the control and once with --prefillers 3.
Each stream has a fixed prompt and seed; output is saved as SSE chunks arrive.
The prefills begin when the first decoder has emitted --trigger-chars text.
"""

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

from garble_docs_probe import build_corpus
from garble_repro2 import API, MODEL, get_tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prompt-tag",
                    help="reuse a previous prompt marker while writing a new output tag")
    ap.add_argument("--context", type=int, default=32000)
    ap.add_argument("--output", type=int, default=2048)
    ap.add_argument("--decoders", type=int, default=3)
    ap.add_argument("--prefillers", type=int, default=3)
    ap.add_argument("--prefill-output", type=int, default=16)
    ap.add_argument("--trigger-chars", type=int, default=1500)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--thinking", action="store_true",
                    help="use the launcher's normal low-thinking chat template")
    ap.add_argument("--audit-logprobs", action="store_true",
                    help="record committed-token rank among the top 20 target logits")
    ap.add_argument("--capture-token-ids", action="store_true",
                    help="save exact prompt and generated token IDs without logprobs")
    args = ap.parse_args()
    prompt_tag = args.prompt_tag or args.tag

    if args.decoders + args.prefillers > 6:
        ap.error("this probe is limited to the production C6 setting")
    key = os.environ.get("VLLM_API_KEY")
    if not key:
        ap.error("VLLM_API_KEY is required")

    tok = get_tok()
    corpus = build_corpus(tok, target=args.context)
    corpus_hash = hashlib.sha256(corpus.encode()).hexdigest()
    out_dir = Path("logs/garble")
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    trigger = threading.Event()
    results = {}

    def stream(role: str, idx: int, max_tokens: int) -> None:
        name = f"{role}{idx}"
        # Include the run tag before the corpus so earlier legs on this same
        # server cannot turn the intended fresh prefill into a prefix hit.
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content":
                f"Run {prompt_tag}, stream {name}. Write exhaustive structured technical release "
                "notes about the following documentation. Continue in detail.\n\n"
                + corpus}],
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": args.repetition_penalty,
            "seed": 301 + idx + (100 if role == "prefill" else 0),
            "max_tokens": max_tokens,
            "stream": True,
            "chat_template_kwargs": (
                {"enable_thinking": True, "reasoning_effort": "low"}
                if args.thinking else {"enable_thinking": False}
            ),
        }
        capture_token_ids = args.audit_logprobs or args.capture_token_ids
        if args.audit_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 20
        if capture_token_ids:
            body["return_token_ids"] = True
        req = urllib.request.Request(
            API,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"},
        )
        content = []
        reasoning = []
        content_chars = 0
        first_token = None
        finish_reason = None
        error = None
        rank_count = 0
        output_token_ids = []
        prompt_token_ids = None
        rank_at_least20 = 0
        low_logprob_count = 0
        min_logprob = None
        sent = time.monotonic() - start
        path = out_dir / f"{args.tag}_{name}.txt"
        reason_path = out_dir / f"{args.tag}_{name}.reasoning.txt"
        rank_path = out_dir / f"{args.tag}_{name}.rank.jsonl"
        chunk_path = out_dir / f"{args.tag}_{name}.token_chunks.jsonl"
        try:
            with (urllib.request.urlopen(req, timeout=1800) as response,
                  path.open("w", encoding="utf-8") as text_file,
                  reason_path.open("w", encoding="utf-8") as reason_file,
                  rank_path.open("w", encoding="utf-8") as rank_file,
                  chunk_path.open("w", encoding="utf-8") as chunk_file):
                for raw in response:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if capture_token_ids and chunk.get("prompt_token_ids") is not None:
                        prompt_token_ids = chunk["prompt_token_ids"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    if capture_token_ids:
                        chunk_ids = choice.get("token_ids") or []
                        if chunk_ids:
                            chunk_file.write(json.dumps({
                                "start": len(output_token_ids),
                                "ids": chunk_ids,
                                "received_s": time.monotonic() - start,
                            }) + "\n")
                            chunk_file.flush()
                            output_token_ids.extend(chunk_ids)
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    part = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or ""
                    if part or thought:
                        first_token = first_token or time.monotonic() - start
                    if part:
                        content.append(part)
                        content_chars += len(part)
                        text_file.write(part)
                        text_file.flush()
                    if thought:
                        reasoning.append(thought)
                        reason_file.write(thought)
                        reason_file.flush()
                    if args.audit_logprobs:
                        for item in (choice.get("logprobs") or {}).get("content") or []:
                            lp = item.get("logprob")
                            if lp is None:
                                continue
                            top = item.get("top_logprobs") or []
                            better = sum(
                                other.get("logprob", float("-inf")) > lp + 1e-4
                                for other in top
                            )
                            rank_count += 1
                            # The API list includes the chosen token first,
                            # leaving at most 19 distinct alternatives here.
                            # 19 better entries prove rank >= 20; the list
                            # cannot distinguish rank 20 from below top 20.
                            rank_at_least20 += better >= 19
                            low_logprob_count += lp < -12
                            min_logprob = lp if min_logprob is None else min(lp, min_logprob)
                            rank_file.write(json.dumps({
                                "i": rank_count - 1, "chars": content_chars,
                                "token": item.get("token"), "lp": lp,
                                "better_in_list": better,
                                "best_listed_lp": max(
                                    (x.get("logprob", float("-inf")) for x in top),
                                    default=None,
                                ),
                            }) + "\n")
                        rank_file.flush()
                    if role == "decode" and content_chars >= args.trigger_chars:
                        trigger.set()
        except Exception as exc:
            error = repr(exc)
        result = {
            "role": role, "index": idx, "sent_s": sent,
            "first_token_s": first_token,
            "finished_s": time.monotonic() - start,
            "content_chars": content_chars,
            "reasoning_chars": sum(map(len, reasoning)),
            "finish_reason": finish_reason, "error": error,
            "rank_count": rank_count,
            "token_id_count": len(output_token_ids),
            "rank_at_least20": rank_at_least20,
            "low_logprob_count": low_logprob_count,
            "min_logprob": min_logprob,
        }
        results[name] = result
        if capture_token_ids:
            (out_dir / f"{args.tag}_{name}.token_ids.json").write_text(
                json.dumps({"prompt_token_ids": prompt_token_ids,
                            "output_token_ids": output_token_ids}),
                encoding="utf-8",
            )
        print(json.dumps(result), flush=True)

    manifest = {
        "tag": args.tag, "prompt_tag": prompt_tag,
        "context_tokens": args.context,
        "corpus_sha256": corpus_hash, "output_tokens": args.output,
        "decoders": args.decoders, "prefillers": args.prefillers,
        "trigger_chars": args.trigger_chars, "thinking": args.thinking,
        "repetition_penalty": args.repetition_penalty,
        "audit_logprobs": args.audit_logprobs,
        "capture_token_ids": args.capture_token_ids,
    }
    print(json.dumps(manifest), flush=True)
    threads = [threading.Thread(target=stream, args=("decode", i, args.output))
               for i in range(args.decoders)]
    for thread in threads:
        thread.start()
    if args.prefillers:
        if not trigger.wait(timeout=900):
            raise RuntimeError("no decoder emitted enough text to trigger prefills")
        manifest["prefill_launch_s"] = time.monotonic() - start
        print(f"prefills launched at {manifest['prefill_launch_s']:.1f}s", flush=True)
        threads.extend(
            threading.Thread(target=stream,
                             args=("prefill", i, args.prefill_output))
            for i in range(args.prefillers)
        )
        for thread in threads[args.decoders:]:
            thread.start()
    for thread in threads:
        thread.join()
    manifest["results"] = results
    (out_dir / f"{args.tag}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
