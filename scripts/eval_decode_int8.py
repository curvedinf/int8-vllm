#!/usr/bin/env python3
"""Aligned raw-logprob probe on decode, not teacher-forced prefill.

Force exactly one reference token with allowed_token_ids and inspect the NEXT
token's raw (pre-mask) distribution. Both arms thus see identical
token histories while exercising a decode-sized step. The
server must use its default raw_logprobs mode. The mask also affects the second
sample; that sample is discarded, not mistaken for a quality measurement.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from urllib.error import HTTPError

from bench_agent_mixed import nginx_down
from eval_mlp_car import post as completion


def post(body):
    try:
        return completion(body)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()}") from exc


def probe(item):
    key, prompt, forced = item
    result = post({
        "model": "qwen3.8-27b-gptq8", "prompt": prompt,
        "max_tokens": 2, "min_tokens": 2, "ignore_eos": True,
        "temperature": 0, "logprobs": 20, "return_token_ids": True,
        "return_tokens_as_token_ids": True, "allowed_token_ids": [forced],
    })
    assert result["token_ids"][0] == forced, (key, result["token_ids"])
    rows = result["logprobs"]["top_logprobs"]
    assert len(rows) == 2
    raw = rows[1]
    assert raw and all(math.isfinite(value) for value in raw.values())
    # A masked/processed distribution would collapse to p(forced) ~ 1.
    # Retain raw rows so the gate's semantics can be audited independently.
    return {"key": key, "raw": raw, "sampled": result["token_ids"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations")
    ap.add_argument("--reference")
    ap.add_argument("--output")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--canary", action="store_true")
    args = ap.parse_args()
    nginx_down()
    if args.canary:
        result = probe(("raw_canary", "The capital of France is", 0))
        if result["raw"].get("token_id:0", 0) >= -1:
            raise RuntimeError("Scores are not raw pre-mask logprobs")
        print("PASS: forced-token sampling preserves raw pre-mask scores", flush=True)
        return
    if args.smoke:
        def request(i):
            result = post({
                "model": "qwen3.8-27b-gptq8",
                "prompt": (f"Request {i}. " + "Some harmless context. " * 80
                           + "\nWhat is 2 + 2? Answer briefly."),
                "max_tokens": 64, "temperature": 0,
                "return_token_ids": True, "logprobs": 1,
            })
            assert result["token_ids"]
            assert all(math.isfinite(p) for p in result["logprobs"]["token_logprobs"])
            return {"tokens": len(result["token_ids"]), "text": result["text"]}
        with ThreadPoolExecutor(max_workers=6) as pool:
            print(json.dumps(list(pool.map(request, range(6)))), flush=True)
        return
    if not all((args.generations, args.reference, args.output)):
        ap.error("--generations, --reference and --output are required without --smoke")
    items = []
    for i, record in enumerate(json.loads(Path(args.generations).read_text())):
        output = record["output_ids"]
        for pos in (1, 2, 4, 8, 16, 32, 64, 128, 255):
            if pos < len(output):
                items.append((f"{i}:{pos}",
                              record["prompt_ids"] + output[:pos - 1],
                              output[pos - 1]))
    canary = probe(("raw_canary", items[0][1], 0))
    if canary["raw"].get("token_id:0", 0) >= -1:
        raise RuntimeError("Raw-logprob canary failed: scores may include the mask")
    with ThreadPoolExecutor(max_workers=6) as pool:
        actual = list(pool.map(probe, items))
    if args.capture:
        Path(args.reference).write_text(json.dumps(actual))
        summary = {"decode_distributions": len(actual), "captured": True}
    else:
        expected = json.loads(Path(args.reference).read_text())
        stats = []
        for a, b in zip(expected, actual, strict=True):
            assert a["key"] == b["key"]
            p = {k: math.exp(v) for k, v in a["raw"].items()}
            q = {k: math.exp(v) for k, v in b["raw"].items()}
            common = p.keys() & q.keys()
            pairs = [(p[k], q[k]) for k in common]
            pairs.append((max(1e-30, 1 - sum(x for x, _ in pairs)),
                          max(1e-30, 1 - sum(y for _, y in pairs))))
            stats.append({"key": a["key"],
                          "coarsened_kl": sum(x * math.log(x / y)
                                              for x, y in pairs if x),
                          "top1_agree": max(p, key=p.get) == max(q, key=q.get)})
        summary = {"decode_distributions": len(stats),
                   "mean_coarsened_kl": (
                       sum(s["coarsened_kl"] for s in stats) / len(stats)),
                   "max_coarsened_kl": max(s["coarsened_kl"] for s in stats),
                   "top1_agreement": sum(s["top1_agree"] for s in stats) / len(stats)}
        summary["positions"] = stats
    Path(args.output).write_text(json.dumps({"summary": summary, "rows": actual,
                                           "raw_canary": canary}))
    brief = {k: v for k, v in summary.items() if k != "positions"}
    print(json.dumps(brief), flush=True)


if __name__ == "__main__":
    main()
