#!/usr/bin/env python3
"""Aligned 52-prompt gate: score the SAME reference tokens in both arms.

Unlike comparing independent greedy continuations, teacher-forced logprobs
share an exact context. KL is over shared top-20 tokens plus one OTHER bucket:
this is a coarsened lower bound, not a full-vocabulary KL claim.
"""
import argparse
import json
import math
import os
import urllib.request
from pathlib import Path

from bench_agent_mixed import nginx_down
from kld_probe_v2 import PROMPTS


def post(body):
    req = urllib.request.Request(
        "http://127.0.0.1:8020/v1/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + os.environ["VLLM_API_KEY"],
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)["choices"][0]


def score(ids, start):
    result = post({"model": "qwen3.8-27b-gptq8", "prompt": ids,
                   "max_tokens": 1, "temperature": 0, "prompt_logprobs": 20})
    rows = result["prompt_logprobs"]
    assert len(rows) == len(ids)
    for row in rows[start:]:
        if not row or any(not math.isfinite(v["logprob"]) for v in row.values()):
            raise RuntimeError("Missing or nonfinite teacher-forced logprobs")
    return rows[start:]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    nginx_down()
    ref_path = Path(args.reference)
    out_path = Path(args.output)
    if args.capture:
        records = []
        for i, prompt in enumerate(PROMPTS):
            result = post({"model": "qwen3.8-27b-gptq8", "prompt": prompt,
                           "max_tokens": args.max_tokens, "temperature": 0,
                           "return_token_ids": True})
            ids, output = result["prompt_token_ids"], result["token_ids"]
            assert ids and output
            rows = score(ids + output, len(ids))
            records.append({"prompt": prompt, "prompt_ids": ids,
                            "output_ids": output, "text": result["text"],
                            "rows": rows})
            ref_path.write_text(json.dumps(records))
            print(f"reference {i+1}/{len(PROMPTS)}: {len(output)} tokens", flush=True)
        out_path.write_text(json.dumps({
            "captured": len(records),
            "tokens": sum(len(r["output_ids"]) for r in records),
        }))
        return
    records = json.loads(ref_path.read_text())
    results = []
    for i, record in enumerate(records):
        ids, output = record["prompt_ids"], record["output_ids"]
        rows = score(ids + output, len(ids))
        stats = []
        for token, base, candidate in zip(output, record["rows"], rows, strict=True):
            a = {k: math.exp(v["logprob"]) for k, v in base.items()}
            b = {k: math.exp(v["logprob"]) for k, v in candidate.items()}
            common = a.keys() & b.keys()
            pairs = [(a[k], b[k]) for k in common]
            pairs.append((max(1e-30, 1-sum(p for p, _ in pairs)),
                          max(1e-30, 1-sum(q for _, q in pairs))))
            kl = sum(p * math.log(p/max(q, 1e-30)) for p, q in pairs if p)
            chosen = candidate[str(token)]
            stats.append({"coarsened_kl": kl,
                          "top1_agree": max(a, key=a.get) == max(b, key=b.get),
                          "chosen_lp_delta": (chosen["logprob"]
                                              - base[str(token)]["logprob"]),
                          "rank": chosen["rank"]})
        results.append({"index": i, "rows": rows, "stats": stats})
        out_path.write_text(json.dumps({"prompts": results}))
        print(f"scored {i+1}/{len(records)}", flush=True)
    flat = [s for r in results for s in r["stats"]]
    summary = {"prompts": len(results), "tokens": len(flat),
               "mean_coarsened_kl": sum(s["coarsened_kl"] for s in flat)/len(flat),
               "top1_agreement": sum(s["top1_agree"] for s in flat)/len(flat),
               "mean_abs_chosen_lp_delta": (
                   sum(abs(s["chosen_lp_delta"]) for s in flat) / len(flat)),
               "rank_gt20": sum(s["rank"] > 20 for s in flat),
               "max_rank": max(s["rank"] for s in flat)}
    out_path.write_text(json.dumps({"summary": summary, "prompts": results}))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
