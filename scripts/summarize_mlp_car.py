#!/usr/bin/env python3
"""Summarize retained MLP CAR campaign artifacts without issuing requests.

Whole-run rates include queueing/prefill. Contention rates are per-stream
token-seconds while another client awaits a first token, not GPU overlap.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

from analyze_agent_mixed import analyze


def summarize_run(path):
    analysis = analyze(path)
    rows = analysis["requests"]
    meta = json.loads((path / "meta.json").read_text())
    spec = [r.get("metrics", {}).get("speculative_decoding", {}) for r in rows]
    accepted = sum(s.get("num_accepted_draft_tokens", 0) for s in spec)
    drafted = sum(s.get("num_draft_tokens", 0) for s in spec)
    contention = [r.get("while_other_awaiting_first_token", {}) for r in rows]
    seconds = sum(r.get("seconds", 0) for r in contention)
    return {
        "completed": analysis["completed"],
        "planned": analysis["planned"],
        "wall_s": analysis["wall_s"],
        "whole_run_output_tok_s": analysis["wall_output_tok_s"],
        "output_tokens": sum(r.get("output_tokens_observed", 0) for r in rows),
        "accounting_errors": sum(r.get("token_accounting_delta") != 0 for r in rows),
        "accepted": accepted,
        "drafted": drafted,
        "acceptance_fraction": accepted / drafted if drafted else None,
        "median_stream_decode_tok_s": statistics.median(
            r["decode_tok_s"] for r in rows if r.get("decode_tok_s") is not None
        ),
        "contention_per_stream_tok_s": (
            sum(r.get("tokens", 0) for r in contention) / seconds if seconds else None
        ),
        "mean_ttft_s_by_turn": {
            str(turn): statistics.mean(
                r["ttft_s"] for r in rows
                if r["name"].endswith(f"-t{turn}") and r.get("ttft_s") is not None
            )
            for turn in sorted({int(r["name"].rsplit("-t", 1)[1]) for r in rows})
        },
        "cache_tokens_by_request": {
            r["name"]: r.get("usage", {}).get("prompt_tokens_details", {})
            for r in rows
        },
        "body_sha256_by_request": {r["name"]: r["body_sha256"] for r in rows},
        "manifest_sha256": meta["manifest_sha256"],
        "max_stream_gap_s": max(r["stream_gap_s"]["max"] for r in rows),
        "stream_events_over_1s": sum(r["stream_gap_s"]["over_1s"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"runs": {}, "oracles": {}, "notes": __doc__}
    for arm in ("baseline", "candidate"):
        for kind in ("greedy_cold", "greedy_warm", "cold", "warm",
                     "profile_cold", "profile_warm"):
            name = f"{arm}_{kind}"
            path = args.campaign / name
            if (path / "summary.json").exists():
                result["runs"][name] = summarize_run(path)
    for path in sorted(args.campaign.glob("oracle_*.json")):
        oracle = json.loads(path.read_text())
        rows = oracle["rows"]
        result["oracles"][path.stem] = {
            "tokens": len(rows),
            "finite": all(math.isfinite(r["lp"]) for r in rows),
            "rank_gt20": sum(r["rank"] > 20 for r in rows),
            "max_rank": max(r["rank"] for r in rows),
            "mean_logprob": statistics.mean(r["lp"] for r in rows),
            "tail_from_token_1024": {
                "tokens": len(rows[1024:]),
                "rank_gt20": sum(r["rank"] > 20 for r in rows[1024:]),
                "max_rank": max((r["rank"] for r in rows[1024:]), default=None),
            },
        }
    quality = args.campaign / "candidate52.json"
    if quality.exists():
        result["aligned_quality"] = json.loads(quality.read_text()).get("summary")
    decode_quality = args.campaign / "candidate_decode_accuracy.json"
    if decode_quality.exists():
        result["decode_aligned_quality"] = json.loads(
            decode_quality.read_text()
        ).get("summary")
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        name: {k: v for k, v in row.items() if not isinstance(v, dict)}
        for name, row in result["runs"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
