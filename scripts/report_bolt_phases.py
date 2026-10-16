#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recalculate aggregate phase rates from recorded streamed Bolt trials."""

import argparse
import json
import statistics
from pathlib import Path

from vllm.benchmarks.streaming_phases import (
    MEASUREMENT_VERSION,
    aggregate_streaming_phases,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    trials = [json.loads(path.read_text()) for path in args.trials]
    if len({trial["manifest_sha256"] for trial in trials}) != 1:
        parser.error("All trials must use the same frozen input manifest")
    phases = [aggregate_streaming_phases(trial["rows"]) for trial in trials]
    report = {
        "trials": len(trials),
        "measurement_version": MEASUREMENT_VERSION,
        "manifest_sha256": trials[0]["manifest_sha256"],
        "median_prefill_tok_s": statistics.median(p["prefill_tok_s"] for p in phases),
        "pooled_prefill_tok_s": sum(
            row["usage"]["prompt_tokens"] for t in trials for row in t["rows"]
        )
        / sum(p["prefill_elapsed_s"] for p in phases),
        "median_aggregate_decode_tok_s": statistics.median(
            p["aggregate_decode_tok_s"] for p in phases
        ),
        "pooled_aggregate_decode_tok_s": sum(p["decode_only_tokens"] for p in phases)
        / sum(p["decode_only_seconds"] for p in phases),
        "pooled_whole_run_output_tok_s": sum(
            row["usage"]["completion_tokens"] for t in trials for row in t["rows"]
        )
        / sum(t["wall_s"] for t in trials),
        "decode_tail_active_requests": [p["decode_phase_requests"] for p in phases],
        "scope": "Input admission through latest first token; then decode-only tail",
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
