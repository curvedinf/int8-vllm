# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate input admission and the decode-only tail of a streamed cohort."""

import math
from collections.abc import Sequence
from typing import Any

MEASUREMENT_VERSION = "aggregate_decode_only_after_all_prefills_v2"


def aggregate_streaming_phases(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Use one shared clock; never sum overlapping request durations.

    Input throughput ends at the latest first-token event, including admission
    delays and any interleaved decode. Decode counts only events strictly after
    that boundary. Staggered admission can leave fewer active requests than the
    submitted concurrency; the result is a decode-only tail, not steady C8.
    """
    if not rows:
        raise ValueError("At least one streamed request is required")
    events = []
    first = []
    last = []
    prompt_tokens = 0
    for row in rows:
        stream = row.get("events", [])
        if not stream:
            raise ValueError("Every request must contain token events")
        previous = -math.inf
        tokens = 0
        for event in stream:
            if "run_elapsed_s" not in event:
                raise ValueError("Token events require a shared run_elapsed_s clock")
            elapsed = event["run_elapsed_s"]
            count = event["tokens"]
            if not math.isfinite(elapsed) or elapsed < 0 or elapsed < previous:
                raise ValueError("Token times must be finite, nonnegative and ordered")
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError("Each event must count positive integer tokens")
            previous = elapsed
            tokens += count
        usage = row["usage"]
        if tokens != usage["completion_tokens"]:
            raise ValueError("Token events disagree with completion usage")
        prompt_tokens += usage["prompt_tokens"]
        first.append(stream[0]["run_elapsed_s"])
        last.append(stream[-1]["run_elapsed_s"])
        events.extend(stream)
    prefill_end = max(first)
    decode_end = max(last)
    decode_tokens = sum(e["tokens"] for e in events if e["run_elapsed_s"] > prefill_end)
    decode_seconds = decode_end - prefill_end
    if prefill_end <= 0 or decode_seconds <= 0 or decode_tokens <= 0:
        raise ValueError("No nonempty decode-only tail after all first-token events")
    return {
        "prefill_tok_s": prompt_tokens / prefill_end,
        "prefill_elapsed_s": prefill_end,
        "aggregate_decode_tok_s": decode_tokens / decode_seconds,
        "decode_only_tokens": decode_tokens,
        "decode_only_seconds": decode_seconds,
        "decode_only_start_s": prefill_end,
        "decode_only_end_s": decode_end,
        "decode_phase_requests": sum(end > prefill_end for end in last),
    }
