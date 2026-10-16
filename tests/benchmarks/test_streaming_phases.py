# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import pytest

from vllm.benchmarks.streaming_phases import aggregate_streaming_phases


def request(prompt_tokens, *events):
    return {
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": sum(count for _, count in events),
        },
        "events": [{"run_elapsed_s": t, "tokens": count} for t, count in events],
    }


def test_shared_clock_excludes_interleaved_decode_and_boundary_bursts():
    rows = [
        request(100, (2.0, 1), (4.0, 4), (6.0, 3), (8.0, 5)),
        request(200, (6.0, 8), (7.0, 6), (10.0, 10)),
    ]
    result = aggregate_streaming_phases(rows)
    assert result["prefill_tok_s"] == 50
    assert result["decode_only_tokens"] == 21
    assert result["decode_only_seconds"] == 4
    assert result["aggregate_decode_tok_s"] == 5.25
    assert result["decode_phase_requests"] == 2


def test_finished_requests_are_not_counted_as_active_tail_concurrency():
    result = aggregate_streaming_phases(
        [request(64, (1.0, 1), (2.0, 9)), request(64, (4.0, 3), (5.0, 7))]
    )
    assert result["prefill_tok_s"] == 32
    assert result["aggregate_decode_tok_s"] == 7
    assert result["decode_phase_requests"] == 1


@pytest.mark.parametrize("bad", ["local_clock", "unordered", "nonfinite", "usage"])
def test_invalid_events_cannot_silently_change_the_denominator(bad):
    rows = [request(128, (1.0, 1), (2.0, 9))]
    if bad == "local_clock":
        for event in rows[0]["events"]:
            event["elapsed_s"] = event.pop("run_elapsed_s")
    elif bad == "unordered":
        rows[0]["events"].reverse()
    elif bad == "nonfinite":
        rows[0]["events"][0]["run_elapsed_s"] = float("nan")
    else:
        rows[0]["usage"]["completion_tokens"] += 1
    with pytest.raises(ValueError):
        aggregate_streaming_phases(rows)


@pytest.mark.parametrize("rows", [[], [{"events": []}], [request(64, (1.0, 8))]])
def test_absent_tail_is_not_reported_as_decode_throughput(rows):
    with pytest.raises(ValueError):
        aggregate_streaming_phases(copy.deepcopy(rows))
