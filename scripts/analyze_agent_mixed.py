#!/usr/bin/env python3
"""Client-side rates, including decode while another request awaits first token.

That contention window includes queue time as well as actual prefill. It is
not a kernel-overlap measure. Speculative tokens arriving together are a burst;
stream-event gap percentiles are not single-token ITL percentiles.
"""

import argparse
import json
from pathlib import Path


def merged(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if result and lo <= result[-1][1]:
            result[-1][1] = max(result[-1][1], hi)
        else:
            result.append([lo, hi])
    return result


def analyze(directory):
    directory = Path(directory)
    results = [json.loads(p.read_text()) for p in directory.glob("*.result.json")]
    summary = json.loads((directory / "summary.json").read_text())
    for r in results:
        tokens = r["tokens"]
        if len(tokens) < 2:
            continue
        lo, hi = tokens[0][0], tokens[-1][0]
        intervals = merged((max(lo, other["sent_ns"]), min(hi,
            other["tokens"][0][0] if other["tokens"] else other["finished_ns"]))
            for other in results if other["name"] != r["name"] and "sent_ns" in other)
        duration = sum(b-a for a, b in intervals) / 1e9
        count = sum(len(ids) for t, ids in tokens[1:]
                    if any(a < t <= b for a, b in intervals))
        r["while_other_awaiting_first_token"] = {
            "seconds": duration, "tokens": count,
            "tok_s": count / duration if duration > 0 else None}
        solo_duration = (hi-lo)/1e9 - duration
        solo_count = sum(len(ids) for _, ids in tokens[1:]) - count
        r["without_other_awaiting_first_token"] = {
            "seconds": solo_duration, "tokens": solo_count,
            "tok_s": solo_count/solo_duration if solo_duration > 1e-6 else None}
    completed = sum(r["status"] == "complete" for r in summary)
    finish = max((r["finished_ns"] for r in results), default=0)
    start = json.loads((directory / "meta.json").read_text())["origin_ns"]
    elapsed = (finish-start)/1e9 if results else 0
    return {"completed": completed, "planned": len(summary), "wall_s": elapsed,
            "wall_output_tok_s": sum(r.get("output_tokens_observed", 0)
                                      for r in results)/elapsed if elapsed else None,
            "requests": [{k: v for k, v in r.items()
                          if k not in ("tokens", "message", "prompt_token_ids")}
                         for r in sorted(results, key=lambda x: x["name"])],
            "notes": __doc__}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(analyze(args.directory), indent=2))
