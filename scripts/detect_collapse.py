#!/usr/bin/env python3
"""Detect draft-acceptance-collapse episodes in a vllm server log.

The true corruption signature on this stack: surface repetition loops
co-occurring with draft acceptance collapse (mean ~1.0, pos-0 <= ~0.05).
All-reasoning output alone is NOT counted (normal model behavior).
"""
import re
import sys

LINE_RE = re.compile(
    r"INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Mean acceptance length: ([0-9.]+), "
    r"Accepted throughput: ([0-9.]+) tokens/s.*Per-position acceptance rate: ([0-9., ]+)"
)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/serve_recipe_qwen38/server.log"
    episodes = []
    for line in open(path, errors="replace"):
        m = LINE_RE.search(line)
        if not m:
            continue
        ts, mean_s, thr_s, pp_s = m.groups()
        mean = float(mean_s)
        thr = float(thr_s)
        pp = [float(x) for x in pp_s.strip().rstrip(",").split(",") if x]
        pos0 = pp[0] if pp else 1.0
        # Collapse: nearly everything rejected while the request is still
        # generating content (accepted throughput > 0 excludes idle gaps).
        if mean <= 1.5 and pos0 <= 0.05 and thr > 0.0:
            episodes.append((ts, mean, pos0, thr))
    print(f"{len(episodes)} collapse windows in {path}")
    for ts, mean, pos0, thr in episodes:
        print(f"  {ts} mean={mean:.2f} pos0={pos0:.3f} acc_thr={thr:.1f} tok/s")


if __name__ == "__main__":
    main()
