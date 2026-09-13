#!/usr/bin/env python3
"""Analyze VLLM_KV_STABLECHECK v2 records.

Rebuilds per-instance step series of (sum, width) for scheduler rows 0/1
and flags sum changes that are NOT explained by stable-window growth
(width increase) — i.e. byte changes in supposedly immutable KV blocks.

Output: per (rank-file, inst, row): total steps, width-stable steps,
unexplained sum changes with step numbers, plus dump-level bid
fingerprints (fp) and seq_lens (sl) for context.
"""
import json
import sys
from collections import defaultdict

def main(paths):
    for path in paths:
        per_inst = defaultdict(lambda: ([], []))  # inst -> (row0 series, row1 series)
        fp_log = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                inst = r["inst"]
                for row in (0, 1):
                    series = per_inst[inst][row]
                    if len(r["sums"]) <= row:
                        continue
                    for step_vals in r["sums"]:
                        # each entry is [sum, width] for one step (row-major)
                        pass
                    break
                # sums is indexed [step][row][2]
                for step_vals in r["sums"]:
                    for row in (0, 1):
                        if len(step_vals) > row and len(step_vals[row]) == 2:
                            per_inst[inst][row].append(
                                (step_vals[row][0], step_vals[row][1]))
                fp_log.append((r["n1"], r["fp"], r["sl"], r["inst"]))

        print(f"== {path}")
        total_bad = 0
        for inst in sorted(per_inst):
            for row in (0, 1):
                series = per_inst[inst][row]
                if not series:
                    continue
                bad = []
                grows = 0
                for i in range(1, len(series)):
                    (s0, w0), (s1, w1) = series[i - 1], series[i]
                    if w1 != w0:
                        grows += 1
                        continue  # window grew: sum change expected
                    if s0 != s1:
                        bad.append((i + 1, s0, s1))
                total_bad += len(bad)
                if bad or grows:
                    print(f"  inst={inst} row={row}: steps={len(series)} "
                          f"width-grow-steps={grows} UNEXPLAINED={len(bad)}")
                    for step, a, b in bad[:12]:
                        print(f"    step {step}: {a} -> {b} (delta {b - a})")
        if total_bad == 0:
            print("  ALL CLEAN: no unexplained sum changes in any instance/row")
        # dump-level fingerprints
        print("  dump fps (first 4, last 4):")
        for e in fp_log[:4] + fp_log[-4:]:
            print("   ", e)

if __name__ == "__main__":
    main(sys.argv[1:])
