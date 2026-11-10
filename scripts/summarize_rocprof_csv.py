#!/usr/bin/env python3
"""Aggregate a rocprofv3 kernel-trace CSV directory by kernel name.

Usage: .venv/bin/python scripts/summarize_rocprof_csv.py <dir> [--top N]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 30
    files = sorted(root.glob("*.csv"))
    print(f"csv files: {len(files)}")
    total_us = 0.0
    for f in files:
        with f.open(newline="") as fh:
            rd = csv.DictReader(fh)
            names = rd.fieldnames or []
            name_col = next((c for c in names if "Kernel_Name" in c), names[0])
            dur_col = next(
                (c for c in names if c.lower() in ("duration", "duration_ns",
                                                   "duration_us")), None)
            st_col = next((c for c in names if "Start_Timestamp" in c), None)
            en_col = next((c for c in names if "End_Timestamp" in c), None)
            agg = defaultdict(lambda: [0, 0.0])
            for row in rd:
                try:
                    if dur_col:
                        d = float(row[dur_col])
                        unit = dur_col.lower()
                        us = d / 1000.0 if unit.endswith("_ns") else (
                            d * 1000.0 if unit.endswith("_us") else d)
                    elif st_col and en_col:
                        us = (float(row[en_col]) - float(row[st_col])) / 1000.0
                    else:
                        continue
                except (TypeError, ValueError):
                    continue
                a = agg[row[name_col]]
                a[0] += 1
                a[1] += us
            if not agg:
                print(f"== {f.name}: no rows")
                continue
            subtotal = sum(v[1] for v in agg.values())
            total_us += subtotal
            print(f"== {f.name}: {sum(v[0] for v in agg.values())} kernels, "
                  f"{subtotal/1000:.1f} ms")
            for name, (n, us) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:top]:
                print(f"  {us/1000:9.2f} ms  {100*us/subtotal:5.1f}%  "
                      f"n={n:<6d} {us/n:8.1f} us  {name[:80]}")
    print(f"TOTAL {total_us/1000:.1f} ms")


if __name__ == "__main__":
    main()
