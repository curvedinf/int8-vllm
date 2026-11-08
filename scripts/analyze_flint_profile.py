#!/usr/bin/env python3
"""Aggregate a rocprofv3 kernel trace into per-kernel time shares.

Usage:
  analyze_flint_profile.py <out_dir> [--steps N]
Finds the newest kernel_trace.csv under out_dir (or takes the csv path),
groups by kernel name, and prints the top consumers with ms total, call
count and (with --steps) us per decode step.
"""

import argparse
import glob
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    path = args.path
    if os.path.isdir(path):
        cands = glob.glob(os.path.join(path, "**", "kernel_trace.csv"),
                          recursive=True)
        if not cands:
            cands = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
        if not cands:
            raise SystemExit(f"no csv under {path}")
        path = max(cands, key=os.path.getmtime)
    df = pd.read_csv(path)
    print("columns:", list(df.columns)[:12], "rows:", len(df))
    name_col = next(c for c in df.columns if "Kernel" in c and "Name" in c)
    dur_col = next(c for c in df.columns if c in ("Duration", "DurationNs")
                   or "Duration" in c)
    df["ms"] = pd.to_numeric(df[dur_col], errors="coerce") / 1e6
    g = (df.groupby(name_col)["ms"].agg(["sum", "count"])
           .sort_values("sum", ascending=False))
    total = g["sum"].sum()
    print(f"total kernel time {total:.1f} ms over {len(df)} dispatches")
    if args.steps:
        print(f"per step ({args.steps} steps): {total / args.steps * 1000:.0f} us")
    for name, row in g.head(args.top).iterrows():
        per = (row["sum"] / args.steps * 1000) if args.steps else 0.0
        short = name if len(name) < 90 else name[:87] + "..."
        print(f"{row['sum']:9.2f} ms {int(row['count']):7d}x "
              f"{(row['sum']/total*100):5.1f}% {per:8.1f} us/step  {short}")


if __name__ == "__main__":
    main()
