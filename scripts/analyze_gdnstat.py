#!/usr/bin/env python3
"""Triage analyzer for VLLM_GDNSTAT captures (per-round GDN window norms).

Reads the .pt shards written by MambaHybridModelStates._gdnstat_snap and
flags, per layer/state and per request slot, rounds where the RESUMED slot's
norm (norms[key][ri]) jumps abnormally relative to its local evolution.
A wrong-slot read (stale/other-position checkpoint) shows as an O(1)
relative jump that persists; normal recurrence evolves smoothly.

This is a TRIAGE script: it narrows where to look (round index, layer,
slot rel) but does not decide quality — output text is read separately.

Usage:
  python scripts/analyze_gdnstat.py logs/garble/gdnstat_nat [--rs RS] [--jump 0.35]
"""
import argparse
import glob
import os

import torch


def load_rounds(d):
    shards = sorted(glob.glob(os.path.join(d, "gdnstat_*.pt")),
                    key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    rounds = []
    layout = None
    for p in shards:
        s = torch.load(p, weights_only=False)
        layout = s["layout"]
        rounds.extend(s["rounds"])
    return layout, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--rs", type=int, default=None,
                    help="request-state slot to follow (default: first)")
    ap.add_argument("--jump", type=float, default=0.35,
                    help="relative norm jump threshold")
    args = ap.parse_args()

    layout, rounds = load_rounds(args.dir)
    if not rounds:
        print("no rounds found")
        return
    rss = sorted({r["rs"] for r in rounds})
    rs = args.rs if args.rs is not None else rss[0]
    rows = [r for r in rounds if r["rs"] == rs]
    print(f"{len(rows)} rounds for rs={rs} (available: {rss})")
    print(f"first nct={rows[0]['nct']} last nct={rows[-1]['nct']}")

    # Per (layer,state): relative change of the resumed-slot norm between
    # consecutive rounds, plus whether the window base column moved (crossing).
    prev = {}
    hits = []
    for r in rows:
        ri = r["ri"]
        crossed = r["col"] != prev.get("col", r["col"])
        prev["col"] = r["col"]
        for key in layout:
            norms = r["norms"].get(key)
            if not norms or ri >= len(norms):
                continue
            v = norms[ri]
            pv = prev.get(key)
            prev[key] = v
            if pv is None or pv == 0.0:
                continue
            rel = abs(v - pv) / pv
            if rel > args.jump:
                hits.append((r["n"], r["nct"], key, ri, round(rel, 3),
                             r["T"], int(crossed)))

    print(f"\nresumed-slot norm jumps > {args.jump}:")
    if not hits:
        print("  NONE")
    for h in hits[:80]:
        print(f"  round {h[0]:5d} nct={h[1]:6d} {h[2]:40s} ri={h[3]:2d} "
              f"rel={h[4]:5.3f} T={h[5]:2d} crossed={h[6]}")
    if len(hits) > 80:
        print(f"  ... {len(hits) - 80} more")

    # Layer-level summary: max consecutive-round relative change of the
    # resumed-slot norm, to spot a single noisy layer even below threshold.
    print("\nper-layer max rel change (resumed slot):")
    stats = {}
    prev = {}
    for r in rows:
        ri = r["ri"]
        for key in layout:
            norms = r["norms"].get(key)
            if not norms or ri >= len(norms):
                continue
            v = norms[ri]
            pv = prev.get(key)
            prev[key] = v
            if pv is None or pv == 0.0:
                continue
            rel = abs(v - pv) / pv
            m = stats.get(key)
            if m is None or rel > m[0]:
                stats[key] = (rel, r["n"])
    for key, (rel, n) in sorted(stats.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"  {key:40s} max={rel:6.3f} at round {n}")


if __name__ == "__main__":
    main()
