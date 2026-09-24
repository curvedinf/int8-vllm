#!/usr/bin/env python3
"""Aggregate a PyTorch chrome trace: kernel time by category/name.

Usage: trace_kernel_summary.py <trace.json.gz> [--top 25]
"""
import gzip
import json
import re
import sys
from collections import defaultdict


def main() -> None:
    path = sys.argv[1]
    top = int(sys.argv.argv[2 - 1]) if False else 25
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])

    with gzip.open(path, "rt") as f:
        data = json.load(f)

    events = data.get("traceEvents", [])
    per_name = defaultdict(lambda: [0.0, 0])  # us, count
    t_min, t_max = None, None
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        dur = ev.get("dur", 0)
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            per_name[name][0] += dur
            per_name[name][1] += 1
            ts = ev.get("ts", 0)
            t_min = ts if t_min is None else min(t_min, ts)
            t_max = max(t_max or 0, ts + dur)

    def category(name: str) -> str:
        n = name.lower()
        if "sigmoid_gating" in n or "delta_rule" in n:
            return "GDN_recurrence"
        if "causal_conv1d" in n or "conv1d" in n:
            return "GDN_conv"
        if "gluon" in n or "g128" in n:
            return "gluon_g128"
        if "unified_attention" in n or "fmha" in n or "attention" in n:
            return "attention"
        if "aiter" in n or "ck" in n.split("_")[0:1] or re.search(r"\bck\b", n):
            return "AITER"
        if "gemm" in n or "matmul" in n or "s16816" in n or "xnack" in n:
            return "GEMM_other"
        if "elementwise" in n:
            return "elementwise"
        if "reduce" in n or "reduction" in n:
            return "reduction"
        if "memcpy" in n or "memset" in n:
            return "memcpy/memset"
        if "quant" in n or "int8" in n:
            return "quant"
        if "cross_device_reduce" in n or "all_reduce" in n or "allreduce" in n:
            return "all_reduce"
        return "other"

    per_cat = defaultdict(lambda: [0.0, 0])
    for name, (us, cnt) in per_name.items():
        c = category(name)
        per_cat[c][0] += us
        per_cat[c][1] += cnt

    window = (t_max - t_min) / 1e6 if t_min is not None else 0.0
    total = sum(us for us, _ in per_name.values()) / 1e6
    print(f"trace window {window:.2f}s; total GPU kernel time {total:.2f}s "
          f"(busy {100*total/window if window else 0:.0f}%)")
    print("\n== by category ==")
    for c, (us, cnt) in sorted(per_cat.items(), key=lambda kv: -kv[1][0]):
        print(f"{c:22s} {us/1e3:10.1f} ms {cnt:7d} calls")

    print(f"\n== top {top} kernels ==")
    for name, (us, cnt) in sorted(per_name.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"{us/1e3:9.1f} ms {cnt:6d}x  {name[:110]}")


if __name__ == "__main__":
    main()
