#!/usr/bin/env python3
"""Aggregate a torch-profiler chrome trace into a decode-step budget.

Usage: .venv/bin/python scripts/analyze_flint_profile.py <trace_dir>
Groups kernels by name, sums microseconds, and reports the top
contributors plus totals partitioned into: MoE GEMMs, collectives,
attention, dense GEMMs, elementwise/other. This attributes a vLLM
decode step's wall time to concrete mechanisms (run-experiment input).
"""
import json, sys
from pathlib import Path

def find_traces(root: Path):
    out = []
    for p in root.rglob("*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        evs = d.get("traceEvents", [])
        if any(e.get("ph") == "X" and e.get("dur") for e in evs):
            out.append(p)
    return out

def main():
    root = Path(sys.argv[1])
    files = find_traces(root)
    print(f"traces: {len(files)}")
    if not files:
        return
    for f in files:
        print(f"== {f}")
    agg = {}
    walls = []
    for f in files:
        d = json.loads(f.read_text())
        for e in d.get("traceEvents", []):
            if e.get("ph") != "X" or not e.get("dur"):
                continue
            name = e.get("name", "?")
            agg[name] = agg.get(name, 0) + e["dur"]
            if name in ("vllm::execute_model", "execute_model"):
                walls.append(e["dur"])
    total = sum(agg.values())
    print(f"self-time total: {total/1e3:.1f} ms")
    if walls:
        print(f"execute_model calls: {len(walls)} mean {walls and (sum(walls)/len(walls)/1e3):.2f} ms")
    cats = {"moe": 0.0, "coll": 0.0, "attn": 0.0, "gemm": 0.0, "other": 0.0}
    for n, t in agg.items():
        ln = n.lower()
        if "moe" in ln or "_moe" in ln or "grouped" in ln:
            cats["moe"] += t
        elif any(s in ln for s in ("all_reduce", "reduce", "collective", "ar_", "car")):
            cats["coll"] += t
        elif "attention" in ln or "attn" in ln or "flash" in ln or "unified" in ln:
            cats["attn"] += t
        elif any(s in ln for s in ("gemm", "matmul", "ck_", "aiter")):
            cats["gemm"] += t
        else:
            cats["other"] += t
    for k, v in cats.items():
        print(f"  {k:6s} {v/total*100:5.1f}%  {v/1e3:7.2f} ms")
    print("top kernels:")
    for n, t in sorted(agg.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {t/1e3:7.3f} ms  {n[:90]}")

if __name__ == "__main__":
    main()
