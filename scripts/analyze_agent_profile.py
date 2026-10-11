#!/usr/bin/env python3
"""Summarize GPU CSV + lifecycle JSONL without summing TP ranks as wall time.

GPU busy time is the union of kernel intervals per device/process. Category
times are sums (overlap can exceed busy time). Step phase assignment uses the
correlated HIP launch timestamp and scheduler/worker metadata, falling back to
GPU start when no launch correlation exists. Host spans are asynchronous.
Keep raw traces for exact correlation-ID analysis in a timeline viewer.
"""

import argparse
import bisect
import csv
import functools
import json
import math
from collections import defaultdict
from pathlib import Path


@functools.cache
def category(name):
    name = name.lower()
    if any(s in name for s in ("allreduce", "all_reduce", "cross_device", "car_",
                               "nccl", "rccl")):
        return "collective"
    if any(s in name for s in ("g128", "unified_attention", "fmha", "reduce_segments")):
        return "attention"
    if any(s in name for s in ("delta_rule", "fwd_h", "fwd_o", "fwd_intra", "wy_",
                               "sigmoid_gating", "solve_tril", "cumsum",
                               "chunk_fwd", "fused_recurrent", "recompute_w_u",
                               "merge_16x16")):
        return "GDN_recurrence"
    if "conv1d" in name:
        return "GDN_convolution"
    if any(s in name for s in ("gemm", "matmul", "cijk_", "tiny_ba")):
        return "GEMM"
    if any(s in name for s in ("norm", "rms")):
        return "normalization"
    if "quant" in name:
        return "quantization"
    if any(s in name for s in ("copy", "offload", "memset", "fillbuffer")):
        return "copy_store"
    if any(s in name for s in ("sample", "gumbel", "rejection", "softmax", "topk")):
        return "sampling"
    return "other"


def union_ns(intervals):
    end = -1
    total = 0
    for lo, hi in sorted(intervals):
        total += max(0, hi - max(lo, end))
        end = max(end, hi)
    return total


def stats(values):
    values = sorted(values)
    return {"count": len(values), "sum_s": sum(values),
            "p50_s": (values[(len(values)-1)//2] + values[len(values)//2])/2,
            "p95_s": values[math.ceil(len(values)*.95)-1], "max_s": values[-1]}


def phase(row):
    reqs = row.get("requests", {})
    if reqs:
        prefill = [r["output"] == 0 for r in reqs.values()]
    else:
        prefill = [n > 7 for n in row.get("scheduled", {}).values()]
    if not prefill:
        return "empty"
    return "prefill" if all(prefill) else "mixed" if any(prefill) else "decode"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-csv", "--rocprof", dest="rocprof", required=True,
                        help="Native or Kineto-converted trace CSV directory")
    parser.add_argument("--host", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", help="Client run directory; excludes smoke/startup spans")
    args = parser.parse_args()
    first_ns, last_ns = 0, 2**63 - 1
    if args.run:
        meta = json.loads((Path(args.run) / "meta.json").read_text())
        first_ns, last_ns = meta["origin_ns"], meta["finished_ns"]
    hosts, host_stats, steps = {}, defaultdict(list), {}
    scheduler = []
    for path in Path(args.host).glob("*.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [r for r in rows if first_ns <= r["start_ns"] <= last_ns]
        hosts[path.stem] = rows
        for row in rows:
            host_stats[f"{path.stem}:{row['name']}"].append(
                (row["end_ns"] - row["start_ns"]) / 1e9)
            if row["name"] == "scheduler.schedule":
                scheduler.append(row)
    scheduler.sort(key=lambda x: x["start_ns"])
    sched_times = [r["start_ns"] for r in scheduler]
    for pid, rows in hosts.items():
        selected = sorted((r for r in rows if r["name"] == "worker.execute_model"),
                          key=lambda x: x["start_ns"])
        for row in selected:
            idx = bisect.bisect_right(sched_times, row["start_ns"]) - 1
            # Pipeline lookahead means the latest scheduler call may belong to
            # the next batch. Match request/count mappings in recent calls.
            candidates = scheduler[max(0, idx-4):idx+1]
            match = next((r for r in reversed(candidates)
                          if r.get("scheduled") == row.get("scheduled")), row)
            row["phase"] = phase(match)
        steps[pid] = selected
    result = {"host_spans": {k: stats(v) for k, v in host_stats.items()},
              "ranks": {}, "api": {}, "copies": {}, "notes": __doc__}
    launches = defaultdict(dict)
    chunk_first_launch = defaultdict(dict)
    for path in Path(args.rocprof).rglob("*_hip_api_trace.csv"):
        pid = path.name.split("_")[0]
        stage_spans = {}
        for stage in ("draft.propose", "runner.sample", "runner.prepare_attn",
                      "runner.prepare_inputs", "runner.postprocess_sampled",
                      "worker.execute_model", "worker.sample_tokens"):
            spans = sorted((r for r in hosts.get(pid, []) if r["name"] == stage),
                           key=lambda r: r["start_ns"])
            stage_spans[stage] = ([r["start_ns"] for r in spans], spans)
        totals = defaultdict(lambda: [0, 0, 0])
        with path.open() as file:
            for row in csv.DictReader(file):
                if not first_ns <= int(row["Start_Timestamp"]) <= last_ns:
                    continue
                value = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
                entry = totals[row["Function"]]
                entry[0] += value
                entry[1] += 1
                entry[2] = max(entry[2], value)
                if "Launch" in row["Function"]:
                    timestamp = int(row["Start_Timestamp"])
                    origin = "unattributed"
                    for stage, (times, spans) in stage_spans.items():
                        i = bisect.bisect_right(times, timestamp) - 1
                        if i >= 0 and timestamp <= spans[i]["end_ns"]:
                            origin = stage
                            break
                    launches[pid][row["Correlation_Id"]] = (timestamp, origin)
                    chunk = (row["Correlation_Id"].split(":", 1)[0]
                             if ":" in row["Correlation_Id"] else "capture")
                    chunk_first_launch[pid][chunk] = min(
                        timestamp, chunk_first_launch[pid].get(chunk, timestamp))
        result["api"][path.name] = [
            {"name": name, "sum_s": v[0]/1e9, "count": v[1], "max_s": v[2]/1e9}
            for name, v in sorted(totals.items(), key=lambda x: -x[1][0])]
    for path in Path(args.rocprof).rglob("*_memory_copy_trace.csv"):
        totals = defaultdict(lambda: [0, 0, 0])
        with path.open() as file:
            for row in csv.DictReader(file):
                if not first_ns <= int(row["Start_Timestamp"]) <= last_ns:
                    continue
                label = row.get("Direction", row.get("Kind", "copy"))
                entry = totals[label]
                entry[0] += int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
                entry[1] += 1
                entry[2] += int(row.get("Size", row.get("Bytes", 0)))
        result["copies"][path.name] = {
            name: {"sum_s": v[0]/1e9, "count": v[1], "bytes": v[2]}
            for name, v in totals.items()}
    for path in Path(args.rocprof).rglob("*_kernel_trace.csv"):
        pid = path.name.split("_")[0]
        rows = steps.get(pid, [])
        starts = [r["start_ns"] for r in rows]
        by_name = defaultdict(lambda: [0, 0, 0])
        phase_names = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
        by_phase = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        by_phase_steady = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        by_origin = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        intervals = defaultdict(list)
        attribution_counts = defaultdict(int)
        step_gpu = {}
        chunk_windows = {}
        boundary_steps = {bisect.bisect_right(starts, timestamp)-1
                          for timestamp in chunk_first_launch[pid].values()}
        with path.open() as file:
            for row in csv.DictReader(file):
                lo, hi = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
                if not first_ns <= lo <= last_ns:
                    continue
                name = row["Kernel_Name"]
                duration = hi-lo
                by_name[name][0] += duration
                by_name[name][1] += 1
                by_name[name][2] = max(by_name[name][2], duration)
                correlated = row.get("Correlation_Id") in launches[pid]
                attribution_counts["launch_correlated" if correlated else
                                   "gpu_start_fallback"] += 1
                launch, origin = launches[pid].get(
                    row.get("Correlation_Id"), (lo, "unattributed"))
                idx = bisect.bisect_right(starts, launch)-1
                label = rows[idx]["phase"] if idx >= 0 else "unattributed"
                entry = phase_names[label][name]
                entry[0] += duration
                entry[1] += 1
                entry[2] = max(entry[2], duration)
                correlation = row.get("Correlation_Id", "")
                chunk = correlation.split(":", 1)[0] if ":" in correlation else "capture"
                window = chunk_windows.setdefault((row["Agent_Id"], chunk), [lo, hi])
                window[0], window[1] = min(window[0], lo), max(window[1], hi)
                if idx >= 0:
                    step = step_gpu.setdefault(idx, {
                        "first_ns": lo, "last_ns": hi, "kernel_count": 0,
                        "category_s": defaultdict(float)})
                    step["first_ns"] = min(step["first_ns"], lo)
                    step["last_ns"] = max(step["last_ns"], hi)
                    step["kernel_count"] += 1
                    step["category_s"][category(name)] += duration / 1e9
                by_phase[label][category(name)][0] += duration
                by_phase[label][category(name)][1] += 1
                if idx not in boundary_steps:
                    by_phase_steady[label][category(name)][0] += duration
                    by_phase_steady[label][category(name)][1] += 1
                by_origin[origin][category(name)][0] += duration
                by_origin[origin][category(name)][1] += 1
                intervals[row["Agent_Id"]].append((lo, hi))
        devices = {}
        for agent, spans in intervals.items():
            first, last = min(s[0] for s in spans), max(s[1] for s in spans)
            busy = union_ns(spans)
            devices[agent] = {"first_ns": first, "last_ns": last,
                              "window_s": (last-first)/1e9,
                              "kernel_busy_s": busy/1e9,
                              "kernel_busy_fraction": busy/max(1, last-first)}
            captured_window = sum(hi-lo for (device, _), (lo, hi)
                                  in chunk_windows.items() if device == agent)
            devices[agent].update(
                summed_chunk_kernel_windows_s=captured_window/1e9,
                kernel_busy_fraction_inside_chunks=busy/max(1, captured_window))
        result["ranks"][pid] = {
            "devices": devices, "kernels": len(by_name),
            "kernel_count": sum(v[1] for v in by_name.values()),
            "attribution_counts": dict(attribution_counts),
            "by_phase": {p: {c: {"sum_s": v[0]/1e9, "count": v[1]}
                              for c, v in cats.items()} for p, cats in by_phase.items()},
            "by_phase_excluding_chunk_boundary_steps": {
                p: {c: {"sum_s": v[0]/1e9, "count": v[1]}
                    for c, v in cats.items()} for p, cats in by_phase_steady.items()},
            "by_origin": {p: {c: {"sum_s": v[0]/1e9, "count": v[1]}
                               for c, v in cats.items()} for p, cats in by_origin.items()},
            "top_kernels": [{"name": n, "category": category(n), "sum_s": v[0]/1e9,
                             "count": v[1], "mean_us": v[0]/v[1]/1e3,
                             "max_us": v[2]/1e3}
                            for n, v in sorted(by_name.items(), key=lambda x: -x[1][0])[:60]]}
        step_groups = defaultdict(list)
        for idx, step in step_gpu.items():
            step.update(phase=rows[idx]["phase"],
                        host_start_ns=rows[idx]["start_ns"],
                        scheduled=rows[idx].get("scheduled", {}),
                        gpu_window_s=(step["last_ns"]-step["first_ns"])/1e9)
            step_groups[step["phase"]].append(step["gpu_window_s"])
        result["ranks"][pid]["step_gpu_windows"] = {
            p: stats(v) for p, v in step_groups.items()}
        result["ranks"][pid]["chunk_boundary_steps"] = sorted(boundary_steps)
        result["ranks"][pid]["top_kernels_by_phase"] = {
            p: [{"name": n, "category": category(n), "sum_s": v[0]/1e9,
                 "count": v[1], "mean_us": v[0]/v[1]/1e3, "max_us": v[2]/1e3}
                for n, v in sorted(names.items(), key=lambda x: -x[1][0])[:20]]
            for p, names in phase_names.items()}
        result["ranks"][pid]["longest_steps"] = sorted(
            step_gpu.values(), key=lambda s: -s["gpu_window_s"])[:20]
        print(pid, devices, flush=True)
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
