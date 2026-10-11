#!/usr/bin/env python3
"""Stream a GPU-only Kineto trace into bounded-memory lifecycle-analysis CSVs.

Kineto timestamps are epoch-relative to baseTimeNanoseconds. Explicit worker
clock calibration maps them to the host/client CLOCK_MONOTONIC domain.
Original .json.gz files remain directly loadable as Chrome/Perfetto traces.
"""

import argparse
import csv
import gzip
import json
from pathlib import Path

import ijson


def convert(path, output, append=False):
    clock = json.loads(Path(str(path).replace(".json.gz", ".clock.json")).read_text())
    with gzip.open(path, "rb") as file:
        base = next(ijson.items(file, "baseTimeNanoseconds"))
    offset = clock["start"]["wall_minus_monotonic_ns"]
    pid = clock["pid"]
    output.mkdir(parents=True, exist_ok=True)
    files = [open(output / f"{pid}_{suffix}.csv", "a" if append else "x") for suffix in
             ("kernel_trace", "hip_api_trace", "memory_copy_trace")]
    kernels, apis, copies = [csv.writer(f) for f in files]
    if files[0].tell() == 0:
        kernels.writerow(["Agent_Id", "Kernel_Name", "Start_Timestamp", "End_Timestamp",
                      "Correlation_Id", "Stream_Id"])
        apis.writerow(["Function", "Start_Timestamp", "End_Timestamp", "Correlation_Id"])
        copies.writerow(["Direction", "Start_Timestamp", "End_Timestamp", "Bytes"])
    counts = {"kernel": 0, "runtime": 0, "copy": 0}
    try:
        with gzip.open(path, "rb") as file:
            for event in ijson.items(file, "traceEvents.item", use_float=True):
                if event.get("ph") != "X":
                    continue
                start = base + round(event["ts"] * 1000) - offset
                end = start + round(event["dur"] * 1000)
                args = event.get("args", {})
                cat = event.get("cat")
                if cat == "kernel":
                    kernels.writerow([f"Agent {args.get('device', event['pid'])}",
                        event["name"], start, end, f"{path.stem}:{args.get('correlation', '')}",
                        args.get("stream", "")])
                    counts["kernel"] += 1
                elif cat in ("cuda_runtime", "hip_runtime", "cuda_driver"):
                    apis.writerow([event["name"], start, end,
                                   f"{path.stem}:{args.get('correlation', '')}"])
                    counts["runtime"] += 1
                elif cat in ("gpu_memcpy", "gpu_memset"):
                    copies.writerow([event["name"], start, end, args.get("bytes", 0)])
                    counts["copy"] += 1
    finally:
        for file in files:
            file.close()
    print(path.name, counts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=False)
    for trace in args.traces:
        convert(Path(trace), Path(args.output), append=True)
