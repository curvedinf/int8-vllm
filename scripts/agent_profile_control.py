#!/usr/bin/env python3
"""Idle-only GPU capture control for an isolated lifecycle-enabled recipe boot."""

import argparse
import json
import os
import urllib.request
from pathlib import Path

from bench_agent_mixed import nginx_down


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop"))
    parser.add_argument("--tag", default="capture")
    parser.add_argument("--ranks", default="0",
                        help="Comma-separated TP ranks; full-duration default is rank 0")
    parser.add_argument("--url", default="http://127.0.0.1:8020")
    parser.add_argument("--output", help="Save the complete worker capture manifest")
    args = parser.parse_args()
    nginx_down()
    headers = {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"],
               "Content-Type": "application/json"}
    request = urllib.request.Request(args.url + "/metrics", headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        metrics = response.read().decode()
    gauges = {}
    for line in metrics.splitlines():
        if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")):
            name = line.split("{", 1)[0]
            gauges[name] = gauges.get(name, 0) + float(line.rsplit(" ", 1)[1])
    if len(gauges) != 2 or any(gauges.values()):
        raise RuntimeError(f"Backend must be idle before capture control: {gauges}")
    body = {"method": f"{args.action}_lifecycle_gpu_profile", "timeout": 300,
            "args": [args.tag, args.ranks] if args.action == "start" else []}
    request = urllib.request.Request(args.url + "/collective_rpc", headers=headers,
                                     data=json.dumps(body).encode())
    with urllib.request.urlopen(request, timeout=330) as response:
        result = json.load(response)
    if len(result.get("results", [])) != 4:
        raise RuntimeError(f"Expected four worker acknowledgments: {result}")
    if args.output:
        with Path(args.output).open("x") as file:
            json.dump(result, file, indent=2)
    if args.action == "stop":
        print(json.dumps({"results": [{"pid": r["pid"],
            "trace_chunks": sum("path" in c for c in r.get("chunks", [])),
            "skipped_chunks": sum(c.get("skipped", False) for c in r.get("chunks", []))}
            for r in result["results"]]}))
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
