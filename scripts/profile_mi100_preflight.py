"""Bounded TP4 profiler preflight; run with torch.distributed.run, never serving.

Initialize communications before enabling selected-region tracing. Exercises
the recipe's custom AR graph replay and small RCCL broadcast, not a replacement
production configuration. The constant reference needs no large RCCL reduction.
"""

import ctypes
import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("roctx", "kineto", "none"), default="kineto")
    parser.add_argument("--output", default="logs/agent_mixed/kineto_preflight")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    print(rank, "device ready", flush=True)
    dist.init_process_group("gloo")
    group = dist.new_group(backend="nccl")
    token = torch.tensor([rank], dtype=torch.int64, device="cuda")
    dist.broadcast(token, 0, group=group)
    torch.cuda.synchronize()
    print(rank, "broadcast ready", flush=True)
    ar = CustomAllreduce(group=dist.group.WORLD, device=rank)
    assert not ar.disabled
    data = torch.full((262144,), rank+1, dtype=torch.float16, device="cuda")
    out = torch.empty_like(data)
    ar.all_reduce(data, out=out, registered=False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with ar.capture(), torch.cuda.graph(graph):
        ar.all_reduce(data, out=out, registered=False)
    print(rank, "graph ready", flush=True)
    if args.backend == "roctx":
        roctx = ctypes.CDLL("/opt/rocm/lib/librocprofiler-sdk-roctx.so")
        roctx.roctxProfilerResume(0)
        roctx.roctxRangePushA(b"preflight replay and broadcast")
    elif args.backend == "kineto":
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
        profiler.start()
    for _ in range(5):
        graph.replay()
        dist.broadcast(token, 0, group=group)
    torch.cuda.synchronize()
    if args.backend == "roctx":
        roctx.roctxRangePop()
        roctx.roctxProfilerPause(0)
    elif args.backend == "kineto":
        profiler.stop()
        Path(args.output).mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(Path(args.output) / f"rank{rank}.json.gz"))
    assert torch.all(out == 10).item()
    assert token.item() == 0
    print(rank, "PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
