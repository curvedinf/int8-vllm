#!/usr/bin/env python3
"""Standalone custom-AR latency at decode shapes on this fabric.

torchrun --nproc_per_node=4 scripts/bench_custom_ar.py
Times vLLM custom AR (and NCCL fallback for reference) at hidden 5120,
the per-layer TP4 reduction size, interleaved with GEMM-like work to
mimic the engine pattern.
"""
import os
import time
import torch
import torch.distributed as dist

rank = int(os.environ.get("LOCAL_RANK", "0"))
world = int(os.environ.get("WORLD_SIZE", "1"))
dist.init_process_group(backend="nccl")
torch.cuda.set_device(rank)
dev = torch.device(f"cuda:{rank}")

from vllm.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
)
cpu_group = dist.new_group(backend="gloo")
ar = CustomAllreduce(group=cpu_group, device=dev)


def bench(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    dist.barrier()
    return e0.elapsed_time(e1) / iters


if rank == 0:
    print("custom-AR vs NCCL at hidden 5120 (bf16):")
for n in (5120, 5120 * 4):
    x = torch.randn(1, n, device=dev, dtype=torch.bfloat16)
    nccl_ms = bench(lambda: dist.all_reduce(x))
    car_ms = bench(lambda: ar.custom_all_reduce(x))
    if rank == 0:
        print(f"  n={n:6d}: custom {car_ms*1000:8.1f} us | nccl {nccl_ms*1000:8.1f} us",
              flush=True)

# burst pattern: 128 ARs back-to-back (engine does ~128/step)
x = torch.randn(1, 5120, device=dev, dtype=torch.bfloat16)
def burst():
    for _ in range(128):
        ar.custom_all_reduce(x)
ms = bench(burst, iters=5)
if rank == 0:
    print(f"  128x AR burst: {ms:.2f} ms total", flush=True)

ar.close()
dist.destroy_process_group()
