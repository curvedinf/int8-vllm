#!/usr/bin/env python3
"""E2 child: time vLLM CUSTOM AR at one (threads, blocks) geometry.

argv: <threads> <blocks>. Env is set before any AR call so the kernel's
static cache picks it up. Prints CSV rows: threads,blocks,T,us.
"""
import os
import sys
import time

threads, blocks = int(sys.argv[1]), int(sys.argv[2])
os.environ["VLLM_GFX908_AR_THREADS"] = str(threads)
os.environ["VLLM_GFX908_AR_BLOCKS"] = str(blocks)
os.environ.update(
    {
        "ROCM_PATH": "/opt/rocm",
        "LD_LIBRARY_PATH": "/opt/rocm/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
        "HIP_VISIBLE_DEVICES": "0,1,2,3",
        "ROCR_VISIBLE_DEVICES": "0,1,2,3",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29517",
    }
)
sys.path.insert(0, "/home/curved/vllm-gfx908")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from vllm.distributed.device_communicators.custom_all_reduce import (  # noqa: E402
    CustomAllreduce,
)

rank = int(os.environ.get("RANK", "0"))
dist.init_process_group("gloo", rank=rank, world_size=4)
torch.cuda.set_device(rank)
car = CustomAllreduce(group=dist.group.WORLD, device=torch.device(f"cuda:{rank}"))
assert not car.disabled

def bench(T: int, iters: int = 200) -> float:
    x = torch.randn(T, 5120, dtype=torch.float16, device="cuda")
    for _ in range(20):
        car.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        car.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - t0) / iters * 1e6

if rank == 0:
    print(f"threads,blocks,T,us")
for T in (1, 2, 4, 8, 16, 32, 64):
    us = bench(T)
    if rank == 0:
        print(f"{threads},{blocks},{T},{us:.1f}", flush=True)
dist.destroy_process_group()
