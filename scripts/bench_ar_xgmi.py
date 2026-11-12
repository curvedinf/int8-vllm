#!/usr/bin/env python3
"""Measure the XGMI all-reduce cost at the Flint per-step payloads.

The step issues 98 collects/step (49 o_proj + 47 MoE + 2 dense) with
~192 KB payloads. This times a 4-GPU RCCL all-reduce on that payload,
eager and captured as the 98-collective step sequence, so the collective
block in the step budget stops being an estimate.

  .venv/bin/python scripts/bench_ar_xgmi.py
"""

import json
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

PAYLOADS = [196608, 49152, 4608]   # 96x2048, 24x2048, 96x48 bytes
STEP_COLLECTIVES = 98


def worker(rank, world, out_path):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    res = {}
    for payload in PAYLOADS:
        t = torch.zeros(payload, dtype=torch.uint8, device=f"cuda:{rank}")
        for _ in range(10):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        iters = 300
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            dist.all_reduce(t)
        e.record()
        torch.cuda.synchronize()
        res[f"eager_us_{payload}"] = round(s.elapsed_time(e) / iters * 1000, 2)
    dist.destroy_process_group()
    if rank == 0:
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)


def main():
    out = "/tmp/ar_xgmi.json"
    mp.spawn(worker, args=(4, out), nprocs=4, join=True)
    print(open(out).read())


if __name__ == "__main__":
    main()
