#!/usr/bin/env python3
"""Measure the production GS128 payload collective (TP4, XGMI).

Builds the native custom-AR IPC allocator + the GS128 transport, verifies
the reduction against a float64 reference, and times 1 / 98 chained calls at
the decode payload (96x2048 int8 + scales). VLLM_GFX908_G128_AR_BLOCKS
selects the reduce grid (default 16, max 64).

  .venv/bin/python scripts/bench_g128_collective.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402


def worker(rank, world, repo):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29601")
    os.environ.setdefault("VLLM_RESIDUAL_G128_LIBRARY",
                          os.path.join(repo, "vllm/_residual_g128.so"))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )
    from vllm.distributed.device_communicators.residual_g128_all_reduce import (
        ResidualG128Allreduce,
    )
    from vllm.model_executor.layers.residual_int8 import ResidualInt8

    native = CustomAllreduce(dist.group.WORLD, device=f"cuda:{rank}")
    comm = ResidualG128Allreduce(native)
    g = torch.Generator(device="cpu").manual_seed(1234 + rank)
    codes = torch.randint(-60, 61, (96, 2048), generator=g,
                          dtype=torch.int8).cuda()
    scales = (torch.rand(96, 16, generator=g) * 0.02 + 0.001).float().cuda()
    val = ResidualInt8(codes, scales, torch.bfloat16)

    # reference from every rank's payload via a CPU all-gather
    q_all = [torch.empty_like(codes, device="cpu") for _ in range(world)]
    s_all = [torch.empty_like(scales, device="cpu") for _ in range(world)]
    dist.all_gather(q_all, codes.cpu())
    dist.all_gather(s_all, scales.cpu())
    out = comm.all_reduce(val)
    torch.cuda.synchronize()
    if rank == 0:
        acc = torch.zeros(96, 16, 128, dtype=torch.float64)
        for q, s in zip(q_all, s_all):
            acc += q.double().view(96, 16, 128) * s.double().unsqueeze(-1)
        amax = acc.abs().amax(dim=-1)
        sr = torch.clamp(amax / 127.0, min=1e-30)
        ref = torch.round(acc / sr.unsqueeze(-1)).clamp(-127, 127)
        ref = ref.to(torch.int8).view(96, 2048)
        print("correct=", torch.equal(out.codes.cpu(), ref),
              "scale_diff=", (out.scales.cpu() - sr.float()).abs().max().item(),
              flush=True)
    for _ in range(5):
        comm.all_reduce(val)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(200):
        comm.all_reduce(val)
    e.record()
    torch.cuda.synchronize()
    if rank == 0:
        print("blocks=", os.environ.get("VLLM_GFX908_G128_AR_BLOCKS", "16"),
              "us_per_call=", round(s.elapsed_time(e) / 200 * 1000, 2), flush=True)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            comm.all_reduce(val)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        for _ in range(98):
            comm.all_reduce(val)
    for _ in range(3):
        g2.replay()
    torch.cuda.synchronize()
    s.record()
    for _ in range(20):
        g2.replay()
    e.record()
    torch.cuda.synchronize()
    if rank == 0:
        print("step98_ms=", round(s.elapsed_time(e) / 20, 2), flush=True)
    dist.destroy_process_group()


def main():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mp.spawn(worker, args=(4, repo), nprocs=4, join=True)


if __name__ == "__main__":
    main()
