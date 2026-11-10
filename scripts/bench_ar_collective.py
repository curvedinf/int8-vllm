#!/usr/bin/env python3
"""Standalone TP4 collective microbench: GS128 int8 payload collective
(vllm/_residual_g128) vs the separate AR+norm chain, decode shapes.

Correctness (compare vs NCCL all_reduce on materialized values) runs first;
timing uses the kernel microbenchmark graph-replay recipe over rotating
buffers (cold HBM, HIP events). Needs 4 GPUs; writes nothing.

Usage: scripts/bench_ar_collective.py
"""
import os, torch
import torch.distributed as dist

rank = int(os.environ.get("LOCAL_RANK", "0"))
from vllm.config import VllmConfig, set_current_vllm_config

set_current_vllm_config(VllmConfig())
dist.init_process_group(backend="nccl")
torch.cuda.set_device(rank)
dev = torch.device(f"cuda:{rank}")
world = dist.get_world_size()
from vllm.distributed.parallel_state import initialize_model_parallel
initialize_model_parallel(tensor_model_parallel_size=world)
from vllm.distributed.device_communicators.residual_g128_all_reduce import all_reduce as rg_ar
from vllm.model_executor.layers.residual_int8 import ResidualInt8, materialize
from vllm.distributed import get_tp_group

H = 2048
def make(M, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randint(-127, 127, (M, H), device=dev, dtype=torch.int8, generator=g)
    s = torch.rand(M, H//128, device=dev, generator=g) * 0.01
    return ResidualInt8(q, s, torch.bfloat16)

def rot_time(build_and_call, slots, inner=2, iters=60):
    # rotate buffers so each call is HBM-cold
    calls = [build_and_call(i) for i in range(slots)]
    for c in calls: c()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(slots*inner):
            calls[i % slots]()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters/(slots*inner)*1000  # us per call

for M in (12, 96, 2048):
    pay = make(M)
    # correctness vs NCCL on the materialized values
    out_nccl = get_tp_group().all_reduce(materialize(pay))
    out_gs = materialize(rg_ar(pay))
    rel = ((out_gs.float()-out_nccl.float()).abs().mean()/out_nccl.float().abs().mean()).item()
    # rotating buffers: 400MB footprint per slot
    slots = max(1, int(400e6 / (M*H*1)))
    pays = [make(M, seed=1000+i) for i in range(slots)]
    def gs(i): return lambda: rg_ar(pays[i])
    us = rot_time(gs, slots)
    if rank == 0:
        print(f"M={M:5d} codes={q.shape} gs128-correct rel={rel:.5f}  gs128={us:8.1f} us/call (slots={slots})", flush=True)
    # production shape note: this measures the bare collective, not the fused epilogue
dist.destroy_process_group()
