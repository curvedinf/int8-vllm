#!/usr/bin/env python3
"""Does activation-pointer alignment change the decode-band a8w8 blockscale
GEMM's speed? Triton specializes each pointer on 16-byte divisibility, so a
process whose activation view lands off 16 bytes may compile and keep a
scalar variant for its whole life (the born-slow hypothesis).

Times the production call at the Flint qkv decode shape (M=96, N=1280,
K=2048, GS128) via CUDA-graph replay so the measurement is kernel time, not
Python dispatch. Int8 only.

  HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD:$PWD/../int8-aiter \
    .venv/bin/python scripts/bench_blockscale_align.py
"""

import sys

import torch

sys.path.insert(0, ".")
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (  # noqa: E402
    gemm_a8w8_blockscale,
)
from vllm.model_executor.kernels.linear.mixed_precision.aiter_w8a16 import (  # noqa: E402
    _get_aiter_w8a8_config,
)

M, N, K = 96, 1280, 2048


def timed_graph(fn, warmup=3, iters=40):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def main():
    torch.manual_seed(0)
    cfg = _get_aiter_w8a8_config(M, N, K, 128)
    w = torch.randint(-60, 61, (N, K), dtype=torch.int8, device="cuda")
    xs = torch.rand(M, K // 128, dtype=torch.float32, device="cuda") * 0.02
    ws = torch.rand(N, K // 128, dtype=torch.float32, device="cuda") * 0.02
    src = torch.randint(-60, 61, (M, K), dtype=torch.int8, device="cuda")
    rows = []
    for off in (0, 1, 2, 4, 8, 16, 64, 128):
        buf = torch.empty(M * K + 128, dtype=torch.int8, device="cuda")
        x = buf[off:off + M * K].view(M, K)
        x.copy_(src)
        us = timed_graph(lambda: gemm_a8w8_blockscale(
            x, w, xs, ws, dtype=torch.bfloat16, config=cfg))
        rows.append((off, us))
        print(f"offset {off:3d} B: {us:8.2f} us", flush=True)
    base = rows[0][1]
    print("--- relative to offset 0")
    for off, us in rows:
        print(f"  off {off:3d}: {us / base:.3f}x")


if __name__ == "__main__":
    main()
