#!/usr/bin/env python3
"""Per-site int8 decode-band GEMM baseline (Triton GS128 blockscale).

Times the exact production call for every dense GEMM at M=12 and M=96 with
CUDA-graph replay, and reports effective weight bandwidth and TFLOP/s
against the MI100 int8 ceilings (184.6 TOPS, 921-953 GB/s measured).

This is the denominator for the stage-2 native small-GEMM gate and the
per-site roofline table. Int8 only.

  HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD:$PWD/../int8-aiter \
    .venv/bin/python scripts/bench_int8_small_gemm.py
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

# per-TP4-rank shapes (N, K)
SITES = {
    "target qkv": (1280, 2048),
    "target o_proj": (2048, 1024),
    "target shared gate_up": (512, 2048),    # 2*(1024/4) x 2048
    "target shared down": (2048, 256),       # 2048 x (1024/4)
    "target lm_head": (37536, 2048),
    "draft qkv": (1536, 2048),
    "draft o_proj": (2048, 1024),
    "draft gate_up": (3072, 2048),           # 2*(6144/4) x 2048
    "draft down": (2048, 1536),              # 2048 x (6144/4)
}
BANDWIDTH = 921e9   # measured per-GPU stream ceiling, bytes/s
TOPS = 184.6e12     # MI100 int8 matrix peak


def timed_graph(fn, warmup=3, iters=30):
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
    print(f"{'site':22s} {'M':>3s} {'us':>8s} {'GB/s':>8s} {'%BW':>6s} "
          f"{'TFLOP/s':>8s} {'%peak':>6s}")
    for name, (N, K) in SITES.items():
        w = torch.randint(-60, 61, (N, K), dtype=torch.int8, device="cuda")
        ws = torch.rand(N, K // 128, dtype=torch.float32, device="cuda") * 0.02
        wbytes = N * K
        for M in (16, 32, 96):
            cfg = _get_aiter_w8a8_config(M, N, K, 128)
            x = torch.randint(-60, 61, (M, K), dtype=torch.int8, device="cuda")
            xs = torch.rand(M, K // 128, dtype=torch.float32, device="cuda") * 0.02
            us = timed_graph(lambda: gemm_a8w8_blockscale(
                x, w, xs, ws, dtype=torch.bfloat16, config=cfg))
            gbs = wbytes / (us * 1e-6) / 1e9
            tflops = 2 * M * N * K / (us * 1e-6) / 1e12
            print(f"{name:22s} {M:3d} {us:8.2f} {gbs:8.1f} "
                  f"{100 * gbs / (BANDWIDTH / 1e9):5.0f}% {tflops:8.2f} "
                  f"{100 * tflops / (TOPS / 1e12):5.1f}%", flush=True)


if __name__ == "__main__":
    main()
