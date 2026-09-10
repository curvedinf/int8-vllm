#!/usr/bin/env python3
"""W8A8 GEMM decode-batch timing: aiter CK vs torch bf16 at M=1/14.

Per-rank decode shapes for Qwen3.8-27B TP4 (5120 hidden, 17408/4 inter).
"""
import sys
import torch

sys.path.insert(0, "/home/curved/aiter")

dev = "cuda"


def bench_bf16(M, K, N, iters=100):
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(N, K, device=dev, dtype=torch.bfloat16).t()
    for _ in range(10):
        a @ b
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        a @ b
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


if __name__ == "__main__":
    H, I = 5120, 4352
    shapes = [
        ("qkvz_proj", H, H * 3 // 2 + I),  # rough combined
        ("o_proj", H, H),
        ("gate_up", H, 2 * I),
        ("down_proj", I, H),
    ]
    for M in (1, 14):
        print(f"M={M}:")
        tot = 0
        for name, K, N in shapes:
            ms = bench_bf16(M, K, N)
            tot += ms
            print(f"  {name:10s} K={K} N={N}: {ms:6.3f} ms", flush=True)
        print(f"  bf16 total/layer: {tot:6.3f} ms; x64 layers = {64*tot:7.1f} ms")
