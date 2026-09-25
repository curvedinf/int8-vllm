#!/usr/bin/env python3
"""Measure gfx908 MFMA rate for fp16 vs bf16 tl.dot (single CTA)."""
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _mma_rate_kernel(
    A, B, C, DTYPE: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
    K: tl.constexpr, ITER: tl.constexpr,
):
    offs_a = tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :]
    offs_b = tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :]
    offs_c = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    a = tl.load(A + offs_a).to(DTYPE)
    b = tl.load(B + offs_b).to(DTYPE)
    if DTYPE == tl.int8:
        c = tl.zeros([M, N], dtype=tl.int32)
        for _ in range(ITER):
            c += tl.dot(a, b, out_dtype=tl.int32)
    else:
        c = tl.zeros([M, N], dtype=tl.float32)
        for _ in range(ITER):
            c += tl.dot(a, b, out_dtype=tl.float32)
    tl.store(C + offs_c, c)


def bench(dtype, M=256, N=256, K=128, iters=40000):
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    _mma_rate_kernel[(1,)](a, b, c, DTYPE=dtype, M=M, N=N, K=K, ITER=iters,
                           num_warps=8)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _mma_rate_kernel[(1,)](a, b, c, DTYPE=dtype, M=M, N=N, K=K, ITER=iters,
                           num_warps=8)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return 2 * M * N * K * iters / dt / 1e12


if __name__ == "__main__":
    tf16 = bench(tl.float16)
    tbf16 = bench(tl.bfloat16)
    print(f"single-CTA tl.dot: fp16={tf16:.2f} TFLOP/s "
          f"bf16={tbf16:.2f} TFLOP/s ratio={tf16 / tbf16:.2f}x")

    import time as _time

    def bench_i8(M=256, N=256, K=128, iters=40000):
        a = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8)
        b = torch.randint(-127, 127, (K, N), device="cuda", dtype=torch.int8)
        c = torch.empty(M, N, device="cuda", dtype=torch.int32)
        _mma_rate_kernel[(1,)](a, b, c, DTYPE=tl.int8, M=M, N=N, K=K,
                               ITER=iters, num_warps=8)
        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        _mma_rate_kernel[(1,)](a, b, c, DTYPE=tl.int8, M=M, N=N, K=K,
                               ITER=iters, num_warps=8)
        torch.cuda.synchronize()
        dt = _time.perf_counter() - t0
        ref = (a.float() @ b.float()).round()
        d = (c.float() - ref).abs(); exact = bool((d.max() == 0).item()); print('max_abs_err', d.max().item(), 'rel', (d.max()/ref.abs().max()).item())
        return 2 * M * N * K * iters / dt / 1e12, exact

    ti8, exact = bench_i8()
    print(f"int8={ti8:.2f} TFLOP/s exact={exact} "
          f"(vs fp16 {tf16:.2f}: {ti8 / tf16:.2f}x)")
