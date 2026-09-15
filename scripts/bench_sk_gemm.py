#!/usr/bin/env python3
"""Triton split-K bf16 GEMM for skinny M (GDN projections / draft).

C[M,N] = A[M,K] @ W[N,K]^T. Deterministic: partials [SK, M, N] fp32 +
fixed-order reduce (seam-gate friendly - no atomics). Sweep
BLOCK_N/BLOCK_K/SK/warps on the production shapes and compare against
rocBLAS (F.linear) numbers.
"""
import sys
import torch
import triton
import triton.language as tl

dev = "cuda"


@triton.jit
def _sk_gemm(
    A, W, P,  # A [M,K] bf16, W [N,K] bf16, P [SK,M,N] fp32
    M, N, K,
    stride_am, stride_wn, stride_pk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_per = tl.cdiv(K, SK)
    k_lo = pid_k * k_per
    k_hi = tl.minimum(k_lo + k_per, K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            A + offs_m[:, None] * stride_am + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_hi),
            other=0.0,
        )
        w = tl.load(
            W + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < k_hi),
            other=0.0,
        )
        acc += tl.dot(a, tl.trans(w))
    tl.store(
        P + pid_k.to(tl.int64) * stride_pk
        + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _sk_reduce(P, C, M, N, stride_pk, SK: tl.constexpr,
               BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * M
    mask = offs < total
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SK):
        acc += tl.load(
            P + s * stride_pk + offs, mask=mask, other=0.0
        )
    tl.store(C + offs, acc.to(tl.bfloat16), mask=mask)


def sk_linear(x, w, out=None, BLOCK_N=128, BLOCK_K=128, SK=3, warps=8):
    M, K = x.shape
    N = w.shape[0]
    BM = max(triton.next_power_of_2(M), 16)
    P = torch.empty(SK, M, N, dtype=torch.float32, device=x.device)
    if out is None:
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    _sk_gemm[(triton.cdiv(N, BLOCK_N), SK)](
        x, w, P, M, N, K,
        x.stride(0), w.stride(0), M * N,
        BLOCK_M=BM, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SK=SK,
        num_warps=warps,
    )
    total = M * N
    _sk_reduce[(triton.cdiv(total, 1024),)](
        P, out, M, N, M * N, SK=SK, BLOCK=1024, num_warps=4,
    )
    return out


def bench(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1000


if __name__ == "__main__":
    torch.manual_seed(0)
    H, I = 5120, 4352
    shapes = [
        ("qkvz_proj", H, H * 3 // 2 + I, 623.1),
        ("o_proj", H, H, 233.8),
        ("ba_proj", H, 2 * H // 5, 234.4),
        ("gate_up", H, 2 * I, 392.3),
        ("down_proj", I, H, 200.8),
    ]
    M = 42
    for name, K, N, blas_us in shapes:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        ref = torch.nn.functional.linear(x, w)
        best = None
        for BN in (64, 128, 256):
            for BK in (64, 128):
                for SK in (1, 2, 3, 4, 6):
                    for wp in (4, 8):
                        try:
                            o = sk_linear(x, w, BLOCK_N=BN, BLOCK_K=BK,
                                          SK=SK, warps=wp)
                            d = (o.float() - ref.float()).abs().max()
                        except Exception:
                            continue
                        t = bench(
                            lambda: sk_linear(
                                x, w, BLOCK_N=BN, BLOCK_K=BK, SK=SK,
                                warps=wp)
                        )
                        if best is None or t < best[0]:
                            best = (t, BN, BK, SK, wp, float(d))
        if best:
            t, BN, BK, SK, wp, d = best
            print(f"{name:10s} M={M} K={K} N={N}: triton={t:7.1f}us "
                  f"(BN{BN}/BK{BK}/SK{SK}/w{wp}, maxdiff {d:.1e}) "
                  f"vs blas={blas_us:7.1f}us -> x{blas_us / t:.2f}")
