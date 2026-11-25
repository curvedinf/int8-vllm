# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Narrow decode GEMMs that map to one-compute-unit rocBLAS kernels.

The expert router's FP32 logits (`tokens x experts`) and the unquantized
relative-position projection (96 output columns per rank at TP4) each launch
a single-CTA rocBLAS kernel at decode batch sizes, ~130 us for a few MFLOP.
The kernels here split K and tile N to fill the compute units. Opt-in: their
accumulation order differs from rocBLAS in the last bits.
"""

import torch

from vllm.triton_utils import tl, triton

ROUTER_SPLITS = 4
ROUTER_BLOCK_K = 64
# rocBLAS spreads larger batches over more compute units than these FP32 FMA
# tiles do; only decode-sized router calls map to a single unit.
DECODE_TOKENS = 128


@triton.jit
def _split_k_fp32(
    A,
    B,
    Part,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_pm,
    stride_pn,
    stride_ps,
    S: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    rows = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    cols = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    # Every split covers the same number of whole K blocks: the partial sums
    # are added in fixed split order, so the result is deterministic.
    for k0 in range(pid_s * (K // S), (pid_s + 1) * (K // S), BK):
        ks = k0 + tl.arange(0, BK)
        a = tl.load(
            A + rows[:, None] * stride_am + ks[None, :] * stride_ak,
            mask=rows[:, None] < M,
            other=0.0,
        )
        b = tl.load(
            B + ks[:, None] * stride_bk + cols[None, :] * stride_bn,
            mask=cols[None, :] < N,
            other=0.0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(
        Part
        + pid_s * stride_ps
        + rows[:, None] * stride_pm
        + cols[None, :] * stride_pn,
        acc,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _sum_splits(
    Part,
    Out,
    numel,
    stride_ps,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(S):
        acc += tl.load(Part + s * stride_ps + offs, mask=mask, other=0.0)
    tl.store(Out + offs, acc, mask=mask)


@triton.jit
def _fp16_gemm(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        a = tl.load(
            A + rows[:, None].to(tl.int64) * stride_am + ks[None, :] * stride_ak,
            mask=rows[:, None] < M,
            other=0.0,
        )
        b = tl.load(
            B + ks[:, None].to(tl.int64) * stride_bk + cols[None, :] * stride_bn,
            mask=cols[None, :] < N,
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
    tl.store(
        C + rows[:, None].to(tl.int64) * stride_cm + cols[None, :] * stride_cn,
        acc.to(C.dtype.element_ty),
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


_partials: dict[tuple[int, int, int, torch.device], torch.Tensor] = {}


def router_logits(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return the router's FP32 logits `x.float() @ weight.T`.

    `weight` is `(experts, hidden)` FP32. Falls back to torch when the hidden
    size does not divide into equal whole K blocks.
    """
    x = x.float()
    tokens, hidden = x.shape
    experts = weight.shape[0]
    splits = ROUTER_SPLITS
    while splits > 1 and hidden % (splits * ROUTER_BLOCK_K):
        splits //= 2
    if hidden % (splits * ROUTER_BLOCK_K):
        return torch.nn.functional.linear(x, weight)
    key = (tokens, experts, splits, x.device)
    parts = _partials.get(key)
    if parts is None:
        parts = torch.empty(
            (splits, tokens, experts), dtype=torch.float32, device=x.device
        )
        _partials[key] = parts
    out = torch.empty((tokens, experts), dtype=torch.float32, device=x.device)
    weight_t = weight.t()
    _split_k_fp32[(triton.cdiv(tokens, 16), triton.cdiv(experts, 64), splits)](
        x,
        weight_t,
        parts,
        tokens,
        experts,
        hidden,
        x.stride(0),
        x.stride(1),
        weight_t.stride(0),
        weight_t.stride(1),
        parts.stride(1),
        parts.stride(2),
        parts.stride(0),
        S=splits,
        BM=16,
        BN=64,
        BK=ROUTER_BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    _sum_splits[(triton.cdiv(tokens * experts, 1024),)](
        parts,
        out,
        tokens * experts,
        parts.stride(0),
        S=splits,
        BLOCK=1024,
        num_warps=4,
    )
    return out


def fp16_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return the FP16 `x @ weight.T` for a narrow `weight` (`(N, K)` FP16).

    K must be a multiple of 64.
    """
    tokens, hidden = x.shape
    n = weight.shape[0]
    assert hidden % 64 == 0, hidden
    out = torch.empty((tokens, n), dtype=torch.float16, device=x.device)
    weight_t = weight.t()
    _fp16_gemm[(triton.cdiv(tokens, 16), triton.cdiv(n, 16))](
        x,
        weight_t,
        out,
        tokens,
        n,
        hidden,
        x.stride(0),
        x.stride(1),
        weight_t.stride(0),
        weight_t.stride(1),
        out.stride(0),
        out.stride(1),
        BM=16,
        BN=16,
        BK=256 if hidden % 256 == 0 else 64,
        num_warps=1,
        num_stages=2,
    )
    return out
