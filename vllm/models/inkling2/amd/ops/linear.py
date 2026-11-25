# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp16 linears as the reference `linear`: fp32 accumulation in ordered
1024-wide K groups."""

import torch

from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _paired_norm(X, Message, Out, sx, sm, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    side = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    if side == 0:
        value = tl.load(X + row * sx + col, col < H, other=0).to(tl.float32)
    else:
        value = tl.load(Message + row * sm + col, col < H, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.div_rn(tl.sum(value * value, 0), H) + 1e-6)
    tl.store(Out + (row * 2 + side) * H + col, value * inv, col < H)


@triton.jit
def _message_mix(
    Low,
    Forget,
    Gain,
    Dest,
    Message,
    Out,
    M,
    sd,
    sm,
    H: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, K)
    low = tl.load(Low + row[:, None] * K + k[None, :], row[:, None] < M, other=0)
    offset = col[:, None] * K + k[None, :]
    forget_weight = tl.load(Forget + offset, col[:, None] < H, other=0)
    gain_weight = tl.load(Gain + offset, col[:, None] < H, other=0)
    forget = tldevice.tanh(tl.dot(low, tl.trans(forget_weight)))
    gain = tldevice.tanh(tl.dot(low, tl.trans(gain_weight)))
    valid = (row < M)[:, None] & (col < H)[None, :]
    dest = tl.load(Dest + row[:, None] * sd + col[None, :], valid, other=0).to(
        tl.float32
    )
    msg = tl.load(Message + row[:, None] * sm + col[None, :], valid, other=0).to(
        tl.float32
    )
    tl.store(
        Out + row[:, None] * H + col[None, :], dest * (1 - forget) + msg * gain, valid
    )


@triton.jit
def _controller_output(
    Dest,
    Message,
    Forget,
    Gain,
    Out,
    sd,
    sm,
    sf,
    sg,
    H: tl.constexpr,
    TILE: tl.constexpr,
    TANH: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    valid = col < H
    dest = tl.load(Dest + row * sd + col, valid, 0).to(tl.float32)
    message = tl.load(Message + row * sm + col, valid, 0).to(tl.float32)
    forget = tl.load(Forget + row * sf + col, valid, 0)
    gain = tl.load(Gain + row * sg + col, valid, 0)
    if TANH:
        forget = tldevice.tanh(forget)
        gain = tldevice.tanh(gain)
    tl.store(Out + row * H + col, dest * (1.0 - forget) + message * gain, valid)


def controller_output(dest, message, forget, gain, *, fuse_tanh, tile=256):
    """Apply the two FP32 gates while preserving separate product rounding."""
    if not fuse_tanh:
        forget, gain = forget.tanh(), gain.tanh()
    tokens, width = dest.shape
    out = gain.new_empty((tokens, width))
    _controller_output[(tokens, triton.cdiv(width, tile))](
        dest,
        message,
        forget,
        gain,
        out,
        dest.stride(0),
        message.stride(0),
        forget.stride(0),
        gain.stride(0),
        width,
        tile,
        fuse_tanh,
        enable_fp_fusion=False,
    )
    return out


def mix_message(dest, message, down_weight, forget_weight, gain_weight):
    """Normalize a pair of residuals and apply the loop's learned message gates."""
    m, h = dest.shape
    rank = down_weight.shape[0]
    pair = down_weight.new_empty((m, 2 * h))
    _paired_norm[(m, 2)](
        dest,
        message,
        pair,
        dest.stride(0),
        message.stride(0),
        h,
        triton.next_power_of_2(h),
        num_warps=4,
        enable_fp_fusion=False,
    )
    low = torch.nn.functional.linear(pair, down_weight)
    out = dest.new_empty((m, h), dtype=torch.float32)
    _message_mix[(triton.cdiv(m, 16), triton.cdiv(h, 64))](
        low,
        forget_weight,
        gain_weight,
        dest,
        message,
        out,
        m,
        dest.stride(0),
        message.stride(0),
        h,
        rank,
        16,
        64,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _k_group_tile(
    X,
    W,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLIT: tl.constexpr,
    pid_n,
    pid_m,
    pid_k,
):
    # With SPLIT, each program computes one K group's partial; otherwise it
    # sums the groups in order.
    cols = pid_n * BN + tl.arange(0, BN)
    rows = pid_m * BM + tl.arange(0, BM)
    ks = tl.arange(0, 128)
    first = pid_k if SPLIT else 0
    NUM_GROUPS: tl.constexpr = 1 if SPLIT else (K + 1023) // 1024
    total = tl.zeros((BM, BN), tl.float32)
    for g in tl.static_range(NUM_GROUPS):
        group = first + g
        partial = tl.zeros((BM, BN), tl.float32)
        for tile in range(8):
            k = group * 1024 + tile * 128 + ks
            x = tl.load(
                X + rows[:, None] * K + k[None, :],
                (rows[:, None] < M) & (k[None, :] < K),
                other=0,
            )
            w = tl.load(
                W + cols[None, :] * K + k[:, None],
                (cols[None, :] < N) & (k[:, None] < K),
                other=0,
            )
            partial = tl.dot(x, w, partial)
        total = partial if g == 0 else total + partial
    tl.store(
        Y + (first * M + rows[:, None]) * N + cols[None, :],
        total.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _k_group_kernel(
    X,
    W,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLIT: tl.constexpr,
):
    _k_group_tile(
        X,
        W,
        Y,
        M,
        N,
        K,
        BM,
        BN,
        SPLIT,
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )


@triton.jit
def _k_group_sum_kernel(Partials, Y, MN, GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    total = tl.load(Partials + offs, mask)
    for group in range(1, GROUPS):
        total = total + tl.load(Partials + group * MN + offs, mask)
    tl.store(Y + offs, total, mask)


def k_group_partials(
    x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """fp32 partials [G, T, N] of `x @ weight.T` whose in-order sum over G is
    the reference `linear`. G is 1 (already summed, in `out_dtype`) when
    there are enough rows to fill the GPU without splitting K."""
    assert x.is_contiguous() and weight.is_contiguous()
    tokens, hidden = x.shape
    n = weight.shape[0]
    block_m = 16 if tokens <= 128 else 64
    block_n = 32 if tokens <= 16 else 64 if tokens <= 256 else 128
    groups = triton.cdiv(hidden, 1024)
    split = groups > 1 and tokens < 4096
    out = torch.empty(
        (groups if split else 1, tokens, n),
        dtype=torch.float32 if split else out_dtype,
        device=x.device,
    )
    grid = (
        triton.cdiv(n, block_n),
        triton.cdiv(tokens, block_m),
        groups if split else 1,
    )
    _k_group_kernel[grid](
        x,
        weight,
        out,
        tokens,
        n,
        hidden,
        block_m,
        block_n,
        split,
        num_warps=4 if block_n <= 64 else 8,
        # gfx950's 160 KB of LDS fits two stages of the largest tiles.
        num_stages=2,
        enable_fp_fusion=False,
    )
    return out


def linear_k_groups(
    x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """`x @ weight.T` for fp16 `x` [T, K] and `weight` [N, K].

    Each 1024-wide K group is accumulated in fp32 and the partials are summed in
    order, as in the reference `linear`.
    """
    partials = k_group_partials(x, weight, out_dtype)
    if partials.shape[0] == 1:
        return partials[0]
    out = torch.empty(partials.shape[1:], dtype=out_dtype, device=x.device)
    _k_group_sum_kernel[(triton.cdiv(out.numel(), 1024),)](
        partials, out, out.numel(), partials.shape[0], 1024
    )
    return out
