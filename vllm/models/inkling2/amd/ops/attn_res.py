# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block attention residual: softmax-weighted mix of residual stream blocks.

For each token, sources `x_n` (stored blocks, then the partial block plus an
optional addend, all in fp32) are scored as `(q . x_n) * rsqrt(mean(x_n^2) +
eps)` per pseudo-query `q`; the output is `sum_n softmax_n * x_n`, with
`x_n` RMS-normalized when `normalize_values` is set. The fp32 mix then
goes through the following sublayer's RMSNorm in the same kernel.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _residual_source(
    Blocks,
    Last,
    Addend,
    state,
    row,
    col,
    mask,
    stride_bs,
    stride_bt,
    stride_lt,
    stride_at,
    STACK: tl.constexpr,
    ADD: tl.constexpr,
):
    if state < STACK:
        value = tl.load(Blocks + state * stride_bs + row * stride_bt + col, mask, 0).to(
            tl.float32
        )
    else:
        value = tl.load(Last + row * stride_lt + col, mask, 0).to(tl.float32)
        if ADD:
            value += tl.load(Addend + row * stride_at + col, mask, 0).to(tl.float32)
    return value


@triton.jit
def _normalize_residual_rows(
    Blocks,
    Last,
    Addend,
    Out,
    stride_bs,
    stride_bt,
    stride_lt,
    stride_at,
    stride_os,
    stride_ot,
    STACK: tl.constexpr,
    ADD: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    state = tl.program_id(1)
    source = (Blocks, Last, Addend, state, row)
    strides = (stride_bs, stride_bt, stride_lt, stride_at)
    # Preserve the FP32 mean reduction order used by ATen on gfx908 for
    # contiguous 2304-element rows. Each lane reduces four columns.
    offset = tl.arange(0, 256)
    partial = tl.full((256,), 0.0, tl.float32)
    for chunk in tl.static_range(9):
        x = _residual_source(*source, chunk * 256 + offset, True, *strides, STACK, ADD)
        partial += x * x
    lanes = tl.arange(0, 64)
    total = tl.gather(partial, lanes * 4, 0)
    for i in tl.static_range(1, 4):
        total += tl.gather(partial, lanes * 4 + i, 0)
    for i in tl.static_range(6):
        total += tl.gather(total, lanes ^ (1 << i), 0)
    sum_squares = tl.sum(tl.where(lanes == 0, total, 0.0), 0)
    mean_eps = sum_squares * (1.0 / 2304) + 1e-6
    inv = tl.rsqrt(mean_eps.to(tl.float64)).to(tl.float32)

    cols = tl.arange(0, 4096)
    value = _residual_source(*source, cols, cols < 2304, *strides, STACK, ADD)
    if Out.dtype.element_ty == tl.float16:
        # Prevent a direct-to-half multiply with different tie rounding.
        norm = tl.inline_asm_elementwise(
            "v_mul_f32 $0, $1, $2",
            "=v,v,v",
            [value, inv],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        norm = value * inv
    tl.store(Out + state * stride_os + row * stride_ot + cols, norm, cols < 2304)


def normalize_residual_rows(
    blocks, last, addend=None, *, controller=False, dtype=None
):
    """Normalize residual sources along width; a controller takes FP16 rows.

    `dtype` rounds the normalized stack rows (FP16 halves the bytes the
    residual query reads); the normalization itself stays FP32.
    """
    tokens, width = last.shape
    if (
        width != 2304
        or tokens < 8
        or blocks.stride(-1) != 1
        or last.stride(-1) != 1
        or (addend is not None and addend.stride(-1) != 1)
    ):
        tail = last.float()
        if addend is not None:
            tail = tail + addend.float()
        values = torch.cat((blocks.float(), tail.unsqueeze(0)))
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
        if controller:
            return values.transpose(0, 1).contiguous().half()
        return values.to(dtype) if dtype is not None else values
    states = blocks.shape[0] + 1
    if controller:
        out = last.new_empty((tokens, states, width), dtype=torch.float16)
        state_stride, token_stride = out.stride(1), out.stride(0)
    else:
        out = last.new_empty((states, tokens, width), dtype=dtype or torch.float32)
        state_stride, token_stride = out.stride(0), out.stride(1)
    _normalize_residual_rows[(tokens, states)](
        blocks,
        last,
        addend,
        out,
        blocks.stride(0),
        blocks.stride(1),
        last.stride(0),
        addend.stride(0) if addend is not None else 0,
        state_stride,
        token_stride,
        states - 1,
        addend is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _weighted_values(
    Values,
    Weights,
    Out,
    stride_vs,
    stride_vt,
    stride_wq,
    stride_wt,
    stride_ot,
    stride_oq,
    H: tl.constexpr,
    S: tl.constexpr,
    B: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    q = tl.program_id(1)
    col = tl.program_id(2) * B + tl.arange(0, B)
    values = Values + token * stride_vt + col
    weights = Weights + q * stride_wq + token * stride_wt
    result = tl.full((B,), 0, tl.float32)
    for state in tl.static_range(S):
        value = tl.load(values + state * stride_vs, col < H, other=0)
        # rocBLAS uses eight split-K lanes for this skinny product. With
        # at most eight states each lane contributes one rounded product.
        result = result + value * tl.load(weights + state)
    tl.store(Out + token * stride_ot + q * stride_oq + col, result, col < H)


def weighted_values(weights, values):
    """Sum the few residual states in their original FP32 accumulation order."""
    states, tokens, hidden = values.shape
    queries = weights.shape[0]
    # The reference selects a different contraction for small batches. Use
    # this specialization only for the Inkling 2 Mini shapes checked on MI100.
    if (
        tokens < 16
        or hidden != 2304
        or not 2 <= states <= 5
        or queries not in (1, 3, 4)
        or values.dtype != torch.float32
        or weights.dtype != torch.float32
        or values.stride(-1) != 1
        or weights.stride(-1) != 1
    ):
        return torch.einsum("qts,std->qtd", weights, values)
    out = values.new_empty((tokens, queries, hidden)).transpose(0, 1)
    block = 256
    _weighted_values[(tokens, queries, triton.cdiv(hidden, block))](
        values,
        weights,
        out,
        values.stride(0),
        values.stride(1),
        weights.stride(0),
        weights.stride(1),
        out.stride(1),
        out.stride(0),
        hidden,
        states,
        block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _load_source(
    Stack, stride_s, Last, Addend, row_off, offs, mask, n, N_STACK, HAS_ADDEND
):
    if n < N_STACK:
        x = tl.load(Stack + n * stride_s + row_off + offs, mask, other=0)
        x = x.to(tl.float32)
    else:
        x = tl.load(Last + row_off + offs, mask, other=0).to(tl.float32)
        if HAS_ADDEND:
            x = x + tl.load(Addend + row_off + offs, mask, other=0).to(tl.float32)
    return x


@triton.jit
def _mix(
    Stack,
    stride_s,
    Last,
    Addend,
    row_off,
    offs,
    mask,
    probs,
    n_idx,
    N_STACK,
    HAS_ADDEND,
    BLOCK_D,
):
    acc = tl.zeros((BLOCK_D,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        x = _load_source(
            Stack, stride_s, Last, Addend, row_off, offs, mask, n, N_STACK, HAS_ADDEND
        )
        acc = tl.fma(tl.sum(tl.where(n_idx == n, probs, 0.0)), x, acc)
    return acc


@triton.jit
def _attn_res_kernel(
    Stack,
    Last,
    Addend,
    Queries,
    NormW,
    Out,
    stride_s,
    T,
    eps,
    norm_eps,
    D: tl.constexpr,
    N_STACK: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # One program per (token, query).
    t = tl.program_id(1).to(tl.int64)
    q_off = tl.program_id(0) * D
    row_off = t * D
    n_idx = tl.arange(0, BLOCK_N)
    d_idx = tl.arange(0, BLOCK_D)

    # Pass 1: per-source inverse RMS and query dot products.
    logits = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    inv_rms = tl.zeros((BLOCK_N,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        sum_sq = tl.zeros((BLOCK_D,), tl.float32)
        dots = tl.zeros((BLOCK_D,), tl.float32)
        for d0 in tl.range(0, D, BLOCK_D):
            offs = d0 + d_idx
            mask = offs < D
            x = _load_source(
                Stack,
                stride_s,
                Last,
                Addend,
                row_off,
                offs,
                mask,
                n,
                N_STACK,
                HAS_ADDEND,
            )
            q = tl.load(Queries + q_off + offs, mask, other=0)
            sum_sq = tl.fma(x, x, sum_sq)
            dots = tl.fma(q, x, dots)
        rstd = tl.rsqrt(tl.fma(tl.sum(sum_sq), 1.0 / D, eps))
        inv_rms = tl.where(n_idx == n, rstd, inv_rms)
        logits = tl.where(n_idx == n, tl.sum(dots) * rstd, logits)

    probs = tldevice.exp(logits - tl.max(logits))
    inv_sum = tl.div_rn(1.0, tl.sum(probs))
    if NORMALIZE:
        probs = probs * inv_rms

    # Pass 2: RMS of the probability-weighted sum; pass 3: normalized sum.
    sum_sq = 0.0
    for d0 in tl.range(0, D, BLOCK_D):
        offs = d0 + d_idx
        acc = (
            _mix(
                Stack,
                stride_s,
                Last,
                Addend,
                row_off,
                offs,
                offs < D,
                probs,
                n_idx,
                N_STACK,
                HAS_ADDEND,
                BLOCK_D,
            )
            * inv_sum
        )
        sum_sq += tl.sum(acc * acc)
    out_rstd = tl.rsqrt(tl.div_rn(sum_sq, D) + norm_eps)
    out = Out + tl.program_id(0).to(tl.int64) * T * D + row_off
    for d0 in tl.range(0, D, BLOCK_D):
        offs = d0 + d_idx
        mask = offs < D
        acc = (
            _mix(
                Stack,
                stride_s,
                Last,
                Addend,
                row_off,
                offs,
                mask,
                probs,
                n_idx,
                N_STACK,
                HAS_ADDEND,
                BLOCK_D,
            )
            * inv_sum
        )
        w = tl.load(NormW + q_off + offs, mask, other=0).to(tl.float32)
        acc = acc * out_rstd * w
        tl.store(out + offs, acc.to(Out.dtype.element_ty), mask)


@triton.jit
def _attn_res_row_kernel(
    Stack,
    Last,
    Addend,
    Queries,
    NormW,
    Out,
    stride_s,
    T,
    eps,
    norm_eps,
    D: tl.constexpr,
    N_STACK: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_B: tl.constexpr,
    APPLY_NORM: tl.constexpr,
):
    # Small batches: one program per (token, query) holds the row in registers
    # as a power-of-2 head [0, BLOCK_A) and a masked tail [BLOCK_A, D), so the
    # sources are read twice instead of three times.
    t = tl.program_id(1).to(tl.int64)
    q_off = tl.program_id(0) * D
    row_off = t * D
    da = tl.arange(0, BLOCK_A)
    db = BLOCK_A + tl.arange(0, BLOCK_B)
    mask_a = da < D
    mask_b = db < D
    n_idx = tl.arange(0, BLOCK_N)
    qa = tl.load(Queries + q_off + da, mask_a, other=0)
    qb = tl.load(Queries + q_off + db, mask_b, other=0)
    src = (Stack, stride_s, Last, Addend, row_off)

    logits = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    inv_rms = tl.zeros((BLOCK_N,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        xa = _load_source(*src, da, mask_a, n, N_STACK, HAS_ADDEND)
        xb = _load_source(*src, db, mask_b, n, N_STACK, HAS_ADDEND)
        rstd = tl.rsqrt(tl.fma(tl.sum(xa * xa) + tl.sum(xb * xb), 1.0 / D, eps))
        inv_rms = tl.where(n_idx == n, rstd, inv_rms)
        dot = tl.sum(qa * xa) + tl.sum(qb * xb)
        logits = tl.where(n_idx == n, dot * rstd, logits)

    probs = tldevice.exp(logits - tl.max(logits))
    inv_sum = tl.div_rn(1.0, tl.sum(probs))
    if NORMALIZE:
        probs = probs * inv_rms

    acc_a = tl.zeros((BLOCK_A,), tl.float32)
    acc_b = tl.zeros((BLOCK_B,), tl.float32)
    for n in tl.static_range(N_STACK + 1):
        p = tl.sum(tl.where(n_idx == n, probs, 0.0))
        xa = _load_source(*src, da, mask_a, n, N_STACK, HAS_ADDEND)
        xb = _load_source(*src, db, mask_b, n, N_STACK, HAS_ADDEND)
        acc_a = tl.fma(p, xa, acc_a)
        acc_b = tl.fma(p, xb, acc_b)
    acc_a = acc_a * inv_sum
    acc_b = acc_b * inv_sum
    if APPLY_NORM:
        sum_sq = tl.sum(acc_a * acc_a) + tl.sum(acc_b * acc_b)
        out_rstd = tl.rsqrt(tl.div_rn(sum_sq, D) + norm_eps)
        wa = tl.load(NormW + q_off + da, mask_a, other=0).to(tl.float32)
        wb = tl.load(NormW + q_off + db, mask_b, other=0).to(tl.float32)
        acc_a = acc_a * out_rstd * wa
        acc_b = acc_b * out_rstd * wb
    out = Out + tl.program_id(0).to(tl.int64) * T * D + row_off
    tl.store(out + da, acc_a.to(Out.dtype.element_ty), mask_a)
    tl.store(out + db, acc_b.to(Out.dtype.element_ty), mask_b)


def attn_res(
    stack: torch.Tensor,
    last: torch.Tensor,
    addend: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    normalize_values: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mix `[*stack, last (+ addend)]` with each pseudo-query.

    `queries` are fp32 `(Q, D)`, already scaled by `D**-0.5`; each fp32
    mix is RMS-normalized with its row of `norm_weight` `(Q, D)`. Returns
    `(Q, T, D)` in `last.dtype`.
    """
    num_tokens, d = last.shape
    num_stack = stack.shape[0]
    num_q = queries.shape[0]
    assert last.is_contiguous() and queries.is_contiguous()
    assert queries.dtype == torch.float32
    assert stack.stride(1) == d and stack.stride(2) == 1
    if addend is not None:
        assert addend.is_contiguous() and addend.shape == last.shape
    if norm_weight is not None:
        assert norm_weight.shape == queries.shape and norm_weight.is_contiguous()
    dtype = last.dtype if norm_weight is not None else torch.float32
    out = torch.empty((num_q, num_tokens, d), dtype=dtype, device=last.device)
    args = (
        stack,
        last,
        addend,
        queries,
        norm_weight,
        out,
        stack.stride(0),
        num_tokens,
        eps,
        norm_eps,
    )
    meta = dict(
        D=d,
        N_STACK=num_stack,
        HAS_ADDEND=addend is not None,
        NORMALIZE=normalize_values,
        BLOCK_N=triton.next_power_of_2(num_stack + 1),
        enable_fp_fusion=False,
    )
    # The queries of a token run back to back and share its sources in L2.
    grid = (num_q, num_tokens)
    if norm_weight is None or num_tokens * num_q <= 256:
        block_a = triton.next_power_of_2(d + 1) // 2
        block_b = max(triton.next_power_of_2(d - block_a), 16)
        _attn_res_row_kernel[grid](
            *args,
            **meta,
            BLOCK_A=block_a,
            BLOCK_B=block_b,
            APPLY_NORM=norm_weight is not None,
            num_warps=8,
        )
    else:
        _attn_res_kernel[grid](*args, **meta, BLOCK_D=1024, num_warps=4)
    return out
