# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 paged relative attention, gate/sink, and log-scaling.

A relative attention program owns `BLOCK_Q` query tokens of one request and
the query heads of one KV head (rows token-major), and walks the request's
keys from its first query's window (local layers) or the start (full layers)
to its last query. Decode and verify batches split those keys over several
programs (split-KV) and merge the partial outputs in a second launch.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.platform_utils import num_compute_units

_LOG2E = 1.4426950408889634
_BLOCK_N = 64
# Query rows (tokens x GQA group, padded to a power of two) of a split-KV
# program, which holds all queries of a request.
_DECODE_BLOCK_M = 64
# Split-KV programs per compute unit that a decode batch launches, and the
# most splits of a request and KV head (fitted on MI355X at TP4, for 1 to
# 256 requests of 1 and 8 queries over 1K to 100K keys).
_SPLIT_PROGRAMS_PER_CU = 2
_MAX_SPLITS = 128


@triton.jit(do_not_specialize=["num_splits", "num_tokens"])
def _rel_attention_kernel(
    Q,
    K,
    V,
    Bias,
    Out,
    Lse,
    BlockTable,
    SeqLens,
    CuSeqlensQ,
    scale_log2,
    num_splits,
    num_tokens,
    stride_qt,
    stride_qh,
    stride_kb,
    stride_kp,
    stride_kh,
    stride_vb,
    stride_vp,
    stride_vh,
    stride_bt,
    stride_bh,
    stride_ot,
    stride_oh,
    stride_lh,
    stride_table,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    E_PAD: tl.constexpr,
    WINDOW: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Gate=None,
    Sink=None,
    stride_gate=0,
    stride_sink=0,
    DRAFT: tl.constexpr = False,
):
    # `WINDOW` is the largest visible query-key distance (-1 for full attention).
    # `SPLIT`: program `s` of `num_splits` holds all queries of the request,
    # takes key blocks `s, s + num_splits, ...` and writes row
    # `s * num_tokens + token` of the partial output and its log2 LSE
    # (-inf without keys). Otherwise programs tile the queries, and write the
    # output and its natural-log LSE.
    BLOCK_M: tl.constexpr = BLOCK_Q * BLOCK_H
    kv_head = tl.program_id(1)
    req = tl.program_id(2)
    q_start = tl.load(CuSeqlensQ + req)
    q_len = tl.load(CuSeqlensQ + req + 1) - q_start
    if SPLIT:
        split = tl.program_id(0)
        step = num_splits
        q0 = 0
    else:
        split = 0
        step = 1
        q0 = tl.program_id(0) * BLOCK_Q
    if q0 >= q_len:
        return
    seq_len = tl.load(SeqLens + req)
    past = seq_len - q_len

    rows = tl.arange(0, BLOCK_M)
    q_idx = q0 + rows // BLOCK_H
    head = kv_head * GROUP + rows % BLOCK_H
    row_valid = (q_idx < q_len) & (rows % BLOCK_H < GROUP)
    token = (q_start + q_idx).to(tl.int64)
    d = tl.arange(0, D)
    q = tl.load(
        Q + token[:, None] * stride_qt + head[:, None] * stride_qh + d[None, :],
        row_valid[:, None],
        other=0.0,
    )
    if DRAFT:
        # All queries in the draft block see all block keys and the same history.
        pos = tl.full([BLOCK_M], 0, tl.int32) + seq_len - 1
        first_pos = seq_len - 1
        hi = seq_len
    else:
        pos = past + q_idx
        first_pos = past + q0
        hi = tl.minimum(past + q0 + BLOCK_Q, seq_len)
    # Sheared column of key j is j - key0: key0 starts the E_PAD keys
    # ending at the key blocks of the token's 128-token query tile.
    tile_end = tl.minimum((q_idx // 128 + 1) * 128, q_len)
    key0 = tl.cdiv(past + tile_end, 128) * 128 - E_PAD
    bias_rows = Bias + token * stride_bt + head * stride_bh - key0
    lo = 0
    if WINDOW >= 0:
        lo = tl.maximum(first_pos - WINDOW, 0) // BLOCK_N * BLOCK_N
    table = BlockTable + req * stride_table
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    n_first = lo // BLOCK_N + split
    num_iters = tl.maximum(tl.cdiv(tl.cdiv(hi, BLOCK_N) - n_first, step), 0)
    for i in range(num_iters):
        keys = (n_first + i * step) * BLOCK_N + tl.arange(0, BLOCK_N)
        key_valid = keys < hi
        page = tl.load(table + keys // PAGE, key_valid, other=0).to(tl.int64)
        slot = keys % PAGE
        k = tl.load(
            K
            + page[:, None] * stride_kb
            + slot[:, None] * stride_kp
            + kv_head * stride_kh
            + d[None, :],
            key_valid[:, None],
            other=0.0,
        )
        if K.dtype.element_ty == tl.int8:
            scale_addr = (
                K + page * stride_kb + slot * stride_kp + kv_head * stride_kh + D
            ).to(tl.pointer_type(tl.float16))
            scale = tl.load(scale_addr, key_valid, other=0.0)
            k = (k.to(tl.float32) * scale[:, None]).to(q.dtype)
        s = tl.dot(q, tl.trans(k.to(q.dtype))) * scale_log2
        dist = pos[:, None] - keys[None, :]
        valid = row_valid[:, None] & key_valid[None, :] & (dist >= 0)
        if WINDOW >= 0:
            valid &= dist <= WINDOW
        # Keys before key0 are past the extent: no bias (full layers) or
        # out of the window.
        if not DRAFT:
            in_table = valid & (keys[None, :] >= key0[:, None])
            bias = tl.load(bias_rows[:, None] + keys[None, :], in_table, other=0.0)
            s = s + bias.to(tl.float32) * 1.4426950408889634
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        # Rows without a live key so far keep m = -inf: no -inf - -inf.
        m_ref = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_ref)
        p = tl.math.exp2(s - m_ref[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            V
            + page[:, None] * stride_vb
            + slot[:, None] * stride_vp
            + kv_head * stride_vh
            + d[None, :],
            key_valid[:, None],
            other=0.0,
        )
        if V.dtype.element_ty == tl.int8:
            scale_addr = (
                V + page * stride_vb + slot * stride_vp + kv_head * stride_vh + D
            ).to(tl.pointer_type(tl.float16))
            scale = tl.load(scale_addr, key_valid, other=0.0)
            v = (v.to(tl.float32) * scale[:, None]).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    has_keys = l_i > 0.0
    out = acc / tl.where(has_keys, l_i, 1.0)[:, None]
    out_token = token + split * num_tokens
    lse = tl.where(has_keys, m_i + tl.math.log2(l_i), float("-inf"))
    if not SPLIT:
        lse *= 0.6931471805599453
    if Gate is not None:
        gate = tl.load(
            Gate + token[:, None] * stride_gate + head[:, None] * D + d[None, :],
            row_valid[:, None],
            other=0,
        ).to(tl.float32)
        sink = tl.load(Sink + token * stride_sink + head, row_valid, other=0).to(
            tl.float32
        )
        out = (out * _sigmoid(gate)) * _sigmoid(lse - sink)[:, None]
    else:
        tl.store(Lse + head * stride_lh + out_token, lse, row_valid)
    tl.store(
        Out + out_token[:, None] * stride_ot + head[:, None] * stride_oh + d[None, :],
        out.to(Out.dtype.element_ty),
        row_valid[:, None],
    )


def rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    window: int,
    sheared_bias: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    """Write the fp32 relative attention `out` `(T, H, D)` and its
    natural-log LSE `lse` `(H, T)` of the tokens of each request.

    Each query sees the request's keys up to its own, at most `window`
    positions back (`window` -1: all). `sheared_bias` holds the relative
    logits in the layout of `attention_prologue`; `max_seqlen_q` bounds the
    query lengths. Rows of tokens outside the requests are not written.
    """
    assert lse.stride(1) == 1
    _check_sheared_bias(sheared_bias, q)
    _launch(
        q,
        key_cache,
        value_cache,
        sheared_bias,
        out,
        lse,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        max_seqlen_q,
        softmax_scale,
        window,
    )


def gated_rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    sheared_bias: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    result: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    num_actual_tokens: int,
    softmax_scale: float,
    window: int,
    fused_epilogue: bool = False,
    key_tile: int = _BLOCK_N,
    max_rows: int = 128,
) -> None:
    """Write the gated `(T, H * D)` attention output (`gate_sink` of
    `rel_attention`) to `result`. Tokens from `num_actual_tokens` on are
    zeroed."""
    num_tokens, num_heads = q.shape[:2]
    nt = num_actual_tokens
    if fused_epilogue:
        assert result.is_contiguous() and result.shape == gate.shape
        assert gate.stride(1) == sinks.stride(1) == 1
        _check_sheared_bias(sheared_bias, q)
        _launch(
            q[:nt],
            key_cache,
            value_cache,
            sheared_bias,
            result[:nt].view(nt, num_heads, q.shape[-1]),
            None,
            block_table,
            cache_seqlens,
            cu_seqlens_q,
            max_seqlen_q,
            softmax_scale,
            window,
            gate=gate,
            sinks=sinks,
            key_tile=key_tile,
            max_rows=max_rows,
        )
        result[nt:].zero_()
        return
    out = q.new_empty(q.shape, dtype=torch.float32)
    lse = q.new_empty((num_heads, num_tokens), dtype=torch.float32)
    _check_sheared_bias(sheared_bias, q)
    _launch(
        q[:nt],
        key_cache,
        value_cache,
        sheared_bias[: nt + 128],
        out[:nt],
        lse[:, :nt],
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        max_seqlen_q,
        softmax_scale,
        window,
        key_tile=key_tile,
        max_rows=max_rows,
    )
    out[nt:].zero_()
    lse[:, nt:].zero_()
    gate_sink(out, gate, lse.t(), sinks, result)


def plan_decode_attention(
    is_local: bool,
    num_reqs: int,
    max_query_len: int,
    num_heads: int,
    num_kv_heads: int,
    rel_extent: int,
) -> int:
    """Return the decode split count, or zero to use prefill attention.

    A split-KV program holds all queries of one request for one KV head.
    Local layers (`rel_extent`: the window) split the key blocks of the
    window, full layers the whole context, so that the batch launches two
    programs per compute unit. The count depends only on the batch, not the
    KV lengths, so it holds for every replay of a CUDA graph.
    """
    block_h = triton.next_power_of_2(num_heads // num_kv_heads)
    if max_query_len * block_h > _DECODE_BLOCK_M:
        return 0
    work = max(1, num_reqs * num_kv_heads)
    num_cus = num_compute_units()
    max_splits = _MAX_SPLITS
    if is_local:
        # One key block of the window per split, at most.
        max_splits = triton.cdiv(rel_extent - 1, _BLOCK_N) + 1
    return max(1, min(max_splits, _SPLIT_PROGRAMS_PER_CU * num_cus // work))


def rel_decode_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    rel_logits: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    num_actual_tokens: int,
    softmax_scale: float,
    is_local: bool,
    num_splits: int,
    key_tile: int = _BLOCK_N,
) -> None:
    """Write the gated `(T, H * D)` attention output for small batches.

    Split-KV relative attention (see `plan_decode_attention`) reading the
    sheared relative logits; a second launch merges the fp32 partials and
    applies the output gate and attention sink. Tokens after the last
    request's are zeroed: the merge reads their count from `cu_seqlens_q`,
    so this also holds when a CUDA graph is replayed on a smaller batch.
    """
    num_tokens = q.shape[0]
    assert num_actual_tokens <= num_tokens
    assert gate.stride(1) == sinks.stride(1) == 1
    assert out.shape == gate.shape and out.is_contiguous()
    _check_sheared_bias(rel_logits, q)
    extent = rel_logits.shape[2] - 256
    o_part, lse_part = _launch(
        q,
        key_cache,
        value_cache,
        rel_logits,
        None,
        None,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        max_query_len,
        softmax_scale,
        extent - 1 if is_local else -1,
        num_splits=num_splits,
        key_tile=key_tile,
    )
    assert lse_part is not None
    _combine(o_part, lse_part, gate, sinks, out, cu_seqlens_q, num_splits)


def block_decode_attention(
    q,
    key_cache,
    value_cache,
    out,
    *,
    block_table,
    cache_seqlens,
    cu_seqlens_q,
    block,
    window,
    num_splits,
):
    """DSpark block attention with a common bidirectional window."""
    partial, lse = _launch(
        q,
        key_cache,
        value_cache,
        q,
        None,
        None,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        block,
        q.shape[-1] ** -0.5,
        window - 1 if window else -1,
        num_splits=num_splits,
        draft=True,
    )
    assert lse is not None
    _combine(partial, lse, None, None, out, cu_seqlens_q, num_splits)


def _check_sheared_bias(sheared_bias: torch.Tensor, q: torch.Tensor) -> None:
    assert sheared_bias.shape[1] == q.shape[1] and sheared_bias.stride(2) == 1
    assert sheared_bias.shape[2] % 128 == 0


def _launch(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor | None,
    lse: torch.Tensor | None,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    window: int,
    num_splits: int = 0,
    *,
    gate: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    key_tile: int = _BLOCK_N,
    max_rows: int = 128,
    draft: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch `_rel_attention_kernel` and return `(out, lse)`.

    Without `out` and `lse` this is split-KV over `num_splits` programs:
    the fp32 partial outputs `(num_splits * T, H, D)` and their log2 LSEs
    `(H, num_splits * T)` are allocated here.
    """
    num_tokens, num_heads, head_dim = q.shape
    page, num_kv_heads = key_cache.shape[1:3]
    group = num_heads // num_kv_heads
    assert q.stride(2) == key_cache.stride(3) == value_cache.stride(3) == 1
    block_h = triton.next_power_of_2(group)
    # Rows (tokens x heads) per program: at least one 16-row MFMA tile.
    block_m = min(max_rows, max(16, triton.next_power_of_2(max_seqlen_q * block_h)))
    block_q = max(1, block_m // block_h)
    num_reqs = cache_seqlens.shape[0]
    split = out is None
    if split:
        assert max_seqlen_q <= block_q and num_splits > 0
        out = q.new_empty((num_splits * num_tokens, *q.shape[1:]), dtype=torch.float32)
        lse = q.new_empty((num_heads, num_splits * num_tokens), dtype=torch.float32)
        grid = (num_splits, num_kv_heads, num_reqs)
    else:
        grid = (triton.cdiv(max_seqlen_q, block_q), num_kv_heads, num_reqs)
    assert out is not None and (lse is not None or gate is not None)
    assert out.stride(2) == 1
    if num_tokens == 0:
        return out, lse
    _rel_attention_kernel[grid](
        q,
        key_cache,
        value_cache,
        bias,
        out,
        lse,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        softmax_scale * _LOG2E,
        num_splits,
        num_tokens,
        q.stride(0),
        q.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        bias.stride(0),
        bias.stride(1),
        out.stride(0),
        out.stride(1),
        lse.stride(0) if lse is not None else 0,
        block_table.stride(0),
        GROUP=group,
        D=head_dim,
        PAGE=page,
        E_PAD=bias.shape[2],
        WINDOW=window,
        SPLIT=split,
        DRAFT=draft,
        BLOCK_Q=block_q,
        BLOCK_H=block_h,
        BLOCK_N=key_tile,
        Gate=gate,
        Sink=sinks,
        stride_gate=gate.stride(0) if gate is not None else 0,
        stride_sink=sinks.stride(0) if sinks is not None else 0,
        num_warps=4 if block_m <= 64 else 8,
        num_stages=1,
    )
    return out, lse


@triton.jit(do_not_specialize=["num_reqs", "num_splits", "num_tokens"])
def _split_combine_kernel(
    OPart,
    LsePart,
    Gate,
    Sink,
    Y,
    CuSeqlensQ,
    num_reqs,
    num_splits,
    num_tokens,
    stride_lh,
    stride_gate,
    stride_sink,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Merge the split partials of one token and head, then apply the output
    gate and attention sink."""
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    offs_d = tl.arange(0, D)
    y_ptrs = Y + row * H * D + head * D + offs_d
    # Tokens of no request (padding) have no partials.
    if row >= tl.load(CuSeqlensQ + num_reqs):
        tl.store(y_ptrs, tl.zeros([D], Y.dtype.element_ty))
        return
    m_i = tl.full([1], float("-inf"), tl.float32)
    l_i = tl.zeros([1], tl.float32)
    acc = tl.zeros([D], tl.float32)
    for first in range(0, num_splits, BLOCK_S):
        splits = first + tl.arange(0, BLOCK_S)
        part = splits * num_tokens + row
        lse = tl.load(
            LsePart + head * stride_lh + part, splits < num_splits, other=float("-inf")
        )
        m_new = tl.maximum(m_i, tl.max(lse, 0))
        # Splits without keys have LSE -inf and weight 0: no -inf - -inf.
        m_ref = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_ref)
        w = tl.math.exp2(lse - m_ref)
        o = tl.load(
            OPart + (part * H + head)[:, None] * D + offs_d[None, :],
            (w > 0.0)[:, None],
            other=0.0,
        )
        acc = acc * alpha + tl.sum(o * w[:, None], 0)
        l_i = l_i * alpha + tl.sum(w, 0)
        m_i = m_new
    # Requests without keys come out zero.
    has_keys = l_i > 0.0
    y = acc / tl.where(has_keys, l_i, 1.0)
    if Gate is not None:
        gate = tl.load(Gate + row * stride_gate + head * D + offs_d).to(tl.float32)
        sink = tl.load(Sink + row * stride_sink + head).to(tl.float32)
        lse_e = tl.where(
            has_keys, (m_i + tl.math.log2(l_i)) * 0.6931471805599453, float("-inf")
        )
        y = y * _sigmoid(gate)
        y = y * _sigmoid(lse_e - sink)
    tl.store(y_ptrs, y.to(Y.dtype.element_ty))


def _combine(
    o_part: torch.Tensor,
    lse_part: torch.Tensor,
    gate: torch.Tensor | None,
    sinks: torch.Tensor | None,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    num_splits: int,
) -> None:
    num_heads, head_dim = o_part.shape[1:]
    num_tokens = o_part.shape[0] // num_splits
    if num_tokens == 0:
        return
    _split_combine_kernel[(num_tokens, num_heads)](
        o_part,
        lse_part,
        gate,
        sinks,
        out,
        cu_seqlens_q,
        # The last entry counts the tokens of all requests.
        cu_seqlens_q.shape[0] - 1,
        num_splits,
        num_tokens,
        lse_part.stride(0),
        gate.stride(0) if gate is not None else 0,
        sinks.stride(0) if sinks is not None else 0,
        H=num_heads,
        D=head_dim,
        # Splits merged per iteration: few splits come with large batches.
        BLOCK_S=4 if num_splits <= 4 else 64,
        num_warps=1,
        enable_fp_fusion=False,
    )


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.math.exp2(-x * 1.4426950408889634))


@triton.jit
def _gate_sink_kernel(
    X,
    Gate,
    LSE,
    Sink,
    Y,
    stride_x,
    stride_gate,
    stride_lse_t,
    stride_lse_h,
    stride_sink,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H * D
    head = col // D
    x = tl.load(X + row * stride_x + col, mask, other=0)
    gate = tl.load(Gate + row * stride_gate + col, mask, other=0).to(tl.float32)
    lse = tl.load(LSE + row * stride_lse_t + head * stride_lse_h, mask, other=0)
    sink = tl.load(Sink + row * stride_sink + head, mask, other=0).to(tl.float32)
    result = x.to(tl.float32) * _sigmoid(gate)
    result = result * _sigmoid(lse.to(tl.float32) - sink)
    tl.store(Y + row * H * D + col, result, mask)


def gate_sink(
    output: torch.Tensor,
    gate: torch.Tensor,
    lse: torch.Tensor,
    sinks: torch.Tensor,
    result: torch.Tensor,
) -> None:
    """Write `output * sigmoid(gate) * sigmoid(lse - sink)` per head to
    `result`.

    `gate` and `result` are `(T, H * D)`, `sinks` and `lse` are
    `(T, H)` and `output` is `(T, H, D)`.
    """
    num_tokens, num_heads = sinks.shape
    assert lse.shape == (num_tokens, num_heads)
    assert output.shape[1] == num_heads and output.is_contiguous()
    assert gate.stride(1) == sinks.stride(1) == 1
    assert result.shape == gate.shape and result.is_contiguous()
    width = gate.shape[1]
    _gate_sink_kernel[(num_tokens, triton.cdiv(width, 512))](
        output,
        gate,
        lse,
        sinks,
        result,
        output.stride(0),
        gate.stride(0),
        lse.stride(0),
        lse.stride(1),
        sinks.stride(0),
        num_heads,
        width // num_heads,
        512,
        enable_fp_fusion=False,
    )


@triton.jit
def _log_scaling_tau_kernel(
    Pos,
    Out,
    n,
    FLOOR: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(Pos + i, i < n, other=0)
    ratio = tl.maximum((pos + 1).to(tl.float32) * (1.0 / FLOOR), 1.0)
    tau = 1.0 + ALPHA * tldevice.log(ratio)
    tl.store(Out + i, tau, i < n)


def compute_log_scaling_tau(
    positions: torch.Tensor, n_floor: int, alpha: float
) -> torch.Tensor:
    """Return fp32 `1 + alpha * log(max((positions + 1) / n_floor, 1))`."""
    output = torch.empty(positions.shape, device=positions.device, dtype=torch.float32)
    n = positions.numel()
    _log_scaling_tau_kernel[(triton.cdiv(n, 256),)](
        positions, output, n, n_floor, alpha, 256, enable_fp_fusion=False
    )
    return output
