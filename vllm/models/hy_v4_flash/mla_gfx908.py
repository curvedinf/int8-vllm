# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared paged, sink-aware MLA for the sliding and sparse HY4 layers on gfx908."""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _sparse_attn_decode_reduce_kernel,
)

LATENT, ROTARY = 512, 64


def cache_row_bytes(group, rope_group):
    payload = LATENT + ROTARY + 2 * (LATENT // group + ROTARY // rope_group)
    return triton.cdiv(payload, 16) * 16


def quantize_vector(x, group):
    groups = x.float().reshape(*x.shape[:-1], x.shape[-1] // group, group)
    scales = (groups.abs().amax(-1) / 127).clamp_min(2**-24).to(torch.float16)
    codes = (groups / scales.float().unsqueeze(-1)).round().clamp_(-127, 127)
    return codes.to(torch.int8).reshape(x.shape), scales


@triton.jit
def _quantize_int8(x, SIZE: tl.constexpr, GROUP: tl.constexpr):
    x = tl.reshape(x, (SIZE // GROUP, GROUP))
    scale = tl.maximum(tl.max(tl.abs(x), 1) / 127.0, 5.960464477539063e-8).to(
        tl.float16
    )
    normalized = tl.div_rn(x, scale[:, None].to(tl.float32))
    rounded = tl.inline_asm_elementwise(
        "v_rndne_f32 $0, $1",
        "=v,v",
        [normalized],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    code = tl.maximum(-127.0, tl.minimum(127.0, rounded)).to(tl.int8)
    return code.reshape((SIZE,)), scale


@triton.jit
def _encode_group(
    X, Q, S, valid, STRIDE: tl.constexpr, SIZE: tl.constexpr, GROUP: tl.constexpr
):
    cols = tl.arange(0, SIZE)
    x = tl.load(X + cols * STRIDE).to(tl.float32)
    code, scale = _quantize_int8(x, SIZE, GROUP)
    tl.store(Q + cols, code, valid)
    tl.store(S + tl.arange(0, SIZE // GROUP), scale, valid)


@triton.jit
def _encode_mla(
    L,
    R,
    Q,
    S,
    SLOTS,
    LT: tl.constexpr,
    LH: tl.constexpr,
    LD: tl.constexpr,
    RT: tl.constexpr,
    RH: tl.constexpr,
    RD: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    RG: tl.constexpr,
    CACHE: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    if CACHE:
        slot = tl.load(SLOTS + t).to(tl.int64)
        valid = slot >= 0
        base = (slot // PAGE) * PAGE_STRIDE + (slot % PAGE) * ROW_STRIDE
        dest = Q + base
        scales = S + base // 2 + 288
    else:
        valid = True
        dest = Q + (t * H + h) * 576
        scales = S + (t * H + h) * (512 // G + 64 // RG)
    _encode_group(L + t * LT + h * LH, dest, scales, valid, LD, 512, G)
    _encode_group(R + t * RT + h * RH, dest + 512, scales + 512 // G, valid, RD, 64, RG)


def quantize_mla(latent, rope, group, rope_group):
    if (
        latent.shape[:-1] != rope.shape[:-1]
        or latent.shape[-1] != LATENT
        or rope.shape[-1] != ROTARY
    ):
        raise ValueError("HY4 MLA requires matching latent and rotary input rows")
    shape = latent.shape[:-1]
    if latent.ndim == 2:
        latent, rope = latent.unsqueeze(1), rope.unsqueeze(1)
    elif latent.ndim != 3:
        raise ValueError("Expected a token or token-by-head HY4 MLA input")
    tokens, heads = latent.shape[:2]
    q = torch.empty((*shape, LATENT + ROTARY), dtype=torch.int8, device=latent.device)
    scales = torch.empty(
        (*shape, LATENT // group + ROTARY // rope_group),
        dtype=torch.float16,
        device=latent.device,
    )
    if tokens:
        _encode_mla[(tokens, heads)](
            latent,
            rope,
            q,
            scales,
            q,
            *latent.stride(),
            *rope.stride(),
            heads,
            group,
            rope_group,
            False,
            1,
            0,
            0,
            num_warps=2,
        )
    return q, scales


def write_mla_cache(latent, rope, cache, slots, group, rope_group):
    count = slots.numel()
    if count:
        latent_rows = latent[:count].reshape(count, LATENT)
        rotary_rows = rope[:count].reshape(count, ROTARY)
        _encode_mla[(count, 1)](
            latent_rows,
            rotary_rows,
            cache.view(torch.int8),
            cache.view(torch.float16),
            slots,
            latent_rows.stride(0),
            0,
            latent_rows.stride(1),
            rotary_rows.stride(0),
            0,
            rotary_rows.stride(1),
            1,
            group,
            rope_group,
            True,
            cache.shape[1],
            cache.stride(0),
            cache.stride(1),
            num_warps=2,
        )


@triton.jit
def _group_scores(
    Q,
    QS,
    KV,
    KS,
    head,
    good_head,
    offsets,
    valid,
    LABEL: tl.constexpr,
    SIZE: tl.constexpr,
    GROUP: tl.constexpr,
    SCALE_OFFSET: tl.constexpr,
    NS: tl.constexpr,
    B: tl.constexpr,
):
    scores = tl.full((head.shape[0], B), 0, tl.float32)
    channels = tl.arange(0, max(32, GROUP))
    # Unrolling many small scale groups allocates all dot operand conversions
    # at once and can exceed CDNA1's 64 KiB LDS.
    for g in tl.range(
        SIZE // GROUP, loop_unroll_factor=1 if GROUP < 32 else SIZE // GROUP
    ):
        axis = LABEL + g * GROUP + channels
        q = tl.load(
            Q + head[:, None] * 576 + axis[None, :],
            good_head[:, None] & (channels[None, :] < GROUP),
            0,
        )
        k = tl.load(
            KV + offsets[:, None] + axis[None, :],
            valid[:, None] & (channels[None, :] < GROUP),
            0,
        )
        qs = tl.load(QS + head * NS + SCALE_OFFSET + g, good_head, 0).to(tl.float32)
        ks = tl.load(KS + offsets // 2 + 288 + SCALE_OFFSET + g, valid, 0).to(
            tl.float32
        )
        dot = tl.dot(q, tl.trans(k), out_dtype=tl.int32).to(tl.float32)
        scores += dot * qs[:, None] * ks[None, :]
    return scores


@triton.jit
def _probability_sum(prob):
    # A fixed reduction tree keeps fused and materialized score tiles in step.
    rows: tl.constexpr = prob.shape[0]
    cols: tl.constexpr = prob.shape[1]
    for i in tl.static_range(5 if cols == 32 else 6):
        left, right = tl.split(tl.reshape(prob, (rows, cols // (1 << (i + 1)), 2)))
        prob = left + right
    return tl.reshape(prob, (rows,))


@triton.jit
def _accumulate_values(
    score,
    valid,
    offsets,
    dv,
    KV,
    KS,
    mx,
    sm,
    acc,
    GROUP: tl.constexpr,
    BV: tl.constexpr,
    PV_I8: tl.constexpr,
    vtile,
):
    new_mx = tl.maximum(mx, tl.max(score, 1))
    retain = tl.exp(mx - new_mx)
    prob = tl.where(score > -3.4028234663852886e38, tl.exp(score - new_mx[:, None]), 0)
    sm = sm * retain + _probability_sum(prob)
    v = tl.load(KV + offsets[:, None] + dv[None, :], valid[:, None], 0)
    if PV_I8:
        tl.static_assert(BV == GROUP and GROUP >= 32)
        vs = tl.load(KS + offsets // 2 + 288 + vtile, valid, 0).to(tl.float32)
        weighted = prob * vs[None, :]
        ps = tl.maximum(tl.max(weighted, 1) / 127, 1e-30)
        p8 = (weighted / ps[:, None] + 0.5).to(tl.int8)
        part = tl.dot(p8, v, out_dtype=tl.int32).to(tl.float32) * ps[:, None]
    else:
        vs = tl.load(
            KS + offsets[:, None] // 2 + 288 + dv[None, :] // GROUP, valid[:, None], 0
        ).to(tl.float16)
        part = tl.dot(prob.to(tl.float16), v.to(tl.float16) * vs)
    acc = acc * retain[:, None] + part
    return new_mx, sm, acc


@triton.jit
def _window_start(STARTS, LENGTHS, owner, token, WINDOW: tl.constexpr):
    if WINDOW:
        start = tl.load(STARTS + owner)
        end = tl.load(STARTS + owner + 1)
        position = tl.load(LENGTHS + owner) - (end - token)
        first = tl.where(
            (token >= start) & (token < end), position - WINDOW + 1, -WINDOW
        )
    else:
        first = 0
    return first


@triton.jit
def _selected_offsets(
    IDS,
    TABLE,
    col,
    owner,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TABLE_WIDTH,
    BLOCK_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    WINDOW: tl.constexpr,
    first,
):
    if WINDOW:
        logical = tl.where(col < WINDOW, first + col, -1)
    else:
        logical = tl.load(IDS + tl.minimum(col, TOPK - 1))
    valid = (col < TOPK) & (logical >= 0) & (logical < TABLE_WIDTH * PAGE)
    logical = tl.where(valid, logical, 0)
    page = tl.load(TABLE + owner * TABLE_STRIDE + logical // PAGE, valid, -1)
    valid &= page >= 0
    offsets = tl.where(
        valid, page.to(tl.int64) * BLOCK_STRIDE + (logical % PAGE) * ROW_STRIDE, 0
    )
    return offsets, valid


@triton.jit
def _mla_scores(
    Q,
    QS,
    KV,
    KS,
    head,
    active,
    offsets,
    valid,
    GROUP: tl.constexpr,
    RG: tl.constexpr,
    NS: tl.constexpr,
    B: tl.constexpr,
    SCALE: tl.constexpr,
):
    score = _group_scores(
        Q, QS, KV, KS, head, active, offsets, valid, 0, 512, GROUP, 0, NS, B
    )
    score += _group_scores(
        Q, QS, KV, KS, head, active, offsets, valid, 512, 64, RG, 512 // GROUP, NS, B
    )
    return tl.where(
        valid[None, :] & active[:, None], score * SCALE, -3.4028234663852886e38
    )


@triton.jit
def _selected_scores(
    Q,
    QS,
    KV,
    KS,
    IDS,
    OWNER,
    TABLE,
    OUT,
    ADDRESS,
    STARTS,
    LENGTHS,
    ROW_OFFSET,
    WINDOW: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TABLE_WIDTH,
    ID_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    GROUP: tl.constexpr,
    RG: tl.constexpr,
    SCALE: tl.constexpr,
    B: tl.constexpr,
    PITCH: tl.constexpr,
    SAVE_ADDRESS: tl.constexpr,
    SKIP_EMPTY: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * B + tl.arange(0, B)
    head = tl.arange(0, 16)
    owner = tl.load(OWNER + token).to(tl.int64)
    first = _window_start(STARTS, LENGTHS, owner, token + ROW_OFFSET, WINDOW)
    offsets, valid = _selected_offsets(
        IDS + token * ID_STRIDE,
        TABLE,
        col,
        owner,
        TOPK,
        PAGE,
        TABLE_STRIDE,
        TABLE_WIDTH,
        BLOCK_STRIDE,
        ROW_STRIDE,
        WINDOW,
        first,
    )
    ns: tl.constexpr = 512 // GROUP + 64 // RG
    q = Q + token * HEADS * 576
    qs = QS + token * HEADS * ns
    if not SKIP_EMPTY or tl.sum(valid.to(tl.int32), 0) > 0:
        score = _mla_scores(
            q, qs, KV, KS, head, head < HEADS, offsets, valid, GROUP, RG, ns, B, SCALE
        )
    else:
        score = tl.full((16, B), -3.4028234663852886e38, tl.float32)
    if SAVE_ADDRESS:
        tl.store(
            ADDRESS + token * PITCH + col, tl.where(valid, offsets, -1), col < TOPK
        )
    tl.store(
        OUT + (token * HEADS + head[:, None]) * PITCH + col[None, :],
        score,
        (head < HEADS)[:, None] & (col < TOPK)[None, :],
    )


@triton.jit
def _paged_mla(
    Q,
    QS,
    KV,
    KS,
    IDS,
    OWNER,
    TABLE,
    SINK,
    OUT,
    M,
    L,
    PART,
    SCORES,
    ADDRESS,
    STARTS,
    LENGTHS,
    ROW_OFFSET,
    WINDOW: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TABLE_WIDTH,
    ID_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    GROUP: tl.constexpr,
    RG: tl.constexpr,
    SCALE: tl.constexpr,
    B: tl.constexpr,
    BV: tl.constexpr,
    SPLIT: tl.constexpr,
    PV_I8: tl.constexpr,
    SKIP_EMPTY: tl.constexpr,
    SCORE_PITCH: tl.constexpr,
    SAVED_ADDRESS: tl.constexpr,
):
    token = tl.program_id(0)
    split = tl.program_id(1)
    vtile = tl.program_id(2)
    head = tl.arange(0, 16)
    good_head = head < HEADS
    dv = vtile * BV + tl.arange(0, BV)
    idx = tl.arange(0, B)
    nsc: tl.constexpr = 512 // GROUP + 64 // RG
    q = Q + token.to(tl.int64) * HEADS * 576
    qs = QS + token.to(tl.int64) * HEADS * nsc
    owner = tl.load(OWNER + token).to(tl.int64)
    first = _window_start(STARTS, LENGTHS, owner, token + ROW_OFFSET, WINDOW)
    neg = -3.4028234663852886e38
    mx = tl.full((16,), neg, tl.float32)
    sm = tl.zeros((16,), tl.float32)
    acc = tl.zeros((16, BV), tl.float32)
    span: tl.constexpr = triton.cdiv(TOPK, SPLIT * B) * B
    for i in range(triton.cdiv(span, B)):
        col = split * span + i * B + idx
        if SAVED_ADDRESS:
            offsets = tl.load(
                ADDRESS + token.to(tl.int64) * SCORE_PITCH + col, col < TOPK, -1
            )
            valid = offsets >= 0
            offsets = tl.where(valid, offsets, 0)
        else:
            offsets, valid = _selected_offsets(
                IDS + token.to(tl.int64) * ID_STRIDE,
                TABLE,
                col,
                owner,
                TOPK,
                PAGE,
                TABLE_STRIDE,
                TABLE_WIDTH,
                BLOCK_STRIDE,
                ROW_STRIDE,
                WINDOW,
                first,
            )
        if not SKIP_EMPTY or tl.sum(valid.to(tl.int32), 0) > 0:
            if SCORE_PITCH:
                score = tl.load(
                    SCORES
                    + (token.to(tl.int64) * HEADS + head[:, None]) * SCORE_PITCH
                    + col[None, :],
                    good_head[:, None] & (col < TOPK)[None, :],
                    neg,
                )
            else:
                score = _mla_scores(
                    q,
                    qs,
                    KV,
                    KS,
                    head,
                    good_head,
                    offsets,
                    valid,
                    GROUP,
                    RG,
                    nsc,
                    B,
                    SCALE,
                )
            mx, sm, acc = _accumulate_values(
                score, valid, offsets, dv, KV, KS, mx, sm, acc, GROUP, BV, PV_I8, vtile
            )
    if SPLIT == 1:
        sink = tl.load(SINK + head, good_head, 0).to(tl.float32)
        final_mx = tl.maximum(mx, sink)
        factor = tl.exp(mx - final_mx)
        den = sm * factor + tl.exp(sink - final_mx)
        result = acc * (factor / den)[:, None]
        tl.store(
            OUT + (token.to(tl.int64) * HEADS + head[:, None]) * 512 + dv[None, :],
            result,
            good_head[:, None],
        )
    else:
        base = (token.to(tl.int64) * SPLIT + split) * HEADS + head
        if vtile == 0:
            tl.store(M + base, mx, good_head)
            tl.store(L + base, sm, good_head)
        tl.store(PART + base[:, None] * 512 + dv[None, :], acc, good_head[:, None])


@triton.jit
def _causal_mla(
    Q,
    QS,
    KV,
    KS,
    STARTS,
    LENGTHS,
    TABLE,
    SINK,
    OUT,
    BLOCK_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TABLE_WIDTH,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    G: tl.constexpr,
    RG: tl.constexpr,
    SCALE: tl.constexpr,
    WINDOW: tl.constexpr,
    QT: tl.constexpr,
    B: tl.constexpr,
    BV: tl.constexpr,
    PAGE_PAIR: tl.constexpr = False,
):
    # Neighboring tokens of this request share KV reads and MFMA operands.
    local = tl.program_id(0) * QT
    request = tl.program_id(1)
    vtile = tl.program_id(2)
    start = tl.load(STARTS + request)
    end = tl.load(STARTS + request + 1)
    if start + local >= end:
        return
    length = tl.load(LENGTHS + request)
    context = length - (end - start)
    lanes = tl.arange(0, QT * HEADS)
    token = start + local + lanes // HEADS
    live = token < end
    head = token.to(tl.int64) * HEADS + lanes % HEADS
    position = context + local + lanes // HEADS
    dv = vtile * BV + tl.arange(0, BV)
    nsc: tl.constexpr = 512 // G + 64 // RG
    neg = -3.4028234663852886e38
    mx = tl.full((QT * HEADS,), neg, tl.float32)
    sm = tl.zeros((QT * HEADS,), tl.float32)
    acc = tl.zeros((QT * HEADS, BV), tl.float32)
    low = tl.maximum(0, context + local - WINDOW + 1) if WINDOW else 0
    high = tl.minimum(length, context + local + QT)
    for first in range(low, high, B):
        logical = first + tl.arange(0, B)
        valid = (logical < high) & (logical < TABLE_WIDTH * PAGE)
        if PAGE_PAIR:
            # A consecutive key tile spans at most two logical pages.
            first_page = first // PAGE
            page_entry = TABLE + request * TABLE_STRIDE + first_page
            left = tl.load(page_entry, first_page < TABLE_WIDTH, -1)
            if WINDOW:
                right = tl.load(page_entry + 1, first_page + 1 < TABLE_WIDTH, -1)
                page = tl.where(logical // PAGE == first_page, left, right)
            else:
                page = tl.full((B,), 0, tl.int32) + left
        else:
            page = tl.load(TABLE + request * TABLE_STRIDE + logical // PAGE, valid, -1)
        valid = valid & (page >= 0)
        offsets = tl.where(
            valid, page.to(tl.int64) * BLOCK_STRIDE + (logical % PAGE) * ROW_STRIDE, 0
        )
        score = _group_scores(
            Q, QS, KV, KS, head, live, offsets, valid, 0, 512, G, 0, nsc, B
        )
        score += _group_scores(
            Q, QS, KV, KS, head, live, offsets, valid, 512, 64, RG, 512 // G, nsc, B
        )
        allowed = (
            live[:, None] & valid[None, :] & (logical[None, :] <= position[:, None])
        )
        if WINDOW:
            allowed = allowed & (logical[None, :] > position[:, None] - WINDOW)
        score = tl.where(allowed, score * SCALE, neg)
        mx, sm, acc = _accumulate_values(
            score, valid, offsets, dv, KV, KS, mx, sm, acc, G, BV, False, vtile
        )
    sink = tl.load(SINK + lanes % HEADS).to(tl.float32)
    peak = tl.maximum(mx, sink)
    factor = tl.exp(mx - peak)
    den = sm * factor + tl.exp(sink - peak)
    value = acc * (factor / den)[:, None]
    tl.store(OUT + head[:, None] * 512 + dv[None, :], value, live[:, None])


def causal_mla(
    query,
    scales,
    cache,
    starts,
    lengths,
    table,
    sinks,
    scale,
    group,
    rope_group,
    max_query_len,
    window,
    *,
    query_tile=4,
    key_tile=32,
    value_tile=128,
    warps=8,
    page_pair=False,
):
    """Shared chronological MLA for sliding layers and complete sparse prefixes."""
    tokens, heads, width = query.shape
    if (
        heads not in (8, 16)
        or width != 576
        or heads * query_tile not in (16, 32, 64, 128)
    ):
        raise ValueError("Unsupported MI100 causal MLA head tile")
    if key_tile not in (32, 64) or value_tile not in (64, 128, 256, 512):
        raise ValueError("Unsupported MI100 causal MLA channel tile")
    out = torch.empty((tokens, heads, 512), device=query.device, dtype=torch.float16)
    if tokens and lengths.numel():
        _causal_mla[
            (triton.cdiv(max_query_len, query_tile), lengths.numel(), 512 // value_tile)
        ](
            query,
            scales,
            cache.view(torch.int8),
            cache.view(torch.float16),
            starts,
            lengths,
            table,
            sinks,
            out,
            cache.stride(0),
            cache.stride(1),
            table.stride(0),
            table.shape[1],
            heads,
            cache.shape[1],
            group,
            rope_group,
            scale,
            window,
            query_tile,
            key_tile,
            value_tile,
            PAGE_PAIR=page_pair and cache.shape[1] % key_tile == 0,
            num_warps=warps,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


def mla_splits(tokens, topk, tile=32):
    return max(1, min(triton.cdiv(topk, tile), triton.cdiv(120, max(1, tokens))))


def paged_mla(
    query,
    qscales,
    cache,
    ids,
    owner,
    table,
    sinks,
    scale,
    group,
    rope_group,
    *,
    tile=32,
    value_tile=128,
    splits=None,
    warps=8,
    pv_int8=False,
    skip_empty=False,
    score_tile=0,
    score_warps=4,
    fly_scores=False,
    fly_values=False,
    save_addresses=False,
    out=None,
    window=0,
    starts=None,
    lengths=None,
    row_offset=0,
):
    tokens, heads, width = query.shape
    topk = ids.shape[1]
    count = 512 // group + 64 // rope_group
    if (
        width != 576
        or heads > 16
        or qscales.shape != (tokens, heads, count)
        or cache.shape[-1] < 576 + 2 * count
        or ids.shape[0] != tokens
        or group not in (4, 8, 16, 32, 64, 128)
        or rope_group not in (4, 8, 16, 32, 64)
    ):
        raise ValueError("Incompatible HY4 quantized MLA layout")
    if splits is None:
        splits = mla_splits(tokens, topk, tile)
    if tile != 32 or value_tile not in (32, 64, 128, 256):
        raise ValueError("Unsupported gfx908 paged MLA tile")
    if window and (starts is None or lengths is None or not 0 < window <= topk):
        raise ValueError(
            "Implicit sliding MLA needs request lengths and a valid window"
        )
    if not window:
        starts = lengths = owner
    if out is None:
        out = torch.empty(
            (tokens, heads, LATENT), dtype=torch.float16, device=query.device
        )
    if not tokens:
        return out
    if splits > 1:
        maxima = torch.empty(
            (tokens, splits, heads), device=query.device, dtype=torch.float32
        )
        normalizers = torch.empty_like(maxima)
        partial = torch.empty(
            (*maxima.shape, LATENT), device=query.device, dtype=torch.float32
        )
    else:
        maxima = normalizers = partial = out
    pitch = 0
    scores = addresses = out
    if score_tile:
        if score_tile not in (32, 64, 128):
            raise ValueError("Unsupported MI100 sparse QK tile")
        pitch = triton.cdiv(topk, score_tile) * score_tile
        if fly_scores and heads in (8, 16) and group >= 32 and rope_group >= 32:
            from aiter.ops.flydsl.kernels.gfx908_mla_scores import gfx908_mla_scores

            scores, addresses = gfx908_mla_scores(
                query,
                qscales,
                cache,
                ids,
                owner,
                table,
                starts,
                lengths,
                scale,
                group,
                rope_group,
                pitch,
                window,
                row_offset,
                save_addresses,
                skip_empty,
            )
        else:
            scores = torch.empty(
                (tokens, heads, pitch), dtype=torch.float32, device=query.device
            )
            if save_addresses:
                addresses = torch.empty(
                    (tokens, pitch), dtype=torch.int64, device=query.device
                )
            _selected_scores[(tokens, triton.cdiv(topk, score_tile))](
                query,
                qscales,
                cache.view(torch.int8),
                cache.view(torch.float16),
                ids,
                owner,
                table,
                scores,
                addresses,
                starts,
                lengths,
                row_offset,
                window,
                cache.stride(0),
                cache.stride(1),
                table.stride(0),
                table.shape[1],
                ids.stride(0),
                topk,
                heads,
                cache.shape[1],
                group,
                rope_group,
                scale,
                score_tile,
                pitch,
                save_addresses,
                skip_empty,
                num_warps=score_warps,
                num_stages=1,
                enable_fp_fusion=False,
            )
    if (
        fly_values
        and score_tile
        and save_addresses
        and heads == 8
        and group >= 32
        and rope_group >= 32
        and topk >= 1024
        and not pv_int8
        and not skip_empty
    ):
        from aiter.ops.flydsl.kernels.gfx908_mla_values import gfx908_mla_values

        gfx908_mla_values(
            cache,
            scores,
            addresses,
            sinks,
            out,
            maxima if splits > 1 else None,
            normalizers if splits > 1 else None,
            partial if splits > 1 else None,
            topk,
            group,
            splits,
            2 if tokens >= 32 else 1,
            tokens < 32,
        )
    else:
        _paged_mla[(tokens, splits, LATENT // value_tile)](
            query,
            qscales,
            cache.view(torch.int8),
            cache.view(torch.float16),
            ids,
            owner,
            table,
            sinks,
            out,
            maxima,
            normalizers,
            partial,
            scores,
            addresses,
            starts,
            lengths,
            row_offset,
            window,
            cache.stride(0),
            cache.stride(1),
            table.stride(0),
            table.shape[1],
            ids.stride(0),
            topk,
            heads,
            cache.shape[1],
            group,
            rope_group,
            scale,
            tile,
            value_tile,
            splits,
            pv_int8,
            skip_empty,
            pitch,
            bool(score_tile and save_addresses),
            num_warps=warps,
            num_stages=1,
            enable_fp_fusion=False,
        )
    if splits > 1:
        _sparse_attn_decode_reduce_kernel[(tokens, triton.cdiv(heads, 4))](
            maxima,
            normalizers,
            partial,
            sinks,
            out,
            out,
            out,
            out,
            out.stride(0),
            out.stride(1),
            0,
            0,
            maxima.stride(0),
            maxima.stride(1),
            partial.stride(0),
            partial.stride(1),
            partial.stride(2),
            0,
            heads,
            True,
            False,
            LATENT,
            4,
            splits,
            triton.next_power_of_2(splits),
            False,
            LATENT,
            0,
            False,
            num_warps=4,
        )
    return out
