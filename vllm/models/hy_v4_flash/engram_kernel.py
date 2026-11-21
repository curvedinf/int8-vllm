# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fetch and dequantize only the 54-byte Q4_0 Engram rows a token needs."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _lookup_q4(
    PAGES,
    IDS,
    DEST,
    OUT,
    N,
    WIDTH_IDS: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    FIRST_PAGE,
    LOW,
    HIGH,
    SCATTER: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_MAJOR: tl.constexpr,
    INDEXED: tl.constexpr = False,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    col = tl.arange(0, 128)
    if ROW_MAJOR:
        slot = tl.load(IDS + row, row < N, -1) if INDEXED else row
        valid = (row < N) & (slot >= 0)
        address = tl.maximum(slot, 0).to(tl.int64) * 54
    else:
        idx = tl.load(
            IDS + (row // WIDTH_IDS) * STRIDE0 + (row % WIDTH_IDS) * STRIDE1, row < N, 0
        )
        valid = (row < N) & (idx >= LOW) & (idx < HIGH)
        safe = tl.where(valid, idx, LOW).to(tl.int64)
        address = (safe // 75 - FIRST_PAGE) * 4096 + (safe % 75) * 54
    # Each of the three 18-byte blocks starts with an aligned little-endian
    # half scale. The same byte packs columns i and i+16.
    base = PAGES + address[:, None] + (col[None, :] // 32) * 18
    mask = valid[:, None] & (col[None, :] < 96)
    scale = tl.load(base.to(tl.pointer_type(tl.float16)), mask, 0)
    codes = tl.load(base + 2 + col[None, :] % 16, mask, 0)
    nibble = tl.where((col[None, :] % 32) < 16, codes & 15, codes >> 4)
    y = (nibble.to(tl.int16) - 8).to(tl.float16) * scale
    dest = tl.load(DEST + row, row < N, 0) if SCATTER else row
    tl.store(
        OUT + dest[:, None].to(tl.int64) * 96 + col[None, :],
        y,
        (row < N)[:, None] & (col < 96)[None, :],
    )


def lookup_q4_pages(pages, ids, first_page, low, high, *, out=None, destinations=None):
    if out is None:
        out = torch.empty((*ids.shape, 96), device=ids.device, dtype=torch.float16)
    if not ids.numel():
        return out
    width = ids.shape[-1] if ids.dim() > 1 else 1
    matrix = ids.reshape(-1, width)
    block = 8
    _lookup_q4[(triton.cdiv(ids.numel(), block),)](
        pages,
        matrix,
        destinations if destinations is not None else ids,
        out,
        ids.numel(),
        width,
        matrix.stride(0),
        matrix.stride(1),
        first_page,
        low,
        high,
        destinations is not None,
        block,
        False,
        num_warps=4,
    )
    return out


def decompress_q4_rows(packed, out, indices=None):
    """Decode contiguous 54-byte rows with the same arithmetic as a paged lookup."""
    count = packed.shape[0] if indices is None else indices.numel()
    if count:
        _lookup_q4[(triton.cdiv(count, 8),)](
            packed,
            packed if indices is None else indices,
            packed,
            out,
            count,
            1,
            1,
            1,
            0,
            0,
            count,
            False,
            8,
            True,
            INDEXED=indices is not None,
            num_warps=4,
        )


@triton.jit
def _residual_mix(
    HIDDEN,
    KEYS,
    VALUE,
    HASHES,
    KEY_NORM,
    QUERY_NORM,
    GAMMA,
    SCALES,
    OUT,
    H0: tl.constexpr,
    H1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    V0: tl.constexpr,
    ID0: tl.constexpr,
    WIDTH: tl.constexpr,
    STREAMS: tl.constexpr,
    HASH_COUNT: tl.constexpr,
    HASH_TILE: tl.constexpr,
    TILE: tl.constexpr,
):
    token, stream = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, TILE)
    valid = col < WIDTH
    hidden = tl.load(HIDDEN + token * H0 + stream * H1 + col, valid, 0).to(tl.float32)
    key = tl.load(KEYS + token * K0 + stream * K1 + col, valid, 0).to(tl.float32)
    value = tl.load(VALUE + token * V0 + col, valid, 0).to(tl.float32)
    query_gamma = tl.load(QUERY_NORM + stream * WIDTH + col, valid, 0).to(tl.float32)
    key_gamma = tl.load(KEY_NORM + stream * WIDTH + col, valid, 0).to(tl.float32)
    eps: tl.constexpr = 1.1920928955078125e-7
    key_normalized = (
        key * tl.rsqrt(tl.sum(key * key, 0) / WIDTH + eps) * key_gamma
    ).to(tl.float16)
    query_normalized = (
        hidden * tl.rsqrt(tl.sum(hidden * hidden, 0) / WIDTH + eps) * query_gamma
    ).to(tl.float16)
    # Match the original FP16 elementwise multiply, sum output, nonlinearity
    # inputs and residual rounding. Reductions and the value stream use FP32.
    dot = tl.sum((key_normalized * query_normalized).to(tl.float32), 0).to(tl.float16)
    gate = (dot.to(tl.float32) * (WIDTH**-0.5)).to(tl.float16).to(tl.float32)
    min_gate = tl.full((), 1.0e-6, tl.float16).to(tl.float32)
    root = tl.sqrt(tl.maximum(tl.abs(gate), min_gate)).to(tl.float16).to(tl.float32)
    gate = (
        tl.sigmoid(tl.where(gate > 0, root, tl.where(gate < 0, -root, 0.0)))
        .to(tl.float16)
        .to(tl.float32)
    )
    value_gamma = tl.load(GAMMA + col, valid, 0).to(tl.float32)
    scales = tl.load(SCALES + stream * WIDTH + col, valid, 0).to(tl.float32)
    signal = value_gamma * value * tl.rsqrt(tl.sum(value * value, 0) / WIDTH + eps)
    delta = ((gate + scales) * signal).to(tl.float16)
    hc = tl.arange(0, HASH_TILE)
    alive = tl.max(tl.load(HASHES + token * ID0 + hc, hc < HASH_COUNT, -1), 0) >= 0
    tl.store(
        OUT + (token * STREAMS + stream) * WIDTH + col,
        hidden.to(tl.float16) + tl.where(alive, delta, 0.0).to(tl.float16),
        valid,
    )


def mix_engram_residual(
    hidden, keys, value, hashes, key_norm, query_norm, gamma, scales, *, warps=4
):
    rows, streams, width = hidden.shape
    output = torch.empty_like(hidden, memory_format=torch.contiguous_format)
    if rows:
        _residual_mix[(rows, streams)](
            hidden,
            keys,
            value,
            hashes,
            key_norm,
            query_norm,
            gamma,
            scales,
            output,
            hidden.stride(0),
            hidden.stride(1),
            keys.stride(0),
            width,
            value.stride(0),
            hashes.stride(0),
            width,
            streams,
            hashes.shape[1],
            triton.next_power_of_2(hashes.shape[1]),
            triton.next_power_of_2(width),
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return output
