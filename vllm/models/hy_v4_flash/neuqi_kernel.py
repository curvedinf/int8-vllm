# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused GPU evaluation of the NeUQI scale/offset candidates for W4 G128."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _offset(x, h):
    lo = tl.min(x, 0) - 16.0
    hi = tl.max(x, 0) + 1.0
    for _ in range(20):
        b = (lo + hi) * 0.5
        delta = b - x
        slope = tl.sum(
            h
            * tl.where(delta < -15.5, delta + 15.0, tl.where(delta > 0.5, delta, 0.0)),
            0,
        )
        lo, hi = tl.where(slope < 0.0, b, lo), tl.where(slope < 0.0, hi, b)
    left = (lo + hi) * 0.5 - 1.0
    right = left + 2.0
    level = tl.floor(x + 0.5 - left)
    y = x - tl.minimum(15.0, tl.maximum(0.0, level))
    init1 = tl.sum(h * y, 0)
    init2 = tl.sum(h * y * y, 0)
    events = tl.arange(0, 256)
    rows = events % 128
    after = (events // 128).to(tl.float32)
    old = tl.gather(level, rows, 0) - after
    time = tl.gather(x, rows, 0) + 0.5 - old
    live = (old >= 1.0) & (old <= 15.0) & (time <= right)
    time = tl.where(live, time, right)
    mass = tl.where(live, tl.gather(h, rows, 0), 0.0)
    # Sort time and carry its original lane for gathering event weights.
    bits = time.to(tl.uint32, bitcast=True)
    key = bits ^ tl.where((bits >> 31) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
    pair = (key.to(tl.uint64) << 32) | events.to(tl.uint64)
    order = tl.sort(pair, descending=False).to(tl.uint32).to(tl.int32)
    time = tl.gather(time, order, 0)
    mass = tl.gather(mass, order, 0)
    s1 = init1 + tl.cumsum(mass)
    s2 = init2 + tl.cumsum(mass * (2.0 * time))
    end = tl.where(events < 255, tl.gather(time, tl.minimum(events + 1, 255), 0), right)
    best = tl.minimum(tl.maximum(s1, time), end)
    loss = s2 + best * (best - 2.0 * s1)
    idx = tl.argmin(loss, 0, tie_break_left=True)
    choice = tl.sum(tl.where(events == idx, best, 0.0), 0)
    first = tl.minimum(tl.maximum(init1, left), tl.min(time, 0))
    first_loss = init2 + first * (first - 2.0 * init1)
    return tl.where(first_loss <= tl.min(loss, 0), first, choice)


@triton.jit
def _fit(w, h, step):
    scale = tl.maximum(step, 5.960464477539063e-8).to(tl.float16).to(tl.float32)
    x = w / scale
    zero = (-_offset(x, h)).to(tl.float16).to(tl.float32)
    # Rounding ties to even matches torch.round on the deployed grids.
    from_trunc = x + zero
    low = tl.floor(from_trunc)
    frac = from_trunc - low
    q = low + ((frac > 0.5) | ((frac == 0.5) & ((low.to(tl.int32) & 1) != 0))).to(
        tl.float32
    )
    q = tl.maximum(0.0, tl.minimum(15.0, q))
    err = (q - zero) * scale - w
    return scale, zero, tl.sum(h * err * err, 0)


@triton.jit
def _neuqi(
    W,
    H,
    OUT,
    S,
    Z,
    K: tl.constexpr,
    H_ROW: tl.constexpr,
    COARSE: tl.constexpr,
    FINE: tl.constexpr,
):
    group = tl.program_id(0).to(tl.int64)
    idx = tl.arange(0, 128)
    w = tl.load(W + group * 128 + idx).to(tl.float32)
    row = group // (K // 128)
    hi = row * H_ROW + group % (K // 128) * 128 + idx
    h = tl.maximum(tl.load(H + hi).to(tl.float32), 0.0)
    norm = tl.sum(h, 0)
    h = tl.where(norm > 0.0, h / tl.maximum(norm, 1.0e-30), 1.0 / 128.0)
    bottom, top = tl.min(w, 0), tl.max(w, 0)
    full = (top - bottom) / 15.0
    best, step, zero = float("inf"), 1.0, 0.0
    for j in range(max(1, COARSE // 2), COARSE + 1):
        s, z, loss = _fit(w, h, full * (j / COARSE))
        step, zero = tl.where(loss < best, s, step), tl.where(loss < best, z, zero)
        best = tl.minimum(loss, best)
    center = step
    for i in range(1, FINE // COARSE // 2 + 1):
        for sign in tl.static_range(2):
            s, z, loss = _fit(w, h, center + (1 - 2 * sign) * full * (i / FINE))
            step, zero = tl.where(loss < best, s, step), tl.where(loss < best, z, zero)
            best = tl.minimum(best, loss)
    step, zero = (
        tl.where(bottom == top, 1.0, step),
        tl.where(bottom == top, -bottom, zero),
    )
    v = w / step + zero
    low = tl.floor(v)
    frac = v - low
    code = low + ((frac > 0.5) | ((frac == 0.5) & ((low.to(tl.int32) & 1) != 0))).to(
        tl.float32
    )
    q = tl.maximum(0.0, tl.minimum(15.0, code)).to(tl.uint8)
    first, second = tl.split(tl.reshape(q, (64, 2)))
    tl.store(OUT + group * 64 + tl.arange(0, 64), first | (second << 4))
    tl.store(S + group, step)
    tl.store(Z + group, zero)


def fit_gpu(weights, diagonal, coarse, fine):
    if weights.ndim != 2 or not weights.is_contiguous() or diagonal.stride(-1) != 1:
        raise ValueError(
            "The GPU NeUQI fitter requires contiguous two-dimensional weight tiles"
        )
    m, k = weights.shape
    expanded = diagonal.broadcast_to(weights.shape)
    codes = torch.empty((m, k // 2), dtype=torch.uint8, device=weights.device)
    scales = torch.empty((m, k // 128), dtype=torch.float16, device=weights.device)
    zeros = torch.empty_like(scales)
    _neuqi[(m * (k // 128),)](
        weights,
        expanded,
        codes,
        scales,
        zeros,
        k,
        expanded.stride(0),
        coarse,
        fine,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return codes, scales, zeros
