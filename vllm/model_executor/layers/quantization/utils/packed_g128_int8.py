# SPDX-License-Identifier: Apache-2.0
"""Experimental packed IQ2-XXS/Q8_0/NVFP4 -> G128 INT8 MFMA for gfx908.

Weights remain in their official packed format in device memory. Only a tile is expanded
inside the GEMM. ModelOpt selects this path for gfx908 packed checkpoints.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _decode(W, GRID, rows, ks, N: tl.constexpr, K: tl.constexpr):
    base = rows[:, None].to(tl.int64) * (K // 256) * 66 + (ks[None, :] // 256) * 66
    valid = (rows[:, None] < N) & (ks[None, :] < K)
    lo = tl.load(W + base, valid, 0).to(tl.uint16)
    hi = tl.load(W + base + 1, valid, 0).to(tl.uint16)
    d = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    sub = (ks[None, :] % 256) // 32
    lane = (ks[None, :] % 32) // 8
    j = ks[None, :] % 8
    code = tl.load(W + base + 2 + sub * 8 + lane, valid, 0).to(tl.int32)
    bits = tl.full(base.shape, 0, tl.uint32)
    for b in tl.static_range(4):
        bits |= tl.load(W + base + 2 + sub * 8 + 4 + b, valid, 0).to(tl.uint32) << (
            8 * b
        )
    sign7 = (bits >> (7 * lane)) & 127
    parity = sign7 ^ (sign7 >> 4)
    parity ^= parity >> 2
    parity ^= parity >> 1
    signs = sign7 | ((parity & 1) << 7)
    grid = tl.load(GRID + code * 8 + j).to(tl.float32)
    value = d * (0.5 + (bits >> 28).to(tl.float32)) * 0.25 * grid
    return tl.where(valid, tl.where(((signs >> j) & 1) != 0, -value, value), 0.0)


@triton.jit
def _decode_q8(W, rows, ks, N: tl.constexpr, K: tl.constexpr):
    base = rows[:, None].to(tl.int64) * (K // 32) * 34 + (ks[None, :] // 32) * 34
    valid = (rows[:, None] < N) & (ks[None, :] < K)
    lo = tl.load(W + base, valid, 0).to(tl.uint16)
    hi = tl.load(W + base + 1, valid, 0).to(tl.uint16)
    scale = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    q = tl.load(W + base + 2 + ks[None, :] % 32, valid, 0).to(tl.int8, bitcast=True)
    return q.to(tl.float32) * scale


@triton.jit
def _decode_nvfp4(
    W, WS, WG, rows, ks, N: tl.constexpr, K: tl.constexpr, WG_PER_ROW: tl.constexpr
):
    valid = (rows[:, None] < N) & (ks[None, :] < K)
    byte = tl.load(
        W + rows[:, None].to(tl.int64) * (K // 2) + ks[None, :] // 2, valid, 0
    ).to(tl.int32)
    nibble = (byte >> ((ks[None, :] % 2) * 4)) & 15
    mag = nibble & 7
    value = tl.where(
        mag < 2,
        mag.to(tl.float32) * 0.5,
        tldevice.ldexp((2 + (mag & 1)).to(tl.float32), (mag >> 1) - 2),
    )
    value = tl.where((nibble & 8) != 0, -value, value)
    # E4M3 scales are read as bytes: gfx908 has no native FP8 load/compute.
    bits = tl.load(
        WS + rows[:, None].to(tl.int64) * (K // 16) + ks[None, :] // 16, valid, 0
    ).to(tl.int32)
    exp = (bits >> 3) & 15
    mantissa = bits & 7
    scale = tl.where(
        exp == 0,
        mantissa.to(tl.float32) * (2.0**-9),
        tldevice.ldexp((8 + mantissa).to(tl.float32), exp - 10),
    )
    scale = tl.where((bits & 128) != 0, -scale, scale)
    if WG_PER_ROW:
        global_scale = tl.load(WG + rows[:, None], rows[:, None] < N, 0)
    else:
        global_scale = tl.load(WG)
    return value * scale * global_scale.to(tl.float32)


@triton.jit
def _quant(X, Q, S, M: tl.constexpr, K: tl.constexpr):
    row, group = tl.program_id(0), tl.program_id(1)
    k = group * 128 + tl.arange(0, 128)
    v = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    scale = tl.maximum(tl.div_rn(tl.max(tl.abs(v), 0), 127.0), 1.0e-30)
    q = tldevice.nearbyint(tl.div_rn(v, scale))
    tl.store(Q + row * K + k, q.to(tl.int8), k < K)
    tl.store(S + row * tl.cdiv(K, 128) + group, scale)


@triton.jit
def _gemm(
    X,
    XS,
    W,
    GRID,
    WS,
    WG,
    Y,
    YS,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    START: tl.constexpr,
    LENGTH: tl.constexpr,
    RELU2: tl.constexpr,
    QUANT_OUT: tl.constexpr,
    BM: tl.constexpr,
    CODEC: tl.constexpr,
    IDS=None,
    TOPK: tl.constexpr = 1,
    ROUTED: tl.constexpr = False,
    DOWN: tl.constexpr = False,
    WSTRIDE: tl.constexpr = 0,
    SORTED=None,
    COUNT=None,
    GROUPED: tl.constexpr = False,
    WG_PER_ROW: tl.constexpr = False,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    if ROUTED:
        route = tl.program_id(0)
        if GROUPED:
            if route * BM >= tl.load(COUNT):
                return
            expert = tl.load(IDS + route).to(tl.int64)
            rm = tl.load(SORTED + route * BM + tl.arange(0, BM))
        else:
            expert = tl.load(IDS + route).to(tl.int64)
            rm = tl.where(tl.arange(0, BM) == 0, route, M)
        W += expert * WSTRIDE
    xm = rm
    if ROUTED and not DOWN:
        xm = rm // TOPK
    rn = tl.program_id(1) * 128 + tl.arange(0, 128)
    rk = tl.arange(0, 128)
    acc = tl.zeros((BM, 128), tl.float32)
    for group in range(tl.cdiv(LENGTH, 128)):
        local_k = group * 128 + rk
        if CODEC == "iq2_xxs":
            w = _decode(W, GRID, rn, START + local_k, N, K)
        elif CODEC == "q8_0":
            w = _decode_q8(W, rn, START + local_k, N, K)
        else:
            w = _decode_nvfp4(W, WS, WG, rn, START + local_k, N, K, WG_PER_ROW)
        w = tl.where(local_k[None, :] < LENGTH, w, 0.0)
        sw = tl.maximum(tl.div_rn(tl.max(tl.abs(w), 1), 127.0), 1.0e-30)
        wq = tldevice.nearbyint(tl.div_rn(w, sw[:, None])).to(tl.int8)
        x = tl.load(
            X + xm[:, None] * LENGTH + local_k[None, :],
            (rm[:, None] < M) & (local_k[None, :] < LENGTH),
            0,
        )
        sx = tl.load(XS + xm * tl.cdiv(LENGTH, 128) + group, rm < M, 0)
        dot = tl.dot(x, tl.trans(wq), out_dtype=tl.int32)
        acc += dot.to(tl.float32) * sx[:, None] * sw[None, :]
    if RELU2:
        acc = tl.maximum(acc, 0.0)
        acc *= acc
    if QUANT_OUT:
        acc = tl.where(rn[None, :] < N, acc, 0.0)
        sy = tl.maximum(tl.div_rn(tl.max(tl.abs(acc), 1), 127.0), 1.0e-30)
        q = tldevice.nearbyint(tl.div_rn(acc, sy[:, None])).to(tl.int8)
        tl.store(
            Y + rm[:, None] * N + rn[None, :], q, (rm[:, None] < M) & (rn[None, :] < N)
        )
        tl.store(YS + rm * tl.cdiv(N, 128) + tl.program_id(1), sy, rm < M)
    else:
        tl.store(
            Y + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )


def make_grid(device):
    from .iq2_xxs_grid import IQ2_XXS_GRID

    return torch.tensor(
        [v >> (8 * j) & 255 for v in IQ2_XXS_GRID for j in range(8)],
        dtype=torch.uint8,
        device=device,
    )


def quantize(x):
    if x.ndim != 2 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("Expected contiguous GPU activation matrix")
    m, k = x.shape
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty((m, triton.cdiv(k, 128)), device=x.device, dtype=torch.float32)
    _quant[(m, triton.cdiv(k, 128))](x, q, s, m, k)
    return q, s


def packed_gemm(
    q,
    scales,
    packed,
    grid,
    *,
    start=0,
    relu2=False,
    quant_out=False,
    codec="iq2_xxs",
    weight_scales=None,
    global_scale=None,
):
    """Multiply a logical K slice, including slices crossing source blocks.

    IQ2 uses [N, K/256, 66], Q8_0 [N, K/32, 34], NVFP4 [N, K/2].
    The activation matrix q has [M, local_K].
    Input and output activation groups use local (not global) coordinates.
    """
    if codec not in ("iq2_xxs", "q8_0", "nvfp4"):
        raise ValueError("Unsupported packed codec")
    if codec == "nvfp4":
        if packed.ndim != 2 or weight_scales is None or global_scale is None:
            raise ValueError("NVFP4 requires packed matrix and both scale tensors")
        n, half_k = packed.shape
        full_k = half_k * 2
        if (
            full_k % 16
            or weight_scales.shape != (n, full_k // 16)
            or weight_scales.dtype != torch.float8_e4m3fn
            or global_scale.numel() not in (1, n)
            or global_scale.dtype != torch.float32
        ):
            raise ValueError("Invalid NVFP4 scale geometry")
        ws = weight_scales.view(torch.uint8)
        wg = global_scale
    else:
        block_bytes, block_size = (66, 256) if codec == "iq2_xxs" else (34, 32)
        if packed.ndim != 3 or packed.shape[-1] != block_bytes:
            raise ValueError("Invalid packed block geometry")
        n, blocks, _ = packed.shape
        full_k = blocks * block_size
        ws, wg = scales, scales  # Unused pointers for codecs with inline scales.
    m, length = q.shape
    if (
        q.dtype != torch.int8
        or packed.dtype != torch.uint8
        or scales.dtype != torch.float32
        or grid.dtype != torch.uint8
        or scales.shape != (m, triton.cdiv(length, 128))
        or grid.numel() != 2048
        or start < 0
        or start + length > full_k
        or min(m, n, length) < 1
        or not all(
            t.is_cuda and t.is_contiguous() and t.device == q.device
            for t in (q, scales, packed, grid, ws, wg)
        )
    ):
        raise ValueError("Unsupported packed IQ2 G128 GEMM contract")
    y = torch.empty(
        (m, n), device=q.device, dtype=torch.int8 if quant_out else torch.float32
    )
    ys = torch.empty((m, triton.cdiv(n, 128)), device=q.device, dtype=torch.float32)
    _gemm[(triton.cdiv(m, 16), triton.cdiv(n, 128))](
        q,
        scales,
        packed,
        grid,
        ws,
        wg,
        y,
        ys,
        m,
        n,
        full_k,
        start,
        length,
        relu2,
        quant_out,
        16,
        codec,
        WG_PER_ROW=codec == "nvfp4" and wg.numel() != 1,
        num_warps=4,
        num_stages=1,
        matrix_instr_nonkdim=16,
        enable_fp_fusion=False,
    )
    return (y, ys) if quant_out else y


@triton.jit
def _embedding(
    W, IDS, OUT, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    ks = tl.program_id(1) * B + tl.arange(0, B)
    idx = tl.load(IDS + row)
    value = _decode_q8(W, tl.full((1,), 0, tl.int32) + idx, ks, N, K)
    tl.store(OUT + row * K + ks, tl.reshape(value, (B,)), ks < K)


def packed_embedding(weight, ids, width, dtype=torch.bfloat16):
    out = torch.empty((*ids.shape, width), device=weight.device, dtype=dtype)
    _embedding[(ids.numel(), triton.cdiv(width, 256))](
        weight, ids.contiguous(), out, ids.numel(), width, weight.shape[0], 256
    )
    return out


def packed_experts(q, scales, packed, grid, ids, *, start=0, down=False, plan=None):
    m, topk = ids.shape
    routes = m * topk
    n, blocks = packed.shape[1:3]
    length = q.shape[-1]
    y = torch.empty(
        (routes, n), device=q.device, dtype=torch.float32 if down else torch.int8
    )
    ys = torch.empty(
        (routes, triton.cdiv(n, 128)), device=q.device, dtype=torch.float32
    )
    sorted_ids, experts, count = (None, ids, None) if plan is None else plan
    tiles = routes if plan is None else experts.numel()
    _gemm[(tiles, triton.cdiv(n, 128))](
        q,
        scales,
        packed,
        grid,
        scales,
        scales,
        y,
        ys,
        routes,
        n,
        blocks * 256,
        start,
        length,
        not down,
        not down,
        16,
        "iq2_xxs",
        experts,
        topk,
        True,
        down,
        packed.stride(0),
        sorted_ids,
        count,
        plan is not None,
        num_warps=4,
        num_stages=1,
        matrix_instr_nonkdim=16,
        enable_fp_fusion=False,
    )
    return y if down else (y, ys)
