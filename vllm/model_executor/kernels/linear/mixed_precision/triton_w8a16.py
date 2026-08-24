# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton-based W8A16 GEMM kernel for ROCm gfx908 (MI100).

Forked from triton_w4a16.py for W8 (uint8b128). Key differences vs W4:
  - 4 weights per int32 (vs 8); shifts [0,8,16,24] (vs [0,4,...,28])
  - mask 0xFF (vs 0xF); 2 interleaves (vs 3)
  - qweight column dim is N//4 (vs N//8); zeros also N//4

Plugs into the MPLinearKernel selection system. Required for dense
GPTQ-8bit on gfx908 — the default ops.gptq_gemm path runs at 2-7× off
HBM bandwidth for typical decode shapes; this kernel uses tl.dot
(MFMA) and is bandwidth-limited.

Weight layout expected by this kernel (post-process_weights_after_loading):
  qweight: [K, N//4]  int32  — rows=K (input), cols=N//4 (N is packed)
  scales:  [K//G, N]  fp16/bf16
  qzeros:  [K//G, N//4]  int32  (optional; None for symmetric uint8b128)

Checkpoint layout from GPTQ create_weights (PackedvLLMParameter):
  qweight: [K//4, N]  int32  (input_dim=0, output_dim=1, packed_dim=0; K packed)
  scales:  [K//G, N]  fp16   (output_dim=1, input_dim=0)
  qzeros:  [K//G, N//4]  int32 (output_dim=1, packed_dim=1)
"""

import os

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

TRITON_W8A16_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128, 256]
TRITON_W8A16_SUPPORTED_QUANT_TYPES = [
    scalar_types.uint8b128,  # symmetric GPTQ-8bit (bias=128)
]


@triton.jit
def triton_w8a16_decode_kernel(
    # M=1 decode-specialized W8A16 kernel for gfx908.
    # No MFMA (tl.dot wastes 15/16 lanes for M=1); pure scalar reduction.
    # Split-K via atomicAdd to maximize CU utilization.
    a_ptr,           # [K] fp16 (single row of activations)
    b_ptr,           # [K, N//4] int32 packed weights (N-packed, 4 int8 per int32)
    scales_ptr,      # [K//G, N] fp16
    zeros_ptr,       # [K//G, N//4] int32 (unused when HAS_ZP=False)
    c_ptr,           # [N] fp16 output (initialized to 0; we atomicAdd)
    N, K,
    stride_bk, stride_bn,
    group_size,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    ZERO_OFFSET: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Decode-only W8A16 GEMM with split-K + atomicAdd reduction.

    Grid: (cdiv(N, BLOCK_N), SPLIT_K)
    Each program reduces one (BLOCK_N) × (K_chunk = K // SPLIT_K) tile.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # b/zeros: N//4 int32 columns per K row
    offs_bn = pid_n * (BLOCK_N // 4) + tl.arange(0, BLOCK_N // 4)
    mask_bn = offs_bn < (N // 4)

    # 8-bit shifts tiled across BLOCK_N: [0,8,16,24] repeating
    shifts_row = tl.arange(0, 4) * 8  # [4]
    shifts_1d_2d = tl.broadcast_to(shifts_row[None, :], (BLOCK_N // 4, 4))
    shifts_1d = tl.reshape(shifts_1d_2d, (BLOCK_N,))

    # Determine this program's K range
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start_offset = pid_k * k_per_split
    k_end = tl.minimum(k_start_offset + k_per_split, K)

    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    k = k_start_offset
    while k < k_end:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_end

        # Load activation row segment: [BLOCK_K] fp16
        a = tl.load(a_ptr + offs_k, mask=mask_k, other=0.0)
        a_f32 = a.to(tl.float32)

        # Load packed weight tile: [BLOCK_K, BLOCK_N//4] int32
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        b_packed = tl.load(b_ptrs, mask=mask_k[:, None] & mask_bn[None, :], other=0)

        # Unpack int8: 2 interleaves expand last dim by 4
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = (b >> shifts_1d[None, :]) & 0xFF  # [BLOCK_K, BLOCK_N]

        # Load scales (one group per BLOCK_K iter; BLOCK_K <= group_size)
        g_idx = k // group_size
        scales = tl.load(scales_ptr + g_idx * N + offs_n, mask=mask_n, other=1.0)

        # Dequant + accumulate
        if HAS_ZP:
            zero_offset = g_idx * (N // 4) + offs_bn
            z_packed = tl.load(zeros_ptr + zero_offset, mask=mask_bn, other=0)
            z = tl.interleave(z_packed, z_packed)
            z = tl.interleave(z, z)
            z = (z >> shifts_1d) & 0xFF  # [BLOCK_N]
            b_centered_f32 = (
                b.to(tl.float32) - (z[None, :] + ZERO_OFFSET).to(tl.float32)
            )
        else:
            b_centered_f32 = (b.to(tl.float32) - ZP_BIAS)

        # Multiply by scale, then reduce K and accumulate
        # b_centered_f32: [BLOCK_K, BLOCK_N]; scales[None, :]: [1, BLOCK_N]
        b_fp = b_centered_f32 * scales[None, :].to(tl.float32)
        # a_f32[:, None] * b_fp → [BLOCK_K, BLOCK_N]; sum(0) → [BLOCK_N]
        accumulator += tl.sum(a_f32[:, None] * b_fp, axis=0)

        k += BLOCK_K

    # Atomic reduce into output (split-K)
    if SPLIT_K == 1:
        tl.store(c_ptr + offs_n, accumulator.to(c_ptr.type.element_ty), mask=mask_n)
    else:
        tl.atomic_add(c_ptr + offs_n, accumulator.to(c_ptr.type.element_ty), mask=mask_n)


@triton.jit
def triton_w8a16_gemm_kernel(
    a_ptr,
    b_ptr,
    scales_ptr,
    zeros_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    group_size,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    ZERO_OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused W8A16 GEMM: C[M,N] = A[M,K] @ dequant(B)[K,N]

    B is stored as [K, N//4] int32 using GPTQ sequential 8-bit packing:
      each int32 packs 4 consecutive N-values at bit offsets [0,8,16,24].

    Dequant: w_fp = (w_int8 - zero) * scale
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # b/zeros: N//4 int32 columns per K row
    offs_bn = pid_n * (BLOCK_N // 4) + tl.arange(0, BLOCK_N // 4)

    # GPTQ sequential 8-bit shifts tiled across BLOCK_N:
    #   [0,8,16,24] repeating for every group of 4 N-values.
    shifts_row = tl.arange(0, 4) * 8  # [4]
    shifts_1d_2d = tl.broadcast_to(shifts_row[None, :], (BLOCK_N // 4, 4))
    shifts_1d = tl.reshape(shifts_1d_2d, (BLOCK_N,))  # [BLOCK_N]
    shifts = tl.broadcast_to(shifts_1d[None, :], (BLOCK_K, BLOCK_N))

    # Scales column offsets: full N-width
    offs_sn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load activations A: [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # Load packed weights B: [BLOCK_K, BLOCK_N//4] int32
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        mask_b = mask_k[:, None] & (offs_bn[None, :] < N // 4)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

        # Unpack int8 weights → [BLOCK_K, BLOCK_N]
        # Two interleaves multiply last dim by 4 (vs 3 for W4 / by 8).
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = (b >> shifts) & 0xFF

        # Group row index for this K tile
        g_idx = (k_start * BLOCK_K) // group_size

        # Load scales: [BLOCK_N] → broadcast to [BLOCK_K, BLOCK_N]
        scale_offset = g_idx * N + offs_sn
        scale_mask = offs_sn < N
        scales = tl.load(scales_ptr + scale_offset, mask=scale_mask, other=1.0)
        scales = tl.broadcast_to(scales[None, :], (BLOCK_K, BLOCK_N))

        if HAS_ZP:
            # Load packed zeros row: [BLOCK_N//4] int32
            zero_offset = g_idx * (N // 4) + offs_bn
            zero_mask = offs_bn < N // 4
            z_packed = tl.load(zeros_ptr + zero_offset, mask=zero_mask, other=0)
            z = tl.interleave(z_packed, z_packed)
            z = tl.interleave(z, z)
            z = (z >> shifts_1d) & 0xFF
            z += ZERO_OFFSET
            z = tl.broadcast_to(z[None, :], (BLOCK_K, BLOCK_N))
        else:
            z = tl.full((BLOCK_K, BLOCK_N), ZP_BIAS, dtype=tl.int32)

        b_fp = (b - z).to(a.dtype) * scales

        accumulator += tl.dot(a, b_fp, out_dtype=tl.float32)

    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


@triton.jit
def triton_w8a8_gemm_kernel(
    a_ptr,          # [M, K] int8 activations (per-128-block quantized)
    b_ptr,          # [K, N//4] int32 GPTQ-packed weights
    a_scales_ptr,   # [M, K//128] fp16 per-block activation scales
    b_scales_ptr,   # [K//G, N] fp16 weight scales
    zeros_ptr,      # packed zeros or None
    c_ptr,          # [M, N] fp16 out
    M,
    N,
    K,
    stride_am,
    stride_ask,     # a_scales row stride (K//128)
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    group_size,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    ZERO_OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,   # must equal group_size (== 128) — one scale pair per tile
):
    """True INT8 GEMM: C = (A_q * as) @ (B_q - z)^T * bs per K-block.

    Both A and B carry one scale per 128-wide K block. With BLOCK_K == 128 ==
    group_size, each K-tile needs exactly one A-block scale and one B-group
    scale; the int8xint8 dot accumulates exactly one block of products, so
    descaling once per tile is exact (no cross-block integer overflow: max
    |sum| = 128 * 127 * 128 < 2^31).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_bn = pid_n * (BLOCK_N // 4) + tl.arange(0, BLOCK_N // 4)
    offs_sn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    shifts_row = tl.arange(0, 4) * 8
    shifts_1d_2d = tl.broadcast_to(shifts_row[None, :], (BLOCK_N // 4, 4))
    shifts_1d = tl.reshape(shifts_1d_2d, (BLOCK_N,))
    shifts = tl.broadcast_to(shifts_1d[None, :], (BLOCK_K, BLOCK_N))

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # A: int8 activations [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :]
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a_q = tl.load(a_ptrs, mask=mask_a, other=0)

        # B: packed weights [BLOCK_K, BLOCK_N//4] -> int8 [BLOCK_K, BLOCK_N]
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        mask_b = mask_k[:, None] & (offs_bn[None, :] < N // 4)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)
        b_q = tl.interleave(b_packed, b_packed)
        b_q = tl.interleave(b_q, b_q)
        b_q = (b_q >> shifts) & 0xFF
        b_q = (b_q - ZP_BIAS).to(tl.int8) if not HAS_ZP else b_q

        g_idx = k_start  # BLOCK_K == group_size -> tile index == group index

        # zeros (if symmetric-with-bias layout: uint8b128, w = q - 128)
        if HAS_ZP:
            zero_offset = g_idx * (N // 4) + offs_bn
            zero_mask = offs_bn < N // 4
            z_packed = tl.load(zeros_ptr + zero_offset, mask=zero_mask, other=0)
            z = tl.interleave(z_packed, z_packed)
            z = tl.interleave(z, z)
            z = ((z >> shifts_1d) & 0xFF) + ZERO_OFFSET
            z = tl.broadcast_to(z[None, :], (BLOCK_K, BLOCK_N))
            b_q = (b_q - z).to(tl.int8)

        # per-block scales: A [BLOCK_M], B [BLOCK_N]
        a_s = tl.load(
            a_scales_ptr + offs_m * stride_ask + g_idx,
            mask=offs_m < M, other=1.0,
        )  # [BLOCK_M]
        b_s = tl.load(
            b_scales_ptr + g_idx * N + offs_sn,
            mask=offs_sn < N, other=1.0,
        )  # [BLOCK_N]

        acc_i32 = tl.dot(a_q, b_q, out_dtype=tl.int32)  # exact int8xint8
        accumulator += acc_i32.to(tl.float32) * (a_s[:, None] * b_s[None, :])

    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


def _quantize_activation_per_block(x: torch.Tensor, block_k: int = 128):
    """Dynamic per-row per-128-K-block int8 quantization (matches sweep sim).

    Returns (x_q [M,K] int8, scales [M, K//block_k] fp16)."""
    M, K = x.shape
    assert K % block_k == 0
    xb = x.reshape(M, K // block_k, block_k)
    absmax = xb.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
    x_q = (xb / scale).clamp_(-128, 127).round_().to(torch.int8)
    return x_q.reshape(M, K), scale.squeeze(-1).to(torch.float16)


def _pick_block_sizes(M: int, N: int, K: int, group_size: int):
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx1x

        if on_gfx1x():
            # gfx1x branch from W4 kernel (RDNA, 32-wide wavefronts).
            if M <= 32:
                return 32, 32, 64
            if M <= 64:
                return 64, 64, 32
            return 128, 32, 64

        # Detect gfx908 via on_gfx908() (set in vllm/platforms/rocm.py)
        try:
            from vllm.platforms.rocm import on_gfx908
            is_gfx908 = on_gfx908()
        except Exception:
            is_gfx908 = False

        if is_gfx908:
            # gfx908: 120 CUs, 64-wide wavefronts, ~1.2 TB/s HBM2.
            # For M=1 decode (padded to BLOCK_M=16 for MFMA), favor more
            # N-tiles to saturate CUs.
            if M <= 16:
                return 16, 64, 32   # decode hot path
            if M <= 32:
                return 32, 64, 32
            if M <= 64:
                return 64, 64, 32
            return 128, 64, 32

        # MI300/gfx942 default
        if M <= 32:
            return 32, 64, 32
        if M <= 64:
            return 64, 64, 32
        return 128, 128, 32

    # Non-ROCm fallback
    if M <= 32:
        return 32, 64, 32
    if M <= 64:
        return 64, 64, 32
    return 128, 128, 32


def triton_w8a16_decode(
    a: torch.Tensor,           # [1, K] fp16/bf16 (M=1 only)
    b_q: torch.Tensor,         # [K, N//4] int32
    scales: torch.Tensor,      # [K//G, N] fp16/bf16
    qzeros: torch.Tensor | None,
    group_size: int,
    zp_bias: int = 128,
    zero_offset: int = 0,
    split_k: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Decode-specialized W8A16 (M=1) using split-K + atomicAdd reduction.

    Optimized for gfx908 + decode hot path. Avoids tl.dot (no MFMA waste at
    M=1) and uses split-K to keep all 120 CUs busy.
    """
    assert a.shape[0] == 1, f"decode kernel requires M=1, got M={a.shape[0]}"
    M, K = a.shape
    N = b_q.shape[1] * 4

    # Pick split-K to target ~120 programs total (gfx908 CU count)
    if block_n is None:
        block_n = 64
    if block_k is None:
        block_k = min(group_size, 32)
    if split_k is None:
        n_tiles = (N + block_n - 1) // block_n
        target_programs = 120
        split_k = max(1, min(8, target_programs // max(1, n_tiles)))
        # Round split_k to a divisor of K/block_k for clean partitioning
        while split_k > 1 and (K // split_k) % block_k != 0:
            split_k -= 1

    # Output initialized to 0 (atomicAdd accumulates)
    c = torch.zeros((1, N), dtype=a.dtype, device=a.device)

    has_zp = qzeros is not None
    zeros_ptr = qzeros if has_zp else b_q  # dummy ptr when unused

    grid = (triton.cdiv(N, block_n), split_k)
    triton_w8a16_decode_kernel[grid](
        a, b_q, scales, zeros_ptr, c,
        N, K,
        b_q.stride(0), b_q.stride(1),
        group_size=group_size,
        HAS_ZP=has_zp,
        ZP_BIAS=zp_bias,
        ZERO_OFFSET=zero_offset,
        SPLIT_K=split_k,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


def triton_w8a16_gemm(
    a: torch.Tensor,
    b_q: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    group_size: int,
    zp_bias: int = 128,
    zero_offset: int = 0,
) -> torch.Tensor:
    """Fused W8A16 GEMM using GPTQ-packed int8 weights.

    Args:
        a:          [M, K] fp16/bf16 activations.
        b_q:        [K, N//4] int32 packed weights (4 int8 per int32).
        scales:     [K//G, N] fp16/bf16 per-group scales.
        qzeros:     [K//G, N//4] int32 packed zero points, or None for
                    symmetric (uses zp_bias=128 for uint8b128).
        group_size: Group size (resolve -1 → K before calling).
        zp_bias:    Constant zero used when qzeros is None (default 128).
    """
    assert a.is_contiguous(), "a must be contiguous"
    assert b_q.is_contiguous(), "b_q must be contiguous"
    assert scales.is_contiguous(), "scales must be contiguous"

    M, K = a.shape
    N = b_q.shape[1] * 4

    assert b_q.shape == (K, N // 4), (
        f"b_q shape mismatch: {b_q.shape} vs ({K}, {N // 4})"
    )
    assert scales.shape == (K // group_size, N), (
        f"scales shape mismatch: {scales.shape} vs ({K // group_size}, {N})"
    )
    if qzeros is not None:
        assert qzeros.shape == (K // group_size, N // 4), (
            f"qzeros shape mismatch: {qzeros.shape}"
        )

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    has_zp = qzeros is not None
    zeros_ptr = qzeros if has_zp else b_q  # dummy ptr when unused

    BLOCK_M, BLOCK_N, BLOCK_K = _pick_block_sizes(M, N, K, group_size)

    # The kernel loads scales/zeros for one group per BLOCK_K tile.
    # Clamp BLOCK_K to group_size so each tile sees one scale group.
    if group_size < BLOCK_K:
        BLOCK_K = group_size

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    triton_w8a16_gemm_kernel[grid](
        a, b_q, scales, zeros_ptr, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_q.stride(0), b_q.stride(1),
        c.stride(0), c.stride(1),
        group_size=group_size,
        HAS_ZP=has_zp,
        ZP_BIAS=zp_bias,
        ZERO_OFFSET=zero_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


def triton_w8a8_gemm(
    a: torch.Tensor,          # [M, K] fp16 raw activations
    b_q: torch.Tensor,        # [K, N//4] int32 packed weights
    scales: torch.Tensor,     # [K//G, N] weight scales
    qzeros: torch.Tensor | None,
    group_size: int,          # must be 128 (or K for -1)
    zp_bias: int = 128,
    zero_offset: int = 0,
) -> torch.Tensor:
    """A8W8: dynamic per-128-block activation quant + true int8xint8 dot.

    Requires group_size == 128 (kernel assumes one weight group + one
    activation block per BLOCK_K=128 tile)."""
    assert group_size == 128, f"W8A8 requires gs=128, got {group_size}"
    M, K = a.shape
    N = b_q.shape[1] * 4
    a_q, a_s = _quantize_activation_per_block(a, block_k=128)

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    has_zp = qzeros is not None
    zeros_ptr = qzeros if has_zp else b_q

    # int8 dot wants BLOCK_M/BLOCK_N >= 16. Decode (M<=16) pads to one
    # BLOCK_M=16 tile: the wasted rows cost ~nothing on the MFMA pipe while
    # halving the activation read vs W8A16; narrow-N shapes keep BLOCK_N=64
    # for CU occupancy on gfx908's 120 CUs.
    if M <= 16:
        BLOCK_M, BLOCK_N = 16, 128
    else:
        BLOCK_M, BLOCK_N = 64, 128
    BLOCK_K = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    triton_w8a8_gemm_kernel[grid](
        a_q, b_q, a_s, scales, zeros_ptr, c,
        M, N, K,
        a_q.stride(0), a_s.stride(0),
        b_q.stride(0), b_q.stride(1),
        c.stride(0), c.stride(1),
        group_size=group_size,
        HAS_ZP=has_zp,
        ZP_BIAS=zp_bias,
        ZERO_OFFSET=zero_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


_W8A8_DISPATCH_MIN_M = 256  # decode (M<256) stays on W8A16 (bandwidth-bound)
# W8A8 only wins on the fat GEMMs (microbench 2026-08-22: 17408-wide MLP up/gate
# 1.62x, 5120-wide 0.81-0.95x, small-N 0.61x). Gate on N too — the win scales
# with arithmetic intensity, and narrow GEMMs pay the quant overhead back.
_W8A8_DISPATCH_MIN_N = 8192


class TritonW8A16LinearKernel(MPLinearKernel):
    """Triton W8A16 GEMM kernel for ROCm (gfx908 / gfx942)."""

    SUPPORTED_QUANT_TYPES = TRITON_W8A16_SUPPORTED_QUANT_TYPES
    use_v2_format: bool = False

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "TritonW8A16LinearKernel only targets ROCm"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type {c.weight_type} not supported; "
                f"supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "Only float16/bfloat16 activations are supported"

        N = c.partition_weight_shape[1]
        if N % 4 != 0:
            return (
                False,
                f"Output features ({N}) must be divisible by 4 "
                "(4 int8 values packed per int32)",
            )

        if c.has_g_idx:
            return False, "Activation reordering (g_idx) not supported"

        gs = c.group_size
        if (
            gs not in TRITON_W8A16_SUPPORTED_GROUP_SIZES
            and gs != c.full_weight_shape[0]
        ):
            return (
                False,
                f"Group size {gs} not supported; "
                f"supported: {TRITON_W8A16_SUPPORTED_GROUP_SIZES}",
            )

        K = c.partition_weight_shape[0]
        eff_gs = gs if gs != -1 else K
        if K % eff_gs != 0:
            return False, f"Input features {K} not divisible by group_size {eff_gs}"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert GPTQ checkpoint layout to kernel layout.

        Checkpoint qweight (PackedvLLMParameter):
          [K//4, N]  int32  input_dim=0, output_dim=1, packed_dim=0
        Kernel needs:
          [K, N//4]  int32  K at dim 0, N packed at dim 1
        """
        self._process_gptq_layout(layer)

        # #51581-class fix hook: consumers that slice raw rows off
        # layer.weight (DFlash's fused context-KV projection) need a dense
        # dequant of the kernel-layout packed weights. Computed lazily once,
        # cached on the layer; int8 bias (uint8b128) folded in.
        def _dequant_rows(w_packed_or_param) -> torch.Tensor:
            w_q, w_s, w_zp, _ = self._get_weight_params(layer)
            # w_q: [K, N//4] int32 (kernel layout), w_s: [K//G, N]
            K, N4 = w_q.shape
            N = N4 * 4
            shifts = torch.arange(4, device=w_q.device, dtype=torch.int32) * 8
            w = ((w_q.unsqueeze(-1) >> shifts) & 0xFF).reshape(K, N)  # [K, N]
            w = w.to(torch.int16) - 128  # uint8b128 bias
            gs = self.config.group_size if self.config.group_size != -1 else K
            w = w.view(K // gs, gs, N).to(w_s.dtype)
            w = w * w_s.unsqueeze(1)  # [K//G, 1, N] broadcasts over gs
            return w.reshape(K, N).t().contiguous()  # dense [N, K]

        layer.dequant_kv_rows = _dequant_rows

        if (
            os.environ.get("VLLM_GFX908_CK_W8A8", "1") == "1"
            and current_platform.is_rocm()
            and layer.__class__.__name__ not in ("RowParallelLinear",)
            and getattr(layer, "input_size_per_partition", 0)
            == layer.input_size  # unsharded only: CK path is per-rank whole-K
        ):
            # aiter CK int8 W8A8 (gfx908): requantize the GPTQ weight from
            # per-128-K-group scales to per-output-channel int8 + [N,1] fp32
            # scales — the gemm_a8w8_CK contract (x side is per-token [M,1],
            # quantized at runtime by the same pertoken semantics). The
            # per-channel requant strictly coarsens the weight quant; KLD-gate
            # before making this the default off the sweep values.
            try:
                dense = _dequant_rows(None).float()  # [N, K]
                wmax = dense.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
                w_s_ch = (wmax / 127.0).to(torch.float32)
                w_q_ch = (dense / w_s_ch).round().clamp(-127, 127).to(torch.int8)
                layer._ck_w8a8_q = w_q_ch.contiguous()
                layer._ck_w8a8_s = w_s_ch.contiguous()
            except Exception:
                # Never break the dequant hook / normal serving over an
                # optional requant.
                layer.__dict__.pop("_ck_w8a8_q", None)
                layer.__dict__.pop("_ck_w8a8_s", None)

    def _process_gptq_layout(self, layer: torch.nn.Module) -> None:

        def repack_w_q(x: BasevLLMParameter) -> BasevLLMParameter:
            # Bring to [N, K//4] (output at dim 0, K packed at dim 1)
            permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=1)
            w = x.data  # [N, K//4] int32

            N_dim, K4 = w.shape
            K_dim = K4 * 4
            # Unpack to [N, K] int32
            shifts = torch.arange(4, device=w.device, dtype=torch.int32) * 8
            w_unpacked = ((w.unsqueeze(-1) >> shifts) & 0xFF).reshape(N_dim, K_dim)
            # Transpose to [K, N]
            w_KN = w_unpacked.t().contiguous()
            # Repack N into N//4 int32 → [K, N//4]
            N4 = N_dim // 4
            w_repacked = torch.sum(
                (w_KN.view(K_dim, N4, 4) & 0xFF) << shifts,
                dim=2,
                dtype=torch.int32,
            )
            x.data = w_repacked.contiguous()
            return x

        def repack_w_s(x: BasevLLMParameter) -> BasevLLMParameter:
            # [N, K//G] → [K//G, N]
            permute_param_layout_(x, input_dim=1, output_dim=0)
            x.data = x.data.t().contiguous()
            return x

        self._transform_param(layer, self.w_q_name, repack_w_q)
        self._transform_param(layer, self.w_s_name, repack_w_s)

        if self.w_zp_name is not None:
            zp = getattr(layer, self.w_zp_name, None)
            if zp is not None:
                # The vLLM loader leaves GPTQ zero points in kernel layout:
                # [K//G, N//4]. Keep that layout; transposing breaks TP-sharded
                # merged projections such as Qwen GDN in_proj_qkvz.
                replace_parameter(
                    layer,
                    self.w_zp_name,
                    torch.nn.Parameter(zp.data.contiguous(), requires_grad=False),
                )

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        K = c.partition_weight_shape[0]
        group_size = c.group_size if c.group_size != -1 else K

        zp_bias = c.weight_type.bias if c.weight_type.has_bias() else 0

        if (
            os.environ.get("VLLM_GFX908_CK_W8A8", "1") == "1"
            and x_2d.shape[0] <= 64
            and hasattr(layer, "_ck_w8a8_q")
            and current_platform.is_rocm()
        ):
            # aiter CK int8 (gfx908): per-token activation quant + per-channel
            # weight; the 6.8x @ M=8 microbench lives here. Weights were
            # requantized per-channel in process_weights_after_loading.
            from aiter import gemm_a8w8_CK
            from aiter.ops.quant import pertoken_quant

            a_q, a_s = pertoken_quant(
                x_2d.to(torch.float16), quant_dtype=torch.int8
            )
            output = gemm_a8w8_CK(
                a_q,
                layer._ck_w8a8_q,
                a_s,
                layer._ck_w8a8_s,
                None,
                x_2d.dtype,
            )
        elif (
            x_2d.shape[0] <= 8
            and x_2d.dtype == torch.float16
            and c.weight_type == scalar_types.uint8b128
            and w_zp is not None
            and hasattr(torch.ops._C, "gptq_w8a16_repacked_gemm")
        ):
            output = ops.gptq_w8a16_repacked_gemm(
                x_2d,
                w_q,
                w_zp,
                w_s,
                self.use_v2_format,
            )
        elif (
            x_2d.shape[0] >= _W8A8_DISPATCH_MIN_M
            and w_q.shape[1] * 4 >= _W8A8_DISPATCH_MIN_N
            and group_size == 128
            and current_platform.is_rocm()
        ):
            # Large-M prefill: true int8xint8 compute (2x MFMA rate at half
            # the activation bandwidth). Decode stays on the paths above —
            # it's bandwidth-bound, so activation quant only adds overhead.
            output = triton_w8a8_gemm(
                a=x_2d,
                b_q=w_q,
                scales=w_s,
                qzeros=w_zp,
                group_size=group_size,
                zp_bias=zp_bias,
            )
        else:
            output = triton_w8a16_gemm(
                a=x_2d,
                b_q=w_q,
                scales=w_s,
                qzeros=w_zp,
                group_size=group_size,
                zp_bias=zp_bias,
                zero_offset=0,
            )

        if bias is not None:
            output.add_(bias)

        return output.reshape(out_shape)
