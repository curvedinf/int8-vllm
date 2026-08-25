# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused RMSNorm + per-token int8 dynamic quant for gfx908 (MI100).

Production W8A8 serving quantizes every post-norm activation with AITER
``pertoken_quant`` (per-row fp32 absmax/127 scale, trunc-toward-zero int8
cast) as a separate launch after the RMSNorm. This module collapses the
pair into one Triton kernel that reproduces the unfused chain bit-for-bit:

- sum-of-squares: fp32 ``tl.sum`` over one ``next_power_of_2(K)`` block,
  the same reduction structure as aiter's non-blocked ``_rms_norm_kernel``
  (``K <= 32768`` for fp16/bf16, which covers every model shape here);
- normed value: rounded once to the 16-bit dtype. Two arithmetic modes:
  aiter (fp32 ``x * rsqrt * w`` then one rounding, fp16 residual add) —
  bit-exact with the eager ``forward_hip`` chain; and native (ir-native
  ``fp16(x * rsqrt) * w`` with the 16-bit weight multiply, fp32 residual
  add) — matches the compiled ``vllm_ir`` baseline up to reduction order.
  The int8 stage re-loads the stored row (each block owns exactly one
  row, so an intra-CTA barrier makes this race-free): the quant consumes
  the stored bits literally, which both pins the reduction layout to the
  plain-norm kernel's (an in-register round-trip lets Triton's layout
  propagation widen the compute layout and reorder the ``tl.sum`` tree)
  and matches the eager chain, where the quant kernel reads the normed
  tensor from memory (bf16 inputs additionally round through the fp16
  copy the CK path makes);
- scale: fp32 ``absmax * float(1/127)`` with the 0 -> 1 guard — torch's
  scalar division lowers to a reciprocal multiply, so the kernel must
  too (true division diverges on ~4.5% of values);
- payload: IEEE division by the scale, trunc-toward-zero int8 cast.
"""

import torch
import triton
import triton.language as tl

from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

# Side-channel attribute: (q, scale) stashed on the normed tensor by the
# eager norm paths so CK linear consumers can skip the separate quant.
PREQUANT_ATTR = "_vllm_int8_prequant"


@triton.jit
def _rms_norm_pertoken_int8_quant_kernel(
    x_ptr,
    res_in_ptr,
    w_ptr,
    y_ptr,
    res_out_ptr,
    q_ptr,
    s_ptr,
    n_rows,
    n_cols,
    epsilon,
    recip_127,
    BLOCK_SIZE: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    QUANT_FROM_BF16: tl.constexpr,
    NATIVE_NUMERICS: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Cap norm-side accesses at 128-bit vectors. Without this, the int8
    # store's 16-element layout propagates through the whole kernel and
    # reorders the tl.sum tree relative to aiter's rms_norm kernel.
    col_hints = tl.max_contiguous(tl.multiple_of(col_offsets, 8), 8)
    mask = col_offsets < n_cols
    input_dtype = x_ptr.dtype.element_ty

    row_base = row * n_cols
    x = tl.load(
        x_ptr + row_base + col_hints, mask=mask, other=0.0, cache_modifier=".cg"
    )
    if HAS_RESIDUAL:
        res_in = tl.load(
            res_in_ptr + row_base + col_hints,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
        )
        if NATIVE_NUMERICS:
            # ir native: fp32 add, residual rounded from the exact sum,
            # sum-of-squares on the unrounded fp32 sum
            row_f32 = x.to(tl.float32) + res_in.to(tl.float32)
            tl.store(
                res_out_ptr + row_base + col_hints, row_f32.to(input_dtype), mask=mask
            )
        else:
            x += res_in
            tl.store(res_out_ptr + row_base + col_hints, x, mask=mask)
            row_f32 = x.to(tl.float32)
    else:
        row_f32 = x.to(tl.float32)

    row_norm = row_f32 * row_f32
    row_norm = tl.sum(row_norm, axis=-1)
    norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

    if NATIVE_NUMERICS:
        # ir native: round the normalized value to the 16-bit dtype
        # before the weight multiply, which runs in 16-bit arithmetic
        y16 = (row_f32 * norm_factor).to(input_dtype)
        w16 = tl.load(w_ptr + col_hints, mask=mask, other=0.0)
        y16 = y16 * w16
    else:
        # aiter: everything in fp32, one final rounding
        g = tl.load(w_ptr + col_hints, mask=mask, other=0.0).to(tl.float32)
        y16 = (row_f32 * norm_factor * g).to(input_dtype)
    tl.store(y_ptr + row_base + col_hints, y16, mask=mask)
    # The quant below must read back the stored row (see module docstring).
    tl.debug_barrier()

    y_reload = tl.load(
        y_ptr + row_base + col_hints, mask=mask, other=0.0, cache_modifier=".cg"
    )
    if QUANT_FROM_BF16:
        # bf16 serving: the CK path quantizes the fp16 copy of the normed
        # tensor (x_2d.to(torch.float16)); replay that conversion exactly.
        y_q = y_reload.to(tl.float16).to(tl.float32)
    else:
        y_q = y_reload.to(tl.float32)

    amax = tl.max(tl.abs(y_q), axis=-1)
    # aiter pertoken_quant computes the scale as a torch scalar division
    # (amax / 127), which lowers to a multiply by the fp32 reciprocal —
    # not true division. Pass the reciprocal as a runtime arg so the
    # compiler cannot turn it back into a division.
    scale = amax * recip_127
    scale = tl.where(scale == 0.0, 1.0, scale)
    q = tl.math.div_rn(y_q, scale).to(tl.int8)

    tl.store(q_ptr + row_base + col_offsets, q, mask=mask)
    tl.store(s_ptr + row, scale)


def _launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor | None,
    native_numerics: bool = False,
) -> tuple[torch.Tensor, ...]:
    assert x.ndim == 2 and x.is_contiguous()
    assert weight.is_contiguous()
    n_rows, n_cols = x.shape
    # Single-block reduction (aiter non-blocked regime); 65536-byte row cap.
    block_size = min(65536 // x.element_size(), triton.next_power_of_2(n_cols))
    assert block_size >= n_cols, f"K={n_cols} exceeds single-block rms_norm limit"

    y = torch.empty_like(x)
    res_out = torch.empty_like(x) if residual is not None else x
    q = torch.empty(n_rows, n_cols, dtype=torch.int8, device=x.device)
    scale = torch.empty(n_rows, 1, dtype=torch.float32, device=x.device)

    grid = (n_rows,)
    # Triton launches on its active device, not the tensors'; pin it so a
    # process that has touched another GPU first cannot fault here.
    with torch.cuda.device(x.device):
        _rms_norm_pertoken_int8_quant_kernel[grid](
            x,
            residual if residual is not None else x,
            weight,
            y,
            res_out,
            q,
            scale,
            n_rows,
            n_cols,
            epsilon,
            1.0 / 127.0,
            BLOCK_SIZE=block_size,
            HAS_RESIDUAL=residual is not None,
            QUANT_FROM_BF16=x.dtype == torch.bfloat16,
            NATIVE_NUMERICS=native_numerics,
            num_warps=4,
        )
    return y, res_out, q, scale


def rms_norm_int8_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor | None = None,
    native_numerics: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Eager entry: one launch producing the normed tensor and its quant.

    Returns ``(normed, q, scale)`` or, with a residual,
    ``(normed, residual_out, q, scale)``. ``native_numerics`` selects the
    ir-native arithmetic (compiled/eager-native chain); the default
    mirrors aiter's rms_norm.
    """
    x = x.contiguous()
    weight = weight.contiguous()
    if residual is not None:
        residual = residual.contiguous()
    y, res_out, q, scale = _launch(x, weight, epsilon, residual, native_numerics)
    if residual is not None:
        return y, res_out, q, scale
    return y, q, scale


def _rocm_rms_norm_int8_quant_impl(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compiled-graph replacement: mirror the ir native arithmetic
    # (16-bit weight multiply, fp32 residual add) so the fused op stays
    # within reduction-order noise of the inductor baseline.
    _, _, q, scale = _launch(
        x.contiguous(), weight.contiguous(), epsilon, None, native_numerics=True
    )
    return q, scale


def _rocm_rms_norm_int8_quant_fake(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(x.shape, dtype=torch.int8, device=x.device),
        torch.empty(x.shape[:-1] + (1,), dtype=torch.float32, device=x.device),
    )


def _rocm_rms_norm_add_int8_quant_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, res_out, q, scale = _launch(
        x.contiguous(),
        weight.contiguous(),
        epsilon,
        residual.contiguous(),
        native_numerics=True,
    )
    return q, scale, res_out


def _rocm_rms_norm_add_int8_quant_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(x.shape, dtype=torch.int8, device=x.device),
        torch.empty(x.shape[:-1] + (1,), dtype=torch.float32, device=x.device),
        torch.empty_like(x),
    )


if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="rocm_rms_norm_int8_quant",
        op_func=_rocm_rms_norm_int8_quant_impl,
        fake_impl=_rocm_rms_norm_int8_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    direct_register_custom_op(
        op_name="rocm_rms_norm_add_int8_quant",
        op_func=_rocm_rms_norm_add_int8_quant_impl,
        fake_impl=_rocm_rms_norm_add_int8_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )
