# SPDX-License-Identifier: Apache-2.0
"""Direct producer/consumer interfaces for GS128 decoder activations.

Attention may opt into its original floating-point weight/compute dtype.
Other projections require resident INT8 weights and consume codes/scales.
"""

import torch
import triton.language.extra.libdevice as lib

from vllm.model_executor.layers.residual_int8 import ResidualInt8, materialize
from vllm.triton_utils import tl, triton


def quantize_input(value):
    """Explicit ingress from embeddings or a reference-dtype attention interior."""
    if isinstance(value, ResidualInt8):
        return value
    from vllm.model_executor.layers.quantization.utils.packed_g128_int8 import quantize

    q, s = quantize(value.reshape(-1, value.shape[-1]).contiguous())
    return ResidualInt8(
        q.view(value.shape),
        s.view(*value.shape[:-1], triton.cdiv(value.shape[-1], 128)),
        value.dtype,
    )


def projection(layer, value, *, relu2=False, attention=False, activation=None):
    """GEMM epilogue emits codes/scales; TP reduction retains that representation."""
    from aiter.ops.triton.gemm.basic.packed_dense_cache import int8_gemm

    shape = value.shape
    dtype = value.compute_dtype if isinstance(value, ResidualInt8) else value.dtype
    bias = getattr(layer, "bias", None)
    tp_size = getattr(layer, "tp_size", 1)
    row_parallel = hasattr(layer, "reduce_results")
    if row_parallel and tp_size > 1 and getattr(layer, "tp_rank", 0) != 0:
        bias = None
    if row_parallel and not layer.input_is_parallel:
        raise ValueError("Direct GS128 projection requires a partitioned input")
    if hasattr(layer, "weight_gs128"):
        value = quantize_input(value)
        q, s = value.flatten_rows()
        if activation is not None:
            if relu2 or activation not in ("silu", "silu_mul"):
                raise ValueError("Unsupported fused INT8 projection activation")
            from aiter.ops.triton.gemm.basic.gemm_silu_quant_out import (
                gemm_silu_quant_out,
            )

            out, scales = gemm_silu_quant_out(
                q,
                s,
                layer.weight,
                layer.weight_gs128,
                gated=activation == "silu_mul",
                bias=bias,
                dtype=dtype,
            )
        else:
            out, scales = int8_gemm(
                q,
                s,
                layer.weight,
                layer.weight_gs128,
                relu2=relu2,
                quant_out=True,
                fused_quant_out=True,
                bias=bias,
            )
    elif attention:
        if relu2 or activation is not None:
            raise ValueError(
                "Attention projection cannot apply a feed-forward activation"
            )
        from aiter.ops.triton.gemm.basic.gemm_float_quant_out import (
            gemm_float_quant_out,
        )

        x = materialize(value).reshape(-1, shape[-1]).contiguous()
        out, scales = gemm_float_quant_out(x, layer.weight, bias)
    else:
        raise ValueError("Decoder INT8 projection requires resident GS128 weights")
    width = out.shape[-1]
    result = ResidualInt8(
        out.view(*shape[:-1], width),
        scales.view(*shape[:-1], triton.cdiv(width, 128)),
        dtype,
    )
    if row_parallel and layer.reduce_results and tp_size > 1:
        from vllm.distributed.device_communicators.residual_g128_all_reduce import (
            all_reduce,
        )

        result = all_reduce(result)
    return result


@triton.jit
def _combine(Q, S, W, OUT, OS, K: tl.constexpr, TOPK: tl.constexpr):
    token, group = tl.program_id(0), tl.program_id(1)
    cols = group * 128 + tl.arange(0, 128)
    total = tl.zeros((128,), tl.float32)
    for r in range(TOPK):
        row = token * TOPK + r
        value = tl.load(Q + row * K + cols).to(tl.float32)
        value *= tl.load(S + row * (K // 128) + group)
        total += value * tl.load(W + row)
    scale = tl.maximum(tl.div_rn(tl.max(tl.abs(total), 0), 127.0), 1.0e-30)
    codes = lib.nearbyint(tl.div_rn(total, scale)).to(tl.int8)
    tl.store(OUT + token * K + cols, codes)
    tl.store(OS + token * (K // 128) + group, scale)


def combine_experts(q, s, weights, dtype):
    m, topk = weights.shape
    k = q.shape[-1]
    if q.shape != (m * topk, k) or k % 128 or not weights.is_contiguous():
        raise ValueError("Invalid INT8 routed output")
    out = torch.empty((m, k), device=q.device, dtype=torch.int8)
    scales = torch.empty((m, k // 128), device=q.device, dtype=torch.float32)
    _combine[(m, k // 128)](q, s, weights, out, scales, k, topk, enable_fp_fusion=False)
    return ResidualInt8(out, scales, dtype)


@triton.jit
def _conv(
    Q,
    S,
    DELTA,
    BASE,
    OUT,
    OS,
    K: tl.constexpr,
    TAPS: tl.constexpr,
    GROUP: tl.constexpr,
    REQUEST: tl.constexpr,
    DS0: tl.constexpr,
    DS1: tl.constexpr,
    DS2: tl.constexpr,
):
    row, group = tl.program_id(0), tl.program_id(1)
    cols = group * 128 + tl.arange(0, 128)
    total = tl.zeros((128,), tl.float32)
    for tap in range(TAPS):
        valid = (row % REQUEST) >= tap
        source = row - tap
        q = tl.load(Q + source * K + cols, valid, 0).to(tl.float32)
        scale = tl.load(S + source * (K // 128) + group, valid, 0)
        x = (q * scale).to(BASE.dtype.element_ty).to(tl.float32)
        b = tl.load(BASE + tap * K + cols).to(tl.float32)
        d = tl.load(DELTA + row * DS0 + tap * DS1 + (cols // GROUP) * DS2).to(
            tl.float32
        )
        coefficient = (b + d).to(BASE.dtype.element_ty).to(tl.float32)
        product = (coefficient * x).to(BASE.dtype.element_ty).to(tl.float32)
        total = (total + product).to(BASE.dtype.element_ty).to(tl.float32)
    scale = tl.maximum(tl.div_rn(tl.max(tl.abs(total), 0), 127.0), 1.0e-30)
    codes = lib.nearbyint(tl.div_rn(total, scale)).to(tl.int8)
    tl.store(OUT + row * K + cols, codes)
    tl.store(OS + row * (K // 128) + group, scale)


def grouped_conv(value, delta, base, block_size, group_size):
    q, s = value.flatten_rows()
    k = q.shape[-1]
    taps = base.shape[0]
    if (
        base.shape != (taps, k)
        or delta.shape != (q.shape[0], taps, k // group_size)
        or not base.is_contiguous()
        or k % group_size
    ):
        raise ValueError("Invalid direct INT8 grouped convolution")
    out, scales = torch.empty_like(q), torch.empty_like(s)
    _conv[(q.shape[0], k // 128)](
        q,
        s,
        delta,
        base,
        out,
        scales,
        k,
        taps,
        group_size,
        block_size,
        *delta.stride(),
        enable_fp_fusion=False,
    )
    return ResidualInt8(
        out.view(value.shape), scales.view(value.scales.shape), value.compute_dtype
    )


@triton.jit
def _gated_norm(
    X,
    G,
    W,
    OUT,
    S,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    GSTRIDE: tl.constexpr,
    EPS: tl.constexpr,
):
    row, group = tl.program_id(0), tl.program_id(1)
    cols = group * GROUP + tl.arange(0, GROUP)
    x = tl.load(X + row * K + cols).to(tl.float32)
    gate = tl.load(G + row * GSTRIDE + cols).to(tl.float32)
    x *= gate / (1.0 + tl.exp(-gate))
    x *= tl.rsqrt(tl.sum(x * x, 0) / GROUP + EPS)
    weight = tl.load(W + cols).to(tl.float32)
    x = (x.to(X.dtype.element_ty).to(tl.float32) * weight).to(X.dtype.element_ty)
    values = tl.reshape(x.to(tl.float32), (GROUP // 128, 128))
    scales = tl.maximum(tl.div_rn(tl.max(tl.abs(values), 1), 127.0), 1.0e-30)
    codes = lib.nearbyint(tl.div_rn(values, scales[:, None])).to(tl.int8)
    tl.store(OUT + row * K + cols, tl.reshape(codes, (GROUP,)))
    tl.store(
        S + row * (K // 128) + group * (GROUP // 128) + tl.arange(0, GROUP // 128),
        scales,
    )


def gated_norm(norm, x, gate):
    k, group = x.shape[-1], norm.group_size
    if (
        not norm.use_rms_norm
        or norm.n_groups % norm.tp_size
        or group % 128
        or k % group
        or gate.shape != x.shape
        or not x.is_contiguous()
        or gate.stride(-1) != 1
    ):
        raise ValueError("Direct Mamba quant-out requires complete local norm groups")
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty((x.shape[0], k // 128), device=x.device, dtype=torch.float32)
    _gated_norm[(x.shape[0], k // group)](
        x,
        gate,
        norm.weight,
        q,
        s,
        k,
        group,
        gate.stride(0),
        norm.variance_epsilon,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return ResidualInt8(q, s, x.dtype)
