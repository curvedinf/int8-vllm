# SPDX-License-Identifier: Apache-2.0
"""Explicit GS128 decoder activation/skip payloads for Bolt and LiLiCorr.

Residual additions and RMS reductions use FP32 arithmetic; decoder seams emit
INT8 values and FP32 scales directly. No carrier retains a floating-point copy.
"""

from typing import NamedTuple

import torch
import triton.language.extra.libdevice as lib

from vllm import envs
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


class ResidualInt8(NamedTuple):
    codes: torch.Tensor
    scales: torch.Tensor
    compute_dtype: torch.dtype

    @property
    def shape(self):
        return self.codes.shape

    @property
    def device(self):
        return self.codes.device

    @property
    def dtype(self):
        """Arithmetic dtype at an explicitly floating-point consumer."""
        return self.compute_dtype

    @property
    def ndim(self):
        return self.codes.ndim

    def flatten_rows(self):
        """Expose GEMM payloads without decoding or re-quantizing."""
        _validate(self)
        k = self.shape[-1]
        return self.codes.view(-1, k), self.scales.view(-1, (k + 127) // 128)


def _validate(value):
    q, s, dtype = value
    if (
        q.dtype != torch.int8
        or s.dtype != torch.float32
        or dtype not in (torch.bfloat16, torch.float32)
        or q.shape[-1] <= 0
        or s.shape != (*q.shape[:-1], (q.shape[-1] + 127) // 128)
        or s.device != q.device
        or not q.is_contiguous()
        or not s.is_contiguous()
    ):
        raise ValueError("Invalid GS128 residual codes/scales")


@triton.jit
def _add_norm(
    X,
    XS,
    R,
    RS,
    W,
    Y,
    Q,
    S,
    YQ,
    YS,
    K: tl.constexpr,
    EPS: tl.constexpr,
    X_QUANT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    NORMALIZE: tl.constexpr,
    EMIT_NORMAL_QUANT: tl.constexpr,
    NATIVE_NORM: tl.constexpr,
    WRITE_NORMAL: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, K)
    value = tl.load(X + row * K + cols).to(tl.float32)
    if X_QUANT:
        value *= tl.load(XS + row * (K // 128) + cols // 128)
    if HAS_RESIDUAL:
        residual = tl.load(R + row * K + cols).to(tl.float32)
        residual *= tl.load(RS + row * (K // 128) + cols // 128)
        value += residual
    # Native small-M RMSNorm reduces the unrounded sum; AITER's large-M
    # path rounds the sum to the compute dtype first. Preserve that distinction.
    stored = value.to(Y.dtype.element_ty).to(tl.float32)
    groups = tl.reshape(stored, (K // 128, 128))
    scale = tl.maximum(tl.div_rn(tl.max(tl.abs(groups), 1), 127.0), 1.0e-30)
    codes = lib.nearbyint(tl.div_rn(groups, scale[:, None]))
    tl.store(Q + row * K + cols, tl.reshape(codes, (K,)).to(tl.int8))
    tl.store(S + row * (K // 128) + tl.arange(0, K // 128), scale)
    if NORMALIZE:
        if not NATIVE_NORM:
            value = stored
        inv = tl.rsqrt(tl.sum(value * value, 0) / K + EPS)
        weight = tl.load(W + cols).to(tl.float32)
        if NATIVE_NORM:
            normalized = (value * inv).to(W.dtype.element_ty).to(tl.float32)
            normal = (normalized * weight).to(Y.dtype.element_ty)
        else:
            normal = (value * inv * weight).to(Y.dtype.element_ty)
        if WRITE_NORMAL:
            tl.store(Y + row * K + cols, normal)
        if EMIT_NORMAL_QUANT:
            ng = tl.reshape(normal.to(tl.float32), (K // 128, 128))
            ns = tl.maximum(tl.div_rn(tl.max(tl.abs(ng), 1), 127.0), 1.0e-30)
            nq = lib.nearbyint(tl.div_rn(ng, ns[:, None]))
            tl.store(YQ + row * K + cols, tl.reshape(nq, (K,)).to(tl.int8))
            tl.store(YS + row * (K // 128) + tl.arange(0, K // 128), ns)


@triton.jit
def _materialize(Q, S, Y, N: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    q = tl.load(Q + offsets, offsets < N, 0).to(tl.float32)
    groups = (offsets // K) * tl.cdiv(K, 128) + (offsets % K) // 128
    s = tl.load(S + groups, offsets < N, 0)
    tl.store(Y + offsets, q * s, offsets < N)


def materialize(value):
    """Materialize only at a consumer requiring floating-point values."""
    if not isinstance(value, ResidualInt8):
        return value
    _validate(value)
    if not value.codes.is_cuda:
        scales = value.scales.repeat_interleave(128, dim=-1)[..., : value.shape[-1]]
        return (value.codes.float() * scales).to(value.compute_dtype)
    out = torch.empty_like(value.codes, dtype=value.compute_dtype)
    _materialize[(triton.cdiv(out.numel(), 1024),)](
        value.codes, value.scales, out, out.numel(), out.shape[-1], 1024
    )
    return out


def _run(
    x,
    residual=None,
    weight=None,
    epsilon=0.0,
    emit_quant=False,
    native=False,
    quantized=False,
):
    packed = isinstance(x, ResidualInt8)
    if packed:
        _validate(x)
    data = x.codes if packed else x
    dtype = x.compute_dtype if packed else x.dtype
    k = data.shape[-1]
    if k not in (1024, 4096) or dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("INT8 residuals require BF16/FP32 and hidden size 1024/4096")
    if residual is not None and (
        not isinstance(residual, ResidualInt8)
        or residual.shape != data.shape
        or residual.codes.device != data.device
        or residual.compute_dtype != dtype
    ):
        raise ValueError("Expected a matching explicit GS128 residual")
    if residual is not None:
        _validate(residual)
    if weight is not None and (weight.shape != (k,) or weight.device != data.device):
        raise ValueError("Expected a matching RMSNorm weight")
    shape = data.shape
    scales_shape = (*shape[:-1], k // 128)
    q = torch.empty_like(data, dtype=torch.int8, memory_format=torch.contiguous_format)
    s = torch.empty(scales_shape, device=data.device, dtype=torch.float32)
    out = torch.empty(
        shape if weight is not None and not quantized else (0,),
        device=data.device,
        dtype=dtype,
    )
    yq = torch.empty_like(q) if emit_quant else q
    ys = torch.empty_like(s) if emit_quant else s
    if data.is_cuda:
        _add_norm[(data.numel() // k,)](
            data.contiguous(),
            x.scales if packed else s,
            residual.codes if residual is not None else q,
            residual.scales if residual is not None else s,
            weight if weight is not None else out,
            out,
            q,
            s,
            yq,
            ys,
            k,
            epsilon,
            packed,
            residual is not None,
            weight is not None,
            emit_quant,
            native,
            not quantized,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        # CPU oracle also exercises complete model integration without a GPU.
        value = (
            x.codes.float() * x.scales.repeat_interleave(128, -1)
            if packed
            else x.float()
        )
        if residual is not None:
            rg = residual.codes.unflatten(-1, (-1, 128)).float()
            value = value + (rg * residual.scales.unsqueeze(-1)).flatten(-2)
        stored = value.to(dtype).float()

        def quantize(v, codes, scales):
            groups = v.unflatten(-1, (-1, 128)).float()
            scales.copy_(groups.abs().amax(-1).div(127).clamp_min(1.0e-30))
            codes.copy_((groups / scales.unsqueeze(-1)).round().flatten(-2))

        quantize(stored, q, s)
        if weight is not None:
            if not native:
                value = stored
            normal = value * torch.rsqrt(
                value.square().mean(-1, keepdim=True) + epsilon
            )
            if native:
                normal = normal.to(weight.dtype).float()
            normal = (normal * weight.float()).to(dtype)
            if not quantized:
                out.copy_(normal)
            if emit_quant:
                quantize(normal, yq, ys)
    if quantized:
        return ResidualInt8(yq, ys, dtype), ResidualInt8(q, s, dtype)
    result = (out, ResidualInt8(q, s, dtype))
    return (*result, yq, ys) if emit_quant else result


def add_residual(x, residual=None):
    return _run(x, residual)[1]


def residual_norm(norm, x, residual=None, *, emit_quant=False, quantized=False):
    """Preserve the baseline path unless the explicit residual switch is on."""
    if not envs.VLLM_GFX908_RESIDUAL_INT8:
        if emit_quant or quantized:
            raise ValueError("Normal-output quantization requires INT8 residual mode")
        if residual is None:
            return norm(x), x
        return norm(x, residual)
    logger.info_once(
        "GS128 INT8 decoder seams active: FP32 scales and reductions; "
        "direct normalized payloads; Bolt and LiLiCorr text stack",
        scope="process",
    )
    epsilon = getattr(norm, "variance_epsilon", None)
    if epsilon is None:
        epsilon = norm.eps
    data = x.codes if isinstance(x, ResidualInt8) else x
    native = hasattr(norm, "variance_epsilon") and data.numel() // data.shape[-1] < 256
    return _run(
        x, residual, norm.weight, epsilon, emit_quant or quantized, native, quantized
    )


def final_norm(norm, x, residual=None):
    if isinstance(x, ResidualInt8) or isinstance(residual, ResidualInt8):
        return residual_norm(norm, x, residual)[0]
    return norm(x) if residual is None else norm(x, residual)[0]


def decoder_norm(norm, x, residual=None):
    """A decoder seam writes both normalized activation and skip as INT8."""
    return residual_norm(norm, x, residual, quantized=envs.VLLM_GFX908_RESIDUAL_INT8)
