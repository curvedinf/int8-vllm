# SPDX-License-Identifier: Apache-2.0
"""Decode int8 intermediates: original G128 weights and a deferred CAR.

Only MLP down projections use this experimental path. Quantized partials
stay INT8 until the next residual/RMSNorm consumer. The carrier is an
explicit pytree, never a hidden attribute on a Tensor.
"""

import os
from typing import NamedTuple

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

MAX_DECODE_ROWS = 64


class DecodeInt8Tensor(NamedTuple):
    codes: torch.Tensor
    scales: torch.Tensor
    conv_delta: torch.Tensor | None = None
    conv_base: torch.Tensor | None = None
    conv_block_size: int = 0
    conv_group_size: int = 0
    dense: torch.Tensor | None = None

    @property
    def shape(self):
        return self.codes.shape


@triton.jit
def _quantize_g128(X, Q, S, K: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1)
    col = group * 128 + tl.arange(0, 128)
    x = tl.load(X + row * K + col).to(tl.float32)
    amax = tl.max(tl.abs(x), 0)
    scale = tl.where(amax == 0, 1.0, amax / 127.0)
    q = tl.minimum(127.0, tl.maximum(-127.0, tl.floor(x / scale + 0.5)))
    tl.store(Q + row * K + col, q.to(tl.int8))
    tl.store(S + row * (K // 128) + group, scale)


@triton.jit
def _gemm_g128_quant_out(
    X, XS, W, WS, Y, YS,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr,
):
    # One CTA owns all 128 output columns in a quantization group. No
    # split-K scratch or BF16 output tensor is produced.
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    kk = tl.arange(0, 128)
    acc = tl.zeros((BM, 128), tl.float32)
    for group in range(K // 128):
        ks = group * 128 + kk
        a = tl.load(X + rows[:, None] * K + ks[None, :],
                    mask=rows[:, None] < M, other=0)
        b = tl.load(W + cols[None, :] * K + ks[:, None])
        sx = tl.load(XS + rows * (K // 128) + group, mask=rows < M, other=0)
        sw = tl.load(WS + cols * (K // 128) + group).to(tl.float32)
        dot = tl.dot(a, b, out_dtype=tl.int32)
        acc = acc + (dot.to(tl.float32) * sx[:, None]) * sw[None, :]
    amax = tl.max(tl.abs(acc), 1)
    scale = tl.where(amax == 0, 1.0, amax / 127.0)
    quant = tl.minimum(
        127.0, tl.maximum(-127.0, tl.floor(acc / scale[:, None] + 0.5))
    )
    tl.store(Y + rows[:, None] * N + cols[None, :], quant.to(tl.int8),
             mask=rows[:, None] < M)
    tl.store(YS + rows * (N // 128) + tl.program_id(1), scale, mask=rows < M)


def _gemm(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor
          ) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    n = w.shape[0]
    if (x.dtype != torch.bfloat16 or w.dtype != torch.int8
            or ws.dtype != torch.float32 or k % 128 or n % 128
            or w.shape != (n, k) or ws.shape != (n, k // 128)
            or not 0 < m <= MAX_DECODE_ROWS
            or not all(t.is_cuda and t.is_contiguous() for t in (x, w, ws))):
        raise ValueError("Unsupported decode INT8 G128 GEMM contract")
    xq = torch.empty((m, k), dtype=torch.int8, device=x.device)
    xs = torch.empty((m, k // 128), dtype=torch.float32, device=x.device)
    q, s = _gemm_fake(x, w, ws)
    _quantize_g128[(m, k // 128)](x, xq, xs, k, num_warps=4)
    _gemm_g128_quant_out[(triton.cdiv(m, 16), n // 128)](
        xq, xs, w, ws, q, s, m, n, k, 16,
        num_warps=4, num_stages=2, matrix_instr_nonkdim=16,
        enable_fp_fusion=False,
    )
    return q, s


def _gemm_fake(x, w, ws):
    return (torch.empty((x.shape[0], w.shape[0]), device=x.device,
                        dtype=torch.int8),
            torch.empty((x.shape[0], w.shape[0] // 128), device=x.device,
                        dtype=torch.float32))


direct_register_custom_op(
    op_name="decode_int8_g128_gemm", op_func=_gemm, fake_impl=_gemm_fake
)


def down_projection(layer, x):
    if layer.bias is not None or not layer.reduce_results or layer.tp_size != 4:
        raise ValueError("Decode INT8 down projection requires bias-free TP4 CAR")
    q, s, dense = torch.ops.vllm.decode_int8_projection(
        x, layer._decode_i8_wq, layer._decode_i8_ws, layer._ck_q, layer._ck_s
    )
    return DecodeInt8Tensor(q, s, dense=dense)


def _projection(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor,
                ck_w: torch.Tensor, ck_s: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Runtime dispatch is inside this opaque op. vLLM deliberately drops
    # Dynamo's shape guards, so a Python size branch in model.forward would
    # silently freeze the prefill/decode decision made during warmup.
    if x.dtype != torch.bfloat16:
        raise ValueError("Decode INT8 intermediates requires the BF16 model stream")
    if 0 < x.shape[0] <= MAX_DECODE_ROWS:
        q, s = _gemm(x, w, ws)
        # Shape-stable ABI placeholder, never written or consumed at small
        # M. No GEMM or reduced values are materialized in BF16 here.
        dense = torch.empty(q.shape, device=x.device, dtype=torch.bfloat16)
    else:
        from vllm.distributed import tensor_model_parallel_all_reduce
        from vllm.model_executor.kernels.linear.mixed_precision.act_quant_rn import (
            pertoken_quant_rn,
        )

        xq, xs = pertoken_quant_rn(x)
        dense = torch.ops.vllm.rocm_aiter_gemm_a8w8_ck(
            xq, ck_w, xs, ck_s, torch.bfloat16
        )
        dense = tensor_model_parallel_all_reduce(dense)
        q, s = _gemm_fake(x, w, ws)
    return q, s, dense


def _projection_fake(x, w, ws, ck_w, ck_s):
    q, s = _gemm_fake(x, w, ws)
    return q, s, torch.empty(q.shape, device=x.device, dtype=torch.bfloat16)


direct_register_custom_op(
    op_name="decode_int8_projection", op_func=_projection, fake_impl=_projection_fake
)


@torch.compile(dynamic=True, fullgraph=True)
def _prefill_consume(dense, residual, weight, epsilon: float, gemma: bool,
                     delta, base, block_size: int, group_size: int):
    # Same native norm arithmetic as the existing compiled path. Compile
    # this small fallback separately rather than freezing a size branch in
    # vLLM's guard-free graph. No quantization is applied at large M.
    if delta is not None:
        from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv

        dense = _grouped_conv(dense, delta, base, block_size,
                              dense.shape[-1] // group_size, group_size, 2)
    if weight.numel() == 0:
        return torch.empty_like(residual), dense + residual
    effective_weight = weight.float() + 1.0 if gemma else weight
    # Native vllm.ir.ops.fused_add_rms_norm, including its rounding points.
    total = dense.float() + residual.float()
    variance = total.pow(2).mean(dim=-1, keepdim=True)
    normed = total * torch.rsqrt(variance + epsilon)
    normed = normed.to(effective_weight.dtype) * effective_weight
    return normed.to(dense.dtype), total.to(dense.dtype)


@torch.compile(dynamic=True, fullgraph=True)
def _prefill_conv(dense, delta, base, block_size: int, group_size: int):
    from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv

    return _grouped_conv(dense, delta, base, block_size,
                         dense.shape[-1] // group_size, group_size, 2)


def _norm(q: torch.Tensor, s: torch.Tensor, residual: torch.Tensor,
          weight: torch.Tensor, epsilon: float, gemma: bool,
          delta: torch.Tensor | None, base: torch.Tensor | None,
          block_size: int, group_size: int,
          dense: torch.Tensor | None, hip: bool) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.distributed import get_tp_group

    if q.shape[0] > MAX_DECODE_ROWS:
        if dense is None:
            raise ValueError("The prefill carrier requires a dense output")
        if hip and weight.numel():
            if delta is not None:
                dense = _prefill_conv(dense, delta, base, block_size, group_size)
                delta = base = None
            if dense.shape[0] >= 256:
                from aiter.ops.triton.normalization.rmsnorm import (
                    rmsnorm2d_fwd_with_add,
                )

                w = (weight.float() + 1).to(dense.dtype) if gemma else weight
                out, res = torch.empty_like(dense), torch.empty_like(residual)
                rmsnorm2d_fwd_with_add(out, dense, residual, res, w, epsilon)
                return out, res
            if (gemma and os.environ.get("VLLM_GFX908_SMALLM_NORM", "fused")
                    != "native"):
                from vllm.model_executor.layers.layernorm import (
                    _gemma_fused_add_rmsnorm,
                )

                return _gemma_fused_add_rmsnorm(
                    dense, residual, weight.float() + 1, epsilon
                )
        return _prefill_consume(dense, residual, weight, epsilon, gemma,
                                delta, base, block_size, group_size)
    comm = get_tp_group().device_communicator.ca_comm.decode_int8
    if comm is None:
        raise RuntimeError("Decode INT8 CAR is not initialized")
    return comm.norm(q, s, residual, weight, epsilon, gemma,
                     delta, base, block_size, group_size)


def _norm_fake(q, s, residual, weight, epsilon: float, gemma: bool,
               delta, base, block_size: int, group_size: int, dense=None, hip=False):
    return torch.empty_like(residual), torch.empty_like(residual)


direct_register_custom_op(
    op_name="decode_int8_g128_norm", op_func=_norm, fake_impl=_norm_fake
)


def norm(x: DecodeInt8Tensor, residual, weight, epsilon, gemma=False, hip=False):
    if residual is None:
        raise ValueError("A deferred MLP output must have a residual")
    return torch.ops.vllm.decode_int8_g128_norm(
        x.codes, x.scales, residual, weight, epsilon, gemma,
        x.conv_delta, x.conv_base, x.conv_block_size, x.conv_group_size,
        x.dense, hip,
    )


def auxiliary_hidden(x: DecodeInt8Tensor, residual):
    # DFlash requires five dense auxiliary residual snapshots. They are
    # explicit output consumers; the other inter-layer tensors remain INT8.
    unused_weight = torch.empty(0, device=residual.device, dtype=torch.bfloat16)
    _, value = norm(x, residual, unused_weight, 0.0)
    return value
