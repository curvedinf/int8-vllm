# SPDX-License-Identifier: Apache-2.0
"""INT8 GS128 lm_head: quality probe and optional head method.

Quantizes the unquantized checkpoint lm_head to [N,K] int8 + per-128-group
fp32 scales at first use (warmup) and projects through the AITER Triton
blockscale GEMM with per-group activation quant. Enabled by
VLLM_GFX908_LM_HEAD_I8=1; used to answer whether an int8 GS128 head keeps
quality (see scripts/probe_lm_head_quality.py).
"""

import torch


class Int8GS128HeadMethod:
    """Drop-in replacement for the unquantized lm_head's quant_method."""

    def __init__(self) -> None:
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _weights(self, layer):
        ent = self._cache.get(id(layer))
        if ent is None:
            from vllm.model_executor.layers.quantization.utils.packed_g128_int8 import (
                quantize,
            )

            w = layer.weight.data.float().contiguous()
            wq, ws = quantize(w)
            ent = self._cache[id(layer)] = (wq, ws)
        return ent

    def apply(self, layer, x, bias=None, **kwargs):
        from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
            gemm_a8w8_blockscale,
        )
        from vllm.model_executor.layers.quantization.utils.packed_g128_int8 import (
            quantize,
        )

        wq, ws = self._weights(layer)
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        xq, xs = quantize(x2)
        logits = gemm_a8w8_blockscale(xq, wq, xs, ws, dtype=x.dtype)
        if bias is not None:
            logits = logits + bias
        return logits.reshape(*x.shape[:-1], -1)
