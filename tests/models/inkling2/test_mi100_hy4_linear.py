# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Joined launch of two HY4 column-parallel projections on MI100."""

import pytest
import torch

from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.platforms import current_platform


def _quantized(prefix, config, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.device("cuda"):
        layer = ColumnParallelLinear(
            2304,
            768,
            bias=False,
            params_dtype=torch.float16,
            quant_config=config,
            disable_tp=True,
            prefix=prefix,
        )
    for name, param in layer.named_parameters():
        if param.dtype == torch.uint8:
            codes = torch.randint(
                0, 256, param.shape, device="cuda", generator=gen, dtype=torch.int32
            )
            param.copy_(codes.to(torch.uint8))
        elif "scale" in name:
            param.copy_(torch.rand(param.shape, device="cuda", generator=gen) + 0.5)
        elif "zero" in name:
            param.copy_(torch.rand(param.shape, device="cuda", generator=gen) * 8 - 4)
        else:
            raise AssertionError(f"unexpected parameter {name}")
    layer.quant_method.process_weights_after_loading(layer)
    return layer


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")
@torch.inference_mode()
def test_joined_projections_match_separate_calls(dist_init, default_vllm_config):
    from vllm.model_executor.layers.quantization.hy4_neuqi import HY4Linear
    from vllm.models.inkling2.quantization import Inkling2PTQRConfig
    from vllm.platforms.rocm import on_gfx908

    if not on_gfx908():
        pytest.skip("gfx908 packed G128 GEMM")
    names = ["wq_du", "gate_attn"]
    config = Inkling2PTQRConfig(
        [f"model.llm.layers.0.attn.{name}.weight" for name in names]
    )
    layers = [
        _quantized(f"model.layers.0.attn.{name}", config, seed)
        for seed, name in enumerate(names, 1)
    ]
    assert all(isinstance(layer.quant_method, HY4Linear) for layer in layers)
    gen = torch.Generator(device="cuda").manual_seed(11)
    x = torch.randn(2048, 2304, dtype=torch.float16, device="cuda", generator=gen)
    # Decode sizes launch split-K; 513 rows and above take the large-batch tiles.
    for tokens in (8, 96, 513, 2048):
        rows = x[:tokens]
        expected = [layer(rows)[0] for layer in layers]
        actual = layers[0].quant_method.apply_joined(layers, rows)
        assert all(torch.equal(got, want) for got, want in zip(actual, expected)), (
            tokens
        )
