# SPDX-License-Identifier: Apache-2.0
"""Dense ModelOpt promotion releases sources without changing layer outputs."""

import gc
import weakref

import pytest
import torch
from aiter.ops.triton.gemm.basic import gemm_packed_g128 as ops
from aiter.ops.triton.gemm.basic import packed_dense_cache as caches
from torch import nn

from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4W4A16LinearMethod,
)
from vllm.model_executor.layers.quantization.modelopt_gfx908 import (
    PackedG128LinearMethod,
    prepare_static_dense_weight,
)

pytestmark = pytest.mark.skipif(
    not torch.version.hip
    or not torch.cuda.is_available()
    or "gfx908" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="gfx908 GPU required",
)


@pytest.fixture(autouse=True)
def clean_cache(monkeypatch):
    monkeypatch.setenv("VLLM_BOLT_DENSE_CACHE_GIB", "0.25")
    monkeypatch.setenv("VLLM_BOLT_DENSE_CACHE_DROP_PACKED", "0")
    yield
    caches._CACHE.clear()
    caches._BYTES.clear()
    caches._RELEASED_BYTES.clear()
    ops._weight_scale_cache.clear()
    gc.collect()


@pytest.mark.parametrize("codec", ["q8_0", "iq2_xxs"])
def test_linear_promotion_releases_packed_parameter(codec, monkeypatch):
    block, width = (32, 34) if codec == "q8_0" else (256, 66)
    layer = nn.Module()
    packed = torch.randint(
        0, 256, (768, 1024 // block, width), device="cuda", dtype=torch.uint8
    )
    packed[..., 0] = 0
    packed[..., 1] = 0x30
    layer.weight = nn.Parameter(packed, requires_grad=False)
    method = PackedG128LinearMethod(codec)
    method.k, method.start = 1024, 0
    method.process_weights_after_loading(layer)
    x = torch.randn((8, 1024), device="cuda", dtype=torch.bfloat16)
    reference = method.apply(layer, x)
    old = weakref.ref(layer.weight)
    monkeypatch.setenv("VLLM_BOLT_DENSE_CACHE_DROP_PACKED", "1")
    prepare_static_dense_weight(layer, 1024, codec=codec)
    del packed
    gc.collect()
    assert old() is None
    assert layer.weight.dtype == torch.int8
    assert layer.weight_gs128.dtype == torch.float32
    assert not hasattr(layer, "packed_grid")
    assert not ops._weight_scale_cache
    torch.testing.assert_close(method.apply(layer, x), reference, rtol=0, atol=0)


def test_nvfp4_load_hook_releases_all_original_metadata(monkeypatch):
    layer = nn.Module()
    layer.logical_widths = [768]
    layer.input_size_per_partition = 1024
    layer.weight = nn.Parameter(
        torch.randint(0, 256, (768, 512), device="cuda", dtype=torch.uint8),
        requires_grad=False,
    )
    layer.weight_scale = nn.Parameter(
        torch.ones((768, 64), device="cuda").to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale_2 = nn.Parameter(
        torch.ones(1, device="cuda"), requires_grad=False
    )
    layer.input_scale = nn.Parameter(torch.ones(1, device="cuda"), requires_grad=False)
    method = ModelOptNvFp4W4A16LinearMethod.__new__(ModelOptNvFp4W4A16LinearMethod)
    method.packed_g128 = True
    method.process_weights_after_loading(layer)
    x = torch.randn((8, 1024), device="cuda", dtype=torch.bfloat16)
    reference = method.apply(layer, x)
    old = [
        weakref.ref(getattr(layer, k))
        for k in (
            "weight",
            "weight_scale",
            "weight_scale_2",
            "packed_grid",
            "packed_global_scales",
        )
    ]
    monkeypatch.setenv("VLLM_BOLT_DENSE_CACHE_DROP_PACKED", "1")
    prepare_static_dense_weight(layer, 1024, codec="nvfp4")
    gc.collect()
    assert all(ref() is None for ref in old)
    assert layer.weight.dtype == torch.int8
    assert not hasattr(layer, "input_scale")
    assert not ops._weight_scale_cache
    torch.testing.assert_close(method.apply(layer, x), reference, rtol=0, atol=0)
