# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy and determinism of Inkling 2 Mini's narrow decode GEMM kernels."""

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")
@pytest.mark.parametrize("tokens", [1, 8, 96])
def test_router_logits_split_k(tokens):
    from vllm.models.inkling2.amd.ops.skinny_gemm import router_logits
    from vllm.platforms.rocm import on_gfx908

    if not on_gfx908():
        pytest.skip("gfx908 split-K FP32 contraction")
    rng = torch.Generator(device="cuda").manual_seed(4104)
    x = torch.randn(tokens, 2304, device="cuda", generator=rng).half()
    weight = torch.randn(128, 2304, device="cuda", generator=rng) * 0.02
    truth = x.double() @ weight.double().t()
    result = router_logits(x, weight)
    assert result.dtype == torch.float32
    assert result.shape == (tokens, 128)
    # 2304 FP32 additions: the fixed split order stays a few 1e-6 from exact.
    torch.testing.assert_close(result.double(), truth, atol=1e-5, rtol=0)
    assert torch.equal(result, router_logits(x, weight))


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")
@pytest.mark.parametrize("tokens", [8, 96, 128])
def test_fp16_gemm_matches_rocblas_accuracy(tokens):
    from vllm.models.inkling2.amd.ops.skinny_gemm import fp16_gemm
    from vllm.platforms.rocm import on_gfx908

    if not on_gfx908():
        pytest.skip("gfx908 FP16 MFMA contraction")
    rng = torch.Generator(device="cuda").manual_seed(4105)
    x = torch.randn(tokens, 2304, device="cuda", generator=rng).half()
    weight = (torch.randn(96, 2304, device="cuda", generator=rng) * 0.02).half()
    truth = x.double() @ weight.double().t()
    result = fp16_gemm(x, weight)
    reference = torch.nn.functional.linear(x, weight)
    assert result.dtype == torch.float16
    assert result.shape == (tokens, 96)
    # Both contractions accumulate in FP32 and round once to FP16, so each
    # result lies within one FP16 ulp of the partial-sum scale (2**-10) of
    # the exact value; the kernel it replaces has the same bound.
    for value in (result, reference):
        torch.testing.assert_close(value.double(), truth, atol=2**-10, rtol=0)
    assert torch.equal(result, fp16_gemm(x, weight))
