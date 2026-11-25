# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the accumulation order of Inkling 2 Mini's FP32 residual mix."""

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")
@pytest.mark.parametrize("tokens,states,queries", [(8, 2, 4), (96, 3, 1), (1024, 5, 3)])
def test_weighted_values_matches_rocblas_on_mi100(tokens, states, queries):
    from vllm.models.inkling2.amd.ops.attn_res import weighted_values
    from vllm.platforms.rocm import on_gfx908

    if not on_gfx908():
        pytest.skip("gfx908 rocBLAS contraction")
    device = torch.device("cuda")
    rng = torch.Generator(device=device).manual_seed(53703)
    values = torch.randn(states, tokens, 2304, device=device, generator=rng)
    weights = torch.randn(queries, tokens, states, device=device, generator=rng)
    weights = weights.softmax(-1)
    expected = torch.einsum("qts,std->qtd", weights, values)
    result = weighted_values(weights, values)
    assert torch.equal(result.view(torch.int32), expected.view(torch.int32))


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")
@pytest.mark.parametrize("tokens", [4, 8, 96, 513])
def test_residual_input_normalization(tokens):
    from vllm.models.inkling2.amd.ops.attn_res import normalize_residual_rows
    from vllm.platforms.rocm import on_gfx908

    if not on_gfx908():
        pytest.skip("gfx908 FP32 reduction order")
    rng = torch.Generator(device="cuda").manual_seed(1787)
    stack = torch.randn(3, tokens, 2304, device="cuda", generator=rng).half()
    last = torch.randn(tokens * 2, 2304, device="cuda", generator=rng)[::2]
    addend = torch.randn(tokens, 2304, device="cuda", generator=rng).half()

    def normalize(x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

    expected = normalize(torch.cat((stack.float(), (last + addend).unsqueeze(0))))
    assert torch.equal(normalize_residual_rows(stack, last, addend), expected)
    expected = normalize(torch.stack((last, addend.float()), dim=1)).half()
    assert torch.equal(
        normalize_residual_rows(last.unsqueeze(0), addend, controller=True), expected
    )
