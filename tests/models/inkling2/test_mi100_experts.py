# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check local expert activation, remote row masking and weighted reduction."""

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")
def test_stable_top8_matches_native_weights_and_ties():
    from vllm.models.inkling2.amd.moe import _stable_top8

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(3917)
    logits = torch.randn(96, 128, device=device, generator=gen)
    # Saturation, a boundary tie, and a completely tied row.
    logits[0] = 100
    logits[1, :16] = 100
    logits[2] = -100
    for bias in (torch.zeros_like(logits[0]), logits[-1].clone()):
        probabilities = logits.sigmoid()
        ordered = (probabilities + bias).argsort(dim=-1, descending=True, stable=True)
        ids = ordered[:, :8]
        weights = probabilities.gather(-1, ids)
        weights *= 8.0 / weights.sum(-1, keepdim=True).clamp_min(1e-20)
        actual_ids, actual_weights = _stable_top8(logits, bias, 8.0)
        assert torch.equal(actual_ids, ids.int())
        assert torch.equal(actual_weights.view(torch.int32), weights.view(torch.int32))


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")
@pytest.mark.parametrize("tabulated", [False, True])
def test_expert_activation_and_combine(tabulated):
    from aiter.ops.triton.gemm.basic.gemm_packed_g128 import (
        combine_experts,
        quantize,
        quantize_routed_swiglu,
        routed_swiglu_factors,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(706)
    rows, topk, width = 23, 8, 1152
    ids = torch.randint(0, 128, (rows, topk), device=device, generator=generator)
    weights = torch.rand((rows, topk), device=device, generator=generator)
    local = (ids >= 32) & (ids < 64)
    gate_up = (
        torch.randn(
            (rows * topk, 2 * width),
            device=device,
            dtype=torch.float16,
            generator=generator,
        )
        * 8
    )
    # Remote routes have no GEMM producer. NaNs make accidental reads observable.
    gate_up[~local.flatten()] = float("nan")
    if tabulated:
        # Force this product to determine the group scale. A single multiply
        # rounded straight to half misses the reference's FP32 rounding.
        first = local.flatten().nonzero()[0].item()
        gate_up[first, :128] = 0
        gate_up[first, 0] = 1.6171875
        gate_up[first, width] = 10.1484375
    g, u = gate_up.float().chunk(2, -1)
    mid = ((g / 4).tanh() * 4) * g.sigmoid() * ((u / 25).tanh() * 25)
    baseline = quantize(
        torch.where(local.reshape(-1, 1), mid, 0).half(), with_sums=True
    )
    factors = routed_swiglu_factors(device, 4, 25) if tabulated else None
    fused = quantize_routed_swiglu(
        gate_up, ids, 32, 64, gate_cap=4, up_cap=25, factors=factors
    )
    for a, b in zip(baseline, fused):
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    down = torch.randn(
        (rows * topk, width), device=device, dtype=torch.float16, generator=generator
    )
    down[~local.flatten()] = float("nan")
    reference = (
        torch.where(local[..., None], down.view(rows, topk, width), 0)
        * weights[..., None]
    ).sum(1)
    got = combine_experts(down, weights, torch.float32, ids=ids, start=32, stop=64)
    torch.testing.assert_close(got, reference, atol=2e-6, rtol=1e-6)
