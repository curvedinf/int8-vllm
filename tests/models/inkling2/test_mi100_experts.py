# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check local expert activation, remote row masking and weighted reduction."""

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")
def test_expert_activation_and_combine():
    from aiter.ops.triton.gemm.basic.gemm_packed_g128 import (
        combine_experts,
        quantize,
        quantize_routed_swiglu,
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
    g, u = gate_up.float().chunk(2, -1)
    mid = ((g / 4).tanh() * 4) * g.sigmoid() * ((u / 25).tanh() * 25)
    baseline = quantize(
        torch.where(local.reshape(-1, 1), mid, 0).half(), with_sums=True
    )
    fused = quantize_routed_swiglu(gate_up, ids, 32, 64, gate_cap=4, up_cap=25)
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
