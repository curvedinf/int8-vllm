# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark vocabulary reduction across tile boundaries."""

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")
def test_tiled_argmax_preserves_ties_nan_and_vocab_padding(monkeypatch):
    from vllm.models.inkling2.amd.ops import dspark

    monkeypatch.setattr(dspark, "tensor_model_parallel_all_gather", lambda x: x)
    width, valid = 2137, 2078
    base = torch.full((5, 3, width), -7, device="cuda", dtype=torch.bfloat16)[:, 1]
    bias = torch.zeros_like(base)
    base[:, valid:] = 512
    base[0, 1094] = base[0, 2064] = 5
    base[1, :valid] = torch.nan
    base[2, 91] = -0.0
    base[2, 1800] = 0.0
    base[3, 1509] = torch.inf
    base[4, valid - 1] = 3
    args = (base, bias, 30.0, valid)
    reference = dspark.dspark_argmax(*args)
    assert reference.tolist() == [1094, 0, 91, 1509, valid - 1]
    assert torch.equal(dspark.dspark_argmax(*args, tiled=True), reference)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")
def test_winner_exchange_through_all_reduce_round_trips_every_code(monkeypatch):
    from vllm.models.inkling2.amd.ops import dspark

    # One rank: the sum returns this rank's slot unchanged. The int16 halves
    # keep every bit pattern a normal float, so nothing is rounded away.
    monkeypatch.setattr(dspark, "tensor_model_parallel_all_reduce", lambda x: x)
    dspark._zero_padded.clear()
    rng = torch.Generator(device="cuda").manual_seed(4106)
    codes = torch.randint(
        -(2**31), 2**31 - 1, (96, 50), device="cuda", generator=rng
    ).to(torch.int32)
    codes[0, :5] = torch.tensor(
        [0x007F0001, 0x80000001 - 2**32, -1, 0x7F80FFFF, -(2**31)],
        dtype=torch.int32,
        device="cuda",
    )
    assert torch.equal(dspark._exchange_codes(codes, 1, 0), codes)
    assert not dspark._zero_padded[(96, 100), codes.device].isnan().any()

    # Outside a capture the helper keeps the all-gather.
    monkeypatch.setattr(dspark, "tensor_model_parallel_all_gather", lambda x: x)
    assert torch.equal(dspark._winners_through_all_reduce(codes), codes)
    width, valid = 2137, 2078
    base = torch.randn(96, width, device="cuda", generator=rng).bfloat16()
    bias = torch.randn(96, width, device="cuda", generator=rng).bfloat16()
    base[0, :100] = float("nan")
    base[1, 200:] = float("-inf")
    base[2] = 0
    bias[2] = 0
    expected = dspark.dspark_argmax(base, bias, 30.0, valid, tiled=True)
    actual = dspark.dspark_argmax(base, bias, 30.0, valid, tiled=True, gather="car")
    assert torch.equal(actual, expected)
