# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.vocab_parallel_embedding import (
    _int8_embedding_gather,
    _quantize_vocab_rows,
)


def test_int8_embedding_gathers_before_dequantization() -> None:
    weight = torch.tensor(
        [[-127, -3, 0, 4], [5, 6, 7, 8], [9, 10, 11, 127]],
        dtype=torch.int8,
    )
    scale = torch.tensor([0.25, 0.5, 0.125], dtype=torch.float16)
    input_ids = torch.tensor([[2, 0], [1, 2]], dtype=torch.long)

    output = _int8_embedding_gather(input_ids, weight, scale)
    expected = weight[input_ids].to(scale.dtype) * scale[input_ids].unsqueeze(-1)

    assert output.dtype == scale.dtype
    assert output.shape == (*input_ids.shape, weight.shape[1])
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert weight.dtype == torch.int8


@pytest.mark.parametrize(
    "scale_dtype,min_amax,min_code",
    [(torch.float16, 1e-8, -128), (torch.float32, 1e-12, -127)],
)
def test_chunked_vocab_quant_matches_full_table(scale_dtype, min_amax, min_code):
    weight = torch.randn(1025, 4096, dtype=torch.float16)
    weight[-1].zero_()
    codes, scales = _quantize_vocab_rows(weight, scale_dtype, min_amax, min_code)
    full = weight.float()
    expected = (full.abs().amax(1, keepdim=True).clamp_min(min_amax) / 127).to(
        scale_dtype
    )
    quant = (full / expected.float()).round().clamp(min_code, 127).to(torch.int8)
    torch.testing.assert_close(scales, expected, rtol=0, atol=0)
    torch.testing.assert_close(codes, quant, rtol=0, atol=0)
