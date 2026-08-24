# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.vocab_parallel_embedding import (
    _int8_embedding_gather,
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
