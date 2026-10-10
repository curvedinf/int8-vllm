# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact fresh-state initialization with no GPU->CPU shape/scalar readback."""
import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    zero_prefill_states,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("padding", [0, 13])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_zero_prefill_states_preserves_live_states_and_page_padding(
    padding, index_dtype
):
    source = torch.randn(8, 512 + padding, device="cuda", dtype=torch.float32)
    buffer = source.clone()
    state = buffer[:, :512].view(8, 2, 16, 16)
    # Noncontiguous metadata; duplicate fresh/live index and negative index.
    indices = torch.tensor([1, 0, 3, 0, 5, 0, 3, 0, -1, 0],
                           device="cuda", dtype=index_dtype)[::2]
    mask = torch.tensor([False, True, True, True, False, True,
                         False, True, True, True], device="cuda")[::2]
    expected = source.clone()
    expected[[1, 3, 5], :512] = 0
    zero_prefill_states(state, indices, mask)
    assert torch.equal(buffer, expected)

    # Graph capture rejects host scalar reads and dynamic nonzero operations.
    buffer.copy_(source)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        zero_prefill_states(state, indices, mask)
    graph.replay()
    assert torch.equal(buffer, expected)
