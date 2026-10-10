# SPDX-License-Identifier: Apache-2.0
"""Mixed query partitioning must preserve request and token addresses."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
    RocmAiterUnifiedAttentionMetadataBuilder,
)


@pytest.mark.parametrize("lengths", [
    [7, 7, 2048], [2048, 7, 16, 7], [0, 7, 256, 0], [7, 7, 7], [256, 512],
    [32, 7, 1984, 7],
])
def test_partitions_preserve_offsets(monkeypatch, lengths):
    monkeypatch.setenv("VLLM_G128_MIXED_SPLIT", "1")
    builder = object.__new__(RocmAiterUnifiedAttentionMetadataBuilder)
    builder.headdim = 256
    builder.vllm_config = SimpleNamespace(speculative_config=None)
    starts = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0)
    common = SimpleNamespace(
        num_actual_tokens=sum(lengths), max_query_len=max(lengths),
        max_seq_len=32000, query_start_loc=starts,
        query_start_loc_cpu=starts, seq_lens=torch.full((len(lengths),), 32000),
        block_table_tensor=torch.arange(len(lengths)).view(-1, 1),
        slot_mapping=torch.arange(sum(lengths)), causal=True,
    )
    metadata = builder.build(0, common)
    mixed = any(0 < n <= 32 for n in lengths) and max(lengths) > 32
    if not mixed:
        assert metadata.query_partitions is None
        return
    covered = []
    for first, end, lo, hi, max_q, offsets in metadata.query_partitions:
        assert torch.equal(offsets + lo, starts[first:end + 1])
        assert max_q == max(lengths[first:end])
        assert all(n <= 32 for n in lengths[first:end]) or all(
            n > 32 for n in lengths[first:end]
        )
        covered.extend(range(lo, hi))
    assert covered == list(range(sum(lengths)))
