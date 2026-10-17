# SPDX-License-Identifier: Apache-2.0
"""Packed TP boundaries must reconstruct the original logical expert matrix."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.routed_experts import (
    RoutedExperts,
    _index_expert_mapping,
)
from vllm.model_executor.layers.quantization.modelopt_gfx908 import PackedG128MoEMethod


@pytest.mark.parametrize("width", [1536, 2048, 2304, 2560])
def test_expert_tp_byte_slices(width):
    hidden = 1024
    # Distinct bytes across rows, blocks and experts reveal incorrect offsets.
    gen = torch.Generator().manual_seed(width)
    up = torch.randint(
        256, (width, hidden // 256, 66), dtype=torch.uint8, generator=gen
    )
    down = torch.randint(
        256, (hidden, width // 256, 66), dtype=torch.uint8, generator=gen
    )
    for rank in range(4):
        config = SimpleNamespace(
            activation=MoEActivation.RELU2_NO_MUL,
            moe_parallel_config=SimpleNamespace(ep_size=1),
            tp_rank=rank,
        )
        method = PackedG128MoEMethod(config)
        layer = torch.nn.Module()
        method.create_weights(layer, 2, hidden, width // 4, torch.bfloat16)
        layer.layer_name = "model.layers.1.experts"
        layer.get_expert_mapping = lambda **_: [
            ("experts.w13_", "experts.1.up_proj.", 1, "w1"),
            ("experts.w2_", "experts.1.down_proj.", 1, "w2"),
        ]
        names = list(
            RoutedExperts.load_weights(
                layer, [("1.up_proj.weight", up), ("1.down_proj.weight", down)]
            )
        )
        assert names == ["w13_weight", "w2_weight"]
        start, end = rank * width // 4, (rank + 1) * width // 4
        assert torch.equal(layer.w13_weight[1], up[start:end])
        first, last = start // 256, (end + 255) // 256
        assert torch.equal(layer.w2_weight[1], down[:, first:last])
        assert method.start == start - first * 256
        assert layer.w13_weight.dtype == layer.w2_weight.dtype == torch.uint8


def test_expert_name_index_preserves_replicas():
    mapping = [
        ("w13_weight", "experts.2.up_proj.", 2, "w1"),
        ("w2_weight", "experts.2.down_proj.", 2, "w2"),
        ("w13_weight", "experts.2.up_proj.", 5, "w1"),
        ("w13_weight", "experts.gate_up_proj", 0, "w1"),
    ]
    assert _index_expert_mapping(mapping) == {"2": mapping[:3]}


@pytest.mark.parametrize(
    "tokens,experts,expected_block",
    [
        (8, 240, 16),
        (64, 240, 16),
        (65, 240, 16),
        (1706, 240, 16),
        (1707, 240, 64),
        (4096, 496, 64),
    ],
)
def test_expert_projections_share_route_plan_and_tile(
    tokens, experts, expected_block, monkeypatch
):
    from aiter.ops.triton.gemm.basic import packed_expert_cache

    from vllm.model_executor.layers.fused_moe import moe_align_block_size as alignment
    from vllm.model_executor.layers.quantization import modelopt_gfx908 as packed

    method = PackedG128MoEMethod(
        SimpleNamespace(
            activation=MoEActivation.RELU2_NO_MUL,
            moe_parallel_config=SimpleNamespace(ep_size=1),
            tp_rank=0,
        )
    )
    method.start, method.local_i = 0, 4
    layer = SimpleNamespace(
        apply_router_weight_on_input=False,
        w13_weight=torch.empty((experts, 4, 1, 66), dtype=torch.uint8),
        w2_weight=torch.empty((experts, 4, 1, 66), dtype=torch.uint8),
        packed_grid=torch.empty(2048, dtype=torch.uint8),
    )
    ids = torch.zeros((tokens, 9), dtype=torch.int32)
    plan = (object(), object(), object())
    calls = []

    def align(got_ids, block, got_experts, *, pad_sorted_ids):
        assert got_ids is ids and block == expected_block
        assert got_experts == experts and pad_sorted_ids
        return plan

    def projection(q, scales, weight, grid, got_ids, **kwargs):
        assert got_ids is ids and kwargs["plan"] is plan
        assert kwargs["block_size"] == expected_block
        calls.append(kwargs.get("down", False))
        value = torch.zeros((tokens * 9, 4))
        return value if kwargs.get("down") else (value.to(torch.int8), value)

    monkeypatch.setattr(alignment, "moe_align_block_size", align)
    monkeypatch.setattr(packed_expert_cache, "get_cache", lambda *a: None)
    monkeypatch.setattr(packed.ops, "quantize", lambda x: (x.to(torch.int8), x))
    monkeypatch.setattr(packed.ops, "packed_experts", projection)
    x = torch.zeros((tokens, 4), dtype=torch.bfloat16)
    output = method.apply(layer, x, torch.ones((tokens, 9)), ids, None, None)
    assert calls == [False, True]
    assert output.shape == x.shape and output.dtype == x.dtype
