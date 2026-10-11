# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.distributed.device_communicators.int8_gs32_all_reduce import (
    _fake,
    selected_mlp,
)


def test_only_mlp_down_projections():
    for prefix in ("model.layers.0.mlp.down_proj", "draft.layers.4.mlp.down_proj"):
        assert selected_mlp(prefix)
    for prefix in ("model.layers.0.linear_attn.out_proj",
                   "model.layers.0.self_attn.o_proj", "model.embed_tokens",
                   "model.layers.0.mlp.gate_up_proj", "fc", "down_proj"):
        assert not selected_mlp(prefix)


def test_fake_preserves_contract():
    for rows in (0, 1, 42, 2048):
        x = torch.empty((rows, 5120), device="meta", dtype=torch.bfloat16)
        y = _fake(x)
        assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device


def test_opt_in_changes_compile_hash(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setenv("VLLM_MLP_CAR_INT8_GS32", "0")
    before = envs.compile_factors()
    monkeypatch.setenv("VLLM_MLP_CAR_INT8_GS32", "1")
    after = envs.compile_factors()
    assert before != after
