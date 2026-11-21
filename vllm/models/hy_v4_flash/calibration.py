# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layerwise GPU calibration against Tencent's Transformers HY4-Flash implementation."""

import re
from functools import partial
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


def teacher_name(name):
    for original, reference in (
        ("hc_attn_layer.hc_pre.hc_", "attn_hc."),
        ("hc_mlp_layer.hc_pre.hc_", "ffn_hc."),
        ("self_attn.learnable_sink_param", "self_attn.sinks"),
        ("self_attn.linear_gate.", "self_attn.gate_proj."),
        ("mlp.post_fc2_norm.", "post_mlp_layernorm."),
        ("mlp.routed_expert_down_proj.", "mlp.pre_expert_proj."),
        ("mlp.routed_expert_up_proj.", "mlp.post_expert_proj."),
    ):
        if name.startswith(original):
            return reference + name[len(original) :]
    return name


def target_matrix(name):
    return (
        re.fullmatch(
            r"(?:self_attn\.(?:q_a_proj|kv_a_proj_with_mqa|q_b_proj|kv_b_proj|linear_gate|o_proj|"
            r"compressed_indexer\.(?:query_projection|head_weight_projection|token_key_projection))"
            r"|mlp\.(?:(?:shared_experts\.)?(?:gate_proj|up_proj|down_proj)|routed_expert_(?:up|down)_proj))\.weight",
            name,
        )
        is not None
    )


class _EngramRows(nn.Module):
    def __init__(self, source, name, sizes):
        super().__init__()
        self.source = source
        self.name = name
        self.register_buffer(
            "offsets",
            torch.tensor([0] + sizes[:-1], device=source.device).cumsum(0),
            persistent=False,
        )

    def forward(self, ids):
        values = self.source.read_rows(self.name, ids.clamp_min(0) + self.offsets)
        return values.masked_fill(ids.unsqueeze(-1) < 0, 0)


def load_teacher_layer(source, config, idx):
    from transformers.models.hy_v4_flash.modeling_hy_v4_flash import (
        HYV4FlashDecoderLayer,
    )

    with torch.device("meta"):
        layer = HYV4FlashDecoderLayer(config, idx)
    prefix = f"model.layers.{idx}."
    names = {
        teacher_name(n[len(prefix) :]): n for n in source.index if n.startswith(prefix)
    }
    engram = layer.engram_layer
    if engram is not None:
        full = names.pop("engram_layer.multi_head_embedding.embedding.weight")
        engram.multi_head_embedding = _EngramRows(
            source, full, engram.config_head_vocab_sizes
        )
    # Assign each tensor when read. No initialization of an enormous embedding,
    # no CPU state dict, and no duplicate model-sized device allocation.
    state = dict(layer.named_parameters())
    state.update((k, v) for k, v in layer.named_buffers() if k in layer.state_dict())
    if names.keys() != state.keys():
        raise ValueError(
            f"Layer {idx} key mismatch: {sorted(names.keys() ^ state.keys())}"
        )
    for local, name in names.items():
        mod, _, leaf = local.rpartition(".")
        module = layer.get_submodule(mod) if mod else layer
        target = state[local]
        data = source.read(name)
        if data.shape != target.shape:
            raise ValueError(
                f"Teacher shape mismatch for {name}: {data.shape} vs {target.shape}"
            )
        if data.is_floating_point():
            # FP32 router, bias, IHC statistics and the pool's explicitly float
            # projections retain their checkpoint precision.
            dtype = torch.float32 if data.dtype == torch.float32 else torch.float16
            data = data.to(dtype)
        if leaf in module._parameters:
            module._parameters[leaf] = nn.Parameter(data, requires_grad=False)
        else:
            module._buffers[leaf] = data
    if engram is not None:
        engram.head_vocab_sizes = torch.tensor(
            engram.config_head_vocab_sizes, device=source.device
        )
        engram.hash_multipliers = engram.draw_hash_multipliers(
            int(engram.lookup_table.max()) + 1
        ).to(source.device)
    return layer.eval(), names


def _experts_with_moments(experts, moments, hidden_states, ids, weights):
    result = torch.zeros_like(hidden_states)
    for expert in ids.unique().tolist():
        if expert >= experts.num_experts:
            continue
        rows, ranks = torch.where(ids == expert)
        x = hidden_states[rows]
        inputs, outputs = moments
        inputs[expert].add_(x.float().square().sum(0))
        middle = experts._apply_gate(F.linear(x, experts.gate_up_proj[expert]))
        outputs[expert].add_(middle.float().square().sum(0))
        y = F.linear(middle, experts.down_proj[expert]) * weights[rows, ranks, None]
        result.index_add_(0, rows, y.to(result.dtype))
    return result


def calibrate_layer(source, config, idx, layer, names, samples, rotary):
    from transformers.models.hy_v4_flash.modeling_hy_v4_flash import HYV4FlashTextModel

    prefix = f"model.layers.{idx}."
    diagonals = {}
    hooks = []
    if layer.self_attn.compressed_indexer is not None:
        layer.self_attn.compressed_indexer.rotary_emb = rotary

    def observe(key, module, args):
        x = args[0].detach().reshape(-1, args[0].shape[-1])
        diagonals[key].add_(x.float().square().sum(0))

    for local, name in names.items():
        if target_matrix(name[len(prefix) :]):
            module = layer.get_submodule(local.removesuffix(".weight"))
            diagonals[name] = torch.zeros(
                module.weight.shape[1], device=source.device, dtype=torch.float32
            )
            hooks.append(module.register_forward_pre_hook(partial(observe, name)))
    experts = getattr(layer.mlp, "experts", None)
    if experts is not None:
        keys = [
            prefix + "mlp.experts." + leaf for leaf in ("gate_up_proj", "down_proj")
        ]
        moments = [
            torch.zeros(
                (experts.num_experts, k), device=source.device, dtype=torch.float32
            )
            for k in (experts.hidden_dim, experts.intermediate_dim)
        ]
        diagonals.update(zip(keys, moments))
        experts.forward = partial(_experts_with_moments, experts, moments)
    try:
        for sample in samples:
            ids, h = sample
            b, t = ids.shape
            positions = torch.arange(t, device=source.device).expand(b, t)
            xyz = positions.unsqueeze(0).expand(3, b, t)
            if h is None:
                h = source.read_rows("model.embed_tokens.weight", ids).half()
                h = h.unsqueeze(2).expand(-1, -1, config.hc_mult, -1).contiguous()
            # Use the model's independent rotary maps and finite additive mask.
            kind = layer.attention_type
            pos = rotary(h, xyz, kind)
            distance = positions[:, :, None] - positions[:, None, :]
            allowed = distance >= 0
            if kind == "sliding_attention":
                allowed &= distance < config.sliding_window
            mask = torch.zeros((b, 1, t, t), dtype=h.dtype, device=h.device)
            mask.masked_fill_(~allowed.unsqueeze(1), torch.finfo(h.dtype).min)
            ngrams = HYV4FlashTextModel.build_ngram_tokens(
                SimpleNamespace(config=config), ids, xyz
            )
            sample[1] = layer(
                h,
                attention_mask=mask,
                position_embeddings=pos,
                position_ids=xyz,
                indexer_position_ids=xyz,
                ngram_tokens=ngrams,
                engram_mask=(ids != config.engram_pad_id),
                use_cache=False,
            )
    finally:
        for hook in hooks:
            hook.remove()
        if experts is not None:
            del experts.forward
    return diagonals
