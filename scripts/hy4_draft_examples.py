# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect the target context and hidden states DFly2 actually uses."""

import json

import torch
from torch import nn

from vllm.fs_gpu.safetensor_writer import DirectTensorWriter


class DFlyExamples:
    def __init__(self, path, source, config, span=160):
        self.path = path
        self.source = source
        self.config = config
        cfg = json.loads((source.directory / "config.json").read_text())
        self.layers = set(cfg["dfly_target_kv_layers"])
        self.span = span
        self.path.mkdir(parents=True, exist_ok=True)

    def write(self, filename, values):
        specs = {name: (v.shape, v.dtype) for name, v in values.items()}
        with DirectTensorWriter(self.path / filename, specs, self.source.io) as writer:
            for name, value in values.items():
                writer.append(name, value.contiguous())

    def layer_hook(self, layer, idx):
        if idx not in self.layers:
            return None
        original = layer.self_attn.expand_kv
        rows = {}

        def with_capture(latent, rope):
            index = len(rows)
            count = min(self.span, latent.shape[2])
            rows[f"seq-{index}"] = (
                torch.cat((latent[0, 0, :count], rope[0, 0, :count]), -1)
                .half()
                .contiguous()
            )
            return original(latent, rope)

        layer.self_attn.expand_kv = with_capture

        def finish():
            del layer.self_attn.expand_kv
            self.write(f"target-kv-{idx}.safetensors", rows)

        return finish

    def final(self, corpus):
        from transformers.models.hy_v4_flash.modeling_hy_v4_flash import (
            HYV4FlashHyperHead,
            HYV4FlashRMSNorm,
        )

        with torch.device("meta"):
            head = HYV4FlashHyperHead(self.config)
            norm = HYV4FlashRMSNorm(
                self.config.hidden_size, eps=self.config.rms_norm_eps
            )
        for name in ("fn", "base", "scale"):
            value = self.source.read("model.hc_head.hc_head_" + name)
            setattr(head, "hc_" + name, nn.Parameter(value, requires_grad=False))
        norm.weight = nn.Parameter(
            self.source.read("model.norm.weight").half(), requires_grad=False
        )
        rows = {}
        for idx, (ids, streams) in enumerate(corpus):
            n = min(self.span, ids.shape[1])
            rows[f"ids-{idx}"] = ids[0, :n].contiguous()
            rows[f"hidden-{idx}"] = (
                norm(head(streams[:, :n])).half().squeeze(0).contiguous()
            )
        self.write("target-hidden.safetensors", rows)
