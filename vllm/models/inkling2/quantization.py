# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling PTQR manifest and MI100 G128 linear dispatch."""

import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.hy4_neuqi import HY4Linear
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


class Inkling2PTQRConfig(QuantizationConfig):
    def __init__(self, weights):
        super().__init__()
        self.weights = frozenset(weights)
        if not self.weights:
            raise ValueError("Inkling PTQR checkpoint has no quantized weights")

    @classmethod
    def get_name(cls):
        return "inkling2_ptqr"

    def get_supported_act_dtypes(self):
        return [torch.float16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        if (config.get("bits"), config.get("group_size")) != (4, 128):
            raise ValueError("Inkling PTQR requires INT4 weights with group_size=128")
        return cls(config["quantized_modules"])

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, ParallelLMHead):
            if "model.llm.unembed.weight" not in self.weights:
                raise ValueError("Inkling PTQR checkpoint lacks a trained output head")
            return HY4Linear(4)
        if not isinstance(layer, LinearBase):
            return None
        if prefix.startswith("model.layers."):
            prefix = "model.llm." + prefix.removeprefix("model.")
        elif prefix.startswith("visual."):
            prefix = "model." + prefix
        if prefix.endswith(".attn.query_gate"):
            base = prefix.removesuffix("query_gate")
            if all(
                base + name + ".weight" in self.weights
                for name in ("wq_du", "gate_attn")
            ):
                return HY4Linear(4)
            raise ValueError(f"Missing packed attention weights for {prefix}")
        if prefix.endswith(".self_attn.qkv_proj"):
            base = prefix.removesuffix("qkv_proj")
            if all(
                base + name + ".weight" in self.weights
                for name in ("q_proj", "k_proj", "v_proj")
            ):
                return HY4Linear(4)
            raise ValueError(f"Missing packed vision QKV weights for {prefix}")
        if prefix + ".weight" in self.weights:
            return HY4Linear(4)
        return UnquantizedLinearMethod()
