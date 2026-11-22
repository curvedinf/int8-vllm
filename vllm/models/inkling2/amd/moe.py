# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling expert-parallel W4A8 MoE using the shared gfx908 G128 GEMMs."""

import torch
from torch import nn
from torch.nn import functional as F

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.hy4_neuqi import (
    _activation,
    _prepare_wave,
    _weight_parts,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs


class Inkling2Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.fused_router = get_current_vllm_config().additional_config.get(
            "inkling_fused_router", False
        )
        self.weight = nn.Parameter(
            torch.empty(
                config.n_routed_experts, config.hidden_size, dtype=torch.float32
            ),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32),
            requires_grad=False,
        )
        for param in self.parameters():
            set_weight_attrs(param, {"weight_loader": default_weight_loader})

    def forward(self, x):
        logits = F.linear(x.float(), self.weight)
        if self.fused_router:
            from vllm import _custom_ops as ops

            weights = logits.new_empty((len(x), self.topk))
            ids = torch.empty_like(weights, dtype=torch.int32)
            ops.topk_sigmoid(
                weights,
                ids,
                torch.empty_like(ids),
                logits,
                True,
                self.bias,
                self.route_scale,
            )
            return ids, weights
        scores = logits.sigmoid()
        # A tie in the bias-corrected score goes to the lower expert ID.
        ids = torch.argsort(scores + self.bias, descending=True, stable=True)[
            :, : self.topk
        ]
        weights = scores.gather(-1, ids)
        return ids.int().contiguous(), weights * (
            self.route_scale / weights.sum(-1, keepdim=True).clamp_min(1e-20)
        )


class Inkling2Experts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.world = get_tensor_model_parallel_world_size()
        total = config.n_routed_experts
        if total % self.world:
            raise ValueError("Expert parallelism requires equally sized shards")
        self.local = total // self.world
        self.start = get_tensor_model_parallel_rank() * self.local
        self.inter = config.intermediate_size
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        options = get_current_vllm_config().additional_config
        self.fused_activation = options.get("inkling_fused_expert_activation", False)
        self.symmetric = options.get("inkling_symmetric_experts", False)
        # EP keeps all nine K128 groups of each 1152-wide down projection local.
        rank_ids = torch.arange(total, dtype=torch.int32) - self.start
        self.register_buffer(
            "expert_map",
            torch.where((rank_ids >= 0) & (rank_ids < self.local), rank_ids, -1),
            persistent=False,
        )
        for projection, n, k in (
            ("w13", 2 * self.inter, config.hidden_size),
            ("w2", config.hidden_size, self.inter),
        ):
            for leaf, divisor, dtype in _weight_parts(4):
                param = nn.Parameter(
                    torch.empty(self.local, n, k // divisor, dtype=dtype),
                    requires_grad=False,
                )
                self.register_parameter(f"{projection}_{leaf}", param)
                loader = self._load_gate_up if projection == "w13" else self._load_down
                set_weight_attrs(param, {"weight_loader": loader})

    def _load_gate_up(self, param, value):
        local = value.narrow(0, self.start, self.local)
        # Original checkpoint alternates each gate row with its up row.
        param.data.copy_(
            local.unflatten(1, (self.inter, 2)).transpose(1, 2).flatten(1, 2)
        )

    def _load_down(self, param, value):
        param.data.copy_(value.narrow(0, self.start, self.local))

    def process_weights_after_loading(self):
        if self.symmetric:
            # Fixed-zero arithmetic is exact only for a symmetric checkpoint.
            self.symmetric = bool(
                self.w13_weight_zero.eq(8).all() & self.w2_weight_zero.eq(8).all()
            )
        for prefix in ("w13_", "w2_"):
            _prepare_wave(self, 4, prefix, layout=2)

    def forward(self, x, ids, weights):
        from aiter.ops.gemm_packed_native import affine_g128

        tile = 32 if len(x) >= 768 else 16
        plan = moe_align_block_size(
            ids,
            tile,
            len(self.expert_map),
            expert_map=self.expert_map,
            pad_sorted_ids=True,
            ignore_invalid_experts=True,
        )
        kwargs = dict(
            plan=plan,
            topk=ids.shape[1],
            wave_format=2,
            waves=4,
            triton_tiles=(tile, 64 if len(x) <= 16 else 128, 4),
            packet_load=True,
            output_dtype=torch.float16,
            symmetric_w4=self.symmetric,
        )
        quant_bits = 8 if self.symmetric else 4  # fixed zero needs no activation sums
        q, s, sums = _activation(x.half(), quant_bits)
        gate_up = affine_g128(
            q,
            s,
            sums,
            self.w13_weight,
            self.w13_weight_scale,
            self.w13_weight_zero,
            **kwargs,
        )
        if self.fused_activation:
            from aiter.ops.triton.gemm.basic.gemm_packed_g128 import (
                combine_experts,
                quantize_routed_swiglu,
            )

            q, s, sums = quantize_routed_swiglu(
                gate_up,
                ids,
                self.start,
                self.start + self.local,
                gate_cap=self.gate_cap,
                up_cap=self.up_cap,
                with_sums=not self.symmetric,
            )
        else:
            gate, up = gate_up.float().chunk(2, dim=-1)
            # The sigmoid sees the unclipped gate. The linear factors use tanh caps.
            gate_factor = (gate / self.gate_cap).tanh() * self.gate_cap
            up_factor = (up / self.up_cap).tanh() * self.up_cap
            mid = gate_factor * gate.sigmoid() * up_factor
            q, s, sums = _activation(mid.half(), quant_bits)
        down = affine_g128(
            q,
            s,
            sums,
            self.w2_weight,
            self.w2_weight_scale,
            self.w2_weight_zero,
            down=True,
            **kwargs,
        )
        # GEMMs do not write rows owned by a peer. Never read them into the sum.
        if self.fused_activation:
            output = combine_experts(
                down,
                weights,
                torch.float32,
                ids=ids,
                start=self.start,
                stop=self.start + self.local,
            )
        else:
            mask = self.expert_map[ids.long()] >= 0
            local = torch.where(mask[..., None], down.view(*ids.shape, -1), 0)
            output = (local * weights[..., None]).sum(1)
        return tensor_model_parallel_all_reduce(output).to(x.dtype)


class Inkling2MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = Inkling2Gate(config)
        self.experts = Inkling2Experts(config)

    def forward(self, x):
        ids, weights = self.gate(x)
        return self.experts(x, ids, weights)
