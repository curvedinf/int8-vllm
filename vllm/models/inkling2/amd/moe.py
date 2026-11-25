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
from vllm.model_executor.layers.fused_moe.expert_map_manager import expert_range
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
from vllm.triton_utils import tl, tldevice, triton

from .ops.skinny_gemm import DECODE_TOKENS, router_logits
from .ops.streams import side_stream
from .ops.timeline import mark as timeline_mark


@triton.jit
def _top8_router(Logits, Bias, Indices, Weights, SCALE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    expert = tl.arange(0, 128)
    x = tl.load(Logits + row * 128 + expert)
    scores = tl.div_rn(1.0, 1.0 + tldevice.exp(-x))
    biased = scores + tl.load(Bias + expert)
    bits = biased.to(tl.uint32, bitcast=True)
    ordered = tl.where(bits >> 31 != 0, ~bits, bits ^ 0x80000000)
    ordered = tl.where(biased == 0.0, 0x80000000, ordered)
    # Match stable descending argsort, including ties, signed zero and NaN.
    ordered = tl.where(biased != biased, 0xFFFFFFFF, ordered)
    pair = (ordered.to(tl.uint64) << 32) | (127 - expert).to(tl.uint64)
    indices = 127 - (tl.topk(pair, 8) & 0xFFFFFFFF).to(tl.int32)
    weights = tl.gather(scores, indices, 0)
    lane = tl.arange(0, 8)
    partial = weights
    # Match ATen's increasing-distance warp shuffles for an eight-way sum.
    for bit in tl.static_range(3):
        partial += tl.gather(partial, lane ^ (1 << bit), 0)
    total = tl.sum(tl.where(lane == 0, partial, 0.0), 0)
    factor = tl.div_rn(1.0, tl.where(total < 1e-20, 1e-20, total)) * SCALE
    tl.store(Indices + row * 8 + lane, indices)
    tl.store(Weights + row * 8 + lane, weights * factor)


def _stable_top8(logits, bias, route_scale):
    shape = (len(logits), 8)
    indices = logits.new_empty(shape, dtype=torch.int32)
    weights = logits.new_empty(shape)
    _top8_router[(len(logits),)](
        logits, bias, indices, weights, route_scale, num_warps=1, enable_fp_fusion=False
    )
    return indices, weights


class Inkling2Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        options = get_current_vllm_config().additional_config
        self.stable_top8 = (
            config.n_routed_experts == 128
            and self.topk == 8
            and options.get("inkling_stable_top8", False)
        )
        self.fused_router = options.get("inkling_fused_router", False)
        # "triton": split-K kernel instead of the single-CTA rocBLAS GEMM that
        # FP32 `tokens x experts` selects at decode sizes. The accumulation
        # order differs from rocBLAS in the last FP32 bits.
        self.kernel = options.get("inkling_router_kernel", "rocblas")
        if self.kernel not in ("rocblas", "triton"):
            raise ValueError(f"inkling_router_kernel={self.kernel!r}")
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
        if self.kernel == "triton" and len(x) <= DECODE_TOKENS:
            logits = router_logits(x, self.weight)
        else:
            logits = F.linear(x.float(), self.weight)
        if self.stable_top8:
            return _stable_top8(logits, self.bias, self.route_scale)
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
        options = get_current_vllm_config().additional_config
        # One contiguous expert range per rank. `expert_rank_counts` lets a
        # slower rank own fewer experts; every collective keeps its shape and
        # only the grouping of the expert sum changes.
        self.start, self.local = expert_range(
            self.world,
            get_tensor_model_parallel_rank(),
            config.n_routed_experts,
            get_current_vllm_config().parallel_config.expert_rank_counts,
        )
        self.inter = config.intermediate_size
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        self.fused_activation = options.get("inkling_fused_expert_activation", False)
        self.register_buffer("activation_factors", None, persistent=False)
        self.fused_combine = options.get(
            "inkling_fused_expert_combine", self.fused_activation
        )
        self.prefill_tile_threshold = (
            384 if options.get("inkling_mi100_expert_tiles", False) else 768
        )
        # Large prefills route enough rows per expert for a 64-row tile, which
        # re-reads each expert's weights half as often (same per-element math).
        self.large_tile = options.get("inkling_moe_large_tile")
        if self.large_tile not in (None, 64):
            raise ValueError(f"inkling_moe_large_tile={self.large_tile!r}: use 64")
        self.large_tile_tokens = options.get("inkling_moe_large_tile_tokens", 4096)
        self.symmetric = options.get("inkling_symmetric_experts", False)
        # The combine sums the experts in FP32; the partial may be rounded to
        # FP16 before the TP all-reduce (half the message bytes).
        self.combine_dtype = (
            torch.float16 if options.get("inkling_fp16_moe_allreduce", False)
            else torch.float32
        )
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

    def forward(self, x, ids, weights, quant=None):
        from aiter.ops.gemm_packed_native import affine_g128

        if self.large_tile and len(x) >= self.large_tile_tokens:
            tile = self.large_tile
        else:
            tile = 32 if len(x) >= self.prefill_tile_threshold else 16
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
        if quant is None:
            q, s, sums = _activation(x.half(), quant_bits)
        else:
            q, s, sums = quant
        timeline_mark("aquant")
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
                **(
                    {"factors": self.activation_factors}
                    if self.activation_factors is not None
                    else {}
                ),
            )
        else:
            gate, up = gate_up.float().chunk(2, dim=-1)
            # The sigmoid sees the unclipped gate. The linear factors use tanh caps.
            gate_factor = (gate / self.gate_cap).tanh() * self.gate_cap
            up_factor = (up / self.up_cap).tanh() * self.up_cap
            mid = gate_factor * gate.sigmoid() * up_factor
            q, s, sums = _activation(mid.half(), quant_bits)
        timeline_mark("exp_up")
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
        timeline_mark("exp_dn")
        # GEMMs do not write rows owned by a peer. Never read them into the sum.
        if self.fused_combine:
            from aiter.ops.triton.gemm.basic.gemm_packed_g128 import combine_experts

            output = combine_experts(
                down,
                weights,
                self.combine_dtype,
                ids=ids,
                start=self.start,
                stop=self.start + self.local,
            )
        else:
            mask = self.expert_map[ids.long()] >= 0
            local = torch.where(mask[..., None], down.view(*ids.shape, -1), 0)
            output = (local * weights[..., None]).sum(1)
        timeline_mark("exp_cb")
        output = tensor_model_parallel_all_reduce(output)
        timeline_mark("ar2")
        return output.to(x.dtype)


class Inkling2MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = Inkling2Gate(config)
        self.experts = Inkling2Experts(config)
        # The routed-expert activation quantization only reads `x`; on a side
        # stream it executes beside the router with the same kernel and result.
        options = get_current_vllm_config().additional_config
        self._side_stream = (
            side_stream(0)
            if options.get("inkling_parallel_router_quant", False)
            else None
        )

    def forward(self, x):
        side = self._side_stream
        quant = None
        if side is not None:
            main = torch.cuda.current_stream()
            side.wait_stream(main)
            with torch.cuda.stream(side):
                quant = _activation(x.half(), 8 if self.experts.symmetric else 4)
        ids, weights = self.gate(x)
        timeline_mark("route")
        if side is not None:
            main.wait_stream(side)
        return self.experts(x, ids, weights, quant=quant)
