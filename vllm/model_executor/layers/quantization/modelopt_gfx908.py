# SPDX-License-Identifier: Apache-2.0
"""ModelOpt GS128 INT8 execution with static dense and packed expert storage."""

import os

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import LinearMethodBase, RowParallelLinear
from vllm.model_executor.layers.quantization.utils import packed_g128_int8 as ops
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


def prepare_static_dense_weight(layer, length, *, codec, start=0):
    """Replace an immutable dense weight only when its INT8 copy fits in full."""
    if os.environ.get("VLLM_BOLT_DENSE_CACHE_DROP_PACKED", "0") != "1":
        return
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    if isinstance(layer, VocabParallelEmbedding) and not isinstance(
        layer, ParallelLMHead
    ):
        return
    result = ops.prepare_dense_weight(
        layer.weight,
        layer.packed_grid,
        length,
        start=start,
        codec=codec,
        ws=getattr(layer, "weight_scale", None),
        wg=getattr(layer, "packed_global_scales", None),
    )
    if result is None:
        return
    weight, groups = result
    released = layer.weight.numel() * layer.weight.element_size()
    layer.weight = Parameter(weight, requires_grad=False)
    layer.register_buffer("weight_gs128", groups, persistent=False)
    for name in (
        "packed_grid",
        "weight_scale",
        "weight_scale_2",
        "packed_global_scales",
    ):
        tensor = getattr(layer, name, None)
        if tensor is not None:
            released += tensor.numel() * tensor.element_size()
            delattr(layer, name)
    from aiter.ops.triton.gemm.basic.packed_dense_cache import _RELEASED_BYTES

    device = weight.device
    _RELEASED_BYTES[device] = _RELEASED_BYTES.get(device, 0) + released


class PackedG128LinearMethod(LinearMethodBase):
    def __init__(self, codec):
        self.codec = codec.lower()
        self.block, self.block_bytes = {"iq2_xxs": (256, 66), "q8_0": (32, 34)}[
            self.codec
        ]

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **attrs,
    ):
        self.k = input_size_per_partition
        self.dtype = params_dtype
        self.start = 0
        blocks = (self.k + self.block - 1) // self.block
        loader = attrs["weight_loader"]
        if isinstance(layer, RowParallelLinear):
            begin = layer.tp_rank * self.k
            self.start = begin % self.block
            blocks = (self.start + self.k + self.block - 1) // self.block

            def loader(param, loaded):
                source = loaded.narrow(1, begin // self.block, blocks)
                if source.shape != param.shape or source.dtype != torch.uint8:
                    raise ValueError("Invalid packed row-parallel checkpoint geometry")
                param.data.copy_(source)

        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes), blocks, self.block_bytes, dtype=torch.uint8
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, dict(input_dim=1, output_dim=0, weight_loader=loader))
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer):
        logger.info_once(
            "gfx908 %s: official packed weights -> tile-local INT8 GS128 MFMA",
            self.codec,
        )
        layer.register_buffer(
            "packed_grid", ops.make_grid(layer.weight.device), persistent=False
        )
        prepare_static_dense_weight(layer, self.k, codec=self.codec, start=self.start)

    def apply_quantized(self, layer, q, scales, *, relu2=False, quant_out=False):
        if hasattr(layer, "weight_gs128"):
            return ops.int8_gemm(
                q,
                scales,
                layer.weight,
                layer.weight_gs128,
                relu2=relu2,
                quant_out=quant_out,
            )
        return ops.packed_gemm(
            q,
            scales,
            layer.weight,
            layer.packed_grid,
            start=self.start,
            codec=self.codec,
            relu2=relu2,
            quant_out=quant_out,
        )

    def apply(self, layer, x, bias=None):
        shape = x.shape
        q, s = ops.quantize(x.reshape(-1, shape[-1]).contiguous())
        y = self.apply_quantized(layer, q, s)
        if bias is not None:
            y += bias
        return y.to(x.dtype).view(*shape[:-1], layer.weight.shape[0])

    def embedding(self, layer, input_):
        return ops.packed_embedding(layer.weight, input_, self.k, self.dtype)


class PackedG128MoEMethod(FusedMoEMethodBase):
    def __init__(self, moe_config):
        super().__init__(moe_config)
        if moe_config.activation != MoEActivation.RELU2_NO_MUL:
            raise ValueError("Packed IQ2 experts currently require non-gated ReLU2")
        if moe_config.moe_parallel_config.ep_size != 1:
            raise ValueError("Packed IQ2 experts require tensor parallelism without EP")

    def maybe_roundup_sizes(
        self,
        hidden_size,
        intermediate_size_per_partition,
        act_dtype,
        moe_parallel_config,
    ):
        return hidden_size, intermediate_size_per_partition

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **attrs,
    ):
        self.local_i = intermediate_size_per_partition
        begin = self.moe.tp_rank * self.local_i
        self.start = begin % 256
        self.first_block = begin // 256
        blocks = (self.start + self.local_i + 255) // 256
        for name, shape in (
            ("w13_weight", (num_experts, self.local_i, hidden_size // 256, 66)),
            ("w2_weight", (num_experts, hidden_size, blocks, 66)),
        ):
            param = Parameter(
                torch.empty(shape, dtype=torch.uint8), requires_grad=False
            )
            set_weight_attrs(param, dict(weight_loader=self.weight_loader))
            layer.register_parameter(name, param)

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        if shard_id == "w2":
            source = loaded_weight.narrow(1, self.first_block, param.shape[2])
        elif shard_id == "w1":
            source = loaded_weight.narrow(
                0, self.moe.tp_rank * self.local_i, self.local_i
            )
        else:
            raise ValueError(f"Unexpected non-gated expert shard {shard_id}")
        if source.shape != param[expert_id].shape or source.dtype != torch.uint8:
            raise ValueError(
                f"Invalid packed expert geometry for {weight_name}: "
                f"source {tuple(source.shape)} {source.dtype}, "
                f"destination {tuple(param[expert_id].shape)}"
            )
        param.data[expert_id].copy_(source)
        return True if return_success else None

    def get_fused_moe_quant_config(self, layer):
        return None

    def process_weights_after_loading(self, layer):
        layer.register_buffer(
            "packed_grid", ops.make_grid(layer.w13_weight.device), persistent=False
        )
        if os.environ.get("VLLM_BOLT_EXPERT_CACHE_SLOTS") == "auto":
            from aiter.ops.triton.gemm.basic.packed_expert_cache import register_layer

            register_layer(layer.w13_weight, layer.w2_weight, self.local_i, self.start)

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if layer.apply_router_weight_on_input:
            raise ValueError("Packed IQ2 experts require output router weighting")
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )

        # Share packed weight reconstruction across more prefill rows when
        # average expert occupancy can fill a tile. Keep sparse batches and
        # native decode on their original 16-row alignment plan.
        block_size = (
            64
            if x.shape[0] > 64 and topk_ids.numel() >= layer.w13_weight.shape[0] * 64
            else 16
        )
        plan = moe_align_block_size(
            topk_ids, block_size, layer.w13_weight.shape[0], pad_sorted_ids=True
        )
        cache = None
        prefill_admit = x.shape[0] > 64 and os.environ.get(
            "VLLM_BOLT_PREFILL_ADMIT", "0"
        ) == "1"
        if x.shape[0] <= 64 or prefill_admit:
            from aiter.ops.triton.gemm.basic.packed_expert_cache import get_cache

            cache = get_cache(
                layer.w13_weight,
                layer.w2_weight,
                layer.packed_grid,
                self.local_i,
                self.start,
            )
            if cache is not None:
                if prefill_admit:
                    # Prefill-informed admission: a 64-token subsample keeps
                    # the single-CTA route histogram cheap, and the forced
                    # rerank lands convergence inside this forward.
                    cache.step(topk_ids[: min(64, x.shape[0])], rerank=True)
                else:
                    cache.step(topk_ids)
        q, s = ops.quantize(x.contiguous())
        mid, ms = ops.packed_experts(
            q,
            s,
            layer.w13_weight,
            layer.packed_grid,
            topk_ids,
            plan=plan,
            cache=cache.up if cache is not None else None,
            block_size=block_size,
        )
        y = ops.packed_experts(
            mid,
            ms,
            layer.w2_weight,
            layer.packed_grid,
            topk_ids,
            start=self.start,
            down=True,
            plan=plan,
            cache=cache.down if cache is not None else None,
            block_size=block_size,
        )
        return ops.combine_experts(y, topk_weights, x.dtype)


PackedG128MoEMethod.weight_loader.supports_moe_loading = True
