# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 model (AMD).

Decoder layers are tied together by block attention residuals: instead of a
single residual stream, each sublayer input is a learned softmax mix over the
outputs of previous blocks of `n_layers_per_attn_res_block` layers plus the
running partial block. Residual blocks are stored in fp16; mixes are fp32.
"""

from collections import Counter
from collections.abc import Callable, Hashable, Iterable
from typing import Any

import regex as re
import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsEncoderCudaGraph,
    SupportsMultiModal,
)
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from ..configs import Inkling2MMConfig, Inkling2ModelConfig
from ..processing import (
    IMAGE_PLACEHOLDER,
    Inkling2DummyInputsBuilder,
    Inkling2MultiModalProcessor,
    Inkling2ProcessingInfo,
)
from ..quantization import Inkling2PTQRConfig
from .attention import Inkling2Attention, inkling_kv_spec
from .moe import Inkling2MoE
from .ops.attention import compute_log_scaling_tau
from .ops.attn_res import attn_res, normalize_residual_rows, weighted_values
from .ops.linear import controller_output, mix_message
from .ops.norm import rms_norm
from .ops.streams import side_stream
from .ops.timeline import ENABLED as _GRAPH_TIMELINE
from .ops.timeline import TimelineSink, set_sink
from .sconv import ATTN_STREAM, MLP_STREAM, Inkling2ConvState, Inkling2ShortConv
from .vision import Inkling2VisionEncoder, VisionMetadata


class Inkling2RMSNorm(nn.Module):
    """RMSNorm; `num_inputs > 1` stacks the weights of per-input norms (the
    block AttnRes mixes apply them in the same kernel)."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        out_dtype: torch.dtype | None = torch.float16,
        num_inputs: int = 1,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(num_inputs, hidden_size), requires_grad=False
        )
        self.weight.weight_loader = self._load
        self.eps = eps
        self.out_dtype = out_dtype

    @staticmethod
    def _load(param: nn.Parameter, loaded: torch.Tensor, shard_id: int = 0) -> None:
        param.data[shard_id].copy_(loaded)

    def forward(self, x: torch.Tensor, index: int = 0) -> torch.Tensor:
        return rms_norm(x, self.weight[index], self.eps, self.out_dtype)


class Inkling2AttnRes(nn.Module):
    """Pseudo-queries of the block attention residuals (fp32), pre-scaled by
    `hidden_size**-0.5` at load time."""

    def __init__(self, config: Inkling2ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        local_ids = set(config.local_layer_ids)
        self.attn_queries = nn.ParameterList(
            nn.Parameter(
                torch.zeros(
                    (1, hidden) if i in local_ids else (3, hidden),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            for i in range(1, config.num_hidden_layers)
        )
        self.mlp_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(1, hidden, dtype=torch.float32), False)
            for _ in range(config.num_hidden_layers)
        )
        self.output_query = nn.Parameter(
            torch.zeros(1, hidden, dtype=torch.float32), requires_grad=False
        )
        for param in self.parameters():
            param.weight_loader = self._load_query
        self.scale = hidden**-0.5

    def _load_query(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        param.data.copy_(weight.view(param.shape)).mul_(self.scale)


class MessageController(nn.Module):
    def __init__(self, width: int, rank: int):
        super().__init__()
        options = get_current_vllm_config().additional_config
        self.fused = rank in (32, 64) and options.get(
            "inkling_fused_message_gate", False
        )
        self.fused_normalization = options.get("inkling_residual_normalization", False)
        self.epilogue = options.get("inkling_controller_epilogue")
        self.down_weight = nn.Parameter(torch.empty(rank, 2 * width), False)
        self.forget_weight = nn.Parameter(torch.empty(width, rank), False)
        self.gain_weight = nn.Parameter(torch.empty(width, rank), False)

    def forward(self, dest, message):
        if self.fused:
            return mix_message(
                dest, message, self.down_weight, self.forget_weight, self.gain_weight
            )
        if self.fused_normalization and self.down_weight.dtype == torch.float16:
            pair = normalize_residual_rows(dest.unsqueeze(0), message, controller=True)
        else:
            pair = torch.stack((dest, message), dim=-2).float()
            pair = pair * torch.rsqrt(pair.square().mean(-1, keepdim=True) + 1e-6)
        h = nn.functional.linear(
            pair.flatten(-2).to(self.down_weight.dtype), self.down_weight
        )
        if self.epilogue:
            h = h.float()
            gain = nn.functional.linear(h, self.gain_weight.float())
            forget = nn.functional.linear(h, self.forget_weight.float())
            return controller_output(
                dest, message, forget, gain, fuse_tanh=self.epilogue == "tanh"
            )
        gain = nn.functional.linear(h.float(), self.gain_weight.float()).tanh()
        forget = nn.functional.linear(h.float(), self.forget_weight.float()).tanh()
        return dest * (1 - forget) + message * gain


class Inkling2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2PTQRConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.is_local = layer_id in config.local_layer_ids
        num_kv_heads = (
            config.swa_num_key_value_heads
            if self.is_local
            else config.num_key_value_heads
        )
        head_dim = config.swa_head_dim if self.is_local else config.head_dim
        kv_page_per_head = inkling_kv_spec(
            get_current_vllm_config(), 1, head_dim, None
        ).page_size_bytes
        visits = len(config.layer_visit_ids(layer_id))
        self.conv_states = nn.ModuleList(
            Inkling2ConvState(
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                kv_page_per_head=kv_page_per_head,
                hidden_size=config.hidden_size,
                kernel_size=config.sconv_kernel_size,
                prefix=f"{prefix}.conv_state.pass_{p}",
            )
            for p in range(visits)
        )
        self.attn = Inkling2Attention(
            config,
            self.is_local,
            self.conv_states,
            prefix=f"{prefix}.attn",
            quant_config=quant_config,
        )
        self.controllers = nn.ModuleDict()
        if visits > 1:
            for stream in ("attn", "k", "v", "mlp"):
                self.controllers[stream] = MessageController(
                    config.hidden_size, config.loop_config["gate_rank"]
                )
        eps = config.rms_norm_eps
        # Local layers: attn_norm; full layers: q/k/v_input_norm, stacked.
        self.num_attn_inputs = 1 if self.is_local else 3
        self.attn_norm = Inkling2RMSNorm(
            config.hidden_size, eps, num_inputs=self.num_attn_inputs
        )
        self.mlp_norm = Inkling2RMSNorm(config.hidden_size, eps)
        self.mlp = Inkling2MoE(config)
        self.attn_sconv, self.mlp_sconv = (
            Inkling2ShortConv(
                config.hidden_size,
                config.sconv_kernel_size,
                self.conv_states[0],
                stream,
            )
            for stream in (ATTN_STREAM, MLP_STREAM)
        )


class Inkling2Model(nn.Module, EagleModelMixin):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: Inkling2PTQRConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.block_size = config.n_layers_per_attn_res_block
        self.normalize_values = config.attn_res_normalize_values
        options = get_current_vllm_config().additional_config
        _mix_mode = options.get("inkling_fused_loop_mix", False)
        # True fuses every residual mix; "decode" keeps the unfused path for
        # prefill-sized batches (more than 128 rows).
        self.fused_loop_mix = _mix_mode in (True, "decode")
        self.fused_loop_mix_decode_only = _mix_mode == "decode"
        self.fused_normalization = options.get("inkling_residual_normalization", False)
        self.direct_weighted_sum = options.get("inkling_direct_weighted_sum", False)
        # The attention-side message controllers only read `inputs` and
        # `messages`; on side streams they execute concurrently in the
        # captured graph with the same kernels and results.
        self._side_streams = (
            [side_stream(j) for j in range(3)]
            if options.get("inkling_parallel_controllers", False)
            else []
        )
        # In a second pass the bank and stack mixes are independent; the bank
        # mix (the messages) can run on the shared side stream beside the
        # stack mix (the attention inputs) with the same kernels and results.
        self._message_stream = (
            side_stream(0) if options.get("inkling_parallel_messages", False) else None
        )
        self.cache_message_values = options.get("inkling_cache_message_values", False)
        self.batch_message_max_tokens = options.get(
            "inkling_batch_message_max_tokens", 0
        )
        self.timeline = _GRAPH_TIMELINE
        self.timeline_sink = TimelineSink() if _GRAPH_TIMELINE else None
        self.embed_tokens = VocabParallelEmbedding(
            config.padded_vocab_size, config.hidden_size
        )
        self.embed_norm = Inkling2RMSNorm(
            config.hidden_size, config.rms_norm_eps, out_dtype=None
        )
        self.layers = nn.ModuleList(
            Inkling2DecoderLayer(config, i, quant_config, f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        )
        self.attn_res = Inkling2AttnRes(config)
        self.norm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)
        core = config.loop_core_layers
        self.visits = (
            [(i, 0) for i in range(core.start)]
            + [(i, p) for p in range(config.num_loop_passes) for i in core]
            + [(i, 0) for i in range(core.stop, config.num_hidden_layers)]
        )
        shape = 2 * len(core), config.hidden_size
        self.middle_cycle_cross_pass_queries = nn.Parameter(
            torch.empty(shape, dtype=torch.float32), False
        )
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
            torch.empty(shape, dtype=torch.float32), False
        )
        self.register_buffer(
            "message_queries",
            torch.empty(len(core), 4, config.hidden_size, dtype=torch.float32),
            persistent=False,
        )

    def process_weights_after_loading(self):
        q = self.middle_cycle_cross_pass_queries.unflatten(0, (-1, 2))
        kv = self.middle_cycle_cross_pass_kv_queries.unflatten(0, (-1, 2))
        self.message_queries.copy_(
            torch.cat((q[:, :1], kv, q[:, 1:]), 1) * self.config.hidden_size**-0.5
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(self.embed_tokens(input_ids))

    def _mark(self, label: str, visit: int) -> None:
        if self.timeline_sink is not None:
            self.timeline_sink.mark(label, visit)

    @staticmethod
    def _control_inputs(layer, inputs, messages, j, partial):
        """Attention input `j` after its message controller and norm."""
        return layer.attn_norm(
            layer.controllers[("attn", "k", "v")[j]](
                inputs[0 if layer.is_local else j], messages[j]
            ),
            0 if layer.is_local else j,
        ).to(partial.dtype)

    def _mix(self, blocks, partial, query, norm=None, addend=None):
        if norm is not None or (
            self.fused_loop_mix
            and not (self.fused_loop_mix_decode_only and len(partial) > 128)
        ):
            return attn_res(
                blocks,
                partial,
                addend,
                query,
                norm.weight if norm is not None else None,
                norm.eps if norm is not None else self.config.rms_norm_eps,
                self.normalize_values,
            )
        if self.fused_normalization:
            values = normalize_residual_rows(blocks, partial, addend)
            return self._query_values(values, query)
        last = partial.float()
        if addend is not None:
            last = last + addend.float()
        values = torch.cat((blocks.float(), last.unsqueeze(0)))
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
        return self._query_values(values, query)

    def _query_values(self, values, query):
        scores = torch.einsum("std,qd->qts", values, query)
        weights = scores.softmax(-1)
        if self.direct_weighted_sum:
            return weighted_values(weights, values)
        return torch.einsum("qts,std->qtd", weights, values)

    def forward(self, positions, inputs_embeds):
        core = self.config.loop_core_layers
        depth = (len(self.layers) + self.block_size - 1) // self.block_size
        stack = inputs_embeds.new_empty((depth + 1, *inputs_embeds.shape))
        stack[0].copy_(inputs_embeds)
        count = 0
        tau = compute_log_scaling_tau(
            positions, self.config.log_scaling_n_floor, self.config.log_scaling_alpha
        )
        prelude = bank = pass_messages = message_values = None
        aux = []
        set_sink(self.timeline_sink)
        if self.timeline_sink is not None:
            self.timeline_sink.begin()
            self._mark("start", -1)
        for visit_id, (i, visit) in enumerate(self.visits):
            if i == core.start:
                if not visit:
                    prelude = stack[count].clone()
                    prelude_count = count
                else:
                    previous_pass = stack[prelude_count + 1 : count + 1]
                    if (
                        self.cache_message_values
                        and not self.fused_loop_mix
                        and len(inputs_embeds) > self.batch_message_max_tokens
                    ):
                        # Each core layer queries the same previous-pass values.
                        # Materialize them before the stack is reused for this pass.
                        values = previous_pass.float()
                        message_values = values * torch.rsqrt(
                            values.square().mean(-1, keepdim=True) + 1e-6
                        )
                    else:
                        bank = previous_pass.clone()
                    if len(inputs_embeds) <= self.batch_message_max_tokens:
                        # Every layer in this pass reads the same previous-pass
                        # states. Batch its four message queries on small batches.
                        pass_messages = self._mix(
                            bank[:-1], bank[-1], self.message_queries.flatten(0, 1)
                        ).unflatten(0, (len(core), 4))
                        self._mark("passmsg", -1)
                    count = prelude_count
                    stack[count].copy_(prelude)
            layer = self.layers[i]
            owner = layer.conv_states[visit]
            partial = stack[count]
            if visit_id in self.aux_hidden_state_layers:
                aux.append(partial.clone())
            messages = None
            side = None
            if visit:
                queries = self.message_queries[i - core.start]
                if pass_messages is not None:
                    messages = pass_messages[i - core.start]
                elif message_values is not None:
                    messages = self._query_values(message_values, queries)
                elif self._message_stream is not None:
                    side = self._message_stream
                    main = torch.cuda.current_stream()
                    side.wait_stream(main)
                    with torch.cuda.stream(side):
                        messages = self._mix(bank[:-1], bank[-1], queries)
                else:
                    messages = self._mix(bank[:-1], bank[-1], queries)
                self._mark("msg", visit_id)
            if count:
                query = self.attn_res.attn_queries[i - 1]
                inputs = self._mix(
                    stack[:count], partial, query, None if visit else layer.attn_norm
                )
            else:
                inputs = [
                    layer.attn_norm(partial, j) for j in range(layer.num_attn_inputs)
                ]
            self._mark("inmix", visit_id)
            if side is not None:
                torch.cuda.current_stream().wait_stream(side)
                self._mark("msgjoin", visit_id)
            if messages is not None:
                sources = inputs
                if self._side_streams:
                    main = torch.cuda.current_stream()
                    for stream in self._side_streams:
                        stream.wait_stream(main)
                    controlled = []
                    for j, stream in enumerate(self._side_streams):
                        with torch.cuda.stream(stream):
                            controlled.append(
                                self._control_inputs(
                                    layer, sources, messages, j, partial
                                )
                            )
                    for stream in self._side_streams:
                        main.wait_stream(stream)
                    inputs = controlled
                else:
                    inputs = [
                        self._control_inputs(layer, sources, messages, j, partial)
                        for j in range(3)
                    ]
            q, k, v = (inputs[0],) * 3 if layer.is_local and not visit else inputs
            self._mark("a_mix", visit_id)
            projected = layer.attn(
                positions, q, k, v, None if layer.is_local else tau, pass_idx=visit
            )
            self._mark("attn", visit_id)
            if self.timeline:
                projected = tensor_model_parallel_all_reduce(projected)
                self._mark("ar", visit_id)
                attention = layer.attn_sconv(projected, positions, owner=owner)
            else:
                attention = layer.attn_sconv(
                    tensor_model_parallel_all_reduce(projected), positions, owner=owner
                )
            self._mark("ar_sconv", visit_id)
            first = i % self.block_size == 0
            mixed = self._mix(
                stack[: count + first],
                attention if first else partial,
                self.attn_res.mlp_queries[i],
                None if visit else layer.mlp_norm,
                None if first else attention,
            )[0]
            if messages is not None:
                mixed = layer.mlp_norm(layer.controllers["mlp"](mixed, messages[3])).to(
                    partial.dtype
                )
            self._mark("mix", visit_id)
            count += first
            moe_out = layer.mlp(mixed)
            self._mark("moe", visit_id)
            layer.mlp_sconv(
                moe_out,
                positions,
                addend=attention,
                residual=None if first else partial,
                out=stack[count],
                owner=owner,
            )
            self._mark("moe_sconv", visit_id)
        output = self._mix(
            stack[:count], stack[count], self.attn_res.output_query, self.norm
        )[0]
        if self.aux_hidden_state_layers:
            if len(self.visits) in self.aux_hidden_state_layers:
                aux.append(output)
            self._mark("output", len(self.visits))
            self._finish_timeline()
            return output, aux
        self._mark("output", len(self.visits))
        self._finish_timeline()
        return output

    def _finish_timeline(self) -> None:
        if self.timeline_sink is not None:
            self.timeline_sink.finish()


@MULTIMODAL_REGISTRY.register_processor(
    Inkling2MultiModalProcessor,
    info=Inkling2ProcessingInfo,
    dummy_inputs=Inkling2DummyInputsBuilder,
)
class Inkling2LoopForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsEncoderCudaGraph,
    SupportsEagle3,
):
    # Weights mirror the checkpoint names after these substitutions, except for
    # the fused per-layer weights in ``_shards``.
    _renames = (
        ("model.llm.unembed.", "lm_head."),
        ("model.llm.embed.", "model.embed_tokens."),
        ("model.llm.", "model."),
        (".q_l2_clip.scale", ".q_scale"),
        (".rel_l2_clip.scale", ".r_scale"),
        (".rel_logits_proj.proj", ".rel_logits_proj"),
    )
    _shards = {
        ".q_input_norm.": (".attn_norm.", 0),
        ".k_input_norm.": (".attn_norm.", 1),
        ".v_input_norm.": (".attn_norm.", 2),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Inkling2MMConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        assert quant_config is None or isinstance(quant_config, Inkling2PTQRConfig)
        assert vllm_config.parallel_config.pipeline_parallel_size == 1
        if vllm_config.model_config.enable_prompt_embeds:
            # The reference applies embed_norm to raw input embeddings, which
            # cannot be told apart from already-normalized ones here.
            raise ValueError("Inkling2 does not support prompt embeddings")
        self.config = config
        self.text_config = text_config = config.text_config
        self.tabulate_swiglu = vllm_config.additional_config.get(
            "inkling_tabulate_swiglu", False
        )
        if vllm_config.additional_config.get("inkling_merge_query_gate", False):
            self._shards = {
                **self._shards,
                ".attn.wq_du.": (".attn.query_gate.", 0),
                ".attn.gate_attn.": (".attn.query_gate.", 1),
            }
        if vllm_config.additional_config.get("inkling_merge_rel_sink", False):
            self._shards = {
                **self._shards,
                ".attn.wr_du.": (".attn.rel_sink.", 0),
                ".attn.attn_sink_proj.": (".attn.rel_sink.", 1),
            }
        self.visual = (
            Inkling2VisionEncoder(
                config.vision_config, text_config.hidden_size, quant_config
            )
            if config.vision_config is not None
            and vllm_config.model_config.multimodal_config is not None
            and vllm_config.model_config.multimodal_config.get_limit_per_prompt("image")
            else None
        )
        self.model = Inkling2Model(text_config, quant_config, f"{prefix}model")
        self.lm_head = ParallelLMHead(
            text_config.padded_vocab_size,
            text_config.hidden_size,
            quant_config=quant_config,
            prefix="lm_head",
        )
        self.logits_processor = LogitsProcessor(
            text_config.padded_vocab_size,
            org_vocab_size=text_config.vocab_size,
            soft_cap=text_config.final_logit_softcapping,
        )

    def process_weights_after_loading(self):
        self.model.process_weights_after_loading()
        factors = None
        if self.tabulate_swiglu:
            from aiter.ops.triton.gemm.basic.gemm_packed_g128 import (
                routed_swiglu_factors,
            )

            experts = self.model.layers[0].mlp.experts
            factors = routed_swiglu_factors(
                experts.w13_weight.device, experts.gate_cap, experts.up_cap
            )
        for layer in self.model.layers:
            layer.mlp.experts.activation_factors = factors
            layer.attn.process_weights_after_loading()
            layer.mlp.experts.process_weights_after_loading()
            for controller in layer.controllers.values():
                if not controller.fused:
                    # The reference gates multiply in FP32 on every loop visit.
                    # Cast once after loading into the checkpoint's serving dtype.
                    for weight in (controller.forget_weight, controller.gain_weight):
                        weight.data = weight.data.float()

    def get_language_model(self) -> nn.Module:
        return self

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_PLACEHOLDER
        raise ValueError("Only image modality is supported")

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        patches = kwargs.get("vision_pixels")
        grid_thw = kwargs.get("vision_grid_thw")
        if patches is None or grid_thw is None:
            return ()
        assert isinstance(patches, torch.Tensor)
        assert isinstance(grid_thw, torch.Tensor)
        assert self.visual is not None
        embeds = self.visual(patches, grid_thw)
        return embeds.split(grid_thw.prod(-1).tolist())

    # Encoder CUDA graphs (``compilation_config.cudagraph_mm_encoder``): one
    # graph per output-token budget; padded rows are dead (zero-length
    # sequences, never pooled).
    _graph_metadata_keys = ("meta", "patch_rows", "pixel_offsets", "pool_rows")

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        from vllm.model_executor.models.qwen2_5_vl import (
            _pad_cumulative_seqlens_buffer,
        )

        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=[
                "vision_pixels",
                *self._graph_metadata_keys,
                "cu_seqlens",
            ],
            padding_logics={"cu_seqlens": _pad_cumulative_seqlens_buffer},
            out_hidden_size=self.text_config.hidden_size,
        )

    def get_encoder_cudagraph_budget_range(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int]:
        # 64 tokens: a 320x320 image.
        return 64, min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.model_config.max_model_len,
        )

    def get_encoder_cudagraph_item_specs(
        self, mm_kwargs: dict[str, Any]
    ) -> list[EncoderItemSpec]:
        return [
            EncoderItemSpec(input_size=h * w, output_tokens=h * w)
            for _, h, w in mm_kwargs["vision_grid_thw"].tolist()
        ]

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        grid_thw = mm_kwargs["vision_grid_thw"]
        pixels = mm_kwargs["vision_pixels"]
        patches = pixels.split(grid_thw.prod(-1).tolist())
        return {
            # Empty under data-parallel sharding when a rank gets no images.
            "vision_pixels": torch.cat([pixels[:0], *(patches[i] for i in indices)]),
            "vision_grid_thw": grid_thw[indices],
        }

    def _graph_values(
        self, patches: torch.Tensor, grid: list[list[int]]
    ) -> dict[str, torch.Tensor]:
        assert self.visual is not None
        metadata = self.visual.metadata(grid, patches.device)
        return {
            "vision_pixels": patches,
            **{key: getattr(metadata, key) for key in self._graph_metadata_keys},
            "cu_seqlens": metadata.cu_seqlens,
        }

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ) -> EncoderCudaGraphCaptureInputs:
        assert self.visual is not None
        side = self.visual.patch_size * self.visual.merge_size
        per_item = -(-token_budget // max_batch_size)
        pixels = torch.zeros(
            (max_batch_size * per_item, side, side, 3),
            dtype=torch.float16,
            device=device,
        )
        grid = [[1, 1, per_item]] * max_batch_size
        return EncoderCudaGraphCaptureInputs(values=self._graph_values(pixels, grid))

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        return EncoderCudaGraphReplayBuffers(
            values=self._graph_values(
                mm_kwargs["vision_pixels"],
                mm_kwargs["vision_grid_thw"].tolist(),
            )
        )

    def encoder_cudagraph_forward(
        self, values: dict[str, torch.Tensor], path: str = "default"
    ) -> torch.Tensor:
        assert self.visual is not None
        meta = values["meta"]
        metadata = VisionMetadata(
            **{key: values[key] for key in self._graph_metadata_keys},
            cu_seqlens=values["cu_seqlens"],
            # Bounds every sequence a replay can pack into the buffers.
            max_seqlen=meta.shape[0],
        )
        return self.visual.encode(values["vision_pixels"], metadata)

    def encoder_eager_forward(
        self, mm_kwargs: dict[str, Any], path: str = "default"
    ) -> torch.Tensor:
        assert self.visual is not None
        return self.visual(mm_kwargs["vision_pixels"], mm_kwargs["vision_grid_thw"])

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_input_ids(input_ids)
        if not multimodal_embeddings:
            return inputs_embeds
        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.model.embed_input_ids(input_ids)
        return self.model(positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if (
            self.model.timeline_sink is not None
            and torch.cuda.is_current_stream_capturing()
        ):
            self.model._mark("logits", -1)
            self.model._finish_timeline()
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        num_layers = self.text_config.num_hidden_layers
        # Truncated models (e.g. fewer layers via hf_overrides) skip the rest.
        layer_limits = {
            "layers": num_layers,
            "attn_queries": num_layers - 1,
            "mlp_queries": num_layers,
        }
        loaded: set[str] = set()
        loaded_shards: set[tuple[str, int]] = set()

        def visual_weights() -> Iterable[tuple[str, torch.Tensor]]:
            # Keep the shared loader streaming, including vision checkpoint
            # shards. Each yielded tensor may own an entire GPU read window.
            for name, weight in weights:
                if name.startswith("model.visual."):
                    if self.visual is not None:
                        yield name.removeprefix("model.visual."), weight
                    continue
                for old, new in self._renames:
                    name = name.replace(old, new)
                shard_id = None
                for old, (new, shard) in self._shards.items():
                    if old in name:
                        name = name.replace(old, new)
                        shard_id = shard
                        break
                if name not in params:
                    m = re.search(
                        r"\.(layers|attn_queries|mlp_queries)\.(\d+)(?:\.|$)", name
                    )
                    if m is not None and int(m.group(2)) >= layer_limits[m.group(1)]:
                        continue
                    raise ValueError(f"Unexpected checkpoint tensor: {name}")
                param = params[name]
                if param.dtype != weight.dtype and not (
                    param.is_floating_point() and weight.is_floating_point()
                ):
                    raise ValueError(
                        f"{name}: checkpoint dtype {weight.dtype} != {param.dtype}"
                    )
                loader: Callable[..., None] = getattr(
                    param, "weight_loader", default_weight_loader
                )
                if shard_id is None:
                    loader(param, weight)
                    loaded.add(name)
                else:
                    loader(param, weight, shard_id)
                    loaded_shards.add((name, shard_id))

        if self.visual is not None:
            loaded |= {
                f"visual.{name}" for name in self.visual.load_weights(visual_weights())
            }
        else:
            for _ in visual_weights():
                pass
        num_shards = Counter(new for new, _ in self._shards.values())
        for name, count in Counter(name for name, _ in loaded_shards).items():
            fused = next(f for f in num_shards if f in name)
            if count == num_shards[fused]:
                loaded.add(name)
        if missing := params.keys() - loaded:
            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
        return loaded


__all__ = ["Inkling2LoopForConditionalGeneration"]
