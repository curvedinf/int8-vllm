# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 attention: relative-bias GQA with short-conv K/V, L2-clipped
q/k/r, gated output and per-token attention sinks."""

from typing import cast

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.hy4_neuqi import (
    G128ProjectionInput,
    HY4Linear,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
    RocmAiterUnifiedAttentionBackend,
)
from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    KVQuantMode,
    SlidingWindowSpec,
)

from ..configs import Inkling2ModelConfig
from .ops.attention import (
    gated_rel_attention,
    plan_decode_attention,
    rel_decode_attention,
)
from .ops.sconv import attention_prologue, shift_rel_proj
from .ops.skinny_gemm import DECODE_TOKENS, fp16_gemm
from .ops.streams import side_stream
from .ops.timeline import mark as timeline_mark
from .sconv import K_STREAM, V_STREAM, Inkling2ShortConv, conv_metadata


class Inkling2PassCache(nn.Module, AttentionLayerBase):
    """One logical visit's KV pages. Projection weights belong to the physical layer."""

    def __init__(self, prefix, heads, head_dim, window):
        super().__init__()
        self.prefix = prefix
        self.heads = heads
        self.head_dim = head_dim
        self.window = window
        self.kv_cache = torch.tensor([])
        context = get_current_vllm_config().compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    def forward(self):
        raise RuntimeError("The attention layer owns the forward pass")

    def get_attn_backend(self) -> type[AttentionBackend]:
        return RocmAiterUnifiedAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return inkling_kv_spec(vllm_config, self.heads, self.head_dim, self.window)


def inkling_kv_spec(vllm_config, heads, head_dim, window):
    if vllm_config.cache_config.cache_dtype != "int8_block_g128" or head_dim != 128:
        raise ValueError("Inkling W4A8 uses int8_block_g128 with head_dim=128")
    values = dict(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=heads,
        head_size=head_dim,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_BLOCK_G128,
    )
    spec = (
        SlidingWindowSpec(**values, sliding_window=window)
        if window
        else FullAttentionSpec(**values)
    )
    return RocmAiterUnifiedAttentionBackend.customize_spec(spec)


class Inkling2Attention(nn.Module):
    """Inkling2 attention for one decoder layer.

    Local layers project all of q/k/v from one input; full layers project the
    query side (q, r, gate, sink) from `x_q`, k from `x_k` and v from
    `x_v`, and use dynamic K/V convolutions and log-scaled q/r.
    """

    def __init__(
        self,
        config: Inkling2ModelConfig,
        is_local: bool,
        conv_states: nn.ModuleList,
        prefix: str,
        quant_config,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.is_local = is_local
        num_heads = (
            config.swa_num_attention_heads if is_local else config.num_attention_heads
        )
        num_kv_heads = (
            config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        )
        self.head_dim = config.swa_head_dim if is_local else config.head_dim
        self.d_rel = config.d_rel
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        assert num_heads % tp_size == 0 and num_kv_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window_size
        self.rel_extent = config.sliding_window_size if is_local else config.rel_extent
        self.window = config.sliding_window_size - 1 if is_local else -1
        self.qk_clip = config.qk_l2_clip_threshold
        self.rel_clip = config.rel_l2_clip_threshold
        self.eps = config.rms_norm_eps
        assert config.use_value_norm_no_gamma

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        options = get_current_vllm_config().additional_config
        self.merge_query = options.get("inkling_merge_query_gate", False)
        self.merge_rel_sink = options.get("inkling_merge_rel_sink", False)
        rel_size = self.num_heads * self.d_rel
        for name, width in (
            ("wq_du", q_size),
            ("wr_du", rel_size),
            ("gate_attn", q_size),
            ("attn_sink_proj", self.num_heads),
            ("wk_dv", kv_size),
            ("wv_dv", kv_size),
        ):
            if self.merge_query and name in ("wq_du", "gate_attn"):
                continue
            if self.merge_rel_sink and name in ("wr_du", "attn_sink_proj"):
                continue
            setattr(
                self,
                name,
                ColumnParallelLinear(
                    config.hidden_size,
                    width * tp_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.{name}",
                ),
            )
        if self.merge_query:
            self.query_gate = MergedColumnParallelLinear(
                config.hidden_size,
                [q_size * tp_size] * 2,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.query_gate",
            )
            query_projections = (self.query_gate,)
        else:
            query_projections = (self.wq_du, self.gate_attn)
        if self.merge_rel_sink:
            self.rel_sink = MergedColumnParallelLinear(
                config.hidden_size,
                [rel_size * tp_size, self.num_heads * tp_size],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.rel_sink",
            )
        self.skip_past_bias = options.get("inkling_skip_past_bias", False)
        # One launch for the query/gate pair and for the k/v pair when they
        # share a quantized input; bit-identical to the separate calls.
        self.join_projections = options.get("inkling_join_attention_projections", False)
        # "triton": decode batches project the unquantized relative positions
        # with a tiled kernel instead of the single-CTA rocBLAS GEMM. The
        # accumulation order differs from rocBLAS in the last FP32 bits.
        self.rel_kernel = options.get("inkling_rel_proj_kernel", "rocblas")
        # The relative-position and sink projections only read `x_q`; on a
        # side stream they execute beside the query/gate and k/v projections
        # with the same kernels and results.
        self._side_stream = (
            side_stream(0)
            if options.get("inkling_parallel_projections", False)
            else None
        )
        if self.rel_kernel not in ("rocblas", "triton"):
            raise ValueError(f"inkling_rel_proj_kernel={self.rel_kernel!r}")
        self.mi100_tiles = options.get("inkling_mi100_attention_tiles", False)
        self.prefill_splits = options.get("inkling_prefill_kv_splits", 0)
        self.fused_prefill = options.get("inkling_fused_prefill_epilogue", False)
        self.share_input = options.get("inkling_share_attention_input", False) and all(
            isinstance(projection.quant_method, HY4Linear)
            for projection in (*query_projections, self.wk_dv, self.wv_dv)
        )
        self.wo_ud = RowParallelLinear(
            q_size * tp_size,
            config.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wo_ud",
        )
        self.quantize_output = (
            options.get("inkling_attention_quant_output", False)
            and not self.fused_prefill
            and self.head_dim == 128
            and isinstance(self.wo_ud.quant_method, HY4Linear)
        )
        self.rel_logits_proj = nn.Parameter(
            torch.empty(self.d_rel, self.rel_extent), requires_grad=False
        )
        self.q_scale = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim), requires_grad=False
        )
        self.r_scale = nn.Parameter(
            torch.empty(self.num_heads, self.d_rel), requires_grad=False
        )
        for param in (self.q_scale, self.r_scale):
            set_weight_attrs(
                param,
                {
                    "weight_loader": lambda p, w: p.data.copy_(
                        w.narrow(0, tp_rank * p.shape[0], p.shape[0])
                    )
                },
            )

        self.conv_states = conv_states
        self.k_sconv, self.v_sconv = (
            Inkling2ShortConv(
                kv_size,
                config.sconv_kernel_size,
                conv_states[0],
                stream,
                dynamic_input_size=None if is_local else config.hidden_size,
                num_heads=self.num_kv_heads,
            )
            for stream in (K_STREAM, V_STREAM)
        )

        self.caches = nn.ModuleList(
            Inkling2PassCache(
                f"{prefix}.pass_{i}",
                self.num_kv_heads,
                self.head_dim,
                self.sliding_window if is_local else None,
            )
            for i in range(len(conv_states))
        )

    def process_weights_after_loading(self) -> None:
        self.rel_proj_shifted = shift_rel_proj(self.rel_logits_proj, self.is_local)

    def _rel_projection(self, x_q):
        """Relative-position projection; rocBLAS keeps the larger batches."""
        weight = self.wr_du.weight
        if (
            self.rel_kernel == "triton"
            and weight.dtype == torch.float16
            and len(x_q) <= DECODE_TOKENS
            and weight.shape[1] % 64 == 0
        ):
            return fp16_gemm(x_q, weight)
        return self.wr_du(x_q)[0]

    def _project(self, x_q, x_k, x_v):
        q_input = G128ProjectionInput.from_rows(x_q) if self.share_input else x_q
        joined = self.join_projections and self.share_input
        side = self._side_stream if not self.merge_rel_sink else None
        if side is not None:
            main = torch.cuda.current_stream()
            side.wait_stream(main)
            with torch.cuda.stream(side):
                rel, sink = self._rel_projection(x_q), self.attn_sink_proj(x_q)[0]
        if self.merge_query:
            q, gate = self.query_gate(q_input)[0].chunk(2, dim=-1)
        elif joined:
            q, gate = self.wq_du.quant_method.apply_joined(
                [self.wq_du, self.gate_attn], q_input
            )
        else:
            q = self.wq_du(q_input)[0]
            gate = self.gate_attn(q_input)[0]
        timeline_mark("a_qg")
        if self.merge_rel_sink:
            rel, sink = self.rel_sink(x_q)[0].split(
                (self.num_heads * self.d_rel, self.num_heads), dim=-1
            )
        elif side is None:
            rel, sink = self._rel_projection(x_q), self.attn_sink_proj(x_q)[0]
        k_input = q_input if x_k is x_q else x_k
        v_input = q_input if x_v is x_q else x_v
        if joined and k_input is v_input:
            k, v = self.wk_dv.quant_method.apply_joined(
                [self.wk_dv, self.wv_dv], k_input
            )
        else:
            k = self.wk_dv(k_input)[0]
            v = self.wv_dv(v_input)[0]
        timeline_mark("a_kv")
        if side is not None:
            torch.cuda.current_stream().wait_stream(side)
        return q, rel, gate, sink, k, v

    def _split_kv_cache(self, cache):
        key, value = cache.transpose(1, 2).chunk(2, dim=-1)
        # The FP16 group scale follows the 128 codes in each padded half.
        return key[..., : self.head_dim], value[..., : self.head_dim]

    def forward(
        self,
        positions: torch.Tensor,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        """Return this rank's fp16 `(T, hidden)` partial output (to be
        all-reduced over TP).

        Recurrent and full layers may receive separate q/k/v inputs. Full
        layers apply the fp32 log-scaling `tau` to q/r.
        """
        if x_k is None and x_v is None:
            x_k = x_v = x_q
        assert x_k is not None and x_v is not None
        q, r, gate, sinks, k, v = self._project(x_q, x_k, x_v)
        timeline_mark("proj")
        quant_output = self.quantize_output and q.dtype == torch.float16
        output = gate.new_empty(
            gate.shape, dtype=torch.int8 if quant_output else q.dtype
        )
        scales = sums = None
        if quant_output:
            scales = q.new_empty((len(q), self.num_heads), dtype=torch.float32)
            sums = torch.empty_like(scales, dtype=torch.int32)
        self._attention(
            positions,
            q,
            r,
            gate,
            sinks,
            k,
            v,
            self.k_sconv.dynamic_taps(x_k),
            self.v_sconv.dynamic_taps(x_v),
            tau,
            output,
            pass_idx,
            scales,
            sums,
        )
        timeline_mark("attnk")
        projection_input = (
            G128ProjectionInput(output, scales, sums, q.dtype)
            if quant_output
            else output
        )
        output = self.wo_ud(projection_input)[0]
        timeline_mark("wo")
        return output

    @eager_break_during_capture
    def _attention(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        gate: torch.Tensor,
        sinks: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_dynamic: torch.Tensor | None,
        v_dynamic: torch.Tensor | None,
        tau: torch.Tensor | None,
        output: torch.Tensor,
        pass_idx: int,
        output_scales=None,
        output_sums=None,
    ) -> None:
        """Write the gated attention output to `output`.

        The prologue's layout and the attention plan follow the batch's
        requests, so this runs eagerly in piecewise CUDA graphs, which are
        captured on a dummy batch and replayed for any batch of the same
        token count.
        """
        attn_metadata = get_forward_context().attn_metadata
        cache = self.caches[pass_idx]
        owner = self.conv_states[pass_idx]
        if not isinstance(attn_metadata, dict) or cache.kv_cache.numel() == 0:
            output.zero_()
            if output_scales is not None:
                output_scales.fill_(1e-30)
                output_sums.zero_()
            return
        md = cast(RocmAttentionMetadata, attn_metadata[cache.prefix])
        conv_md = conv_metadata(owner)
        assert conv_md is not None
        num_tokens = q.shape[0]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        key_cache, value_cache = self._split_kv_cache(cache.kv_cache)
        clipped_q = q.new_empty(q.shape)
        # Sheared as `rel_attention` reads it (see attention_prologue).
        rel_logits = q.new_empty(
            (num_tokens + 128, self.num_heads, self.rel_extent + 256)
        )
        attention_prologue(
            q,
            r.view(num_tokens, self.num_heads, self.d_rel),
            k,
            v,
            self.q_scale,
            self.r_scale,
            tau,
            self.rel_proj_shifted,
            self.k_sconv.conv_weight,
            self.v_sconv.conv_weight,
            k_dynamic,
            v_dynamic,
            owner.kv_cache,
            positions,
            conv_md.block_table,
            conv_md.seq_idx,
            conv_md.slot_mapping,
            conv_md.query_start,
            key_cache,
            value_cache,
            md.slot_mapping,
            clipped_q,
            rel_logits,
            md.query_start_loc,
            md.seq_lens,
            clip_threshold=self.qk_clip,
            rel_threshold=self.rel_clip,
            norm_eps=self.eps,
        )
        timeline_mark("prol")
        # Decode and verify batches split the keys; the split count follows
        # the batch alone, so it holds for every replay of a CUDA graph.
        num_splits = plan_decode_attention(
            self.is_local,
            md.seq_lens.shape[0],
            md.max_query_len,
            self.num_heads,
            self.num_kv_heads,
            self.rel_extent,
        )
        if num_splits:
            rel_decode_attention(
                clipped_q,
                key_cache,
                value_cache,
                rel_logits,
                gate,
                sinks,
                output,
                block_table=md.block_table,
                cache_seqlens=md.seq_lens,
                cu_seqlens_q=md.query_start_loc,
                max_query_len=md.max_query_len,
                num_actual_tokens=md.num_actual_tokens,
                softmax_scale=self.scaling,
                is_local=self.is_local,
                num_splits=num_splits,
                skip_past_bias=self.skip_past_bias,
                mi100_tiles=self.mi100_tiles,
                output_scales=output_scales,
                output_sums=output_sums,
            )
            return
        gated_rel_attention(
            clipped_q,
            key_cache,
            value_cache,
            rel_logits,
            gate,
            sinks,
            output,
            block_table=md.block_table,
            cache_seqlens=md.seq_lens,
            cu_seqlens_q=md.query_start_loc,
            max_seqlen_q=md.max_query_len,
            num_actual_tokens=md.num_actual_tokens,
            softmax_scale=self.scaling,
            window=self.window,
            fused_epilogue=self.fused_prefill,
            output_scales=output_scales,
            output_sums=output_sums,
            num_splits=self.prefill_splits if self.window < 0 else 0,
            skip_past_bias=self.skip_past_bias,
            mi100_tiles=self.mi100_tiles,
        )
