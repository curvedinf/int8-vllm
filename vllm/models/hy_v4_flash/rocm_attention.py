# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine integration for the packed gfx908 MLA kernels."""

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadata,
    SparseMLACommonMetadataBuilder,
)
from vllm.models.hy_v4_flash.mla_gfx908 import (
    cache_row_bytes,
    mla_splits,
    paged_mla,
    quantize_mla,
    write_mla_cache,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
)
from vllm.v1.kv_cache_interface import int8_block_group_size


def is_mi100():
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx908

        return on_gfx908()
    return False


class FlashMI100MetadataBuilder(SparseMLACommonMetadataBuilder):
    metadata_cls = SparseMLACommonMetadata
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        layer = vllm_config.compilation_config.static_forward_context[layer_names[0]]
        self.topk_tokens = layer.topk_buffer_width
        self.max_model_len = vllm_config.model_config.max_model_len

    def build_for_cudagraph_capture(self, common_attn_metadata):
        from dataclasses import replace

        metadata = super().build_for_cudagraph_capture(common_attn_metadata)
        # A decode graph may replay at a much longer prefix than its dummy.
        # The dense-prefix specialization may only run if every real key fits.
        return replace(metadata, max_seq_len=self.max_model_len)


class FlashMI100Backend(ROCMAiterMLASparseBackend):
    supported_dtypes = [torch.float16]
    supported_kv_cache_dtypes = [f"int8_block_g{g}" for g in (4, 8, 16, 32, 64, 128)]

    @staticmethod
    def get_name():
        return "HY4_MI100_SPARSE_MLA"

    @staticmethod
    def get_metadata_cls():
        return SparseMLACommonMetadata

    @staticmethod
    def get_builder_cls():
        return FlashMI100MetadataBuilder

    @staticmethod
    def get_impl_cls():
        return FlashMI100Impl

    @classmethod
    def supports_sink(cls):
        return True

    @classmethod
    def supports_sliding_window(cls):
        return True


class FlashMI100Impl(SparseMLACommonImpl):
    supports_dense_mha_prefill = False
    supports_quant_query_input = False

    def __init__(self, *, sinks, **kwargs):
        self.group = int8_block_group_size(kwargs["kv_cache_dtype"])
        if self.group is None:
            raise ValueError("HY4 MI100 requires int8_block_g KV")
        options = get_current_vllm_config().additional_config
        self.rope_group = int(options.get("hy4_rope_group", min(64, self.group)))
        if self.rope_group not in (4, 8, 16, 32, 64):
            raise ValueError("Invalid hy4_rope_group")
        self.state_content_bytes = cache_row_bytes(self.group, self.rope_group)
        self.skip_empty = bool(options.get("hy4_skip_empty", True))
        self.dynamic_empty = bool(options.get("hy4_dynamic_empty_tiles", False))
        self.causal_prefill = bool(options.get("hy4_causal_prefill", True))
        self.causal_key_tile = int(options.get("hy4_causal_key_tile", 32))
        self.causal_value_tile = int(options.get("hy4_causal_value_tile", 128))
        self.causal_warps = int(options.get("hy4_causal_warps", 4))
        self.causal_page_pair = bool(options.get("hy4_causal_page_pair", False))
        self.causal_long_chunk_page_pair = bool(
            options.get("hy4_causal_long_chunk_page_pair", self.causal_page_pair)
        )
        self.causal_long_chunk_value_tile = int(
            options.get("hy4_causal_long_chunk_value_tile", self.causal_value_tile)
        )
        if self.causal_long_chunk_value_tile not in (64, 128, 256, 512):
            raise ValueError("Unsupported HY4 causal prefill value tile")
        # -1 picks the QK tile for this batch. 0 recomputes QK in each value tile.
        self.fly_values = bool(options.get("hy4_flydsl_mla_values", False))
        self.pack_verify_query = bool(options.get("hy4_pack_verify_query", False))
        self.verify_query_tile = int(options.get("hy4_causal_verify_query_tile", 2))
        if self.verify_query_tile not in (2, 4, 8):
            raise ValueError("hy4_causal_verify_query_tile must be 2, 4 or 8")
        self.fly_scores = bool(options.get("hy4_flydsl_mla_scores", False))
        self.score_tile = int(options.get("hy4_sparse_score_tile", -1))
        self.score_workspace = int(options.get("hy4_sparse_score_mib", 32)) << 20
        if (
            self.score_tile not in (-1, 0, 32, 64, 128)
            or self.score_workspace < 1 << 20
        ):
            raise ValueError("Invalid HY4 sparse QK tile or score workspace")
        self.sinks = sinks
        self.softmax_scale = kwargs["scale"]
        super().__init__(**kwargs)

    def do_kv_cache_update(
        self,
        kv_c_normed,
        k_pe,
        kv_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        skip_scale=False,
    ):
        write_mla_cache(
            kv_c_normed, k_pe, kv_cache, slot_mapping, self.group, self.rope_group
        )

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        latent, rope = q if isinstance(q, tuple) else q.split([512, 64], dim=-1)
        query, scales = quantize_mla(latent, rope, self.group, self.rope_group)
        tokens, heads = query.shape[:2]
        indexer = layer.indexer
        inline_window = (
            indexer.window if getattr(indexer, "implicit_window", False) else 0
        )
        if (
            self.causal_prefill
            and heads in (8, 16)
            and attn_metadata.max_query_len >= (16 if inline_window else 8)
            and tokens == attn_metadata.num_actual_tokens
        ):
            from vllm.models.hy_v4_flash.mla_gfx908 import causal_mla
            from vllm.models.hy_v4_flash.nvidia.compressed_indexer import (
                FlashCompressedIndexer,
            )
            from vllm.models.hy_v4_flash.nvidia.swa import FlashWindowIndexer

            window = indexer.window if isinstance(indexer, FlashWindowIndexer) else 0
            complete_prefix = (
                isinstance(indexer, FlashCompressedIndexer)
                and attn_metadata.max_seq_len
                <= (indexer.group_budget + 1) * indexer.ratio
            )
            if window or complete_prefix:
                if heads == 16:
                    qt = 8
                elif attn_metadata.max_query_len >= 64:
                    qt = 16
                else:
                    qt = 4 if window else self.verify_query_tile
                tuned_prefill = (
                    attn_metadata.max_query_len >= 128
                    and self.group == 128
                    and self.rope_group == 64
                )
                value_tile = (
                    self.causal_long_chunk_value_tile
                    if tuned_prefill and tokens >= 1024
                    else self.causal_value_tile
                )
                return causal_mla(
                    query,
                    scales,
                    kv_c_and_k_pe_cache,
                    attn_metadata.query_start_loc,
                    attn_metadata.seq_lens,
                    attn_metadata.block_table,
                    self.sinks,
                    self.softmax_scale,
                    self.group,
                    self.rope_group,
                    attn_metadata.max_query_len,
                    window,
                    query_tile=qt,
                    key_tile=self.causal_key_tile,
                    value_tile=value_tile,
                    warps=self.causal_warps,
                    page_pair=tuned_prefill
                    and (
                        self.causal_long_chunk_page_pair
                        if tokens >= 1024
                        else heads == 16 and bool(window) and self.causal_page_pair
                    ),
                ), None
        ids = self.topk_indices_buffer[:tokens]
        owners = attn_metadata.req_id_per_token[:tokens]
        width = ids.shape[-1]
        short = attn_metadata.max_seq_len * 2 < width
        tile = self.score_tile
        if tile == -1:
            if short or (tokens == 1 and width < 1024):
                tile = 0
            else:
                tile = 64 if width >= 1024 and tokens >= 32 else 32
        save_addresses = bool(tile and heads <= 8 and width >= 1024)
        if tile:
            pitch = ((width + tile - 1) // tile) * tile
            chunk = max(
                1, self.score_workspace // (pitch * (4 * heads + 8 * save_addresses))
            )
        else:
            chunk = max(1, tokens)
        out = torch.empty(
            (tokens, heads, 512), dtype=torch.float16, device=query.device
        )
        # Chunking bounds temporary QK storage. Keep the attention reduction
        # order identical even when the last workspace slice has very few rows.
        splits = mla_splits(tokens, width)
        for start in range(0, tokens, chunk):
            rows = slice(start, min(tokens, start + chunk))
            paged_mla(
                query[rows],
                scales[rows],
                kv_c_and_k_pe_cache,
                ids[rows],
                owners[rows],
                attn_metadata.block_table,
                self.sinks,
                self.softmax_scale,
                self.group,
                self.rope_group,
                skip_empty=self.skip_empty and (short or self.dynamic_empty),
                score_tile=tile,
                fly_scores=self.fly_scores,
                fly_values=self.fly_values,
                splits=splits,
                warps=4 if tile else 8,
                save_addresses=save_addresses,
                out=out[rows],
                window=inline_window,
                starts=attn_metadata.query_start_loc,
                lengths=attn_metadata.seq_lens,
                row_offset=start,
            )
        return out, None
