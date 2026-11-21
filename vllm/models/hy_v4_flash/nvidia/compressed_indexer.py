# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash coarse-only gated IndexPool with bounded paged FP8 scoring."""

import torch
import torch.nn.functional as F
from torch import nn

from vllm import envs
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
    has_deep_gemm,
    mqa_logits_mx_inputs,
    mqa_logits_use_mx_scales,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import num_compute_units

from .indexpool_cache import POOL_PAGE_SIZE, IndexPoolCache, IndexPoolMetadata
from .indexpool_ops import (
    expand_groups,
    fwht_quantize,
    gated_pool,
    rotate_indexer,
    select_groups,
    write_pool,
)
from .indexpool_prepare import prepare_groups, write_tails


class FlashCompressedIndexer(nn.Module):
    def __init__(self, vllm_config, config, topk_indices_buffer, prefix):
        super().__init__()
        self.int_group = 0
        self.fused_indexer = vllm_config.additional_config.get(
            "hy4_fused_indexer", True
        )
        self.fuse_selection = self.fused_indexer and vllm_config.additional_config.get(
            "hy4_fused_group_selection", False
        )
        self.fused_pool_projection = vllm_config.additional_config.get(
            "hy4_fused_pool_projection", self.fused_indexer
        )
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx908

            if on_gfx908():
                self.int_group = int(
                    vllm_config.additional_config.get("hy4_index_group", 128)
                )
                if self.int_group not in (4, 8, 16, 32, 64, 128):
                    raise ValueError(
                        "hy4_index_group must be a power of two from 4 to 128"
                    )
        self.index_tp_min_width = int(
            vllm_config.additional_config.get("hy4_index_tp_min_width", 0)
        )
        if self.index_tp_min_width < 0:
            raise ValueError("hy4_index_tp_min_width must be nonnegative")
        self.index_tp = None
        if self.index_tp_min_width and self.fuse_selection and self.int_group == 128:
            from vllm.distributed import get_tp_group

            tp = get_tp_group()
            communicator = getattr(tp.device_communicator, "ca_comm", None)
            if (
                communicator is not None
                and not communicator.disabled
                and tp.world_size > 1
            ):
                self.index_tp = tp
        if not self.int_group and not has_deep_gemm():
            raise RuntimeError("Flash IndexPool requires DeepGEMM paged MQA")
        self.ratio = config.compressed_indexer_compress_ratio
        self.group_budget = config.compressed_indexer_group_topk - 1
        self.num_heads = config.compressed_indexer_num_attention_heads
        if not 1 <= self.num_heads <= 64:
            raise ValueError("IndexPool requires 1..64 scoring heads")
        self.rope_dim = config.qk_rope_head_dim
        self.rope_parameters = config.rope_parameters
        self.eps = config.rms_norm_eps
        self.topk_indices_buffer = topk_indices_buffer
        self.max_model_len = vllm_config.model_config.max_model_len
        self.logits_budget = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4
        for name, input_size, output_size in (
            ("query_projection", config.q_lora_rank, self.num_heads * 128),
            ("head_weight_projection", config.hidden_size, self.num_heads),
            ("token_key_projection", config.hidden_size, 128),
        ):
            setattr(
                self,
                name,
                ReplicatedLinear(
                    input_size,
                    output_size,
                    bias=False,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.{name}",
                ),
            )
        self.token_key_norm = nn.RMSNorm(128, eps=self.eps)
        projection_dtype = torch.float16 if self.int_group else torch.float32
        self.key_value_projection = nn.Linear(
            128, 128, bias=False, dtype=projection_dtype
        )
        self.key_gate_projection = nn.Linear(
            128, 128, bias=False, dtype=projection_dtype
        )
        self.register_buffer("pool_projection", None, persistent=False)
        self.key_norm = nn.RMSNorm(128, eps=self.eps, dtype=torch.float32)
        self.intra_group_ape = nn.Parameter(
            torch.empty(self.ratio, 128, dtype=torch.float32)
        )
        self.k_cache = IndexPoolCache(
            f"{prefix}.k_cache",
            vllm_config.cache_config,
            self.ratio,
            int_group=self.int_group,
        )
        self.tail_cache = IndexPoolCache(
            f"{prefix}.tail_cache",
            vllm_config.cache_config,
            self.ratio,
            tail=True,
        )
        self.num_sms = num_compute_units()
        self.varlen = current_platform.is_device_capability_family(100)

    def prepare_projections(self):
        if not self.int_group or not self.fused_pool_projection:
            return
        value = self.key_value_projection.weight
        gate = self.key_gate_projection.weight
        self.pool_projection = torch.cat((value.detach(), gate.detach()), dim=0)
        value.data, gate.data = self.pool_projection.split(128, dim=0)

    def forward(self, hidden_states, q_c, positions, rope=None):
        output = self.topk_indices_buffer[: hidden_states.shape[0]]
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return output.fill_(-1)
        if not isinstance(metadata, dict):
            raise TypeError("IndexPool requires per-layer metadata")
        km = metadata[self.k_cache.prefix]
        tm = metadata[self.tail_cache.prefix]
        if not isinstance(km, IndexPoolMetadata) or not isinstance(
            tm, IndexPoolMetadata
        ):
            raise TypeError("IndexPool cache metadata has an incompatible type")
        tokens = km.num_actual_tokens
        if tokens == 0 or km.seq_lens.numel() == 0:
            return output.fill_(-1)
        if tokens < output.shape[0]:
            output[tokens:].fill_(-1)
        keys = self.token_key_norm(self.token_key_projection(hidden_states)[0]).to(
            torch.float16 if self.int_group else torch.bfloat16
        )
        cache, tail = self.k_cache.kv_cache, self.tail_cache.kv_cache
        (
            raw,
            group_positions,
            group_owner,
            groups,
            group_live,
            owner,
            logical,
            upper,
            live,
        ) = prepare_groups(
            keys,
            positions,
            tail,
            tm.block_table,
            km.query_start_loc,
            km.seq_lens,
            tokens,
            self.ratio,
        )
        if raw.shape[0]:
            raw = raw.to(self.key_value_projection.weight.dtype)
            if self.pool_projection is None:
                values = self.key_value_projection(raw)
                gates = self.key_gate_projection(raw)
            else:
                values, gates = F.linear(raw, self.pool_projection).split(128, dim=-1)
            pooled = gated_pool(
                values, gates, self.intra_group_ape, self.key_norm.weight, self.eps
            ).to(torch.float16 if self.int_group else torch.bfloat16)
            quant, scales = fwht_quantize(
                rotate_indexer(
                    pooled, group_positions, self.rope_dim, self.rope_parameters
                ),
                self.int_group,
                fused=self.fused_indexer,
            )
            if not self.int_group and mqa_logits_use_mx_scales():
                scales = (scales.view(torch.int32) >> 23) * 0x01010101
            write_pool(
                cache, km.block_table, group_owner, groups, group_live, quant, scales
            )
        write_tails(
            keys, positions, tail, tm.block_table, km.query_start_loc, km.seq_lens
        )
        # Every completed group and the current tail fit. Scoring cannot
        # change membership or order. Preserve both caches for later queries.
        if self.int_group and km.max_seq_len <= (self.group_budget + 1) * self.ratio:
            columns = torch.arange(
                output.shape[1], dtype=torch.int32, device=output.device
            )
            output[:tokens].copy_(
                torch.where(
                    live[:, None] & (columns[None, :] <= logical[:, None]), columns, -1
                )
            )
            return output
        query = self.query_projection(q_c)[0][:tokens].reshape(
            tokens, self.num_heads, 128
        )
        query = rotate_indexer(
            query, positions[..., :tokens], self.rope_dim, self.rope_parameters
        )
        if self.int_group:
            from vllm.models.hy_v4_flash.indexpool_gfx908 import (
                expand_sorted_groups,
                gather_verify_indices,
                local_group_selection,
                select_and_expand,
            )
            from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
                hy4_int8_paged_mqa_logits,
            )

            quant_q, scales_q = fwht_quantize(
                query.reshape(-1, 128).half(), self.int_group, fused=self.fused_indexer
            )
            quant_q = quant_q.view(tokens, self.num_heads, 128)
            scales_q = scales_q.view(tokens, self.num_heads, 128 // self.int_group)
            weights = self.head_weight_projection(hidden_states)[0][:tokens].float()
            weights = weights * (self.num_heads**-0.5 * 128**-0.5)
            width = max(
                POOL_PAGE_SIZE,
                cdiv(km.max_seq_len, self.ratio * POOL_PAGE_SIZE) * POOL_PAGE_SIZE,
            )
            # Very wide contexts use int64 scores with the logical position
            # as the tie break. The usual workload scans a row of FP32 scores.
            ranked = (
                not self.fuse_selection
                or width > 32768
                or (width > 16384 and tokens < 96)
            )
            chunk = max(1, self.logits_budget // (width * (2 if ranked else 1)))
            tp = self.index_tp
            # At C8, a seven-token draft gives the target 64 verification rows.
            # At wide search widths, 16 local rows per MI100 amortize the gather.
            min_shared_rows = (
                64
                if tokens == 64
                and width >= 32768
                and tp is not None
                and tp.world_size == 4
                else 256
            )
            share_rows = (
                tp is not None
                and width >= self.index_tp_min_width
                and self.num_heads == 16
                and tokens >= min_shared_rows
                and output.is_contiguous()
            )
            begin = 0
            while begin < tokens:
                rows = min(tokens - begin, chunk)
                sharded = share_rows and rows >= max(min_shared_rows, tp.world_size)
                if sharded:
                    rows -= rows % tp.world_size
                batch = slice(begin, begin + rows)
                begin = batch.stop
                sl = batch
                if sharded:
                    # All ranks have the same index keys and queries. Each rank
                    # owns a range of query rows and returns the selected IDs.
                    per_rank = rows // tp.world_size
                    start = batch.start + tp.rank_in_group * per_rank
                    sl = slice(start, start + per_rank)
                local = not ranked and local_group_selection(sl.stop - sl.start, width)
                logits = hy4_int8_paged_mqa_logits(
                    quant_q[sl],
                    scales_q[sl],
                    cache,
                    weights[sl],
                    upper[sl],
                    km.block_table,
                    owner[sl],
                    width,
                    self.int_group,
                    rank=ranked,
                    mask_unused=sharded or not local,
                    keep_all_at_most=self.group_budget if local and not sharded else 0,
                )
                if self.fuse_selection:
                    selected = torch.empty_like(output[sl]) if sharded else output[sl]
                    select_and_expand(
                        logits, logical[sl], self.ratio, live[sl], selected
                    )
                    if sharded:
                        if (
                            min_shared_rows == 64
                            and self.ratio == 4
                            and output.shape[1] == 2048
                            and self.max_model_len <= 1 << 24
                        ):
                            gather_verify_indices(
                                tp, selected, logical[batch], output[batch], self.ratio
                            )
                        else:
                            torch.distributed.all_gather_into_tensor(
                                output[batch], selected, group=tp.device_group
                            )
                    continue
                groups = select_groups(logits, upper[sl], self.group_budget)
                if self.fused_indexer:
                    expand_sorted_groups(
                        groups, logical[sl], self.ratio, live[sl], output[sl]
                    )
                else:
                    output[sl].copy_(
                        expand_groups(groups, logical[sl], self.ratio, active=live[sl])
                    )
            return output
        quant_q, scales_q = fwht_quantize(query.reshape(-1, 128).bfloat16())
        heads = 32 if self.num_heads <= 32 else 64
        quant_q = F.pad(
            quant_q.view(torch.uint8).reshape(tokens, self.num_heads, 128),
            (0, 0, 0, heads - self.num_heads),
        ).view(torch.float8_e4m3fn)
        weights = self.head_weight_projection(hidden_states)[0][:tokens].float()
        weights = weights * (self.num_heads**-0.5 * 128**-0.5)
        weights = F.pad(
            weights * scales_q.reshape(tokens, self.num_heads),
            (0, heads - self.num_heads),
        )
        quant_q = quant_q[:, None]
        mx_scales, weights = mqa_logits_mx_inputs(quant_q, None, weights, torch.float32)
        max_seq_len = (
            self.max_model_len
            if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
            else km.max_seq_len
        )
        width = max(
            POOL_PAGE_SIZE,
            cdiv(max_seq_len, self.ratio * POOL_PAGE_SIZE) * POOL_PAGE_SIZE,
        )
        if self.logits_budget < width:
            raise ValueError("IndexPool logits budget cannot hold a single scoring row")
        chunk = max(1, self.logits_budget // width)
        for begin in range(0, tokens, chunk):
            end = min(tokens, begin + chunk)
            bounds = upper[begin:end]
            lengths = bounds.clamp_min(1).int()[:, None].contiguous()
            indices = owner[begin:end].int().contiguous() if self.varlen else None
            schedule = get_paged_mqa_logits_metadata(
                lengths, POOL_PAGE_SIZE, self.num_sms, indices=indices
            )
            table = km.block_table[owner[begin:end]]
            logits = fp8_fp4_paged_mqa_logits(
                (
                    quant_q[begin:end],
                    mx_scales[begin:end] if mx_scales is not None else None,
                ),
                cache.unsqueeze(2),
                weights[begin:end],
                lengths,
                table,
                schedule,
                max_model_len=width,
                indices=indices,
            )
            recalled = select_groups(logits, bounds, self.group_budget)
            output[begin:end].copy_(
                expand_groups(
                    recalled,
                    logical[begin:end],
                    self.ratio,
                    active=live[begin:end],
                )
            )
        return output
