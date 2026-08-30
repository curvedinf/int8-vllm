# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence, Callable, Mapping

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionState,
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.utils import AttentionGroup


def _prepare_dflash_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    skip_attn: bool,
    causal: bool | Mapping[int, bool],
    draft_slot_rows: torch.Tensor | None = None,
    draft_group_ids: Sequence[int] | None = None,
) -> AttentionState:
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )
    # The draft's per-layer slot dict must reference the DRAFT's private
    # rows for its groups (the shared rows are baked into the TARGET's
    # verify graphs; routing the draft through them misdirects target KV
    # writes under replay).
    if draft_slot_rows is not None:
        for i, gid in enumerate(draft_group_ids or ()):
            private_row = draft_slot_rows[i, :num_tokens]
            for layer_name in kv_cache_config.kv_cache_groups[gid].layer_names:
                slot_mappings_by_layer[layer_name] = private_row

    attn_metadata = None
    if not skip_attn:
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        dcp_local_seq_lens = None
        if block_tables.cp_size > 1:
            prepare_dcp_local_seq_lens(
                input_buffers.dcp_local_seq_lens,
                input_buffers.seq_lens,
                num_reqs,
                block_tables.cp_size,
                block_tables.cp_rank,
                block_tables.cp_interleave,
            )
            dcp_local_seq_lens = input_buffers.dcp_local_seq_lens
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=num_tokens // num_reqs,
            seq_lens=input_batch.seq_lens,
            dcp_local_seq_lens=dcp_local_seq_lens,
            max_seq_len=max_model_len,
            block_tables=input_block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            for_cudagraph_capture=True,
            causal=causal,
        )
    return AttentionState(attn_metadata, slot_mappings_by_layer)


class DFlashCudaGraphManager(CudaGraphManager):
    """DFlash CudaGraphManager for the parallel-drafting query forward,
    building its own attention metadata from scratch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The drafter's vocab-parallel top-k all-gathers may be the first
        # torch.distributed collectives on the TP device group. PyTorch then
        # lazily initializes its NCCL communicator on a background thread,
        # whose event polls / allocations are unsafe CUDA calls under a
        # "global"-mode capture and kill the workers on ROCm
        # (hipErrorStreamCaptureUnsupported). "thread_local" keeps the capture
        # validated on the capturing thread while allowing those background
        # threads to proceed.
        self.capture_error_mode = "thread_local"

    def capture(
        self,
        forward_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        causal: bool | Mapping[int, bool],
        progress_bar_desc: str = "Capturing CUDA graphs",
        draft_slot_rows: torch.Tensor | None = None,
        draft_group_ids: Sequence[int] | None = None,
    ) -> None:
        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
            attn_state = _prepare_dflash_inputs_to_capture(
                num_reqs,
                num_tokens,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                max_model_len,
                skip_attn=(desc.cg_mode == CUDAGraphMode.PIECEWISE),
                draft_slot_rows=draft_slot_rows,
                draft_group_ids=draft_group_ids,
                causal=causal,
            )
            attn_metadata, slot_mappings = attn_state

            return lambda cg_mode: forward_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cg_mode,
            )

        super().capture(create_forward_fn, progress_bar_desc)
