# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare packed n-gram rows before a full HIP graph replay."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch

from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState
from vllm.models.hy_v4_flash.engram_kernel import lookup_q4_pages
from vllm.models.hy_v4_flash.engram_q4 import WIDTH


class FlashMI100ModelState(DeepseekV41ModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        language_model = (
            model.get_language_model()
            if hasattr(model, "get_language_model")
            else model
        )
        backbone = language_model.model
        self.hash_state = backbone.engram_hash
        self.tables = []
        self.readahead = None
        self.fused_hashes = vllm_config.additional_config.get("hy4_fused_hashes", False)
        if self.hash_state is None or not vllm_config.engram_config.packed_table:
            return
        self.tables = [
            backbone.layers[i - 1].engram_layer.multi_head_embedding
            for i in backbone.config.engram_layer_ids
        ]
        local_heads = max(table.head_end - table.head_start for table in self.tables)
        self.hashes = torch.full(
            (len(self.tables), self.max_num_tokens, self.tables[0].n_hash_cols),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.rows = torch.zeros(
            (len(self.tables), self.max_num_tokens, local_heads, WIDTH),
            dtype=torch.float16,
            device=device,
        )
        if (
            vllm_config.engram_config.packed_table == "bcache"
            and vllm_config.additional_config.get("hy4_engram_readahead", True)
        ):
            self.readahead = ThreadPoolExecutor(
                max_workers=len(self.tables), thread_name_prefix="engram-read"
            )

    def _views(self, count):
        return {
            "engram_hashes": self.hashes[:, :count],
            "engram_rows": self.rows[:, :count],
        }

    def prepare_dummy_inputs(self, num_reqs, num_tokens):
        inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self.tables:
            views = self._views(num_tokens)
            views["engram_hashes"].fill_(-1)
            views["engram_rows"].zero_()
            inputs.update(views)
        return inputs

    def prepare_inputs(self, input_batch, req_states):
        inputs = super().prepare_inputs(input_batch, req_states)
        if not self.tables:
            return inputs
        ids = input_batch.input_ids
        views = self._views(ids.numel())
        metadata = SimpleNamespace(
            num_actual_tokens=input_batch.num_tokens,
            query_start_loc=input_batch.query_start_loc[: input_batch.num_reqs + 1],
            seq_lens=input_batch.seq_lens[: input_batch.num_reqs],
        )
        if self.fused_hashes:
            hashes = self.hash_state.prepare_hashes_into(
                ids, inputs["lookback_token_ids"], metadata, views["engram_hashes"]
            )
        else:
            hashes = self.hash_state.prepare_hashes(
                ids, inputs["lookback_token_ids"], metadata
            )
            views["engram_hashes"].copy_(hashes)
        disk = self.tables[0].table.mode != "ram"
        # Submit all independent disk faults before processing the first layer.
        # Only hash indices leave VRAM; decoding the packed rows stays on GPU.
        if disk:
            numbers = hashes.cpu().numpy()
            futures = (
                [
                    self.readahead.submit(
                        emb.table.plan_rows,
                        numbers[i, :, emb.head_start : emb.head_end],
                        readahead=True,
                    )
                    for i, emb in enumerate(self.tables)
                ]
                if self.readahead
                else None
            )
        for i, emb in enumerate(self.tables):
            table = emb.table
            out = views["engram_rows"][i, :, : emb.head_end - emb.head_start]
            dest = (
                out
                if out.is_contiguous()
                else torch.empty_like(out, memory_format=torch.contiguous_format)
            )
            if disk:
                if futures is not None:
                    table.lookup_plan(futures[i].result(), dest)
                else:
                    table.lookup_host(
                        numbers[i, :, emb.head_start : emb.head_end], dest
                    )
            else:
                lookup_q4_pages(
                    table.pages,
                    hashes[i, :, emb.head_start : emb.head_end],
                    table.first_page,
                    table.start,
                    table.stop,
                    out=dest,
                )
            if dest is not out:
                out.copy_(dest)
        inputs.update(views)
        return inputs
