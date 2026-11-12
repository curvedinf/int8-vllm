# SPDX-License-Identifier: Apache-2.0
"""KSDraft speculator: block-draft proposals from a ksdraft export.

The drafter has no attention/KV-cache surface — it is a Rosa/KS3 block model
over a short token window — so this speculator runs it directly instead of
through the framework's draft-model attention path. Each propose() gathers
each request's trailing window, rolls the drafter forward N times (greedy or
temperature-matched), and publishes the tokens for the rejection sampler.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator


class KSDraftSpeculator(BaseSpeculator):
    num_query_per_req = 1
    num_extra_query_per_req = 0

    def __init__(self, vllm_config: VllmConfig, device: torch.device,
                 req_states: Any):
        from vllm.models.ksdraft.proposer import KSDraftProposer

        sc = vllm_config.speculative_config
        assert sc is not None and sc.model is not None
        self.vllm_config = vllm_config
        self.speculative_config = sc
        self.device = device
        self.req_states = req_states
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len
        self.num_speculative_steps = int(sc.num_speculative_tokens or 1)
        # The draft runs on one rank and the tokens are broadcast: TP ranks
        # share the request batch, so redundant copies only burn compute and
        # VRAM (the proposal is ~0.2-0.6 s/step on one MI100).
        from vllm.distributed.parallel_state import get_tp_group

        self._tp = get_tp_group()
        self._is_src = self._tp.rank_in_group == 0
        self.proposer = (KSDraftProposer(sc.model, device=device, int8=True)
                         if self._is_src else None)
        import os

        self.window = int(os.environ.get("KSDRAFT_WINDOW",
                                         self.proposer.window if self.proposer
                                         else 16))
        # Parallel block head is the default proposal path (one forward);
        # KSDRAFT_AR=1 keeps the autoregressive rollout for comparison.
        self.use_block = os.environ.get("KSDRAFT_AR", "0") != "1"
        # EAGLE-Harness-style adaptive scheduling (greedy block path).
        import os as _os

        self.adapt = _os.environ.get("KSDRAFT_ADAPT", "0") == "1"
        self.adapt_tau = float(_os.environ.get("KSDRAFT_CONF_TAU", "0.3"))
        self.adapt_low = int(_os.environ.get("KSDRAFT_LOW_LOAD", "4"))
        self.adapt_high = int(_os.environ.get("KSDRAFT_HIGH_LOAD", "8"))
        self.draft_tokens = torch.zeros(
            self.max_num_reqs, self.num_speculative_steps,
            dtype=torch.int64, device=device)
        self.draft_logits = None
        self.draft_token_confidence_probs = torch.zeros_like(
            self.draft_tokens, dtype=torch.float32)
        self.draft_attn_layer_names: set[str] = set()
        self.pcp_manager = None
        self.supports_mm_inputs = False
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self._win = torch.arange(self.window, device=device)

    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        pass

    def capture(self) -> None:
        pass

    def set_attn(self, *args, **kwargs) -> None:
        pass

    def load_model(self, *args, **kwargs) -> None:
        pass

    def observe_verification(self, *args, **kwargs) -> None:
        pass

    def prepare_watermarking(self, *args, **kwargs) -> None:
        pass

    @torch.no_grad()
    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dp_sync_state=None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs=None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        n = int(num_speculative_tokens or self.num_speculative_steps)
        if dummy_run or num_reqs == 0:
            return self.draft_tokens[:num_reqs, :n]

        token_ids = self.req_states.all_token_ids.gpu
        total = self.req_states.total_len.gpu
        idx = input_batch.idx_mapping[:num_reqs].long()
        lens = total[idx].long().clamp_min(1)
        pos = (lens[:, None] - self.window) + self._win[None, :]
        pos = pos.clamp_(min=0).clamp_max_((lens - 1)[:, None])
        rows = token_ids[idx]
        toks = torch.gather(rows, 1, pos)
        n_eff = n
        if self._is_src:
            temp = temperature[:num_reqs].to(self.device).float()
            adaptive = (self.adapt and self.use_block
                        and bool((temp <= 0).all()))
            if self.use_block and adaptive:
                from vllm.models.ksdraft.proposer import adaptive_draft_len

                out, conf = self.proposer.propose_block_with_conf(toks, n)
                n_eff = adaptive_draft_len(
                    conf, n, tau=self.adapt_tau, low_load=self.adapt_low,
                    high_load=self.adapt_high, load=int(num_reqs))
                self.draft_tokens[:num_reqs, :n].copy_(out)
            elif self.use_block:
                out = self.proposer.propose_block(toks, n, temperature=temp)
                self.draft_tokens[:num_reqs, :n].copy_(out)
            else:
                out = self.proposer.propose(toks, n, temperature=temp)
                self.draft_tokens[:num_reqs, :n].copy_(out)
            if n_eff < n:
                # the rejection sampler stops at negative draft ids
                self.draft_tokens[:num_reqs, n_eff:] = -1
        buf = self.draft_tokens[:num_reqs, :n]
        if self._tp.world_size > 1:
            self._tp.broadcast(buf, src=0)
        return buf
