# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

# Env-gated candidate ring (VLLM_CAND_RING): per-round (candidate_ids,
# scores, draft_tokens) clones from _sample_path, flushed every 500 rounds.
_cand_ring: list = []
_hs_stats: list = []


def _cand_ring_flush(pid: int) -> None:
    import os
    import pickle

    out_dir = os.environ.get("VLLM_CAND_RING")
    if not out_dir or not _cand_ring:
        return
    with open(os.path.join(out_dir, f"cand_ring_{pid}.dump"), "ab") as f:
        for ids, scores, drafts in _cand_ring:
            pickle.dump(
                {
                    "cand": ids.cpu().tolist(),
                    "scores": scores.float().cpu().tolist(),
                    "drafts": drafts.cpu().tolist(),
                },
                f,
            )
    _cand_ring.clear()
    if _hs_stats:
        with open(os.path.join(out_dir, f"hs_stats_{pid}.dump"), "ab") as f:
            for nan_cnt, absmax in _hs_stats:
                pickle.dump(
                    {"nan": nan_cnt.tolist(), "absmax": absmax.tolist()}, f
                )
        _hs_stats.clear()


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    tokens_ptr,
    realized_scores_ptr,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    previous = 0
    for step in range(num_steps):
        flat = row * num_steps + step
        score_base = (flat * top_k + previous) * top_k
        scores = tl.load(
            scores_ptr + score_base + offsets,
            mask=mask & valid,
            other=float("-inf"),
        ).to(tl.float64 if USE_FP64 else tl.float32)
        candidate_base = flat * top_k
        candidates = tl.load(
            candidate_ptr + candidate_base + offsets,
            mask=mask & valid,
            other=0,
        )

        # Candidate ids key the noise, matching the target's own sampling.
        position = tl.load(sample_pos_ptr + flat) - 1
        _, index = gumbel_noised_argmax(
            scores,
            candidates,
            mask & valid,
            seed,
            position,
            temperature if SAMPLE_PROBABILISTIC else 0.0,
            USE_FP64=USE_FP64,
        )

        tl.store(
            realized_scores_ptr + candidate_base + offsets,
            scores,
            mask=mask & valid,
        )
        token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
        tl.store(tokens_ptr + flat, token, mask=valid)
        previous = index


@triton.jit
def _cache_draft_logits_kernel(
    draft_logits_ptr,
    cached_candidate_ptr,
    candidate_ptr,
    scores_ptr,
    req_state_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    flat = tl.program_id(0)
    req_state = tl.load(req_state_ptr + flat)
    step = flat % num_steps
    offsets = tl.arange(0, BLOCK_K)
    mask = (req_state >= 0) & (offsets < top_k)
    candidate_base = flat * top_k
    cache_base = (req_state * num_steps + step) * top_k
    old_token_ids = tl.load(cached_candidate_ptr + cache_base + offsets, mask=mask)
    logits_base = (
        draft_logits_ptr
        + req_state * draft_logits_stride_0
        + step * draft_logits_stride_1
    )
    tl.store(logits_base + old_token_ids, -float("inf"), mask=mask)
    token_ids = tl.load(candidate_ptr + candidate_base + offsets, mask=mask)
    scores = tl.load(scores_ptr + candidate_base + offsets, mask=mask)
    tl.store(logits_base + token_ids, scores, mask=mask)
    tl.store(cached_candidate_ptr + cache_base + offsets, token_ids, mask=mask)


class DFlash2Speculator(DFlashSpeculator):
    _speculator_name = "DFlash2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        draft_config = self.draft_model_config.hf_config.dflash_config
        self.selector_top_k = int(draft_config["selector_top_k"])
        self._anchor_indices = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        self._selector_scores = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            self.selector_top_k,
            dtype=torch.float32,
            device=device,
        )
        self._cached_candidate_ids = torch.zeros(
            self._selector_scores.shape, dtype=torch.int64, device=device
        )

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # fp32 so the walk and the rejection that checks it read the same
        # distribution; -inf because the cache kernel writes only the K
        # candidates.
        return torch.float32, -float("inf")

    def _sample_path(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
    ) -> None:
        if os.environ.get("VLLM_CAND_RING") and not torch.cuda.is_current_stream_capturing():
            _cand_ring.append(
                (candidate_ids.detach().clone(), scores.detach().clone(),
                 self.draft_tokens.detach().clone()))
            if len(_cand_ring) >= 500:
                _cand_ring_flush(os.getpid())
        _selector_walk_kernel[(num_reqs,)](
            scores.contiguous(),
            candidate_ids.contiguous(),
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self._selector_scores,
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            SAMPLE_PROBABILISTIC=self.draft_logits is not None,
            USE_FP64=self.use_fp64_gumbel,
            num_warps=1,
        )

    def _cache_draft_logits(self, candidate_ids: torch.Tensor, num_sample: int) -> None:
        draft_logits = self.draft_logits
        assert draft_logits is not None
        block_k = triton.next_power_of_2(self.selector_top_k)
        _cache_draft_logits_kernel[(num_sample,)](
            draft_logits,
            self._cached_candidate_ids,
            candidate_ids,
            self._selector_scores,
            self.sample_idx_mapping,
            draft_logits.stride(0),
            draft_logits.stride(1),
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            num_warps=1,
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
            num_reqs, self.num_speculative_steps, -1
        )
        if os.environ.get("VLLM_CAND_RING") and not torch.cuda.is_current_stream_capturing():
            # cheap GPU-side stats: per-(req, step) NaN count + absmax of
            # the draft hidden states (flushed with the candidate ring).
            _hs_stats.append(
                (
                    hidden_states.isnan().sum(dim=-1).cpu(),
                    hidden_states.abs().amax(dim=-1).cpu(),
                )
            )
        from vllm import quant_audit_recorder as _qa

        if _qa._enabled() and not torch.cuda.is_current_stream_capturing():
            _qa.record_draft(
                last_hidden_states[:32], self.draft_tokens[:8], 0
            )
        if os.environ.get("VLLM_SPEC_DEBUG_DUMP") and not torch.cuda.is_current_stream_capturing():
            hs = hidden_states[0, 0]
            print(
                f"[SPEC-DBG4] hs[0,0] nan={int(hs.isnan().sum())}/{hs.numel()} "
                f"absmax={hs.abs().max().item():.3f} "
                f"in_ids[:6]={self.input_buffers.input_ids[:6].cpu().tolist()} "
                f"full_hs_nan={int(last_hidden_states.isnan().sum())}/{last_hidden_states.numel()}",
                flush=True,
            )
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(
            num_reqs, self.num_speculative_steps, self.selector_top_k
        )
        unary_logits = unary_logits.view_as(candidate_ids)
        if os.environ.get("VLLM_DFLASH_AUDIT"):
            from vllm import quant_audit_recorder as _qa

            _qa.record_dflash_stage(
                "candidates",
                hidden=hidden_states,
                ids=candidate_ids,
                unary=unary_logits,
            )
        anchor_token_ids = self.input_buffers.input_ids[self._anchor_indices[:num_reqs]]
        if os.environ.get("VLLM_SPEC_DEBUG_DUMP") and not torch.cuda.is_current_stream_capturing():
            n = min(num_reqs, 4)
            print(
                f"[SPEC-DBG3] reqs={num_reqs} cand[0,0,:6]="
                f"{candidate_ids[0,0,:6].cpu().tolist()} "
                f"cand[0,1,:6]={candidate_ids[0,1,:6].cpu().tolist()} "
                f"unary[0,0,:4]={unary_logits[0,0,:4].float().cpu().tolist()} "
                f"anchor[:4]={anchor_token_ids[:n].view(-1)[:4].cpu().tolist() if anchor_token_ids.numel() else 'none'}",
                flush=True,
            )
        scores = self.model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden_states,
            anchor_token_ids,
        )
        self._sample_path(candidate_ids, scores, num_reqs)
        if os.environ.get("VLLM_DFLASH_AUDIT"):
            from vllm import quant_audit_recorder as _qa

            _qa.record_dflash_stage(
                "selector",
                scores=scores,
                selected=self.draft_tokens[:num_reqs],
                anchor=anchor_token_ids,
            )
        if self.draft_logits is not None:
            self._cache_draft_logits(candidate_ids, num_sample)
