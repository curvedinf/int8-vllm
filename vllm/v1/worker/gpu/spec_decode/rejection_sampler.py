# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# GOALOPT lookup-drafting flag handoff: the DFlash2 speculator marks
# which request rows proposed from the context lookup; the rejection
# sampler consumes (and clears) the marks in the same engine step.
# q(proposal) is the one-hot by construction, so the Leviathan ratio
# test and residual resample stay exact for those rows.
_lookup_flag_tensor: torch.Tensor | None = None


def set_lookup_flags(flags: torch.Tensor) -> None:
    global _lookup_flag_tensor
    _lookup_flag_tensor = flags


def _pop_lookup_flags(num_reqs: int, device) -> torch.Tensor | None:
    global _lookup_flag_tensor
    t = _lookup_flag_tensor
    _lookup_flag_tensor = None
    if t is None or t.shape[0] < num_reqs or t.device != device:
        return None
    return t[:num_reqs]

from vllm.config import SpeculativeConfig
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    get_num_sampled_and_rejected,
)
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.output import SamplerOutput, SamplingMaskTensors
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

# Cap on the FP32 target-logits buffer materialized by apply_sampling_params.
# TODO(mgoin): Chunking is a workaround. The rejection kernels already upcast
# per vocab block on load and apply ops like temperature and gumbel, so folding
# sampling-param application into those kernels would remove this buffer and
# its traffic entirely.
MAX_CHUNK_BYTES = 2**30  # 1GB
_FP32_BYTES = 4


def get_max_chunk_logits(vocab_size: int) -> int:
    """Largest number of logits rows one verification chunk may hold."""
    return max(1, MAX_CHUNK_BYTES // (vocab_size * _FP32_BYTES))


def _iter_request_chunks(
    cu_num_logits: np.ndarray, max_chunk_logits: int
) -> Iterator[tuple[int, int]]:
    """Yield maximally packed request ranges without splitting requests."""
    assert max_chunk_logits > 0
    num_reqs = cu_num_logits.size - 1
    start = 0
    while start < num_reqs:
        max_logit = int(cu_num_logits[start]) + max_chunk_logits
        end = int(np.searchsorted(cu_num_logits, max_logit, side="right") - 1)
        end = min(num_reqs, max(start + 1, end))
        yield start, end
        start = end


def _p_ring_append(
    out_dir: str,
    processed_logits: torch.Tensor,
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    pos: torch.Tensor,
    idx_mapping_np: np.ndarray,
    draft_sampled: torch.Tensor | None = None,
    raw_logits: torch.Tensor | None = None,
) -> None:
    """Env-gated (VLLM_P_RING) per-round committed-token probability ring.

    For each request, per committed token c of the round, record the
    probability the verify distribution (processed/truncated logits row
    start+c) assigned to the committed token, plus the row's top-1 prob.
    Decisive for the wall question: at a repetition wall, does the target
    itself put p~1 on the wall token (state attractor) or is the committed
    token a low-p token the sampler should rarely return (resample defect)?

    When ``raw_logits`` is provided, the resample row's RAW distribution
    (pre sampling-param processing) is also recorded — splits a wrong
    verify FORWARD from wrong sampler POST-PROCESSING.
    """
    import pickle

    os.makedirs(out_dir, exist_ok=True)
    lp = torch.log_softmax(processed_logits, dim=-1)
    cu = cu_num_logits.cpu().tolist()
    ns = num_sampled.cpu().tolist()
    pos_l = pos.cpu().tolist()
    with open(os.path.join(out_dir, f"p_ring_{os.getpid()}.dump"), "ab") as f:
        for r, rs in enumerate(idx_mapping_np.tolist()):
            start, n = int(cu[r]), int(ns[r])
            if n <= 0:
                continue
            toks = sampled[r, :n]
            rows = lp[start : start + n]
            p_tok = rows.gather(1, toks.view(-1, 1)).exp().view(-1).cpu().tolist()
            top1 = rows.max(dim=-1).values.exp().cpu().tolist()
            # Per-row magnitude signature of the VERIFY rows (not just the
            # resample row): inf count + max finite |logit| per row —
            # discriminates a true sharp distribution from a +inf point mass
            # (LM-head/conv overflow regime) at p=1.0-garbage events.
            raw = processed_logits[start : start + n].float()
            inf_cnt = torch.isinf(raw).sum(dim=-1).cpu().tolist()
            famax = raw.abs()
            famax = famax.masked_fill(torch.isinf(raw) | raw.isnan(), 0)
            famax = [round(float(x), 2) for x in famax.max(dim=-1).values.cpu().tolist()]
            # The drafts verified this round (draft_sampled rows start+1..start+n-1
            # correspond to committed[0..n-2]; committed[n-1] is the resample).
            drafts = (
                draft_sampled[start + 1 : start + n].cpu().tolist()
                if draft_sampled is not None
                else []
            )
            # The REJECTED drafts at/after the resample position (the stale-KV
            # conviction instrument: the draft whose KV occupies the anchor
            # slot at the next round). draft_sampled[start+n] is the first
            # rejected draft; take up to 3 for context.
            rejected = (
                draft_sampled[start + n : start + n + 3].cpu().tolist()
                if draft_sampled is not None
                else []
            )
            # Magnitude signature of the resample row: inf-scale logits mean
            # an overflow upstream (fp16 cast / GEMM); plain NaN means a
            # poisoned state read.
            rrow = processed_logits[start + n - 1]
            row_nan = int(rrow.isnan().sum().item())
            row_absmax = float(rrow.abs().max().item())
            # Top-5 tokens of the resample row's (truncated) distribution:
            # the support the sampler saw — clean English vs salad.
            top5 = torch.topk(rrow, 5).indices.cpu().tolist()
            rec = {
                "rs": int(rs),
                "pos0": int(pos_l[start]),
                "n": n,
                "tok": toks.cpu().tolist(),
                "p": [round(x, 5) for x in p_tok],
                "top1": [round(x, 5) for x in top1],
                "drafts": drafts,
                "rejected": rejected,
                "row_nan": row_nan,
                "row_absmax": round(row_absmax, 2),
                "row_inf": [int(x) for x in inf_cnt],
                "row_famax": famax,
                "top5": top5,
            }
            if raw_logits is not None:
                # RAW (pre-processing) resample row: p of the committed token
                # and top-5 straight off the forward's logits.
                rraw = raw_logits[start + n - 1].float()
                lraw = torch.log_softmax(rraw, dim=-1)
                raw_p = float(lraw[toks[n - 1]].exp())
                rec["raw_p"] = round(raw_p, 5)
                rec["raw_top5"] = torch.topk(rraw, 5).indices.cpu().tolist()
                rec["raw_absmax"] = round(float(rraw.abs().max()), 2)
            pickle.dump(rec, f)


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    for i in range(num_sampled):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


@triton.jit
def _gather_draft_sampled_kernel(
    # [num_logits]
    draft_sampled_ptr,
    # [num_logits]
    pos_ptr,
    # [num_tokens]
    input_ids_ptr,
    # [num_tokens]
    positions_ptr,
    # [num_logits]
    logits_indices_ptr,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [max_num_reqs]
    prefill_len_ptr,
    num_logits,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < num_logits
    token_idx = tl.load(logits_indices_ptr + block, mask=mask)
    token_id = tl.load(input_ids_ptr + token_idx, mask=mask)
    pos = tl.load(positions_ptr + token_idx, mask=mask)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + block, mask=mask)
    local_pos = tl.load(expanded_local_pos_ptr + block, mask=mask)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx, mask=mask)
    # Drafts are only proposed after sampling past the prefill, so any draft
    # slots in a step starting within the prefill are placeholders (e.g. the
    # padded first step after a P/D remote KV load).
    is_placeholder = (local_pos > 0) & (pos - local_pos < prefill_len)
    token_id = tl.where(is_placeholder, -1, token_id)
    tl.store(draft_sampled_ptr + block, token_id, mask=mask)
    tl.store(pos_ptr + block, pos, mask=mask)


def gather_draft_sampled(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    logits_indices: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the input token and position of each logits row.

    Draft rows of requests that have not yet sampled past their prefill are
    set to -1 so that the rejection kernels reject them.
    """
    num_logits = logits_indices.shape[0]
    draft_sampled = input_ids.new_empty(num_logits)
    pos = positions.new_empty(num_logits)
    BLOCK_SIZE = 1024
    _gather_draft_sampled_kernel[(triton.cdiv(num_logits, BLOCK_SIZE),)](
        draft_sampled,
        pos,
        input_ids,
        positions,
        logits_indices,
        expanded_idx_mapping,
        expanded_local_pos,
        prefill_len,
        num_logits,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return draft_sampled, pos


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
        *,
        watermark_key: int | None = None,
    ):
        self.sampler = sampler
        self.watermark_key = watermark_key
        if watermark_key is not None:
            assert isinstance(sampler, GPUWatermarkSampler)
        self.num_speculative_steps = spec_config.num_speculative_tokens
        self.enable_adaptive_verification = spec_config.enable_adaptive_verification
        rejection_sample_method = spec_config.rejection_sample_method
        self.use_block_verification: bool = False
        self.fly_window_size = 0
        self.fly_entropy_threshold = spec_config.fly_entropy_threshold
        self.fly_entropy_top_k = spec_config.fly_entropy_top_k
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if rejection_sample_method == "synthetic":
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        elif rejection_sample_method == "block":
            self.use_block_verification = True
        elif rejection_sample_method == "fly":
            assert spec_config.fly_window_size is not None
            self.fly_window_size = spec_config.fly_window_size

    def _get_logprobs_tensors(
        self,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
        cu_num_logits: torch.Tensor,
        cu_num_logits_np: np.ndarray,
        max_num_logprobs: int,
    ) -> LogprobsTensors | None:
        if max_num_logprobs == NO_LOGPROBS:
            return None

        num_reqs = cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = torch.zeros(
            num_logits, dtype=sampled.dtype, device=sampled.device
        )
        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            cu_num_logits,
            num_warps=1,
        )
        expanded_logits = num_logits != num_reqs
        cu_num_generated_tokens: list[int] | torch.Tensor | None = None
        if expanded_logits:
            if self.enable_adaptive_verification:
                # Adaptive verification keeps the true per-request boundaries
                # on device only; cu_num_logits_np holds the pre-compacted layout.
                cu_num_generated_tokens = cu_num_logits.clone()
            else:
                cu_num_generated_tokens = cu_num_logits_np.tolist()
        return compute_topk_scores(
            logits,
            max_num_logprobs,
            flat_sampled,
            cu_num_generated_tokens,
            logits_mode=self.sampler.logprobs_mode
            in ("raw_logits", "processed_logits"),
        )

    def _watermarking_kwargs(
        self,
        draft_sampled: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> dict[str, Any]:
        if self.watermark_key is None:
            return {}
        assert isinstance(self.sampler, GPUWatermarkSampler)
        contexts = self.sampler._get_contexts(
            expanded_idx_mapping, expanded_local_pos, draft_sampled
        )
        watermarking_skip_mask = None
        if self.sampler.deduplicate_contexts != "none":
            watermarking_skip_mask = self.sampler._get_repeated_contexts(
                expanded_idx_mapping, contexts, expanded_local_pos
            )
        return {
            "contexts": contexts,
            "watermarking": self.sampler.watermarking.gpu,
            "watermarking_skip_mask": watermarking_skip_mask,
            "watermark_key": self.watermark_key,
        }

    def _verify(
        self,
        logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        seq_lens_upper_bound_np: np.ndarray,
        verify_draft_sampled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            draft_sampled,
            expanded_local_pos,
            seq_lens_upper_bound_np,
        )
        if os.environ.get("VLLM_VERIFY_TRUNC_DEBUG") and not torch.cuda.is_current_stream_capturing():
            # One-shot ground truth: does the verify path actually deliver
            # truncated logits to rejection_sample? For a request with
            # top_k < vocab, every row must have at most top_k finite entries.
            states = self.sampler.sampling_states
            for r, st in enumerate(idx_mapping_np.tolist()):
                tk = int(states.top_k.np[st]) if states.top_k is not None else -1
                if 0 < tk < processed_logits.shape[1]:
                    finite = int(
                        torch.isfinite(processed_logits[r]).sum().item()
                    )
                    if finite > tk:
                        logger.warning(
                            "VERIFY-TRUNC LEAK: row %d req_state %d top_k=%d "
                            "but %d finite logits rows -> rejection_sample "
                            "gets UNTRUNCATED target",
                            r, st, tk, finite,
                        )
        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            verify_draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification,
            salt_resample=bool(os.environ.get("VLLM_RESAMPLE_SALT")),
            salt_u=bool(os.environ.get("VLLM_SALT_U")),
            lookup_req=_pop_lookup_flags(num_reqs=idx_mapping.shape[0],
                                         device=logits.device),
            fly_window_size=self.fly_window_size,
            fly_entropy_threshold=self.fly_entropy_threshold,
            fly_entropy_top_k=self.fly_entropy_top_k,
            **self._watermarking_kwargs(
                draft_sampled, expanded_idx_mapping, expanded_local_pos
            ),
        )
        _p_ring = os.environ.get("VLLM_P_RING")
        if _p_ring and not torch.cuda.is_current_stream_capturing():
            _p_ring_append(
                _p_ring,
                processed_logits,
                sampled,
                num_sampled,
                cu_num_logits,
                pos,
                idx_mapping_np,
                draft_sampled,
                raw_logits=logits,
            )
        return processed_logits, sampled, num_sampled

    def _verify_in_chunks(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        max_chunk_logits: int,
        max_num_logprobs: int,
        verify_draft_sampled: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        LogprobsTensors | None,
        SamplingMaskTensors | None,
    ]:
        if verify_draft_sampled is None:
            verify_draft_sampled = draft_sampled
        cu_num_logits_np = input_batch.cu_num_logits_np
        use_processed_logits = self.sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES
        num_reqs = input_batch.num_reqs

        if logits.shape[0] <= max_chunk_logits:
            # One chunk covers the batch. Adaptive verification compacts the logits
            # without updating cu_num_logits_np (it keeps the pre-compacted layout),
            # so the stale sums must not pick chunk boundaries; its budget cap
            # guarantees the compacted batch always lands here.
            request_chunks: Iterable[tuple[int, int]] = ((0, num_reqs),)
        else:
            assert not self.enable_adaptive_verification
            request_chunks = _iter_request_chunks(cu_num_logits_np, max_chunk_logits)

        sampled_chunks: list[torch.Tensor] = []
        num_sampled_chunks: list[torch.Tensor] = []
        logprobs_chunks: list[LogprobsTensors] = []
        sampling_mask_chunks: list[SamplingMaskTensors] = []
        max_num_kept = None
        if self.sampler.return_sampling_mask:
            max_num_kept = int(
                np.max(
                    self.sampler.sampling_states.top_k.np[input_batch.idx_mapping_np]
                )
            )

        for start, end in request_chunks:
            lo = int(cu_num_logits_np[start])
            hi = int(cu_num_logits_np[end])
            chunk_cu_num_logits_np = cu_num_logits_np[start : end + 1] - lo
            chunk_cu_num_logits = input_batch.cu_num_logits[start : end + 1] - lo
            # draft_logits uses persistent request-state indices and stays global.
            processed_logits, sampled, num_sampled = self._verify(
                logits[lo:hi],
                draft_logits,
                draft_sampled[lo:hi],
                pos[lo:hi],
                chunk_cu_num_logits,
                input_batch.idx_mapping[start:end],
                input_batch.idx_mapping_np[start:end],
                input_batch.expanded_idx_mapping[lo:hi],
                input_batch.expanded_local_pos[lo:hi],
                input_batch.seq_lens_cpu_upper_bound.numpy()[start:end],
                verify_draft_sampled[lo:hi],
            )
            chunk_logprobs = self._get_logprobs_tensors(
                sampled,
                num_sampled,
                processed_logits if use_processed_logits else logits[lo:hi],
                chunk_cu_num_logits,
                chunk_cu_num_logits_np,
                max_num_logprobs,
            )
            if chunk_logprobs is not None:
                logprobs_chunks.append(chunk_logprobs)
            if max_num_kept is not None:
                sampling_mask_chunks.append(
                    SamplingMaskTensors.from_logits(
                        processed_logits,
                        chunk_cu_num_logits,
                        num_sampled,
                        max_num_kept,
                        self.num_speculative_steps + 1,
                    )
                )
            del processed_logits
            sampled_chunks.append(sampled)
            num_sampled_chunks.append(num_sampled)

        if len(sampled_chunks) == 1:
            logprobs_tensors = logprobs_chunks[0] if logprobs_chunks else None
            sampling_mask_tensors = (
                sampling_mask_chunks[0] if sampling_mask_chunks else None
            )
            return (
                sampled_chunks[0],
                num_sampled_chunks[0],
                logprobs_tensors,
                sampling_mask_tensors,
            )

        logprobs_tensors = None
        if logprobs_chunks:
            expanded_logits = logits.shape[0] != input_batch.num_reqs
            logprobs_tensors = LogprobsTensors.cat(
                logprobs_chunks,
                cu_num_generated_tokens=(
                    cu_num_logits_np.tolist() if expanded_logits else None
                ),
            )

        sampled = torch.cat(sampled_chunks)
        num_sampled = torch.cat(num_sampled_chunks)
        sampling_mask_tensors = (
            SamplingMaskTensors.cat(sampling_mask_chunks)
            if sampling_mask_chunks
            else None
        )
        return sampled, num_sampled, logprobs_tensors, sampling_mask_tensors

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
        invalid_drafts: torch.Tensor | None = None,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        draft_sampled, pos = gather_draft_sampled(
            input_batch.input_ids,
            input_batch.positions,
            input_batch.logits_indices,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            self.sampler.req_states.prefill_len.gpu,
        )

        # Marking a draft invalid (`is_valid_draft = draft_sampled >= 0`) pins the
        # accepted length before it reaches a permissive bitmask row. Only
        # verification sees the copy: `apply_sampling_params` and watermarking
        # keep the real draft ids.
        verify_draft_sampled = draft_sampled
        if invalid_drafts is not None:
            verify_draft_sampled = draft_sampled.masked_fill(invalid_drafts, -1)

        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        chunk_logit_limit = get_max_chunk_logits(logits.shape[1])
        (
            sampled,
            num_sampled,
            logprobs_tensors,
            sampling_mask_tensors,
        ) = self._verify_in_chunks(
            logits,
            input_batch,
            draft_logits,
            draft_sampled,
            pos,
            chunk_logit_limit,
            max_num_logprobs,
            verify_draft_sampled,
        )

        if os.environ.get("VLLM_ACCEPT1"):
            # ACCEPT1 diagnostic (same lever as the V1 sampler): force every
            # verify round to commit exactly one token. The speculative
            # compute and temp-1 sampling stay fully active, but the
            # post-round mamba state resume geometry (num_accepted - 1)
            # becomes 0 every round, identical to non-spec decode. Clean leg
            # => the rejection-resume math is the corruption source; drifting
            # leg => the multi-token verify write path itself corrupts state.
            if sampled.shape[-1] > 1:
                sampled[:, 1:].fill_(-1)
            num_sampled.clamp_(max=1)

        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            sampling_mask_tensors=sampling_mask_tensors,
        )
