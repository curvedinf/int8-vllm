# SPDX-License-Identifier: Apache-2.0
"""KSDraft speculator: block-draft proposals from a ksdraft export.

Two draft flavors share this shell:

- The Rosa/KS3 block drafter (no attention surface): runs directly over the
  trailing token window (the historical path).
- The DFly2-style KV-reuse drafter (kv_dim > 0 in the export): each propose()
  additionally reads the target's int8_block_g128 KV cache for the tap
  layers (final loop pass) and gathers INT8 codes and FP16 group scales
  across TP heads on the proposing rank into (B, taps, S, kv_dim) streams —
  the contract the draft's TargetKVAttention consumes.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

# (weight layer, loop pass): prelude tap, two final-pass loop taps, coda tap.
# Must match scripts/mine_inkling_signals.py TAPS in the ksdraft repo.
KS_TAPS = ((4, 0), (10, 2), (16, 2), (22, 0))


def _inline_scale_views(kv_cache: torch.Tensor, head_size: int, group: int):
    """(blocks, heads, bs, content) int8 cache -> (codes, scale) views.

    Replicates the backend's int8_block_g layout: each content half holds
    ``head_size`` int8 codes then one fp16 group scale per g-dim group.
    Returns ((blocks, heads, bs, hs) int8, (blocks, bs, heads, groups) fp16)
    for K and V.
    """
    num_blocks, nkv, block_size, content = kv_cache.shape
    dtype_sz = kv_cache.element_size()
    padded_hs = content // 2
    groups = head_size // group
    raw = kv_cache.untyped_storage()
    base_f16 = torch.tensor([], dtype=torch.float16,
                            device=kv_cache.device).set_(raw)

    def f16_units(elements: int) -> int:
        return (elements * dtype_sz) // 2

    strides = kv_cache.stride()
    block_f16 = f16_units(strides[0])
    head_f16 = f16_units(strides[1])
    slot_f16 = f16_units(strides[2])
    base_off = f16_units(kv_cache.storage_offset())
    k_off = base_off + f16_units(head_size)
    v_off = base_off + f16_units(padded_hs + head_size)

    def scales(off):
        return torch.as_strided(
            base_f16, size=(num_blocks, block_size, nkv, groups),
            stride=(block_f16, slot_f16, head_f16, 1), storage_offset=off)

    codes_k = kv_cache[..., :head_size]
    codes_v = kv_cache[..., padded_hs:padded_hs + head_size]
    return codes_k, scales(k_off), codes_v, scales(v_off)


class KSDraftSpeculator(BaseSpeculator):
    num_query_per_req = 1
    num_extra_query_per_req = 0

    def __init__(self, vllm_config: VllmConfig, device: torch.device,
                 req_states: Any):
        import json
        import os
        from pathlib import Path

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
        # The proposer is constructed in load_model: a shared-table export
        # needs the target's own embedding/lm_head, which only exists then.
        self.proposer = None

        self.window = int(os.environ.get("KSDRAFT_WINDOW", "16"))
        # Parallel block head is the default proposal path (one forward);
        # KSDRAFT_AR=1 keeps the autoregressive rollout for comparison.
        self.use_block = os.environ.get("KSDRAFT_AR", "0") != "1"
        # EAGLE-Harness-style adaptive scheduling (greedy block path).
        self.adapt = os.environ.get("KSDRAFT_ADAPT", "0") == "1"
        self.adapt_tau = float(os.environ.get("KSDRAFT_CONF_TAU", "0.3"))
        self.adapt_low = int(os.environ.get("KSDRAFT_LOW_LOAD", "4"))
        self.adapt_high = int(os.environ.get("KSDRAFT_HIGH_LOAD", "8"))
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
        # KV-reuse drafter state
        self._target_model = None
        self._tap_views = None          # [(codes_k, scale_k, codes_v, scale_v)]
        self._tap_ptrs: tuple = ()
        # every TP rank must know: the KV read + all-gather runs on all ranks
        # (only the proposer itself is rank-0)
        meta = json.loads((Path(sc.model) / "config.json").read_text())
        self.kv_window = int(os.environ.get("KSDRAFT_KV_WINDOW", meta["config"].get("kv_window", 1024)))
        self._kv_ok = bool(meta["config"].get("kv_dim", 0))
        self._shared_export = bool(meta.get("shared_embedding"))
        self._shared_head = bool(meta.get("shared_head"))
        self._vocab_size = meta["config"]["vocab_size"]
        self._block_len = meta["config"].get("block_len", 16)
        if self._kv_ok and self._block_len < 2:
            raise ValueError("KV-reuse drafting needs room for the sampled token and a proposal")

    # ----- target KV reading (DFly2-style reuse) -----

    def load_model(self, target_model, *args, **kwargs) -> None:
        self._target_model = target_model
        # The table gather is collective: every TP rank must enter it, so it
        # runs before the rank guard.
        emb = head = None
        if self._shared_export:
            emb, head = self._target_tables(target_model)
        if not self._is_src:
            return
        from vllm.models.ksdraft.proposer import KSDraftProposer

        self.proposer = KSDraftProposer(
            self.speculative_config.model, device=self.device, int8=True,
            embedding=emb, head=head)

    def _full_table(self, layer):
        weight = layer.weight.detach()
        if getattr(layer, "_int8_embedding", False):
            rows = torch.arange(weight.shape[0], device=weight.device)
            weight = layer.quant_method.embedding(layer, rows)
        elif not weight.is_floating_point():
            weight = layer.quant_method.dequantize_weight(layer)
        full = self._tp.gather(weight.contiguous(), dst=0, dim=0)
        if full is None:
            return None
        mapping = layer.get_sharded_to_full_mapping()
        if mapping is not None:
            full = full[mapping[:self._vocab_size]]
        return full[:self._vocab_size]

    def _target_tables(self, target_model):
        lm = getattr(target_model, "model", target_model)
        embedding = self._full_table(lm.embed_tokens)
        head = self._full_table(target_model.lm_head) if self._shared_head else None
        return embedding, head

    def _resolve_taps(self):
        model = self._target_model
        if model is None:
            raise RuntimeError("KSDraft target model was not initialized")
        layers = getattr(model, "model", model).layers
        caches = []
        for layer_idx, pass_idx in KS_TAPS:
            attn = layers[layer_idx].attn
            cache = attn.caches[pass_idx]
            if cache.kv_cache.numel() == 0:
                raise RuntimeError(f"KSDraft tap {cache.prefix} has no allocated cache")
            caches.append((cache, attn.head_dim))
        ptrs = tuple(cache.kv_cache.data_ptr() for cache, _ in caches)
        if self._tap_views is None or self._tap_ptrs != ptrs:
            group_ids = {name: i for i, group in enumerate(self.kv_cache_config.kv_cache_groups)
                         for name in group.layer_names}
            self._tap_views = [
                (group_ids[cache.prefix], *_inline_scale_views(cache.kv_cache, head_dim, 128))
                for cache, head_dim in caches
            ]
            self._tap_ptrs = ptrs
        return self._tap_views

    def _read_kv_streams(self, num_reqs, kv_end):
        views = self._resolve_taps()
        bts = self.block_tables.input_block_tables
        S = self.kv_window + self.window - 1
        pos = (kv_end[:, None] + 1 - S) + torch.arange(S, device=self.device)[None]
        pos = pos.clamp_min(0).clamp_max(kv_end[:, None])
        # SWA taps hold no keys older than the sliding window; the oldest
        # window-1 stream positions repeat the window edge instead of
        # reading null blocks
        pos = pos.clamp_min((kv_end + 1 - self.kv_window).clamp_min(0)[:, None])

        codes, scales = [], []
        for group_id, codes_k, scale_k, codes_v, scale_v in views:
            bt = bts[group_id][:num_reqs]
            block_size = codes_k.shape[2]
            blk = bt.gather(1, pos // block_size).long()
            off = pos % block_size
            codes.append(torch.stack((codes_k[blk, :, off], codes_v[blk, :, off])))
            scales.append(torch.stack((scale_k[blk, off], scale_v[blk, off])))
        # [K/V, batch, tap, time, local heads, head width or scale groups].
        codes = self._tp.gather(torch.stack(codes, dim=2), dst=0, dim=4)
        scales = self._tp.gather(torch.stack(scales, dim=2), dst=0, dim=4)
        if self._is_src:
            k, v = codes.flatten(-2).unbind()
            ks, vs = scales.flatten(-2).unbind()
            return k, ks, v, vs
        return None

    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        pass

    def capture(self) -> None:
        pass

    def set_attn(self, model_state=None, kv_cache_config=None,
                 block_tables=None, *args, **kwargs) -> None:
        # the KV-reuse path reads the target caches directly; keep the
        # handles the reader needs (no draft attention backend to init)
        self.block_tables = block_tables
        self.kv_cache_config = kv_cache_config

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
        n = self.num_speculative_steps if num_speculative_tokens is None else num_speculative_tokens
        buf = self.draft_tokens[:num_reqs]
        if dummy_run or num_reqs == 0 or n == 0:
            return buf.fill_(-1)

        token_ids = self.req_states.all_token_ids.gpu
        total = self.req_states.total_len.gpu
        idx = input_batch.idx_mapping[:num_reqs].long()
        lens = total[idx].long().clamp_min(1)
        end = lens - 1
        if self._kv_ok:
            computed = self.req_states.num_computed_tokens.gpu[idx].long()
            end = (computed - 1).clamp_min(0)
        pos = (end[:, None] + 1 - self.window + self._win).clamp_min(0)
        toks = token_ids[idx[:, None], pos]
        if self._kv_ok:
            kv = self._read_kv_streams(num_reqs, end)
            if self._is_src:
                # Verification computed KV for all accepted drafts. Only the
                # final sampled (bonus/replacement) token lacks target KV.
                count = min(n, self._block_len - 1)
                sampled = token_ids[idx, lens - 1]
                out = self.proposer.propose_block_with_kv(
                    toks, kv, count + 1, first_token=sampled)
                buf.fill_(-1)
                # Partially-prefilled requests have no sampled continuation.
                ready = (computed > 0) & (lens == computed + 1) & (num_sampled[:num_reqs] > 0)
                buf[:, :count].copy_(torch.where(ready[:, None], out[:, 1:], -1))
        elif self._is_src:
            temp = temperature[idx].float()
            if self.use_block and self.adapt and bool((temp <= 0).all()):
                from vllm.models.ksdraft.proposer import adaptive_draft_len

                out, conf = self.proposer.propose_block_with_conf(toks, n)
                n_eff = adaptive_draft_len(conf, n, self.adapt_tau, self.adapt_low,
                                           self.adapt_high, num_reqs)
                out[:, n_eff:] = -1
            elif self.use_block:
                out = self.proposer.propose_block(toks, n, temperature=temp)
            else:
                out = self.proposer.propose(toks, n, temperature=temp)
            buf[:, :n].copy_(out)
        buf[:, n:] = -1
        if self._tp.world_size > 1:
            self._tp.broadcast(buf, src=0)
        return buf
