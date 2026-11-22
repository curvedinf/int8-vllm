# SPDX-License-Identifier: Apache-2.0
"""KSDraft speculator: block-draft proposals from a ksdraft export.

Two draft flavors share this shell:

- The Rosa/KS3 block drafter (no attention surface): runs directly over the
  trailing token window (the historical path).
- The DFly2-style KV-reuse drafter (kv_dim > 0 in the export): each propose()
  additionally reads the target's int8_block_g128 KV cache for the tap
  layers (final loop pass), dequantizes the inline fp16 group scales, and
  all-gathers the TP-sharded KV heads into (B, taps, S, kv_dim) streams —
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
        # The proposer is constructed in load_model: a shared-table export
        # needs the target's own embedding/lm_head, which only exists then.
        self.proposer = None
        self._shared_export = False
        import os

        self.window = int(os.environ.get("KSDRAFT_WINDOW", "16"))
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
        # KV-reuse drafter state
        self._target_model = None
        self._tap_views = None          # [(codes_k, scale_k, codes_v, scale_v)]
        self._tap_prefixes: list = []   # per-tap pass-cache prefix (group map)
        self._tap_ptrs: tuple = ()
        self.kv_window = int(_os.environ.get("KSDRAFT_KV_WINDOW", "1024"))
        # every TP rank must know: the KV read + all-gather runs on all ranks
        # (only the proposer itself is rank-0)
        self._kv_ok = False
        try:
            import json as _json
            from pathlib import Path as _P
            _meta = _json.loads(
                (_P(sc.model) / "config.json").read_text())
            self._kv_ok = bool(_meta.get("config", {}).get("kv_dim", 0))
            self._shared_export = bool(_meta.get("shared_embedding"))
        except Exception:
            self._kv_ok = False

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
        import os

        self.window = int(os.environ.get("KSDRAFT_WINDOW",
                                         self.proposer.window))

    def _full_table(self, weight: torch.Tensor) -> torch.Tensor:
        """Gather a vocab-parallel [vocab/TP, d] table to the full [vocab, d]."""
        w = weight.detach().reshape(weight.shape[0], -1).contiguous()
        world = self._tp.world_size
        if world == 1:
            return w.float()
        outs = [torch.empty_like(w) for _ in range(world)]
        torch.distributed.all_gather(outs, w, group=self._tp.device_group)
        return torch.cat(outs, dim=0).float()

    def _target_tables(self, target_model):
        lm = (target_model.model
              if hasattr(target_model, "model") else target_model)
        # The export declares the table shape it needs; a packed (W4) output
        # head does not match it and must not be cast to float — its bytes
        # are quantized payload, not values.
        import json
        from pathlib import Path
        meta = json.loads(
            (Path(self.speculative_config.model) / "config.json").read_text())
        s = meta.get("config", {})
        want = (int(s.get("vocab_size", 0)), int(s.get("d_model", 0)))
        w = target_model.lm_head.weight
        if want[0] and (tuple(w.shape) != want):
            raise NotImplementedError(
                "shared-table export needs a dense [vocab, d_model] lm_head; "
                f"the target head is {tuple(w.shape)} (packed W4?) — export "
                "the draft with its own tables instead")
        emb = self._full_table(lm.embed_tokens.weight)
        head = self._full_table(w)
        return emb, head

    def _resolve_taps(self):
        if self._tap_views is not None:
            ptrs = tuple(t.data_ptr() for v in self._tap_views
                         for t in v[:1])
            if ptrs == self._tap_ptrs:
                return self._tap_views
        model = self._target_model
        if model is None:
            return None
        lm = model.model if hasattr(model, "model") else model
        layers = lm.layers
        views = []
        prefixes = []
        for layer_idx, pass_idx in KS_TAPS:
            attn = layers[layer_idx].attn
            caches = getattr(attn, "caches", None)
            if not caches or pass_idx >= len(caches):
                return None
            cache = caches[pass_idx]
            prefixes.append(getattr(cache, "prefix", None))
            kv = getattr(cache, "kv_cache", None)
            if not isinstance(kv, torch.Tensor) or kv.numel() == 0:
                return None
            views.append(_inline_scale_views(kv, attn.head_dim, 128))
        self._tap_prefixes = prefixes
        self._tap_views = views
        self._tap_ptrs = tuple(v[0].data_ptr() for v in views)
        return views

    def _tap_group_index(self, tap_i: int) -> int:
        """KV cache group holding this tap's pages (per-pass cache prefix)."""
        cfg = getattr(self, "kv_cache_config", None)
        if cfg is None or not getattr(self, "_tap_prefixes", None):
            return 0
        names = self._tap_prefixes[tap_i]
        for g, group in enumerate(cfg.kv_cache_groups):
            if names in group.layer_names:
                return g
        return 0

    def _read_kv_streams(self, num_reqs: int, idx: torch.Tensor,
                         lens: torch.Tensor,
                         kv_end: torch.Tensor) -> tuple[torch.Tensor, ...] | None:
        """Gather each request's trailing KV span from the tap caches.

        Returns the int8 cache form — (k_codes, k_scales, v_codes, v_scales),
        codes (num_reqs, taps, S, kv_dim) int8 and scales
        (num_reqs, taps, S, groups) fp16, with the FULL vocab of KV heads
        (TP all-gathered). Nothing is dequantized: the pair is the deployed
        int8_block_g128 content. Streams end at ``kv_end`` (the last position
        the target computed KV for; tokens appended after the last forward
        have no KV yet); S = kv_window + window - 1, early positions clamped
        at the request start (the training-time document-start clamp
        semantics). Each tap reads its own KV cache group's block table.
        """
        views = self._resolve_taps()
        if views is None or self.block_tables is None:
            return None
        bts = self.block_tables.gather_block_tables(idx, num_reqs)
        S = self.kv_window + self.window - 1
        pos = (kv_end[:, None] + 1 - S) + torch.arange(S, device=self.device)[None]
        pos = pos.clamp_min(0).clamp_max(kv_end[:, None])
        # SWA taps hold no keys older than the sliding window; the oldest
        # window-1 stream positions repeat the window edge instead of
        # reading null blocks
        pos = pos.clamp_min((kv_end + 1 - self.kv_window).clamp_min(0)[:, None])

        def gather_heads(t: torch.Tensor) -> torch.Tensor:
            """(B, S, heads, last) -> heads all-gathered across TP ranks."""
            world = self._tp.world_size
            if world == 1:
                return t
            flat = t.reshape(-1).contiguous()
            outs = [torch.empty_like(flat) for _ in range(world)]
            torch.distributed.all_gather(outs, flat,
                                         group=self._tp.device_group)
            return torch.stack(outs).view(world, *t.shape) \
                .permute(1, 2, 0, 3, 4).reshape(
                    t.shape[0], t.shape[1], -1, t.shape[3])

        kc, ksc, vc, vsc = [], [], [], []
        for tap_i, (codes_k, scale_k, codes_v, scale_v) in enumerate(views):
            bt = bts[self._tap_group_index(tap_i)].long()
            block_size = int(codes_k.shape[2])     # (blocks, heads, bs, hs)
            blk = bt.gather(1, (pos // block_size).clamp_max(bt.shape[1] - 1))
            off = pos % block_size
            # (blocks, heads, bs, hs) advanced indexing -> (B, S, heads, hs)
            ck = gather_heads(codes_k[blk, :, off, :])
            cv = gather_heads(codes_v[blk, :, off, :])
            sk = gather_heads(scale_k[blk, off, :, :])   # (B, S, heads, groups)
            sv = gather_heads(scale_v[blk, off, :, :])
            kc.append(ck.reshape(num_reqs, S, -1))
            vc.append(cv.reshape(num_reqs, S, -1))
            ksc.append(sk.reshape(num_reqs, S, -1))
            vsc.append(sv.reshape(num_reqs, S, -1))
        k_codes = torch.stack(kc, dim=1)          # (B, taps, S, D) int8
        v_codes = torch.stack(vc, dim=1)
        k_scales = torch.stack(ksc, dim=1)        # (B, taps, S, G) fp16
        v_scales = torch.stack(vsc, dim=1)
        return k_codes, k_scales, v_codes, v_scales

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
        n = int(num_speculative_tokens or self.num_speculative_steps)
        if dummy_run or num_reqs == 0:
            return self.draft_tokens[:num_reqs, :n]

        token_ids = self.req_states.all_token_ids.gpu
        total = self.req_states.total_len.gpu
        idx = input_batch.idx_mapping[:num_reqs].long()
        lens = total[idx].long().clamp_min(1)
        # Tokens appended after the last target forward (the accepted drafts +
        # bonus sample) are in all_token_ids but have no KV yet: the KV end is
        # lens-1-num_sampled. The token window AND the KV stream both end at
        # the last KV-backed position, so the draft's geometry matches
        # training exactly; proposals then start at kv_end+1, the first
        # already-sampled token, and their first num_sampled entries are
        # dropped per request.
        if torch.is_tensor(num_sampled):
            ns = num_sampled[:num_reqs].to(self.device).long()
        else:
            ns = torch.full((num_reqs,), int(num_sampled or 0),
                            dtype=torch.long, device=self.device)
        kv_end = (lens - 1 - ns).clamp_min(0)
        pos = (kv_end[:, None] + 1 - self.window) + self._win[None, :]
        pos = pos.clamp_(min=0).clamp_max_(kv_end[:, None])
        rows = token_ids[idx]
        toks = torch.gather(rows, 1, pos)
        n_eff = n
        if self._is_src or self._kv_ok:
            kv_streams = None
            if self._kv_ok:
                # all ranks read their KV shard + all-gather (collective)
                kv_streams = self._read_kv_streams(num_reqs, idx, lens, kv_end)
            if kv_streams is not None and self._is_src:
                k_max = int(ns.max())
                n_prop = min(n + k_max, 16)
                out = self.proposer.propose_block_with_kv(
                    toks, kv_streams, n_prop)
                if torch.all(ns == ns[0]):
                    k0 = int(ns[0])
                    out = out[:, k0:k0 + n]
                else:
                    drop = ns.clamp_max(n_prop - n)[:, None]
                    take = drop + torch.arange(n, device=out.device)[None, :]
                    out = torch.gather(out, 1, take)
                self.draft_tokens[:num_reqs, :n].copy_(out)
            elif kv_streams is None and self._is_src:
                if self._kv_ok:
                    # A reuse drafter (kv_dim > 0) cannot forward without
                    # the target's KV; skip speculation this step rather
                    # than crash or draft blind.
                    self.draft_tokens[:num_reqs, :n].fill_(-1)
                else:
                    temp = temperature[:num_reqs].to(self.device).float()
                    out = self.proposer.propose_block(toks, n, temperature=temp)
                    self.draft_tokens[:num_reqs, :n].copy_(out)
            if self._is_src and n_eff < n:
                # the rejection sampler stops at negative draft ids
                self.draft_tokens[:num_reqs, n_eff:] = -1
        buf = self.draft_tokens[:num_reqs, :n]
        if self._tp.world_size > 1:
            self._tp.broadcast(buf, src=0)
        return buf
