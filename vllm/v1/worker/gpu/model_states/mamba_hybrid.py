# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import json
import os

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState
from vllm.v1.worker.mamba_utils import (
    MambaSpecDecodeGPUContext,
    preprocess_mamba_align_fused_kernel,
)
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


@dataclass
class MambaHybridAttnMetadata(ModelSpecificAttnMetadata):
    is_prefilling: torch.Tensor
    num_accepted_tokens: torch.Tensor | None = None
    num_decode_draft_tokens_cpu: torch.Tensor | None = None

    def get_extra_common_attn_kwargs(
        self,
        kv_cache_group_id: int,
        num_reqs: int,
    ) -> dict[str, Any]:
        return {"is_prefilling": self.is_prefilling[:num_reqs]}

    def get_extra_attn_kwargs(
        self,
        attn_metadata_builder: Any,
        num_reqs: int,
    ) -> dict[str, Any]:
        if not isinstance(
            attn_metadata_builder,
            (Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder),
        ):
            return {}
        return {
            "num_accepted_tokens": None
            if self.num_accepted_tokens is None
            else self.num_accepted_tokens[:num_reqs],
            "num_decode_draft_tokens_cpu": None
            if self.num_decode_draft_tokens_cpu is None
            else self.num_decode_draft_tokens_cpu[:num_reqs],
        }


class MambaHybridModelState(DefaultModelState):
    """Model state for hybrid attention + Mamba / linear-attention models."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.cache_config = vllm_config.cache_config
        self.num_accepted_tokens_gpu = torch.ones(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        # Pre-copy "align" prefix-cache state (V2). The migration of each
        # request's mamba state across block boundaries runs as a fused GPU
        # kernel reusing the postprocess copy machinery, so the per-step src
        # columns and the running state_idx are kept GPU-resident.
        self._align_mode = self.cache_config.mamba_cache_mode == "align"
        self.recoverssm = (
            RecoverSSMState() if self.cache_config.use_kda_recoverssm else None
        )
        if self._align_mode:
            self._mamba_state_idx_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            self._mamba_src_col_gpu = torch.full(
                (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
            )
            self._mamba_src_off_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            self._mamba_ctx: MambaSpecDecodeGPUContext | None = None
            self._mamba_group_ids: list[int] = []
            self._mamba_spec: MambaSpec | None = None

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        # Must reset the speculative acceptance count in this idx which could be stale.
        self.num_accepted_tokens_gpu[req_index].fill_(1)
        if self._align_mode:
            # Seed the running state block from the resumed/prefilled position.
            # Divide by the mamba group's block size, NOT cache_config.block_size:
            # the DFlash speculator's dataclasses.replace(vllm_config, ...) re-runs
            # the gfx908 platform hook, which resets the shared
            # cache_config.block_size to 32 while the mamba group runs on the
            # aligned 1728. The mismatch seeded a wild src column and crashed
            # precopy_mamba_align_fused_kernel with hipErrorIllegalAddress on
            # prefix-cache resumes (e.g. nct=5184 seeded (5184-1)//32=161 in a
            # 51-column row instead of (5184-1)//1728=2).
            # _mamba_spec is populated lazily by the first preprocess_state; a
            # resume implies a prior batch, and fresh requests seed -1 either
            # way, so the mamba_block_size fallback is never actually taken.
            mamba_block_size = (
                self._mamba_spec.block_size
                if self._mamba_spec is not None
                else self.cache_config.mamba_block_size
            )
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // mamba_block_size
            )

    def _get_mamba_group_info(
        self, kv_cache_config: KVCacheConfig
    ) -> tuple[list[int], MambaSpec]:
        if self._mamba_spec is None:
            group_ids: list[int] = []
            specs: list[MambaSpec] = []
            for i, group in enumerate(kv_cache_config.kv_cache_groups):
                spec = group.kv_cache_spec
                if isinstance(spec, MambaSpec):
                    group_ids.append(i)
                    specs.append(spec)
            assert specs, "no mamba layers in the model"
            assert all(specs[0] == s for s in specs)
            self._mamba_group_ids = group_ids
            self._mamba_spec = specs[0]
        return self._mamba_group_ids, self._mamba_spec

    def _ensure_align_ctx(
        self,
        kv_cache_config: KVCacheConfig,
        mamba_group_ids: list[int],
        block_tables: tuple[torch.Tensor, ...],
    ) -> MambaSpecDecodeGPUContext:
        if self._mamba_ctx is None:
            copy_funcs = self.model.get_mamba_state_copy_func()
            # Both SD and DS conv layouts support a >0 spec-decode shift: the
            # fused pre-copy kernel (``_copy_mamba_state_block``) applies the
            # ``token_bias = num_accepted - 1`` window shift per conv layout
            # (SD: contiguous slice; DS: per-dim-row strided slice), matching
            # the V1 ``get_conv_copy_spec`` semantics.
            self._mamba_ctx = MambaSpecDecodeGPUContext.create(
                max_num_reqs=self.max_num_reqs,
                kv_cache_config=kv_cache_config,
                num_state_types=len(copy_funcs),
                device=self.device,
                make_buffer=lambda n, dtype: CpuGpuBuffer(
                    n, dtype=dtype, device=self.device
                ),
            )
        ctx = self._mamba_ctx
        if not ctx.is_initialized:
            forward_context = self.vllm_config.compilation_config.static_forward_context
            # block_tables are batch-order slices of the persistent
            # input_block_tables (stable data_ptr), so the metadata is captured
            # once here and reused across steps.
            ctx.initialize_from_forward_context(
                kv_cache_config,
                forward_context,
                self.model.get_mamba_state_copy_func(),
                [block_tables[gid] for gid in mamba_group_ids],
            )
        return ctx

    def preprocess_state(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        """Migrate each request's mamba state across block boundaries before the
        forward (V1 align semantics, done on GPU). Runs on real batches only
        (dummy DP/profiling runs skip preprocess_state), and before
        ``prepare_attn`` gathers ``num_accepted_tokens``, so the boundary reset
        is visible to the forward kernels.
        """
        if not self._align_mode:
            return
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return
        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        ctx = self._ensure_align_ctx(kv_cache_config, mamba_group_ids, block_tables)

        # The state-advance + pre-copy kernels run every step; they fast-exit per
        # request when src_col < 0 or src_col == dst_col, so no copy happens on
        # steps that don't cross a block boundary. (Skipping the launch entirely
        # would need a V1-style async-D2H of the actual num_computed, since
        # num_computed_tokens_np is an optimistic mirror under async scheduling;
        # the launch cost is ~0.3% of TPOT, so the GPU fast-exit suffices.)
        block = 256
        grid = (triton.cdiv(num_reqs, block),)
        preprocess_mamba_align_fused_kernel[grid](
            input_batch.idx_mapping,
            self._mamba_state_idx_gpu,
            num_computed_tokens,
            input_batch.query_start_loc,
            self.num_accepted_tokens_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            num_reqs,
            BLOCK_SIZE=block,
            MAMBA_BLOCK_SIZE=mamba_spec.block_size,
        )
        # Probe phase 1: crossing detection + source-block NaN scan BEFORE the
        # precopy (the source is the previous round's checkpoint column).
        _pre = None
        if os.environ.get("VLLM_ALIGN_PROBE") and not torch.cuda.is_current_stream_capturing():
            _pre = self._align_probe(input_batch, mamba_group_ids, block_tables,
                                     kv_cache_config, phase="pre")
        ctx.run_fused_precopy(
            num_reqs,
            self._mamba_state_idx_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            input_batch.idx_mapping,
        )
        if os.environ.get("VLLM_ALIGN_PROBE") and not torch.cuda.is_current_stream_capturing():
            self._align_probe(input_batch, mamba_group_ids, block_tables,
                              kv_cache_config, phase="post", pre=_pre)
        if os.environ.get("VLLM_GDN_PROBE") and not torch.cuda.is_current_stream_capturing():
            self._gdn_probe(input_batch, mamba_group_ids, kv_cache_config, block_tables)
        if os.environ.get("VLLM_KVLINE3") and not torch.cuda.is_current_stream_capturing():
            try:
                self._kvline3_snap(
                    "pre", input_batch, block_tables, kv_cache_config,
                    mamba_group_ids, num_computed_tokens,
                )
            except Exception:
                if not getattr(self, "_kvl3_err", False):
                    self._kvl3_err = True
                    import traceback
                    traceback.print_exc()
        if os.environ.get("VLLM_GDNSTAT") and not torch.cuda.is_current_stream_capturing():
            try:
                self._gdnstat_snap(
                    input_batch, block_tables, kv_cache_config,
                    mamba_group_ids, num_computed_tokens,
                )
            except Exception:
                if not getattr(self, "_gs_err", False):
                    self._gs_err = True
                    import traceback
                    traceback.print_exc()

    def _gdnstat_snap(self, input_batch, block_tables, kv_cache_config,
                      mamba_group_ids, num_computed_tokens) -> None:
        """Env-gated (VLLM_GDNSTAT) per-round GDN checkpoint-window values.

        At preprocess (after the precopy), for every live request and ALL
        mamba layers/state tensors, record the L2 norm of each of the 14
        window slots (bt[r, col + rel]) plus round metadata (col, read_idx
        = num_accepted - 1 with post-reset semantics, query length T,
        num_computed). Every 8th round also stores a strided value slice of
        the resumed slot (read_idx) for cross-run value comparison.

        Detects: (a) the resumed slot jumping to a stale/wrong checkpoint —
        norm discontinuity at a round that persists; (b) graded value drift
        vs a same-tokens replay run (slice comparison). Scalars only:
        ~6KB/round for 48 layers x 2 states.
        """
        out = os.environ["VLLM_GDNSTAT"]
        self._gs_n = getattr(self, "_gs_n", 0)
        n = self._gs_n
        gid = mamba_group_ids[0]
        bt = block_tables[gid]
        width = bt.shape[1]
        n_req = input_batch.num_reqs
        idx_map = input_batch.idx_mapping[:n_req].cpu().tolist()
        cols = self._mamba_state_idx_gpu.cpu().tolist()
        nas = self.num_accepted_tokens_gpu.cpu().tolist()
        qs = input_batch.query_start_loc.cpu().tolist()
        ncts = num_computed_tokens.cpu().tolist()
        fc = self.vllm_config.compilation_config.static_forward_context

        cache = getattr(self, "_gs_cache", None)
        if cache is None:
            layout = []
            tensors = {}
            layer_names = kv_cache_config.kv_cache_groups[gid].layer_names
            for ln in layer_names:
                impl = fc.get(ln)
                st = getattr(impl, "kv_cache", None) if impl else None
                if not st:
                    continue
                for st_i, t in enumerate(st):
                    if not torch.is_tensor(t) or t.ndim < 1:
                        continue
                    key = f"{ln}#st{st_i}"
                    layout.append(key)
                    tensors[key] = t.reshape(t.shape[0], -1)
            cache = self._gs_cache = (layout, tensors)

        layout, tensors = cache
        do_slice = (n % 8) == 0
        rows = []
        for r in range(n_req):
            rs = idx_map[r]
            if rs < 0:
                continue
            col = cols[rs] if 0 <= rs < len(cols) else -1
            if col < 0:
                continue
            ri = max(nas[rs] - 1, 0) if 0 <= rs < len(nas) else 0
            rels = [rel for rel in range(14) if col + rel < width]
            blks = [int(bt[r, col + rel]) for rel in rels]
            keep = [
                (rel, blk) for rel, blk in zip(rels, blks)
                if 0 < blk < tensors[layout[0]].shape[0]
            ]
            if not keep:
                continue
            rec = {
                "n": n, "rs": int(rs), "col": int(col), "ri": int(ri),
                "T": int(qs[r + 1] - qs[r]) if r + 1 < len(qs) else 0,
                "nct": int(ncts[rs]) if 0 <= rs < len(ncts) else -1,
                "norms": {}, "slice": {},
            }
            for key in layout:
                tf = tensors[key]
                blks_k = [blk for _, blk in keep]
                norms = tf[blks_k].float().norm(dim=tuple(range(1, tf[blks_k].ndim)))
                rec["norms"][key] = [round(v, 4) for v in norms.cpu().tolist()]
                if do_slice and ri in [rel for rel, _ in keep]:
                    blk_ri = blks_k[[rel for rel, _ in keep].index(ri)]
                    v = tf[blk_ri].float().flatten()
                    stride = max(v.numel() // 1024, 1)
                    rec["slice"][key] = v[::stride][:1024].to(torch.float16).cpu().tolist()
            rows.append(rec)

        recs = getattr(self, "_gs_recs", None)
        if recs is None:
            recs = self._gs_recs = []
        recs.extend(rows)
        self._gs_n = n + 1
        if len(recs) >= 200:
            os.makedirs(out, exist_ok=True)
            torch.save(
                {"layout": layout, "rounds": recs},
                os.path.join(out, f"gdnstat_{os.getpid()}_{n}.pt"),
            )
            self._gs_recs = []

    def _kvline3_snap(self, phase, input_batch, block_tables, kv_cache_config,
                      mamba_group_ids, num_computed_tokens) -> None:
        """Env-gated (VLLM_KVLINE3) pre/post-forward mamba window lineage.

        "pre"  (end of preprocess_state, after the precopy): per live request,
               the 14-column checkpoint window (slot + checksum) for sampled
               layers/state tensors, plus read_idx = num_accepted-1 (post-reset
               semantics: 0 at boundary crossings) and this round's query
               length T. This is exactly what the GDN spec kernel is about to
               read (window[read_idx]) and overwrite.
        "post" (start of postprocess_state): the same window after the verify
               forward, before the post-step align copy.

        Lineage invariant (SSM copies are byte-exact): pre[N][read_idx_N]
        content == post[N-1] content at the same absolute column (col_{N-1} +
        read_idx_N), including crossings (the precopy sources bt[src_col +
        token_bias]). A violation = the kernel reads a checkpoint that the
        previous round did not leave there (stale/recycled/wrong column).
        """
        import json as _json

        out = os.environ["VLLM_KVLINE3"]
        self._kvl3_n = getattr(self, "_kvl3_n", 0)
        gid = mamba_group_ids[0]
        bt = block_tables[gid]
        width = bt.shape[1]
        n_req = input_batch.num_reqs
        idx_map = input_batch.idx_mapping[:n_req].cpu().tolist()
        cols = self._mamba_state_idx_gpu.cpu().tolist()
        nas = self.num_accepted_tokens_gpu.cpu().tolist()
        qs = input_batch.query_start_loc.cpu().tolist()
        fc = self.vllm_config.compilation_config.static_forward_context
        layer_names = kv_cache_config.kv_cache_groups[gid].layer_names
        n_ln = len(layer_names)
        win_idx = sorted({0, n_ln // 3, (2 * n_ln) // 3, n_ln - 1})
        rows = []
        for ln_i in win_idx:
            ln = layer_names[ln_i]
            impl = fc.get(ln)
            st = getattr(impl, "kv_cache", None) if impl else None
            if not st:
                continue
            for r in range(n_req):
                rs = idx_map[r]
                col = cols[rs] if 0 <= rs < len(cols) else -1
                if col < 0:
                    continue
                for st_i, t in enumerate(st):
                    if not torch.is_tensor(t) or t.ndim < 1:
                        continue
                    tf = t.reshape(t.shape[0], -1)
                    for rel in range(14):
                        c = col + rel
                        if c >= width:
                            continue
                        blk = int(bt[r, c])
                        if blk <= 0 or blk >= tf.shape[0]:
                            continue
                        rows.append({
                            "phase": phase, "n": self._kvl3_n, "rs": int(rs),
                            "layer": f"{ln}#st{st_i}", "col": int(col),
                            "rel": rel, "slot": blk,
                            "k": round(float(tf[blk].float().sum().item()), 3),
                            "ri": int(nas[rs]) - 1 if 0 <= rs < len(nas) else -1,
                            "T": int(qs[r + 1] - qs[r]) if r + 1 < len(qs) else 0,
                        })
        # Attention-KV tail blocks (pass 103 surface a): per live request,
        # checksum the 32-token KV blocks covering [nct-64, nct+T]. Within a
        # round, blocks covering query positions [nct, nct+T) MUST change
        # pre->post (the verify's int8-PTH write); blocks fully below nct
        # MUST NOT change. A query block that does NOT change = the KV write
        # missed its slot -> the next round reads stale/rejected-draft KV.
        try:
            import re as _re
            attn_set = getattr(self, "_kvl3_attn", None)
            if attn_set is None:
                from vllm.v1.kv_cache_interface import AttentionSpec as _AS
                attn_set = []
                for g, grp in enumerate(kv_cache_config.kv_cache_groups):
                    if g in mamba_group_ids or not isinstance(grp.kv_cache_spec, _AS):
                        continue
                    tgt = [
                        ln for ln in grp.layer_names
                        if (_m := _re.search(r"(\d+)", ln))
                        and int(_m.group(1)) <= 63
                    ]
                    if tgt:
                        attn_set.append((g, tgt))
                self._kvl3_attn = attn_set
            ncts = num_computed_tokens.cpu().tolist()
            self._kvl3_attn_blocks = {}
            for g, lns in attn_set:
                bt_a = block_tables[g]
                w_a = bt_a.shape[1]
                for r in range(n_req):
                    rs = idx_map[r]
                    nct = ncts[rs] if 0 <= rs < len(ncts) else -1
                    T_r = int(qs[r + 1] - qs[r]) if r + 1 < len(qs) else 0
                    if nct < 0 or T_r <= 0:
                        continue
                    # Attention KV blocks are 1728-token (slot ids are
                    # 1728-aligned; the block table width is cdiv(len,1728)).
                    blocks = []
                    for bcol in range(
                        max(0, (nct - 1) // 1728 - 1), (nct + T_r) // 1728 + 1
                    ):
                        if bcol >= w_a:
                            continue
                        bid = int(bt_a[r, bcol])
                        if bid > 0 and bid not in blocks:
                            blocks.append((bcol, bid))
                    self._kvl3_attn_blocks[r] = blocks[:3]
                    # Fixed 8-token-aligned absolute slices around the query
                    # boundary: identical token ranges across rounds, so the
                    # pre[n] -> pre[n+1] join is exact. Slice id s covers
                    # tokens [s*8, s*8+8). Slices fully below nct must never
                    # change; slices covering [nct, nct+T) must change (the
                    # verify write).
                    s_lo = max(0, (nct - 16) // 8)
                    s_hi = (nct + T_r + 7) // 8
                    for ln in lns:
                        impl = fc.get(ln)
                        kvv = getattr(impl, "kv_cache", None) if impl else None
                        if kvv is None:
                            continue
                        ts = list(kvv) if isinstance(kvv, (list, tuple)) else [kvv]
                        ts = [t for t in ts if torch.is_tensor(t)]
                        for t_i, t in enumerate(ts[:2]):
                            tok_ax = 1 if t.dim() >= 2 and t.shape[1] == 1728 else (
                                2 if t.dim() >= 3 and t.shape[2] == 1728 else 0
                            )
                            for s in range(s_lo, s_hi):
                                bcol = (s * 8) // 1728
                                if bcol >= w_a:
                                    continue
                                bid = int(bt_a[r, bcol])
                                if bid <= 0 or bid >= t.shape[0]:
                                    continue
                                lo = s * 8 - bcol * 1728
                                hi = min(1728, lo + 8)
                                if hi <= lo:
                                    continue
                                sl = (
                                    t[bid, lo:hi]
                                    if tok_ax == 1
                                    else (
                                        t[bid, :, lo:hi]
                                        if tok_ax == 2
                                        else t.reshape(t.shape[0], -1)[bid]
                                    )
                                )
                                rows.append({
                                    "phase": phase, "n": self._kvl3_n,
                                    "rs": int(rs), "layer": f"{ln}#kv{t_i}",
                                    "col": int(nct), "s": s, "bc": int(bcol),
                                    "slot": bid,
                                    "k": round(float(sl.float().sum().item()), 3),
                                    "ri": int(nas[rs]) - 1 if 0 <= rs < len(nas) else -1,
                                    "T": T_r,
                                })
        except Exception:
            if not getattr(self, "_kvl3_aerr", False):
                self._kvl3_aerr = True
                import traceback
                traceback.print_exc()
        rec = getattr(self, "_kvl3_recs", None)
        if rec is None:
            rec = self._kvl3_recs = []
        rec.extend(rows)
        if phase == "post":
            self._kvl3_n = getattr(self, "_kvl3_n", 0) + 1
        if len(rec) >= 400:
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, f"kvl3_{os.getpid()}.jsonl"), "a") as f:
                for e in rec:
                    f.write(_json.dumps(e) + "\n")
            self._kvl3_recs = []
        # stash for the post hook (postprocess_state lacks block_tables)
        self._kvl3_bt = block_tables
        self._kvl3_cfg = kv_cache_config
        self._kvl3_gids = mamba_group_ids

    @torch.inference_mode()
    def _kvl3_post(self, idx_mapping) -> None:
        """Post-forward half of VLLM_KVLINE3 (see _kvline3_snap). Uses the
        block tables stashed by the pre hook (persistent buffers, rewritten
        per step by gather_block_tables, so they still hold this step's
        batch-order rows)."""
        import json as _json

        out = os.environ["VLLM_KVLINE3"]
        bt = self._kvl3_bt[self._kvl3_gids[0]]
        width = bt.shape[1]
        n_req = idx_mapping.shape[0]
        idx_map = idx_mapping[:n_req].cpu().tolist()
        cols = self._mamba_state_idx_gpu.cpu().tolist()
        fc = self.vllm_config.compilation_config.static_forward_context
        layer_names = self._kvl3_cfg.kv_cache_groups[
            self._kvl3_gids[0]
        ].layer_names
        n_ln = len(layer_names)
        win_idx = sorted({0, n_ln // 3, (2 * n_ln) // 3, n_ln - 1})
        rows = []
        for ln_i in win_idx:
            ln = layer_names[ln_i]
            impl = fc.get(ln)
            st = getattr(impl, "kv_cache", None) if impl else None
            if not st:
                continue
            for r in range(n_req):
                rs = idx_map[r]
                if rs < 0:
                    continue
                col = cols[rs] if 0 <= rs < len(cols) else -1
                if col < 0:
                    continue
                for st_i, t in enumerate(st):
                    if not torch.is_tensor(t) or t.ndim < 1:
                        continue
                    tf = t.reshape(t.shape[0], -1)
                    for rel in range(14):
                        c = col + rel
                        if c >= width:
                            continue
                        blk = int(bt[r, c])
                        if blk <= 0 or blk >= tf.shape[0]:
                            continue
                        rows.append({
                            "phase": "post", "n": self._kvl3_n, "rs": int(rs),
                            "layer": f"{ln}#st{st_i}", "col": int(col),
                            "rel": rel, "slot": blk,
                            "k": round(float(tf[blk].float().sum().item()), 3),
                        })
        # NOTE: the attention-KV post half was removed — the stashed block
        # list proved unreliable (capture/dummy-phase overwrite + async
        # post_update phasing). The attention tests are pre-only
        # (analyze_kvline3b.py): sub-block slices at the query boundary.
        rec = getattr(self, "_kvl3_recs", None)
        if rec is None:
            rec = self._kvl3_recs = []
        rec.extend(rows)
        self._kvl3_n = getattr(self, "_kvl3_n", 0) + 1
        if len(rec) >= 400:
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, f"kvl3_{os.getpid()}.jsonl"), "a") as f:
                for e in rec:
                    f.write(_json.dumps(e) + "\n")
            self._kvl3_recs = []

    @torch.inference_mode()
    def _gdn_probe(self, input_batch, mamba_group_ids, kv_cache_config, block_tables) -> None:
        """Env-gated (VLLM_GDN_PROBE) per-step NaN scan of every live
        request's RUNNING mamba state block (the column the forward is about
        to read), per layer. Runs eagerly outside the CUDA graph, so it sees
        the replayed forward's actual state. First NaN onset per (rs, layer)
        is logged with the request's position — localizing which state type /
        layer / position produces the production-config logits NaN.
        """
        try:
            fc = self.vllm_config.compilation_config.static_forward_context
            idx_map = input_batch.idx_mapping.cpu().tolist()
            state_idx = self._mamba_state_idx_gpu.cpu().tolist()
            out = os.environ["VLLM_GDN_PROBE"]
            os.makedirs(os.path.dirname(out), exist_ok=True)
            nct = input_batch.num_computed_tokens_np
            self._probe_step = getattr(self, "_probe_step", 0) + 1
            step = self._probe_step
            # --- cross-request window sharing scan (mamba groups) ---
            # The spec checkpoint window [start..start+13] of each request must
            # hold PHYSICALLY PRIVATE blocks; a nonzero block id appearing in
            # two live requests' windows means checkpoint writes cross-
            # contaminate (prefix-shared or recycled block mapped twice).
            mamba_bs = self._mamba_spec.block_size if self._mamba_spec else 1728
            for gid in mamba_group_ids:
                bt = block_tables[gid]
                win_map: dict[int, list] = {}
                for b, rs in enumerate(idx_map[: input_batch.num_reqs]):
                    if rs < 0:
                        continue
                    col = state_idx[rs]
                    if col < 0:
                        continue
                    width = bt.shape[1]
                    lo = max(0, min(col, width - 1))
                    hi = min(width, lo + 1 + 13)
                    for c in range(lo, hi):
                        blk = int(bt[b, c])
                        if blk > 0:
                            win_map.setdefault(blk, []).append((rs, c))
                shared = {
                    blk: owners
                    for blk, owners in win_map.items()
                    if len({o[0] for o in owners}) > 1
                }
                if shared:
                    with open(out, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "kind": "SHARE",
                                    "step": step,
                                    "gid": gid,
                                    "shared": {
                                        str(k): v for k, v in list(shared.items())[:8]
                                    },
                                }
                            )
                            + "\n"
                        )
            # --- per-layer NaN scan of the running state block ---
            for gid in mamba_group_ids:
                layer_names = kv_cache_config.kv_cache_groups[gid].layer_names
                bt = block_tables[gid]
                for layer_name in layer_names:
                    attn = fc[layer_name]
                    for st_i, state in enumerate(attn.kv_cache):
                        # state: [num_blocks, ...] — scan the running block of
                        # each live request (first 4k elements of the block).
                        for b, rs in enumerate(idx_map[: input_batch.num_reqs]):
                            if rs < 0:
                                continue
                            col = state_idx[rs]
                            if col < 0:
                                continue
                            blk = int(bt[b, col])
                            if blk <= 0:
                                continue
                            row = state[blk].reshape(-1)[:4096]
                            nan = int(row.isnan().sum())
                            if nan:
                                entry = {
                                    "kind": "NAN",
                                    "step": step,
                                    "rs": rs, "layer": layer_name,
                                    "state_type": st_i, "col": col,
                                    "block": blk, "nan": nan,
                                    "absmax": float(row.abs().max())
                                        if nan < row.numel() else float("nan"),
                                    "pos": int(nct[b]) if nct is not None else None,
                                }
                                with open(out, "a") as f:
                                    f.write(json.dumps(entry) + "\n")
        except Exception as e:  # never let the probe kill the engine
            logger.warning("GDNPROBE failed: %s", e)

    def _align_probe(self, input_batch, mamba_group_ids, block_tables,
                     kv_cache_config=None, phase="post", pre=None) -> dict | None:
        """Env-gated (VLLM_ALIGN_PROBE) two-phase crossing audit.

        phase="pre": detect each request's migration (src_col != dst_col),
        scan the SOURCE block (column src_col + token_bias — the accepted-token
        checkpoint written by the previous round's forward) for NaN across the
        mamba group's state tensors, log the crossing, and return the per-key
        source-NaN map.
        phase="post": scan the DEST block after the precopy and classify:
        source clean + dest NaN => the copy introduced the NaN (read past the
        valid content); source NaN => the forward wrote NaN checkpoints.
        """
        try:
            idx_map = input_batch.idx_mapping.cpu().tolist()
            src_col = self._mamba_src_col_gpu.cpu().tolist()
            dst_col = self._mamba_state_idx_gpu.cpu().tolist()
            src_off = self._mamba_src_off_gpu.cpu().tolist()
            gid = mamba_group_ids[0]
            bt = block_tables[gid]
            out = os.environ["VLLM_ALIGN_PROBE"]
            os.makedirs(os.path.dirname(out), exist_ok=True)
            fc = self.vllm_config.compilation_config.static_forward_context
            layer_names = (
                kv_cache_config.kv_cache_groups[gid].layer_names
                if kv_cache_config is not None
                else []
            )

            def scan_block(blk: int) -> int:
                """Total NaN count over the group's first-layer state tensors."""
                if blk <= 0:
                    return -1
                total = 0
                for ln in layer_names:
                    for state in fc[ln].kv_cache:
                        total += int(state[blk].reshape(-1)[:4096].isnan().sum())
                return total

            pre_map = {} if pre is None else pre
            result = {}
            # num_accepted per req-state slot (previous round's acceptance,
            # pre-reset — the token_bias source) for the regression analysis.
            na_map = {}
            try:
                na_list = self.num_accepted_tokens_gpu.cpu().tolist()
                na_map = {rs: na_list[rs] for rs in idx_map if rs >= 0}
            except Exception:
                pass
            for b, rs in enumerate(idx_map[: input_batch.num_reqs]):
                if rs < 0:
                    continue
                sc, dc, off = src_col[rs], dst_col[rs], src_off[rs]
                if sc < 0 or sc == dc:
                    continue
                row = bt[b].cpu().tolist()
                width = len(row)
                read_col = sc + off
                read_blk = row[read_col] if read_col < width else -1
                dst_blk = row[dc] if dc < width else -1
                key = (rs, sc, dc, off)
                if phase == "pre":
                    src_nan = scan_block(read_blk)
                    result[key] = src_nan
                    self._align_probe_step = getattr(self, "_align_probe_step", 0) + 1
                    entry = {
                        "phase": "pre", "rs": rs, "src_col": sc,
                        "dst_col": dc, "token_bias": off,
                        "read_col": read_col, "read_block_id": read_blk,
                        "dst_block_id": dst_blk, "src_nan": src_nan,
                        "window_blocks": row[sc : min(sc + 14, width)],
                        # num_computed (optimistic mirror) + last-round
                        # acceptance: the regression-driver measurements.
                        "nct": int(input_batch.num_computed_tokens_np[b]),
                        "na": na_map.get(rs, -1),
                        "pid": os.getpid(),
                        "step": self._align_probe_step,
                    }
                    with open(out, "a") as f:
                        f.write(json.dumps(entry) + "\n")
                else:
                    dst_nan = scan_block(dst_blk)
                    src_nan = pre_map.get(key, -2)
                    verdict = (
                        "COPY_INTRODUCED_NAN"
                        if src_nan == 0 and dst_nan > 0
                        else ("SOURCE_NAN" if src_nan and src_nan > 0 else "clean")
                    )
                    entry = {
                        "phase": "post", "rs": rs, "src_col": sc,
                        "dst_col": dc, "token_bias": off,
                        "read_block_id": read_blk, "dst_block_id": dst_blk,
                        "src_nan": src_nan, "dst_nan": dst_nan,
                        "verdict": verdict,
                    }
                    with open(out, "a") as f:
                        f.write(json.dumps(entry) + "\n")
            return result
        except Exception as e:  # never let the probe kill the engine
            logger.warning("ALIGNPROBE failed: %s", e)
            return None

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            # Capture with worst-case max_seq_len so the graph is valid at any replay.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()

        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(
            input_batch.is_prefilling_np
        )
        # During CUDAGraph capture, num_decode_draft_tokens_cpu and num_accepted_tokens
        # are created by attn_metadata_builder.build_for_cudagraph_capture, so we only
        # compute them during actual (non-capture) forward execution.
        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[
                input_batch.idx_mapping
            ]

            # GDN uses >= 0 to select spec-decode rows, so non-decode rows
            # need the -1 sentinel rather than a raw zero draft count.
            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is not None:
                # A row is a spec-decode row only when its whole prompt is already
                # computed, i.e. exactly one non-draft (decode) token is scheduled.
                is_decode = (
                    input_batch.num_scheduled_tokens == num_draft_tokens_per_req + 1
                )
                spec_decode_mask = (num_draft_tokens_per_req > 0) & is_decode
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask, num_draft_tokens_per_req, -1
                )
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)

            # VLLM_CONVGUARD_PROBE (diagnostic): count spec rows where the OLD
            # conv-kernel guard condition (num_accepted_prev > this round's
            # query_len) would have fired — the collapse_no_cross chain break.
            # With the kernel fix (guard bound = max_query_len) these proceed
            # correctly; this counts how often the broken path was reachable.
            if os.environ.get("VLLM_CONVGUARD_PROBE") and num_draft_tokens_per_req is not None:
                na_np = self.num_accepted_tokens_gpu[input_batch.idx_mapping].cpu().numpy()
                spec_rows = num_decode_draft_tokens_np[: input_batch.num_reqs] >= 0
                if spec_rows.any():
                    na_spec = na_np[: input_batch.num_reqs][spec_rows]
                    seqlen_spec = num_decode_draft_tokens_np[: input_batch.num_reqs][spec_rows] + 1
                    hits = int((na_spec > seqlen_spec).sum())
                    self._convguard_hits = getattr(self, "_convguard_hits", 0) + hits
                    self._convguard_rows = getattr(self, "_convguard_rows", 0) + int(spec_rows.sum())
                    step = getattr(self, "_probe_step", 0)
                    if step % 500 == 0:
                        logger.warning(
                            "CONVGUARD probe: old-guard hits=%d over %d spec rows (step %d)",
                            self._convguard_hits, self._convguard_rows, step,
                        )

        mamba_attn_metadata = MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
        if self.recoverssm is not None:
            self.recoverssm.record_step(
                attn_metadata,
                attn_groups,
                for_capture=for_capture,
            )
        return attn_metadata

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        # Chunked prefill does not sample a token, so num_sampled can be 0.
        # Mamba treats num_accepted_tokens=1 as the neutral non-spec value.
        if (
            os.environ.get("VLLM_KVLINE3")
            and not torch.cuda.is_current_stream_capturing()
            and getattr(self, "_kvl3_bt", None) is not None
            and idx_mapping.shape[0] > 0
            and not isinstance(num_sampled, int)
        ):
            try:
                self._kvl3_post(idx_mapping)
            except Exception:
                if not getattr(self, "_kvl3_err", False):
                    self._kvl3_err = True
                    import traceback
                    traceback.print_exc()
        num_reqs = idx_mapping.shape[0]
        if num_reqs:
            if not isinstance(num_sampled, int):
                # idx_mapping may contain -1 sentinels (filtered rows) under PP; the
                # kernel skips them rather than scattering with a host-side gather.
                _scatter_num_accepted_kernel[(num_reqs,)](
                    idx_mapping,
                    num_sampled,
                    self.num_accepted_tokens_gpu,
                )
            else:
                # Fill with single value.
                _fill_num_accepted_kernel[(num_reqs,)](
                    idx_mapping,
                    self.num_accepted_tokens_gpu,
                    max(num_sampled, 1),
                )

        if self.recoverssm is not None:
            self.recoverssm.commit_step(
                num_sampled,
                idx_mapping,
                state_indices=(self._mamba_state_idx_gpu if self._align_mode else None),
                num_accepted_tokens=self.num_accepted_tokens_gpu,
            )

        if not num_reqs:
            return

        # Align: save the running state to the block-aligned position when
        # spec-decode acceptance leaves the sequence non-block-aligned (mirrors
        # the V1 align postprocess). num_computed_tokens already holds the
        # post-step advanced count.
        if (
            self._align_mode
            and num_computed_tokens is not None
            and self._mamba_ctx is not None
        ):
            self._mamba_ctx.run_fused_postprocess_align(
                num_reqs,
                self.num_accepted_tokens_gpu,
                self._mamba_state_idx_gpu,
                num_computed_tokens,
                idx_mapping,
            )


@triton.jit
def _scatter_num_accepted_kernel(
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_sampled_ptr,  # [num_reqs]
    num_accepted_ptr,  # [max_num_reqs]
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    num_sampled = tl.load(num_sampled_ptr + row)
    tl.store(num_accepted_ptr + req_state_idx, tl.maximum(num_sampled, 1))


@triton.jit
def _fill_num_accepted_kernel(
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_accepted_ptr,  # [max_num_reqs]
    num_sampled,
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    tl.store(num_accepted_ptr + req_state_idx, num_sampled)
