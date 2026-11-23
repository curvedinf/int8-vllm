# SPDX-License-Identifier: Apache-2.0
"""Deep-context signal miner: worker-side mixin (worker_extension_cls).

The hooks must live inside each TP worker, where the target model runs. Per
prefill chunk this streams, per rank:

- the tapped layers' final-pass K/V as written into the int8_block_g128
  cache — post short-conv, K clip and V RMSNorm (codes int8 + fp16 group
  scales) — head shards, concatenated offline;
- per-rank top-32 logits from the LM head (the prompt-logprobs path computes
  logits for every prompt row in 1024-row chunks) — merged offline into the
  global top-32 by logit value, probs = softmax over the merged candidates.

No collectives run inside hooks: each rank writes its shard, and the joiner
(scripts/mine_ksdraft_deep.py) assembles the v2 store.

The LM head sees (num_tokens, hidden) chunks; chunks of one row are decode
steps or short prompts and are skipped — pack mining uses 1M-token prompts.
"""

from __future__ import annotations

from pathlib import Path

import torch

# (weight layer, loop pass): prelude tap, two final-pass loop taps, coda tap.
TAPS: tuple[tuple[int, int], ...] = ((4, 0), (10, 2), (16, 2), (22, 0))
TOPK = 32


def gather_cache_kv(cache_kv: torch.Tensor, slots: torch.Tensor,
                    head_dim: int) -> tuple[torch.Tensor, ...]:
    """Read cache-resident K/V rows for ``slots`` (padding rows zeroed).

    ``cache_kv`` is the worker's int8_block_g128 cache:
    (num_blocks, block_size, heads, 2 * (head_dim + 2)) — per head, the
    head_dim int8 codes followed by the fp16 group scale. Returns
    (kc, ks, vc, vs): codes (T, heads, head_dim) int8 and scales (T, heads)
    fp16, i.e. the exact bytes the deployed drafter reads.
    """
    key, value = cache_kv.transpose(1, 2).chunk(2, dim=-1)
    block_size = key.shape[2]
    valid = slots >= 0
    s = slots.clamp(min=0).long()
    blk = torch.div(s, block_size, rounding_mode="floor")
    pos = s - blk * block_size
    k = key[blk, :, pos].contiguous()
    v = value[blk, :, pos].contiguous()
    kc, vc = k[..., :head_dim], v[..., :head_dim]
    ks = k[..., head_dim:head_dim + 2].contiguous().view(
        torch.float16).squeeze(-1)
    vs = v[..., head_dim:head_dim + 2].contiguous().view(
        torch.float16).squeeze(-1)
    if not bool(valid.all()):
        kc, vc, ks, vs = (t.clone() for t in (kc, vc, ks, vs))
        kc[~valid] = 0
        vc[~valid] = 0
        ks[~valid] = 0
        vs[~valid] = 0
    return kc, ks, vc, vs


class KSDraftMinerMixin:
    def ksdraft_miner_setup(self, out_dir: str, taps=None) -> dict:
        from vllm.distributed.parallel_state import get_tp_group

        self._kdm = {
            "taps": [tuple(t) for t in (taps or TAPS)],
            "dir": Path(out_dir),
            "files": {},
            "rank": get_tp_group().rank_in_group,
            "world": get_tp_group().world_size,
        }
        self._kdm["dir"].mkdir(parents=True, exist_ok=True)
        model = self.model_runner.model
        layers = model.model.layers
        for li, pi in self._kdm["taps"]:
            attn = layers[li].attn
            orig = attn._attention

            def wrap(*args, _orig=orig, _li=li, _pi=pi, _attn=attn, **kw):
                # (positions, q, r, gate, sinks, k, v, kd, vd, tau, out, pass_idx)
                pass_idx = args[11] if len(args) > 11 else kw.get("pass_idx")
                out = _orig(*args, **kw)
                if pass_idx == _pi:
                    # After the prologue: the cache holds the short-convolved,
                    # K-clipped, V-normalized int8 K/V the drafter reads.
                    self._kdm_capture_kv(_attn, _li, _pi, pass_idx)
                return out

            attn._attention = wrap

        # The logits processor calls lm_head.quant_method.apply directly (it
        # does not go through the module __call__), so wrap the quant method:
        # it returns this rank's raw vocab-shard logits.
        import types

        head = model.lm_head
        qm = head.quant_method
        orig_apply = qm.apply

        def apply_wrap(_mod, *a, **kw):
            out = orig_apply(_mod, *a, **kw)
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(tensor):
                self._kdm_capture_logits(tensor)
            return out

        qm.apply = types.MethodType(lambda _self, mod, *a, **kw:
                                    apply_wrap(mod, *a, **kw), qm)
        return {"rank": self._kdm["rank"], "world": self._kdm["world"],
                "taps": self._kdm["taps"]}

    # ----- streaming captures -----

    def _kdm_open(self, key: str, stems: tuple[str, ...]) -> dict:
        ent = self._kdm["files"].get(key)
        if ent is None:
            base = self._kdm["dir"] / (
                f"tap{key[0]}_{key[1]}_rank{self._kdm['rank']}"
                if isinstance(key, tuple) else f"{key}_rank{self._kdm['rank']}")
            ent = {s: open(str(base) + f"_{s}.bin", "wb") for s in stems}
            self._kdm["files"][key] = ent
        return ent

    def _kdm_capture_kv(self, attn, li: int, pi: int, pass_idx: int) -> None:
        """Stream the cache-resident K/V the deployed drafter will read."""
        from vllm.forward_context import get_forward_context

        cache = attn.caches[pass_idx]
        md = get_forward_context().attn_metadata[cache.prefix]
        kc, ks, vc, vs = gather_cache_kv(
            cache.kv_cache, md.slot_mapping, attn.head_dim)
        ent = self._kdm_open((li, pi), ("kc", "ks", "vc", "vs"))
        for stem, t, dt in (("kc", kc, torch.int8), ("ks", ks, torch.float16),
                            ("vc", vc, torch.int8), ("vs", vs, torch.float16)):
            host = t.to("cpu", non_blocking=False).contiguous().to(dt)
            ent[stem].write(host.numpy().tobytes())

    def _kdm_capture_logits(self, logits: torch.Tensor) -> None:
        if logits.shape[0] <= 1 or getattr(self, "_kdm", None) is None:
            return  # decode rows / short prompts are not pack rows
        lg = logits.detach().float()
        vals, ids = torch.topk(lg, TOPK, dim=-1)
        lse = torch.logsumexp(lg, dim=-1, keepdim=True).squeeze(-1)
        ent = self._kdm_open("topk", ("ids", "vals", "lse"))
        ent["ids"].write(ids.to(torch.int32).cpu().numpy().tobytes())
        ent["vals"].write(vals.cpu().numpy().tobytes())
        ent["lse"].write(lse.cpu().numpy().tobytes())

    def ksdraft_miner_finish(self) -> dict:
        out = {}
        for key, ent in self._kdm["files"].items():
            for f in ent.values():
                f.flush()
                f.close()
            out[str(key)] = {s: str(ent[s].name) for s in ent}
        self._kdm = None
        return {"rank": None, "files": out}
