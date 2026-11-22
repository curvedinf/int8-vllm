# SPDX-License-Identifier: Apache-2.0
"""Deep-context signal miner: worker-side mixin (worker_extension_cls).

The hooks must live inside each TP worker, where the target model runs. Per
prefill chunk this streams, per rank:

- the tapped layers' final-pass K/V, rounded onto the int8_block_g128 grid
  (codes int8 + fp16 group scales) — head shards, concatenated offline;
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


class KSDraftMinerMixin:
    def ksdraft_miner_setup(self, out_dir: str, taps=None) -> dict:
        from ksdraft.teacher import blocks_int8_kv  # noqa: F401 (available)

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

            def wrap(*args, _orig=orig, _li=li, _pi=pi, **kw):
                # (positions, q, r, gate, sinks, k, v, kd, vd, tau, out, pass_idx)
                pass_idx = args[11] if len(args) > 11 else kw.get("pass_idx")
                if pass_idx == _pi:
                    self._kdm_capture_kv(_li, _pi, args[5], args[6])
                return _orig(*args, **kw)

            attn._attention = wrap

        def head_hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(tensor):
                self._kdm_capture_logits(tensor)

        model.lm_head.register_forward_hook(head_hook)
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

    def _kdm_capture_kv(self, li: int, pi: int, k: torch.Tensor,
                        v: torch.Tensor) -> None:
        from ksdraft.teacher import blocks_int8_kv

        kc, ks = blocks_int8_kv(k.detach())
        vc, vs = blocks_int8_kv(v.detach())
        ent = self._kdm_open((li, pi), ("kc", "ks", "vc", "vs"))
        for stem, t, dt in (("kc", kc, torch.int8), ("ks", ks, torch.float16),
                            ("vc", vc, torch.int8), ("vs", vs, torch.float16)):
            host = t.to("cpu", non_blocking=False).contiguous()
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
