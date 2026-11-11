# SPDX-License-Identifier: Apache-2.0
"""KSDraft proposal logic shared by the vLLM speculator.

The draft is a block drafter: one forward over a short token window produces
next-token logits at every position. The engine needs N autoregressive draft
tokens per request, so this proposer runs N forwards over the trailing window,
appending each sampled token (the standard standalone-AR rollout; the window
is small, so no KV cache is needed).

Serving tier: the export keeps fp32 masters; when ``int8`` is requested the
model is converted with ``ksdraft.model.to_int8`` at load, which is the same
model-wide INT8 tier the gate battery validates.
"""

from __future__ import annotations

from pathlib import Path

import torch


class KSDraftProposer:
    def __init__(self, export_dir: str | Path, device: str | torch.device = "cuda",
                 int8: bool = True, window: int | None = None):
        from ksdraft.export import load_export

        self.model = load_export(str(export_dir), device=device)
        if int8:
            from ksdraft.model import to_int8

            self.model = to_int8(self.model)
        self.model.eval().requires_grad_(False)
        self.device = torch.device(device)
        # Training windows are short (16 tokens); keep the rollout window the
        # same so proposals see the same context distribution.
        self.window = int(window or 16)

    @torch.no_grad()
    def propose(self, tokens: torch.Tensor, n: int,
                temperature: torch.Tensor | None = None) -> torch.Tensor:
        """tokens: [B, L] int64 (each row ends at that request's last token).

        Returns [B, n] int64 draft tokens. ``temperature`` is per request;
        zero means greedy (the engine's rejection sampler still handles the
        target-side acceptance).
        """
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("tokens must be [B, L] with L >= 1")
        B = tokens.shape[0]
        ctx = tokens[:, -self.window:].to(self.device).long()
        greedy = temperature is None or bool((temperature <= 0).all())
        temp = None if temperature is None else temperature.to(self.device).float()
        out = []
        for _ in range(int(n)):
            logits, _ = self.model(ctx)
            lg = logits[:, -1].float()
            if greedy:
                tok = lg.argmax(dim=-1)
            else:
                probs = torch.softmax(lg / temp.clamp_min(1e-5)[:, None], dim=-1)
                tok = torch.multinomial(probs, 1).squeeze(-1)
            out.append(tok)
            ctx = torch.cat([ctx, tok[:, None]], dim=1)[:, -self.window:]
        return torch.stack(out, dim=1)
