#!/usr/bin/env python3
"""PTQR retraining of the DFlash2 draft for the all-int8 stack (Phase 2).

Pure-torch reimplementation of the DFlash2 draft forward (semantics
extracted from vllm/model_executor/models/qwen3_dflash{,2}.py — see the
SERVING SEMANTICS block below), loadable strictly from the bf16 checkpoint.

Smoke test (random context states, teacher==bf16 draft, no training):
  FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python scripts/ptqr_train_draft.py --smoke
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

DRAFT_DIR = "/home/curved/models/dflash2-bf16-with-tokenizer"
CFG = dict(hidden=5120, inter=17408, heads=32, kv_heads=8, hd=128,
           layers=5, window=2048, eps=1e-6, vocab=248320,
           taps=2, group=16, ns=13, mask_token=248070, rope=1e6)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = CFG["eps"]):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        d = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x).to(d)


def rope_cos_sin(T: int, hd: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(T, device=device).float()
    f = torch.outer(t, inv)
    return f.cos().to(dtype), f.sin().to(dtype)


def apply_rope(x, cos, sin):
    # x: [B, H, T, D]; cos/sin: [T, D/2]
    c = cos[None, None, :, :].to(x.dtype)
    s = sin[None, None, :, :].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class GroupedConv(nn.Module):
    """DFlash grouped conv (dense port of qwen3_dflash2._grouped_conv)."""

    def __init__(self, hidden: int, taps: int, group: int, block: int):
        super().__init__()
        self.taps, self.group, self.block = taps, group, block
        self.ng = hidden // group
        self.base_kernel = nn.Parameter(torch.zeros(2, taps, hidden),
                                        requires_grad=False)
        self.kernel_projection = nn.Linear(hidden, 2 * taps * self.ng, bias=False)

    def _convolve(self, h, delta, side):
        # h: [T, B, H]; delta: [T, B, taps*ng]  (B=1 in training use)
        T, B, H = h.shape
        blocks = h.reshape(T, B, self.ng, self.group)          # [T,B,ng,gs]
        base = self.base_kernel[side]                          # [taps,ng,gs]
        coeff = base.view(1, 1, self.taps, self.ng, self.group) + \
            delta.reshape(T, B, self.taps, self.ng, 1)         # [T,B,taps,ng,gs]
        out = coeff[:, :, 0] * blocks
        pos = torch.arange(T, device=h.device)
        pos = pos & (self.block - 1) if self.block & (self.block - 1) == 0 \
            else pos % self.block
        for tap in range(1, self.taps):
            shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, 0, 0, tap, 0))
            out = out + coeff[:, :, tap] * shifted * (pos >= tap).view(-1, 1, 1, 1)
        return out.reshape(T, B, H)

    def prepare(self, h):
        coeff = self.kernel_projection(h).reshape(
            h.shape[0], h.shape[1], 2, self.taps, self.ng)
        return self._convolve(h, coeff[:, :, 0].flatten(-2), 0), coeff[:, :, 1]

    def finish(self, h, coeff):
        return self._convolve(h, coeff.flatten(-2), 1)


class DraftAttention(nn.Module):
    def __init__(self):
        super().__init__()
        c = CFG
        self.hd, self.heads, self.kvh = c["hd"], c["heads"], c["kv_heads"]
        self.q_proj = nn.Linear(c["hidden"], self.heads * self.hd, bias=False)
        self.k_proj = nn.Linear(c["hidden"], self.kvh * self.hd, bias=False)
        self.v_proj = nn.Linear(c["hidden"], self.kvh * self.hd, bias=False)
        self.o_proj = nn.Linear(self.heads * self.hd, c["hidden"], bias=False)
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.hd)

    def _heads(self, x, h):
        T, B, _ = x.shape
        return x.reshape(T, B, h, self.hd).permute(1, 2, 0, 3)  # [B,h,T,D]

    def forward(self, q_tok, ctx_states, cos, sin, window):
        # q_tok: [T,B,H]; ctx_states: [C,B,H] target states at this exit
        T = q_tok.shape[0]
        C = ctx_states.shape[0] if ctx_states is not None else 0
        q = self._heads(self.q_proj(q_tok), self.heads)
        kq = self._heads(self.k_proj(q_tok), self.kvh)
        vq = self._heads(self.v_proj(q_tok), self.kvh)
        q = self.q_norm(q)
        kq = self.k_norm(kq)
        # query tokens sit AFTER the context: RoPE over positions [C, C+T)
        qcos, qsin = cos[C: C + T], sin[C: C + T]
        q, kq = apply_rope(q, qcos, qsin), apply_rope(kq, qcos, qsin)
        if ctx_states is not None:
            kc = self._heads(self.k_proj(ctx_states), self.kvh)
            kc = self.k_norm(kc)  # per-head norm AFTER head split
            vc = self._heads(self.v_proj(ctx_states), self.kvh)
            kc = apply_rope(kc, cos[:C], sin[:C])
            k = torch.cat([kc, kq], dim=2)
            v = torch.cat([vc, vq], dim=2)
        else:
            k, v = kq, vq
        # causal among query tokens; full attention to context; sliding window
        q_idx = torch.arange(C, C + T, device=q.device)
        k_idx = torch.arange(C + T, device=q.device)
        allowed = (k_idx[None, :] <= q_idx[:, None])
        if window:
            allowed &= (q_idx[:, None] - k_idx[None, :]) < window
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed[None, None, :, :].to(q.dtype),
            scale=1.0 / math.sqrt(self.hd), enable_gqa=True)
        out = attn.permute(2, 0, 1, 3).reshape(T, q_tok.shape[1], -1)
        return self.o_proj(out)


class DraftMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(CFG["hidden"], CFG["inter"], bias=False)
        self.up_proj = nn.Linear(CFG["hidden"], CFG["inter"], bias=False)
        self.down_proj = nn.Linear(CFG["inter"], CFG["hidden"], bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DraftLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_conv = GroupedConv(CFG["hidden"], CFG["taps"], CFG["group"],
                                          1 + CFG["ns"])
        self.mlp_conv = GroupedConv(CFG["hidden"], CFG["taps"], CFG["group"],
                                    1 + CFG["ns"])
        self.self_attn = DraftAttention()
        self.mlp = DraftMLP()
        self.input_layernorm = RMSNorm(CFG["hidden"])
        self.post_attention_layernorm = RMSNorm(CFG["hidden"])

    def forward(self, h, res, ctx_states, cos, sin, window):
        import os as _os
        dbg = _os.environ.get("DRAFT_STAGE_DEBUG")

        def _db(tag, t):
            if dbg:
                print(f"[MY-DBGA] {id(self) % 1000} {tag} nan={int(t.isnan().sum())} "
                      f"absmax={float(t.abs().max()):.3f}", flush=True)
        if res is None:
            res = h
            h = self.input_layernorm(h)
        else:
            # vLLM fused_add_rms_norm: s = x + residual; normed = LN(s)*w;
            # returns (normed, s) — the SUM is the new residual (my earlier
            # version stored the normed value: the 100x/layer explosion)
            h = h + res
            res = h
            h = self.input_layernorm(h)
        h, coeff = self.attention_conv.prepare(h)
        _db("attn_conv_prep", h)
        h = self.self_attn(h, ctx_states, cos, sin, window)
        _db("attn_out", h)
        h = self.attention_conv.finish(h, coeff)
        _db("attn_conv_fin", h)
        h = h + res
        res = h
        h = self.post_attention_layernorm(h)
        h, coeff = self.mlp_conv.prepare(h)
        _db("mlp_conv_prep", h)
        h = self.mlp(h)
        _db("mlp_out", h)
        h = self.mlp_conv.finish(h, coeff)
        _db("mlp_conv_fin", h)
        return h, res


class DraftModel(nn.Module):
    def __init__(self, embed_weight: torch.Tensor, lm_head_weight: torch.Tensor):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(embed_weight.detach().clone(),
                                                  freeze=True)
        self.mask_embedding = nn.Parameter(
            embed_weight[CFG["mask_token"]].detach().clone())
        self.layers = nn.ModuleList(DraftLayer() for _ in range(CFG["layers"]))
        self.norm = RMSNorm(CFG["hidden"])
        self.hidden_norm = RMSNorm(CFG["hidden"])
        # aux-state encoder: concat of the 5 exit hiddens (5*5120) -> 5120
        # ("encoder.fc" in origin naming); the VOCAB head is the TARGET's
        # lm_head (shared), passed in as lm_head_weight [vocab, hidden].
        self.fc = nn.Linear(5 * CFG["hidden"], CFG["hidden"], bias=False)
        self.register_buffer("lm_head_weight", lm_head_weight.detach().clone())
        self.input_embedding_scale = 1.0

    def forward(self, tokens, ctx_states_per_layer, positions0=0):
        """tokens: [T] draft ids (anchor+mask); ctx: list of [C,B,H] per layer.

        Returns per-exit logits list: 5 x [T, vocab].
        """
        T = tokens.shape[0]
        dev = self.embed.weight.device
        total = positions0 + T + max(c.shape[0] for c in ctx_states_per_layer)
        cos, sin = rope_cos_sin(total, CFG["hd"], CFG["rope"], dev,
                                torch.float32)
        h = self.embed(tokens) * self.input_embedding_scale
        h = h * 1.0
        # mask rows use mask_embedding
        h = torch.where((tokens == CFG["mask_token"])[:, None],
                        self.mask_embedding.to(h.dtype), h)
        h = h[:, None, :]  # [T, 1, H]
        res = None
        exit_hiddens = []
        for i, layer in enumerate(self.layers):
            h, res = layer(h, res, ctx_states_per_layer[i], cos, sin,
                           CFG["window"])
            # final norm is fused_add: normed = LN(h + res) * w (per exit)
            exit_hiddens.append(self.norm(h + res)[:, 0])  # [T, H] each
        # aux head: fc over the CONCAT of all 5 exit hiddens, one shared
        # logits tensor (the exits are consumed jointly, not per-exit vocab
        # projections — matches DFlash2's single lm_head compute_candidates)
        aux = self.fc(torch.cat(exit_hiddens, dim=-1))       # [T, H]
        logits = F.linear(self.hidden_norm(aux), self.lm_head_weight)
        return logits, exit_hiddens


def load_draft(model: DraftModel, ckpt_dir: str = DRAFT_DIR):
    from safetensors.torch import load_file
    sd = load_file(f"{ckpt_dir}/model.safetensors")
    # checkpoint keys map 1:1 to module names except selector (not in model)
    sd = {k: v for k, v in sd.items() if not k.startswith("candidate_selector")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # embed/lm_head/mask are provided at construction, not from the draft ckpt
    skip = ("lm_head_weight", "embed.weight", "mask_embedding")
    missing = [m for m in missing if not m.startswith(skip)]
    assert not missing, f"missing: {missing[:6]}"
    assert not unexpected, f"unexpected: {unexpected[:6]}"
    return model


def target_embed_table(base_sd=None) -> torch.Tensor:
    """The draft shares the TARGET's (int8-fake-quantized) embedding."""
    if base_sd is None:
        base_sd = torch.load(
            "/home/curved/models/Qwen3.8-27B-bf16-ref-lm.pt",
            weights_only=True, mmap=True)["model_state_dict"]
    w = base_sd["model.embed_tokens.weight"]
    # replicate the deployed int8 embedding conversion
    import torch as _t
    out = _t.empty_like(w, dtype=_t.float32)
    R = 8192
    with _t.no_grad():
        for r0 in range(0, w.shape[0], R):
            r1 = min(r0 + R, w.shape[0])
            wf = w[r0:r1].float()
            s = (wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0).to(w.dtype)
            q = (wf / s.float()).round().clamp(-128, 127)
            out[r0:r1] = q * s.float()
    return out.to(torch.bfloat16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if not args.smoke:
        print("training loop not implemented yet; use --smoke")
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print("loading target embed + lm_head + draft...", flush=True)
    base = torch.load("/home/curved/models/Qwen3.8-27B-bf16-ref-lm.pt",
                      weights_only=True, mmap=True)["model_state_dict"]
    lm_head = base["lm_head.weight"].to(torch.bfloat16)
    model = load_draft(DraftModel(target_embed_table(base), lm_head)) \
        .to(dev).to(torch.bfloat16)
    model.eval()
    T, C = 1 + CFG["ns"], 256
    tokens = torch.full((T,), CFG["mask_token"], dtype=torch.long, device=dev)
    tokens[0] = 100
    ctx = [torch.randn(C, 1, CFG["hidden"], device=dev, dtype=torch.bfloat16)
           for _ in range(CFG["layers"])]
    with torch.no_grad():
        logits, exits = model(tokens, ctx)
    print(f"logits {tuple(logits.shape)} finite={torch.isfinite(logits).all().item()} "
          f"absmax={logits.abs().max().item():.2f}")
    for i, e in enumerate(exits):
        print(f"exit {i}: {tuple(e.shape)} finite={torch.isfinite(e).all().item()}")
    ok = torch.isfinite(logits).all() and all(torch.isfinite(e).all() for e in exits)
    print("SMOKE OK" if ok else "SMOKE FAIL")


if __name__ == "__main__":
    main()
