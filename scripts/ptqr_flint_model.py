"""Embedded Flint reference model for PTQR training (int8-vllm fork).

Port of the serving semantics in vllm/models/flint/model.py (upstream PR
#62938) into a self-contained torch module for scripts/ptqr_train_flint.py:
Flint has no transformers modeling code, so the trainer drives this exact
reference. The teacher is the frozen BF16 forward; the student is the same
tree after replace_linears_with_ptqr() (scripts/ptqr_train_target.py).

Numerics contract (matches the deployed vLLM forward):
- RMSNorm: fp32 variance, weight applied as (1 + w), same eps (1e-6).
- Per-head QK RMSNorm on head_dim (32 q heads / 4 kv heads, head_dim 128).
- RoPE (theta 1e4, neox, rotary_dim == head_dim) ONLY on sliding-window
  layers; config.no_rope_layer_indices lists the full-attention (no-rope)
  layers.
- Attention sinks: learned per-head additive bias on the scores
  (softmax(qk*scale + sink)); -inf padding heads are masked out entirely.
- Sliding window (513): query i attends keys j with i - j < 513 (and j <=
  i, causal). Full-attention layers: plain causal.
- MoE layers: sigmoid router, top-6, renormalize selected weights
  (norm_topk_prob), 256 routed experts + always-on shared expert summed
  unscaled. Dense layers (0, 1): plain SwiGLU MLP.
- Untied lm_head; logits accumulated in fp32.

Single-process reference; the trainer shards heads/experts/vocab for TP4
(the Qwen trainer's sharding pattern), this module stays rank-agnostic.
"""

from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_flint_config(path: str) -> SimpleNamespace:
    with open(os.path.join(path, "config.json")) as f:
        return SimpleNamespace(**json.load(f))


def flint_tiny_config() -> SimpleNamespace:
    """Wiring smoke-test config: same structure, toy dims, 4 layers."""
    return SimpleNamespace(
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=256,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        num_hidden_layers=4,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        router_score_function="sigmoid",
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        sliding_window=17,
        layer_types=["full_attention", "sliding_attention"] * 2,
        mlp_only_layers=[0],
        no_rope_layer_indices=[0, 3],
        vocab_size=512,
        tie_word_embeddings=False,
        max_position_embeddings=2048,
    )


class RmsNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        v = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(v + self.eps)
        return ((1.0 + self.weight.float()) * x32).to(dtype)


class Rope:
    """Neox-style full-head rotary embedding, fp32 tables."""

    def __init__(self, head_dim: int, max_pos: int, theta: float, device):
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                            device=device) / head_dim))
        t = torch.arange(max_pos, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv)  # [max_pos, head_dim/2]
        self.cos = freqs.cos()
        self.sin = freqs.sin()
        self.head_dim = head_dim

    def apply(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        # q, k: [tokens, heads, head_dim]
        cos = self.cos[positions].to(q.dtype)  # [tokens, hd/2]
        sin = self.sin[positions].to(q.dtype)
        cos = cos[:, None, :]
        sin = sin[:, None, :]

        def rot(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

        return rot(q), rot(k)


class AttentionRef(nn.Module):
    def __init__(self, cfg, layer_idx: int, rope: Rope | None, device):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5
        self.rope = rope
        self.is_sliding = cfg.layer_types[layer_idx] == "sliding_attention"
        self.window = cfg.sliding_window if self.is_sliding else -1
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv * self.head_dim
        self.qkv = nn.Linear(cfg.hidden_size, self.q_size + 2 * self.kv_size,
                             bias=cfg.attention_bias if hasattr(cfg, "attention_bias") else False)
        self.o = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = RmsNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RmsNorm(self.head_dim, cfg.rms_norm_eps)
        # Learned per-head sink; -inf padding (TP head holes) is masked out.
        self.sink = nn.Parameter(torch.zeros(self.num_heads), requires_grad=False)

    def forward(self, x, positions, cu_len=None):
        t = x.shape[0]
        qkv = self.qkv(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(t, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(t, self.num_kv, self.head_dim))
        if self.rope is not None:
            q, k = self.rope.apply(q, k, positions)
        if self.num_kv != self.num_heads:
            rep = self.num_heads // self.num_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.view(t, self.num_kv, self.head_dim).repeat_interleave(rep, dim=1)

        scores = torch.einsum("qhd,khd->hqk", q * self.scale, k).float()
        # sinks: additive per-head bias on in-window scores
        sink = self.sink.float()
        finite = torch.isfinite(sink)
        scores = scores + torch.where(finite, sink, torch.zeros_like(sink))[:, None, None]

        q_idx = torch.arange(t, device=x.device)[:, None]
        k_idx = torch.arange(t, device=x.device)[None, :]
        mask = k_idx > q_idx  # causal
        if self.window > 0:
            mask = mask | (q_idx - k_idx >= self.window)
        mask = mask[None].expand(self.num_heads, -1, -1)
        # -inf sink heads attend nothing (padding heads must not emit NaN)
        dead = ~finite.view(-1, 1, 1)
        mask = mask | dead
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)  # fully-masked dead heads -> 0
        out = torch.einsum("hqk,khd->qhd", attn, v.to(attn.dtype))
        return self.o(out.reshape(t, self.num_heads * self.head_dim))


class MlpRef(nn.Module):
    def __init__(self, cfg, inter: int):
        super().__init__()
        self.gate_up = nn.Linear(cfg.hidden_size, 2 * inter, bias=False)
        self.down = nn.Linear(inter, cfg.hidden_size, bias=False)
        self.inter = inter

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class MoeRef(nn.Module):
    """Sigmoid router, top-k renormalized, routed experts + shared expert."""

    def __init__(self, cfg):
        super().__init__()
        self.top_k = cfg.num_experts_per_tok
        self.router = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MlpRef(cfg, cfg.moe_intermediate_size) for _ in range(cfg.num_experts)]
        )
        self.shared = MlpRef(cfg, cfg.shared_expert_intermediate_size)

    def forward(self, x):
        t = x.shape[0]
        logits = self.router(x).float()
        probs = torch.sigmoid(logits)
        top_p, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_p = top_p / top_p.sum(-1, keepdim=True)
        routed = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            sel = (top_i == e)
            if not sel.any():
                continue
            w = (top_p * sel).sum(-1)
            rows = sel.any(-1)
            routed[rows] += expert(x[rows]) * w[rows, None].to(x.dtype)
        return routed + self.shared(x)


class LayerRef(nn.Module):
    def __init__(self, cfg, layer_idx: int, rope: Rope | None, device):
        super().__init__()
        self.attn = AttentionRef(cfg, layer_idx, rope, device)
        self.norm1 = RmsNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.norm2 = RmsNorm(cfg.hidden_size, cfg.rms_norm_eps)
        dense = layer_idx in (cfg.mlp_only_layers or [])
        self.mlp = (MlpRef(cfg, cfg.intermediate_size) if dense
                    else MoeRef(cfg))

    def forward(self, x, positions):
        x = x + self.attn(self.norm1(x), positions)
        x = x + self.mlp(self.norm2(x))
        return x


class FlintRef(nn.Module):
    """Teacher reference: embed -> layers -> final norm -> lm_head (fp32 logits)."""

    def __init__(self, cfg: SimpleNamespace, device="cpu"):
        super().__init__()
        self.cfg = cfg
        max_pos = cfg.max_position_embeddings
        self.rope = Rope(cfg.head_dim, max_pos, cfg.rope_theta, device) \
            if cfg.rope_theta else None
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [
                LayerRef(cfg, i,
                         self.rope if i not in (cfg.no_rope_layer_indices or []) else None,
                         device)
                for i in range(cfg.num_hidden_layers)
            ]
        )
        self.final_norm = RmsNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor | None = None):
        if positions is None:
            positions = torch.arange(input_ids.shape[0], device=input_ids.device)
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, positions)
        h = self.final_norm(x)
        return self.lm_head(h).float()


_HF_MAP = (  # checkpoint name -> attribute path in FlintRef
    ("model.embed_tokens.weight", "embed.weight"),
    ("lm_head.weight", "lm_head.weight"),
    ("model.norm.weight", "final_norm.weight"),
)


def load_flint_state(model: FlintRef, ckpt_dir: str, strict: bool = True) -> None:
    """Load a RecursiveSI/flint checkpoint into FlintRef.

    Mapping: model.layers.N.self_attn.{q,k,v,o}_proj.weight -> qkv/o;
    softmax_offset -> attn.sink; mlp.{gate,up}_proj -> gate_up;
    mlp.down_proj -> down; shared_expert likewise; mlp.gate.weight ->
    mlp.router.weight; mlp.experts.E.{gate,up,down}_proj.weight -> experts.E.
    """
    import glob

    import safetensors.torch as st

    tensors: dict[str, torch.Tensor] = {}
    for f in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
        tensors.update(st.load_file(f))

    def set_param(path: str, t: torch.Tensor):
        node = model
        for part in path.split("."):
            node = getattr(node, part)
        assert node.shape == t.shape, f"{path}: {node.shape} vs {t.shape}"
        node.data.copy_(t.to(node.dtype))

    seen = set()
    for hf, ref in _HF_MAP:
        if hf in tensors:
            set_param(ref, tensors[hf])
            seen.add(hf)

    cfg = model.cfg
    for n in range(cfg.num_hidden_layers):
        p = f"model.layers.{n}"
        set_param(f"layers.{n}.norm1.weight", tensors[f"{p}.input_layernorm.weight"])
        set_param(f"layers.{n}.norm2.weight",
                  tensors[f"{p}.post_attention_layernorm.weight"])
        sa = f"layers.{n}.attn"
        q = tensors[f"{p}.self_attn.q_proj.weight"]
        k = tensors[f"{p}.self_attn.k_proj.weight"]
        v = tensors[f"{p}.self_attn.v_proj.weight"]
        set_param(f"{sa}.qkv.weight", torch.cat([q, k, v], dim=0))
        set_param(f"{sa}.o.weight", tensors[f"{p}.self_attn.o_proj.weight"])
        set_param(f"{sa}.q_norm.weight", tensors[f"{p}.self_attn.q_norm.weight"])
        set_param(f"{sa}.k_norm.weight", tensors[f"{p}.self_attn.k_norm.weight"])
        if f"{p}.self_attn.softmax_offset" in tensors:
            set_param(f"{sa}.sink", tensors[f"{p}.self_attn.softmax_offset"])
        mlp = f"layers.{n}.mlp"
        if n in (cfg.mlp_only_layers or []):
            set_param(f"{mlp}.gate_up.weight", torch.cat([
                tensors[f"{p}.mlp.gate_proj.weight"],
                tensors[f"{p}.mlp.up_proj.weight"]], dim=0))
            set_param(f"{mlp}.down.weight", tensors[f"{p}.mlp.down_proj.weight"])
        else:
            set_param(f"{mlp}.router.weight", tensors[f"{p}.mlp.gate.weight"])
            set_param(f"{mlp}.shared.gate_up.weight", torch.cat([
                tensors[f"{p}.shared_expert.gate_proj.weight"],
                tensors[f"{p}.shared_expert.up_proj.weight"]], dim=0))
            set_param(f"{mlp}.shared.down.weight",
                      tensors[f"{p}.shared_expert.down_proj.weight"])
            for e in range(cfg.num_experts):
                set_param(f"{mlp}.experts.{e}.gate_up.weight", torch.cat([
                    tensors[f"{p}.experts.{e}.gate_proj.weight"],
                    tensors[f"{p}.experts.{e}.up_proj.weight"]], dim=0))
                set_param(f"{mlp}.experts.{e}.down.weight",
                          tensors[f"{p}.experts.{e}.down_proj.weight"])
    if strict:
        missing = [k for k in ("model.embed_tokens.weight", "lm_head.weight")
                   if k not in tensors]
        if missing:
            raise KeyError(f"checkpoint missing {missing}")


if __name__ == "__main__":
    # CPU wiring smoke: tiny config, random weights, shape + finiteness.
    torch.manual_seed(0)
    cfg = flint_tiny_config()
    cfg.attention_bias = False
    m = FlintRef(cfg)
    ids = torch.randint(0, cfg.vocab_size, (37,))
    out = m(ids)
    assert out.shape == (37, cfg.vocab_size) and torch.isfinite(out).all()
    out2 = m(ids)
    assert torch.equal(out, out2), "forward not deterministic"
    print("FlintRef tiny smoke OK:", tuple(out.shape))
