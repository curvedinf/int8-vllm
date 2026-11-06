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
        num_key_value_heads=4,
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

    def apply_batched(self, q, k, positions):
        # q, k: [B, S, H, hd]; positions: [B, S]
        cos = self.cos[positions].to(q.dtype)[:, :, None, :]  # [B,S,1,hd/2]
        sin = self.sin[positions].to(k.dtype)[:, :, None, :]

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
        # x: [B, S, H] (or [T, H] treated as a single sequence)
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        B, S, _ = x.shape
        if positions is None:
            positions = torch.arange(S, device=x.device)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(B, S)

        qkv = self.qkv(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(B, S, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(B, S, self.num_kv, self.head_dim))
        if self.rope is not None:
            q, k = self.rope.apply_batched(q, k, positions)
        # PTQR student: emulate the deployed int8_block_g{G} KV cache — the
        # write quantizes K/V per (token, kv_head) vector exactly as
        # reshape_and_cache_g8 does; the attention then reads dequantized
        # values. Teacher leaves this unset.
        kvg = getattr(self, "kv_fake_quant_group", None)
        if kvg is not None:
            from ptqr_train_target import kv_block_quant_fake
            k = kv_block_quant_fake(k, group=kvg)
            v = kv_block_quant_fake(v.view(B, S, self.num_kv, self.head_dim),
                                    group=kvg)

        if self.num_kv != self.num_heads:
            rep = self.num_heads // self.num_kv
            k = k.repeat_interleave(rep, dim=2)
            v = v.view(B, S, self.num_kv, self.head_dim).repeat_interleave(
                rep, dim=2)

        # [B, H, S, S] with causal + optional sliding window
        q_ = (q * self.scale).permute(0, 2, 1, 3)
        k_ = k.permute(0, 2, 1, 3)
        v_ = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q_, k_.transpose(-1, -2)).float()
        sink = self.sink.float()
        finite = torch.isfinite(sink)
        scores = scores + torch.where(
            finite, sink, torch.zeros_like(sink)).view(1, -1, 1, 1)

        if getattr(self, "_mask_cache", None) is None or \
                self._mask_cache.shape[-1] < S or self._mask_cache.device != x.device:
            big = max(S, 128)
            q_idx = torch.arange(big, device=x.device)[:, None]
            k_idx = torch.arange(big, device=x.device)[None, :]
            m = k_idx > q_idx
            if self.window > 0:
                m = m | (q_idx - k_idx >= self.window)
            self._mask_cache = ~m  # allowed
        allowed = self._mask_cache[:S, :S].to(x.device)
        dead = ~finite.view(1, -1, 1, 1)
        attn = torch.softmax(scores.masked_fill(
            (~allowed[None, None]) | dead, float("-inf")), dim=-1)
        attn = torch.nan_to_num(attn)
        out = torch.matmul(attn.to(v_.dtype), v_).permute(0, 2, 1, 3)
        out = out.reshape(B, S, self.num_heads * self.head_dim)
        out = self.o(out)
        if getattr(self, "_partial_output", False):
            out = tp_sum(out)
        return out.squeeze(0) if squeeze else out


class MlpRef(nn.Module):
    def __init__(self, cfg, inter: int):
        super().__init__()
        self.gate_up = nn.Linear(cfg.hidden_size, 2 * inter, bias=False)
        self.down = nn.Linear(inter, cfg.hidden_size, bias=False)
        self.inter = inter

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        out = self.down(F.silu(g) * u)
        if getattr(self, "_partial_output", False):
            out = tp_sum(out)
        return out


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
        # under TP sharding self.experts holds this rank's slice but the
        # router emits GLOBAL expert ids; match on the global id.
        expert_ids = getattr(self, "_expert_ids", None) or list(
            range(len(self.experts))
        )
        for e, expert in zip(expert_ids, self.experts):
            sel = (top_i == e)
            if not sel.any():
                continue
            w = (top_p * sel).sum(-1)
            rows = sel.any(-1)
            routed[rows] += expert(x[rows]) * w[rows, None].to(x.dtype)
        out = routed + self.shared(x)
        if getattr(self, "_partial_output", False):
            out = tp_sum(out)
        return out


class LayerRef(nn.Module):
    def __init__(self, cfg, layer_idx: int, rope: Rope | None, device):
        super().__init__()
        self.attn = AttentionRef(cfg, layer_idx, rope, device)
        self.norm1 = RmsNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.norm2 = RmsNorm(cfg.hidden_size, cfg.rms_norm_eps)
        dense = layer_idx in (cfg.mlp_only_layers or [])
        self.mlp = (MlpRef(cfg, cfg.intermediate_size) if dense
                    else MoeRef(cfg))

    def forward(self, x, positions, cu_len=None):
        x = x + self.attn(self.norm1(x), positions)
        x = x + self.mlp(self.norm2(x))
        return x


class FlintRef(nn.Module):
    """Teacher reference: embed -> layers -> final norm -> lm_head (fp32 logits)."""

    def __init__(self, cfg: SimpleNamespace, device="cpu"):
        super().__init__()
        self.cfg = cfg
        self._tp_world = 1
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
        # input_ids: [B, S] (or [T] -> single sequence); logits [B, S, V] fp32
        squeeze = input_ids.dim() == 1
        ids = input_ids.unsqueeze(0) if squeeze else input_ids
        B, S = ids.shape
        if positions is None:
            positions = torch.arange(S, device=ids.device)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(B, S)
        if self._tp_world > 1:
            v0, v1 = self._vocab_range
            local = (ids >= v0) & (ids < v1)
            x = torch.zeros(
                B, S, self.cfg.hidden_size,
                dtype=self.embed.weight.dtype, device=ids.device,
            )
            emb = self.embed(ids[local] - v0)
            full = torch.zeros_like(x)
            full[local] = emb.to(full.dtype)
            x = tp_sum(full)
        else:
            x = self.embed(ids)
        for layer in self.layers:
            x = layer(x, positions)
        h = self.final_norm(x)
        logits = self.lm_head(h).float()
        if self._tp_world > 1:
            logits = _tp_gather_logits(logits, dim=-1)
        return logits.squeeze(0) if squeeze else logits


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




# ---------------------------------------------------------------------------
# TP sharding (trainer-side). Each rank holds: vocab-sliced embed + lm_head,
# its slice of attention heads (and the matching KV heads), row/col slices of
# the dense/shared MLPs, and experts[rank::world]. Forward semantics per rank
# produce PARTIAL outputs wherever the unsharded op would all-reduce; the
# trainer sums partials (autograd-exact, mirroring the Qwen trainer's
# _RowParallelSum pattern) and all-gathers vocab-parallel logits.
# ---------------------------------------------------------------------------

def _shard_linear_rows_cols(lin, world, rank, cols: bool, rows_gate_up: bool = False):
    """Shard an nn.Linear OR a PTQRLinear in place.

    cols=True: keep `rank` slice of the INPUT dim (row-parallel output
    partial); cols=False: keep the rank slice of the OUTPUT dim. For fused
    gate_up linears the output dim halves (gate|up) are sliced in lockstep
    so chunk(2) stays aligned. PTQRLinear uses its own shard API so weight
    and scale shards stay consistent.
    """
    P = lin.__class__.__name__ == "PTQRLinear"
    if rows_gate_up:
        if P:
            # fused gate/up rows: PTQRLinear scales live [out, in//group];
            # slice the gate and up halves in lockstep on weight and scale.
            w, sc = lin.weight.data, lin.scale.data
            g = w.shape[0] // 2
            r0, r1 = rank * g // world, (rank + 1) * g // world
            lin.weight = torch.nn.Parameter(
                torch.cat([w[:g][r0:r1], w[g:][r0:r1]], dim=0).contiguous())
            lin.scale = torch.nn.Parameter(
                torch.cat([sc[:g][r0:r1], sc[g:][r0:r1]], dim=0).contiguous())
            return
        _shard_mlp_rows_cols_lin(lin, world, rank)
        return
    if cols:
        n_in = lin.weight.shape[1]
        c0, c1 = rank * n_in // world, (rank + 1) * n_in // world
        if P:
            lin.shard_cols((c0, c1))
        else:
            lin.weight = torch.nn.Parameter(
                lin.weight.data[:, c0:c1].contiguous())
    else:
        n_out = lin.weight.shape[0]
        r0, r1 = rank * n_out // world, (rank + 1) * n_out // world
        if P:
            lin.shard_rows([(r0, r1)])
        else:
            lin.weight = torch.nn.Parameter(
                lin.weight.data[r0:r1].contiguous())


def _shard_mlp_rows_cols_lin(lin, world, rank):
    w = lin.weight.data
    g = w.shape[0] // 2
    r0, r1 = rank * g // world, (rank + 1) * g // world
    lin.weight = torch.nn.Parameter(
        torch.cat([w[:g][r0:r1], w[g:][r0:r1]], dim=0).contiguous())


def shard_flint_tp(model: FlintRef, world: int, rank: int, group=None) -> None:
    """Shard a FlintRef in place for TP rank `rank` of `world`.

    Requires torch.distributed to be initialized when world > 1 (the model's
    forward installs the reductions through the provided `group`).
    """
    import torch.distributed as dist

    model._tp_world, model._tp_rank, model._tp_group = world, rank, group
    _TP_STATE.update(world=world, rank=rank, group=group)
    V, H = model.cfg.vocab_size, model.cfg.hidden_size
    v0, v1 = rank * V // world, (rank + 1) * V // world

    # embedding: vocab rows (id >= v1 mapped to v0 to stay in-bounds; ranks
    # contribute only their own rows to the sum)
    w = model.embed.weight.data[v0:v1].contiguous()
    model.embed = nn.Embedding(v1 - v0, H, _weight=w)
    model._vocab_range = (v0, v1)

    # lm_head: keep full rows but mask to this rank's vocab slice at forward
    model.lm_head.weight.data = model.lm_head.weight.data[v0:v1].contiguous()
    model._lm_full_rows = V

    for layer in model.layers:
        att = layer.attn
        nh, nkv = att.num_heads, att.num_kv
        assert nh % world == 0 and nkv % world == 0
        qh, kvh = nh // world, nkv // world
        h0 = rank * qh
        if att.qkv.__class__.__name__ == "PTQRLinear":
            # student path: qkv rows are fused per-head slices; PTQR scales
            # are per-output-row groups -> slice weight+scale rows together
            w, sc = att.qkv.weight.data, att.qkv.scale.data
            q, k, v = w.split([att.q_size, att.kv_size, att.kv_size], dim=0)
            sq, sk, sv = sc.split(
                [att.q_size, att.kv_size, att.kv_size], dim=0)
            att.qkv.weight = torch.nn.Parameter(torch.cat([
                q[h0 * att.head_dim:(h0 + qh) * att.head_dim],
                k[rank * kvh * att.head_dim:(rank + 1) * kvh * att.head_dim],
                v[rank * kvh * att.head_dim:(rank + 1) * kvh * att.head_dim],
            ], dim=0).contiguous())
            att.qkv.scale = torch.nn.Parameter(torch.cat([
                sq[h0 * att.head_dim:(h0 + qh) * att.head_dim],
                sk[rank * kvh * att.head_dim:(rank + 1) * kvh * att.head_dim],
                sv[rank * kvh * att.head_dim:(rank + 1) * kvh * att.head_dim],
            ], dim=0).contiguous())
        else:
            # qkv rows: [q(h0..h0+qh) | k(rank*kvh..) | v(rank*kvh..)]
            qkv_w = att.qkv.weight.data
            q, k, v = qkv_w.split(
                [att.q_size, att.kv_size, att.kv_size], dim=0)
        att.qkv.weight = nn.Parameter(torch.cat([
            q[h0 * att.head_dim:(h0 + qh) * att.head_dim],
            k[rank * kvh * att.head_dim:(rank + 1) * kvh * att.head_dim],
            v[rank * kvh * att.head_dim:(rank + 1) * kvh * att.head_dim],
        ], dim=0).contiguous())
        att.num_heads, att.num_kv = qh, kvh
        att.q_size, att.kv_size = qh * att.head_dim, kvh * att.head_dim
        att.sink = nn.Parameter(att.sink.data[h0:h0 + qh].clone())
        # o_proj: column slice over this rank's head outputs; its output is a
        # partial sum across ranks
        if att.o.__class__.__name__ == "PTQRLinear":
            att.o.shard_cols(
                (h0 * att.head_dim, (h0 + qh) * att.head_dim))
        else:
            att.o.weight = nn.Parameter(
                att.o.weight.data[
                    :, h0 * att.head_dim:(h0 + qh) * att.head_dim
                ].contiguous())
        att._partial_output = True

        mlp = layer.mlp
        if mlp.__class__.__name__ == "MoeRef":
            keep = list(range(rank, mlp.experts.__len__(), world))
            mlp.experts = nn.ModuleList([mlp.experts[e] for e in keep])
            mlp._expert_ids = keep
            mlp._partial_output = True
        # MlpRef row/col splits (dense MLP + shared expert + each kept
        # routed expert); plain nn.Linear in the teacher is sliced directly.
        if mlp.__class__.__name__ == "MlpRef":
            _shard_mlp_rows_cols(mlp, world, rank)
        else:
            # Routed experts are sharded BY EXPERT (keep whole on their
            # rank); only the shared expert intermediate-slices. Reducing
            # both schemes together double-shards and corrupts the sum.
            _shard_mlp_rows_cols(mlp.shared, world, rank)
            mlp.shared._partial_output = False
    # final norm + all layers otherwise replicated


def _shard_mlp_rows_cols(mlp, world, rank):
    inter = mlp.inter
    gate_up = mlp.gate_up.weight.data
    g = gate_up.shape[0] // 2
    r0, r1 = rank * g // world, (rank + 1) * g // world
    mlp.gate_up.weight = nn.Parameter(
        torch.cat([gate_up[:g][r0:r1], gate_up[g:][r0:r1]], dim=0).contiguous())
    mlp.inter = r1 - r0
    mlp.down.weight = nn.Parameter(
        mlp.down.weight.data[:, r0:r1].contiguous())
    mlp._partial_output = True


# ---------------------------------------------------------------------------
# TP forward reductions. shard_flint_tp sets _TP_STATE; module partials sum
# across ranks (all_reduce forward == identity backward, autograd-exact), and
# vocab-parallel logits all-gather.
# ---------------------------------------------------------------------------

_TP_STATE = {"world": 1, "rank": 0, "group": None}


class _TPSum(torch.autograd.Function):
    """y = sum_r x_r; dy/dx_r = 1 -> identity backward (grad of the summed
    tensor passes through to each rank's partial unchanged)."""

    @staticmethod
    def forward(ctx, x):
        import torch.distributed as dist

        if _TP_STATE["world"] > 1 and dist.is_initialized():
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=_TP_STATE["group"])
        return x

    @staticmethod
    def backward(ctx, g):
        return g


def tp_sum(x: torch.Tensor) -> torch.Tensor:
    return _TPSum.apply(x.contiguous())


def _tp_gather_logits(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    import torch.distributed as dist

    world = _TP_STATE["world"]
    if world == 1 or not dist.is_initialized():
        return x
    parts = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(parts, x.contiguous(), group=_TP_STATE["group"])
    return torch.cat(parts, dim=dim)


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




def load_flint_shards(model: FlintRef, ckpt_dir: str, world: int, rank: int,
                      dtype=torch.bfloat16) -> None:
    """Stream+shard a RecursiveSI/flint checkpoint into a TP-sharded FlintRef.

    One tensor materializes at a time (safe_open mmap -> slice -> copy ->
    free), so host RAM stays far below the 80 GB checkpoint. Slicing mirrors
    shard_flint_tp exactly; norms/routers/sinks replicate on every rank.
    """
    import glob

    from safetensors import safe_open

    files = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    cfg = model.cfg
    V, H = cfg.vocab_size, cfg.hidden_size
    v0, v1 = rank * V // world, (rank + 1) * V // world
    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    qh, kvh = nh // world, nkv // world
    h0 = rank * qh
    hd = cfg.head_dim
    g_inter = cfg.intermediate_size
    r0, r1 = rank * g_inter // world, (rank + 1) * g_inter // world

    def put(path, t):
        node = model
        for part in path.split("."):
            node = getattr(node, part)
        assert node.shape == t.shape, f"{path}: {node.shape} vs {t.shape}"
        node.data.copy_(t.to(node.dtype))

    def shard_gate_up(t):  # [2*inter, H] -> keep rank slice of both halves
        g = t.shape[0] // 2
        return torch.cat([t[:g][r0:r1], t[g:][r0:r1]], dim=0)

    keep = list(range(rank, cfg.num_experts, world))
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as sf:
            for name in sf.keys():
                t = sf.get_tensor(name).to(dtype)
                if name == "model.embed_tokens.weight":
                    put("embed.weight", t[v0:v1])
                elif name == "lm_head.weight":
                    put("lm_head.weight", t[v0:v1])
                elif name == "model.norm.weight":
                    put("final_norm.weight", t)
                elif ".self_attn." in name:
                    li = int(name.split(".")[2])
                    a = f"layers.{li}.attn"
                    if name.endswith("q_proj.weight"):
                        k = sf.get_tensor(
                            name.replace("q_proj", "k_proj"))
                        v = sf.get_tensor(
                            name.replace("q_proj", "v_proj"))
                        put(f"{a}.qkv.weight", torch.cat([
                            t[h0 * hd:(h0 + qh) * hd],
                            k[rank * kvh * hd:(rank + 1) * kvh * hd],
                            v[rank * kvh * hd:(rank + 1) * kvh * hd],
                        ], dim=0))
                    elif name.endswith("k_proj.weight"):
                        pass  # consumed by the q_proj branch
                    elif name.endswith("v_proj.weight"):
                        pass
                    elif name.endswith("o_proj.weight"):
                        put(f"{a}.o.weight",
                            t[:, h0 * hd:(h0 + qh) * hd])
                    elif name.endswith("q_norm.weight"):
                        put(f"{a}.q_norm.weight", t)
                    elif name.endswith("k_norm.weight"):
                        put(f"{a}.k_norm.weight", t)
                    elif name.endswith("softmax_offset"):
                        put(f"{a}.sink", t[h0:h0 + qh])
                elif ".mlp." in name or ".shared_expert." in name:
                    li = int(name.split(".")[2])
                    dense = li in (cfg.mlp_only_layers or [])
                    if name.endswith("mlp.gate.weight"):
                        put(f"layers.{li}.mlp.router.weight", t)
                        continue
                    if dense:
                        tgt = f"layers.{li}.mlp"
                        src = name.split(".mlp.")[1]
                        if src.startswith("gate_proj"):
                            u = sf.get_tensor(name.replace("gate_proj", "up_proj"))
                            put(f"{tgt}.gate_up.weight",
                                shard_gate_up(torch.cat([t, u], dim=0)))
                        elif src.startswith("up_proj"):
                            pass
                        elif src.startswith("down_proj"):
                            put(f"{tgt}.down.weight", t[:, r0:r1])
                    else:
                        tgt = f"layers.{li}.mlp.shared"
                        src = name.split(".shared_expert.")[1]
                        if src.startswith("gate_proj"):
                            u = sf.get_tensor(name.replace("gate_proj", "up_proj"))
                            put(f"{tgt}.gate_up.weight",
                                shard_gate_up(torch.cat([t, u], dim=0)))
                        elif src.startswith("up_proj"):
                            pass
                        elif src.startswith("down_proj"):
                            put(f"{tgt}.down.weight", t[:, r0:r1])
                elif ".experts." in name:
                    li = int(name.split(".")[2])
                    e = int(name.split(".experts.")[1].split(".")[0])
                    if e not in keep:
                        continue
                    local = keep.index(e)
                    src = name.split(f".experts.{e}.")[1]
                    tgt = f"layers.{li}.mlp.experts.{local}"
                    if src.startswith("gate_proj"):
                        u = sf.get_tensor(name.replace("gate_proj", "up_proj"))
                        put(f"{tgt}.gate_up.weight", torch.cat([t, u], dim=0))
                    elif src.startswith("up_proj"):
                        pass
                    elif src.startswith("down_proj"):
                        put(f"{tgt}.down.weight", t)
                elif name.endswith(("input_layernorm.weight",
                                    "post_attention_layernorm.weight")):
                    li = int(name.split(".")[2])
                    put(f"layers.{li}."
                        + ("norm1" if "input_" in name else "norm2") + ".weight",
                        t)
                del t
