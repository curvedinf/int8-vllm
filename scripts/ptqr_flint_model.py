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
        # q, k: [B, S, H, hd]; positions: [B, S]. The fp32 tables ride the
        # module .to(device); index on the tables' own device, then move.
        pos = positions.to(self.cos.device)
        cos = self.cos[pos].to(device=q.device, dtype=q.dtype)[:, :, None, :]
        sin = self.sin[pos].to(device=k.device, dtype=k.dtype)[:, :, None, :]

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
            pos = positions.to(q.device)
            q, k = self.rope.apply_batched(q, k, pos)
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

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.rope is not None:
            dev = self.embed.weight.device
            self.rope.cos = self.rope.cos.to(dev)
            self.rope.sin = self.rope.sin.to(dev)
        return self

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
        node.data.copy_(t.to(device=node.device, dtype=node.dtype))

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

    # group lives ONLY in _TP_STATE: a module-level ProcessGroup attr would
    # break deepcopy(teacher) when building the student
    model._tp_world, model._tp_rank = world, rank
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
        # Dense MLP / shared expert: slice gate_up rows + down cols through
        # the PTQR-aware helper (nn.Linear and PTQRLinear both supported;
        # PTQR scales stay in lockstep with the weight shards).
        # Routed experts are sharded BY EXPERT (whole) - double-slicing
        # corrupts the sum (measured during the TP equivalence debug).
        if mlp.__class__.__name__ == "MlpRef":
            _shard_linear_rows_cols(mlp.gate_up, world, rank,
                                    cols=False, rows_gate_up=True)
            _shard_linear_rows_cols(mlp.down, world, rank, cols=True)
            mlp._partial_output = True
        else:
            _shard_linear_rows_cols(mlp.shared.gate_up, world, rank,
                                    cols=False, rows_gate_up=True)
            _shard_linear_rows_cols(mlp.shared.down, world, rank, cols=True)
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


class _TPGather(torch.autograd.Function):
    """all_gather with autograd: backward scatters each rank's grad slice
    back to its local shard (grad of gather w.r.t. the local part is the
    selection matrix)."""

    @staticmethod
    def forward(ctx, x, dim, world, group):
        import torch.distributed as dist

        ctx.dim, ctx.world, ctx.group = dim, world, group
        parts = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(parts, x.contiguous(), group=group)
        return torch.cat(parts, dim=dim)

    @staticmethod
    def backward(ctx, g):
        import torch.distributed as dist

        world, group = ctx.world, ctx.group
        rank = dist.get_rank(group=group) if group is not None else dist.get_rank()
        n = g.shape[ctx.dim] // world
        return (g.narrow(ctx.dim, rank * n, n).contiguous(), None, None, None)


def _tp_gather_logits(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    import torch.distributed as dist

    world = _TP_STATE["world"]
    if world == 1 or not dist.is_initialized():
        return x
    return _TPGather.apply(x, dim, world, _TP_STATE["group"])


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
    # NOTE(PTQR memory math): a bf16 TP4 teacher shard is ~20 GB/rank; four
    # CPU-resident shards OOM the 61 GB host and teacher+student cannot share
    # a 32 GB MI100. Real runs need the layer-streamed teacher (TODO); this
    # loader is for reduced-expert smoke configs and CPU-equivalence checks.
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
    sh_inter = cfg.shared_expert_intermediate_size
    sr0, sr1 = rank * sh_inter // world, (rank + 1) * sh_inter // world

    def slice_gu(t, a0, a1):  # fused gate|up rows in lockstep
        g = t.shape[0] // 2
        return torch.cat([t[:g][a0:a1], t[g:][a0:a1]], dim=0)

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
                elif ".mlp." in name or ".shared_expert." in name:
                    li = int(name.split(".")[2])
                    dense = li in (cfg.mlp_only_layers or [])
                    if name.endswith("mlp.gate.weight"):
                        n_e = cfg.num_experts
                        put(f"layers.{li}.mlp.router.weight",
                            t[:n_e] if n_e < t.shape[0] else t)
                        continue
                    if dense:
                        tgt = f"layers.{li}.mlp"
                        src = name.split(".mlp.")[1]
                        if src.startswith("gate_proj"):
                            u = sf.get_tensor(name.replace("gate_proj", "up_proj"))
                            put(f"{tgt}.gate_up.weight",
                                slice_gu(torch.cat([t, u], dim=0), r0, r1))
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
                                slice_gu(torch.cat([t, u], dim=0), sr0, sr1))
                        elif src.startswith("up_proj"):
                            pass
                        elif src.startswith("down_proj"):
                            put(f"{tgt}.down.weight", t[:, sr0:sr1])
                elif name.endswith(("input_layernorm.weight",
                                    "post_attention_layernorm.weight")):
                    li = int(name.split(".")[2])
                    put(f"layers.{li}."
                        + ("norm1" if "input_" in name else "norm2") + ".weight",
                        t)
                del t


class StreamedFlintTeacher(nn.Module):
    """Layer-streamed TP teacher: weights never resident (host or device).

    The 39.8B bf16 teacher is 80 GB - 4 TP shards (20 GB each) OOM the
    61 GB host and cannot share a 32 GB MI100 with the student (whose bf16
    masters alone are ~20 GB/rank). This teacher keeps ONE layer's rank
    shard staged on the GPU at a time (~0.5 GB), pulled straight from the
    mmap'd safetensors; the rest of the model never materializes.

    Numerically identical to the resident sharded FlintRef: each layer is
    the same modules with the same shard_flint_tp slicing, and partials
    reduce through tp_sum like the resident path.
    """

    def __init__(self, cfg, ckpt_dir: str, world: int, rank: int,
                 device="cuda", dtype=torch.bfloat16):
        super().__init__()
        import glob
        import json
        import os

        from safetensors import safe_open

        import sys as _sys
        with open(f"/proc/{os.getpid()}/status") as _f:
            _r = [l.split()[1] for l in _f if l.startswith("VmRSS")][0]
        print(f"[SFT r{rank}] init-entry rss={int(_r)//1048576} GB",
              file=_sys.stderr, flush=True)
        self.cfg, self.world, self.rank, self.device = cfg, world, rank, device
        self.dtype = dtype
        shard = os.path.join(ckpt_dir, f"rank{rank}.safetensors")
        if os.path.exists(shard):  # presplit dir: single-file weight map
            from safetensors import safe_open as _so

            with _so(shard, framework="pt", device="cpu") as _sf:
                names = list(_sf.keys())
            self.weight_map = {n: f"rank{rank}.safetensors" for n in names}
        else:
            idx_path = os.path.join(ckpt_dir,
                                    "model.safetensors.index.json")
            self.weight_map = json.load(open(idx_path))["weight_map"]
        self.ckpt_dir = ckpt_dir
        self._handles: dict[str, object] = {}
        with open(f"/proc/{os.getpid()}/status") as _f:
            _r = [l.split()[1] for l in _f if l.startswith("VmRSS")][0]
        print(f"[SFT r{rank}] init-weightmap-done rss={int(_r)//1048576} GB",
              file=_sys.stderr, flush=True)

        # rank sharding plan (mirrors shard_flint_tp)
        V = cfg.vocab_size
        self.v0, self.v1 = rank * V // world, (rank + 1) * V // world
        nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.qh, self.kvh = nh // world, nkv // world
        self.h0 = rank * self.qh
        self.hd = cfg.head_dim
        self.keep = list(range(rank, cfg.num_experts, world))
        g_inter = cfg.intermediate_size
        self.r0, self.r1 = rank * g_inter // world, (rank + 1) * g_inter // world
        sh_inter = cfg.shared_expert_intermediate_size
        self.sr0, self.sr1 = rank * sh_inter // world, (rank + 1) * sh_inter // world

        # rope tables (replicated; small)
        rope = Rope(cfg.head_dim, cfg.max_position_embeddings,
                    cfg.rope_theta, device)
        self.cos, self.sin = rope.cos, rope.sin
        self._norm_eps = cfg.rms_norm_eps
        self._window = cfg.sliding_window
        self._layer_types = cfg.layer_types
        self._no_rope = set(cfg.no_rope_layer_indices or [])
        self._mlp_only = set(cfg.mlp_only_layers or [])
        self._top_k = cfg.num_experts_per_tok
        self._V = V

    def _t(self, name: str, sl=None) -> torch.Tensor:
        f = self.weight_map[name]
        h = self._handles.get(f)
        if h is None:
            from safetensors import safe_open

            h = safe_open(os.path.join(self.ckpt_dir, f),
                          framework="pt", device="cpu")
            self._handles[f] = h
        t = h.get_tensor(name).to(self.dtype)
        return t[sl] if sl is not None else t

    def _rms(self, x, w):
        x32 = x.float()
        v = x32.pow(2).mean(-1, keepdim=True)
        return ((1.0 + w.float()) * (x32 * torch.rsqrt(v + self._norm_eps))
                ).to(x.dtype)

    @torch.no_grad()
    def forward(self, input_ids, positions=None):
        B, S = input_ids.shape
        dev = self.device
        if positions is None:
            positions = torch.arange(S, device=dev)
        pos = positions.unsqueeze(0).expand(B, S) if positions.dim() == 1 \
            else positions

        # embed (vocab rows; partial sum)
        local = (input_ids >= self.v0) & (input_ids < self.v1)
        embw = self._t("model.embed_tokens.weight",
                       slice(self.v0, self.v1)).to(dev)
        x = torch.zeros(B, S, self.cfg.hidden_size, dtype=self.dtype, device=dev)
        rows = local
        if rows.any():
            x[rows] = torch.nn.functional.embedding(
                input_ids[rows] - self.v0, embw)
        x = tp_sum(x)

        for li in range(self.cfg.num_hidden_layers):
            x = self._layer(li, x, pos)

        h = self._rms(x, self._t("model.norm.weight").to(dev))
        lhw = self._t("lm_head.weight", slice(self.v0, self.v1)).to(dev)
        logits = torch.nn.functional.linear(h, lhw).float()
        return _tp_gather_logits(logits, dim=-1)

    def _layer(self, li, x, pos):
        dev = self.device
        p = f"model.layers.{li}"
        # --- attention ---
        a = f"{p}.self_attn"
        qw = self._t(f"{a}.q_proj.weight")
        kw = self._t(f"{a}.k_proj.weight")
        vw = self._t(f"{a}.v_proj.weight")
        qkv = torch.cat([qw, kw, vw], dim=0).to(dev)
        hd, qh, kvh = self.hd, self.qh, self.kvh
        h0 = self.h0
        qs, ks, vs = qkv.shape[0] // (qh + 2 * kvh) // hd, qh * hd, kvh * hd
        q, k, v = qkv.split([qh * hd, kvh * hd, kvh * hd], dim=0)
        ow = self._t(f"{a}.o_proj.weight").to(dev)
        sink = self._t(f"{a}.softmax_offset").to(dev)  # [all heads]
        sink = sink[h0:h0 + qh]

        qn = self._t(f"{a}.q_norm.weight").to(dev)
        kn = self._t(f"{a}.k_norm.weight").to(dev)

        t = x.shape[0] * x.shape[1]
        x2 = x.reshape(t, -1)
        qkv_o = torch.nn.functional.linear(x2, qkv)
        q, k, v = qkv_o.split([qh * hd, kvh * hd, kvh * hd], dim=-1)
        q = self._rms(q.view(t, qh, hd), qn)
        k = self._rms(k.view(t, kvh, hd), kn)
        if li not in self._no_rope:
            cos = self.cos[pos].to(dev, q.dtype)[:, :, None, :].reshape(-1, 1, hd // 2)
            sin = self.sin[pos].to(dev, q.dtype)[:, :, None, :].reshape(-1, 1, hd // 2)
            def rot(z):
                z1, z2 = z[..., :hd // 2], z[..., hd // 2:]
                return torch.cat((z1 * cos - z2 * sin, z1 * sin + z2 * cos), -1)
            q, k = rot(q), rot(k)
        rep = qh // kvh
        k = k.repeat_interleave(rep, dim=1)
        v = v.view(t, kvh, hd).repeat_interleave(rep, dim=1)
        scores = torch.einsum("thd,thd->hqk",
                              q * (hd ** -0.5), k).float().view(t, qh, t)
        scores = scores + torch.where(torch.isfinite(sink), sink,
                                      torch.zeros_like(sink)).view(1, -1, 1, 1)
        qi = torch.arange(t, device=dev)[:, None]
        ki = torch.arange(t, device=dev)[None, :]
        mask = ki > qi
        if self._layer_types[li] == "sliding_attention":
            mask = mask | (qi - ki >= self._window)
        mask = mask[None].expand(qh, -1, -1)
        dead = ~torch.isfinite(sink).view(-1, 1, 1)
        attn = torch.softmax(
            scores.masked_fill(mask | dead, float("-inf")), dim=-1)
        attn = torch.nan_to_num(attn)
        out = torch.einsum("hqk,khd->qhd", attn, v).reshape(t, qh * hd)
        attn_out = torch.nn.functional.linear(out, ow)  # partial
        x = x + tp_sum(attn_out).view(x.shape)

        # --- mlp ---
        if li in self._mlp_only:
            x = x + self._mlp_dense(f"{p}.mlp", x, self.r0, self.r1)
        else:
            # router (replicated)
            rw = self._t(f"{p}.mlp.gate.weight").to(dev)
            logits_r = torch.nn.functional.linear(x.reshape(t, -1), rw).float()
            probs = torch.sigmoid(logits_r.reshape(x.shape[0], x.shape[1], -1)
                                  .reshape(-1, rw.shape[0]))
            top_p, top_i = torch.topk(probs, self._top_k, dim=-1)
            top_p = top_p / top_p.sum(-1, keepdim=True)
            routed = torch.zeros_like(x)
            for gid in self.keep:
                sel = (top_i == gid)
                if not sel.any():
                    continue
                w = (top_p * sel).sum(-1)
                rowsm = sel.any(-1)
                ex = self._expert(f"{p}.mlp.experts.{gid}",
                                  x.reshape(t, -1)[rowsm])
                routed.reshape(t, -1)[rowsm] += ex * w[rowsm, None].to(x.dtype)
            shared = self._mlp_dense(f"{p}.mlp.shared_expert", x,
                                     self.sr0, self.sr1)
            x = x + tp_sum(routed + shared)
        return x

    def _mlp_dense(self, prefix, x, a0, a1):
        dev = self.device
        t = x.shape[0] * x.shape[1]
        g = self._t(f"{prefix}.gate_proj.weight", slice(a0, a1)).to(dev)
        u = self._t(f"{prefix}.up_proj.weight", slice(a0, a1)).to(dev)
        d = self._t(f"{prefix}.down_proj.weight",
                    (slice(None), slice(a0, a1))).to(dev)
        h = torch.nn.functional.linear(x.reshape(t, -1), g)
        h = torch.nn.functional.silu(h) * torch.nn.functional.linear(
            x.reshape(t, -1), u)
        return torch.nn.functional.linear(h, d).view(x.shape)

    def _expert(self, prefix, xr):
        dev = self.device
        g = self._t(f"{prefix}.gate_proj.weight").to(dev)
        u = self._t(f"{prefix}.up_proj.weight").to(dev)
        d = self._t(f"{prefix}.down_proj.weight").to(dev)
        h = torch.nn.functional.silu(xr @ g.t()) * (xr @ u.t())
        return h @ d.t()


def presplit_flint_shards(ckpt_dir: str, out_root: str, world: int) -> list[str]:
    """Split the Flint checkpoint into per-rank safetensors (one rank/pass).

    Each output holds exactly one TP rank's tensors: non-expert slices
    (vocab rows, qkv head slices, o cols, intermediate slices) + its
    rank::world experts (whole), original names preserved. Peak anon =
    ONE rank's shard (~20GB); the checkpoint is read world times.
    """
    import glob as _glob
    import os as _os

    import torch as _torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    cfg = load_flint_config(ckpt_dir)
    V = cfg.vocab_size
    hd = cfg.head_dim
    qh = cfg.num_attention_heads // world
    kvh = cfg.num_key_value_heads // world
    g_inter = cfg.intermediate_size
    sh_inter = cfg.shared_expert_intermediate_size
    files = sorted(_glob.glob(_os.path.join(ckpt_dir, "*.safetensors")))
    outs = []
    for r in range(world):
        out = _os.path.join(out_root, f"rank{r}.safetensors")
        outs.append(out)
        if _os.path.exists(out):
            continue
        _os.makedirs(out_root, exist_ok=True)
        buf: dict = {}
        v0, v1 = r * V // world, (r + 1) * V // world
        h0 = r * qh
        r0, r1 = r * g_inter // world, (r + 1) * g_inter // world
        sr0 = r * sh_inter // world
        sr1 = (r + 1) * sh_inter // world
        for f in files:
            with safe_open(f, framework="pt", device="cpu") as sf:
                for name in sf.keys():
                    if name.endswith(("k_proj.weight", "v_proj.weight")):
                        continue  # folded into q_proj below
                    t = sf.get_tensor(name)
                    if name in ("model.embed_tokens.weight", "lm_head.weight"):
                        buf[name] = t[v0:v1].contiguous()
                    elif ".self_attn.q_proj.weight" in name:
                        k = sf.get_tensor(name.replace("q_proj", "k_proj"))
                        v = sf.get_tensor(name.replace("q_proj", "v_proj"))
                        buf[name] = _torch.cat([
                            t[h0 * hd:(h0 + qh) * hd],
                            k[r * kvh * hd:(r + 1) * kvh * hd],
                            v[r * kvh * hd:(r + 1) * kvh * hd]], dim=0).contiguous()
                    elif ".self_attn.o_proj.weight" in name:
                        buf[name] = t[:, h0 * hd:(h0 + qh) * hd].contiguous()
                    elif ".experts." in name:
                        e = int(name.split(".experts.")[1].split(".")[0])
                        if e % world == r:
                            buf[name] = t
                    elif ".shared_expert." in name or ".mlp." in name:
                        li = int(name.split(".")[2])
                        a0, a1 = ((r0, r1) if li in (cfg.mlp_only_layers or [])
                                  else (sr0, sr1))
                        if name.endswith(".mlp.gate.weight"):
                            buf[name] = t  # router, full
                        elif name.endswith("gate_proj.weight"):
                            u = sf.get_tensor(name.replace("gate_proj", "up_proj"))
                            g = t.shape[0]
                            buf[name] = _torch.cat(
                                [t[:g][a0:a1], u[g:][a0:a1]], dim=0).contiguous()
                        elif name.endswith("down_proj.weight"):
                            buf[name] = t[:, a0:a1].contiguous()
                        else:
                            buf[name] = t
                    else:
                        buf[name] = t
                    del t
        save_file(buf, out, metadata={"format": "pt"})
        buf.clear()
    return outs


def load_flint_rank_shard(model: FlintRef, shard_path: str,
                          dtype=torch.bfloat16) -> None:
    """Load a presplit per-rank shard (original names, slices pre-applied)
    into a skeleton already shrunk by shard_flint_tp. Pure put-by-name."""

    from safetensors import safe_open

    def put(path, t):
        node = model
        for part in path.split("."):
            node = getattr(node, part)
        assert node.shape == t.shape, f"{path}: {node.shape} vs {t.shape}"
        node.data.copy_(t.to(node.dtype))

    w = _TP_STATE["world"]
    r = _TP_STATE["rank"]
    qh = model.cfg.num_attention_heads // w

    with safe_open(shard_path, framework="pt", device="cpu") as sf:
        for name in sf.keys():
            t = sf.get_tensor(name).to(dtype)
            if name == "model.embed_tokens.weight":
                put("embed.weight", t)
            elif name == "lm_head.weight":
                put("lm_head.weight", t)
            elif name == "model.norm.weight":
                put("final_norm.weight", t)
            elif ".self_attn." in name:
                li = int(name.split(".")[2])
                a = f"layers.{li}.attn"
                if name.endswith("q_proj.weight"):
                    put(f"{a}.qkv.weight", t)
                elif name.endswith("o_proj.weight"):
                    put(f"{a}.o.weight", t)
                elif name.endswith("q_norm.weight"):
                    put(f"{a}.q_norm.weight", t)
                elif name.endswith("k_norm.weight"):
                    put(f"{a}.k_norm.weight", t)
                elif name.endswith("softmax_offset"):
                    h0 = r * qh
                    put(f"{a}.sink",
                        t[h0:h0 + qh] if t.shape[0] != qh else t)
            elif ".experts." in name:
                li = int(name.split(".")[2])
                e = int(name.split(".experts.")[1].split(".")[0])
                local = e // _TP_STATE["world"]
                tgt = f"layers.{li}.mlp.experts.{local}"
                if name.endswith("gate_proj.weight"):
                    u = sf.get_tensor(name.replace("gate_proj", "up_proj"))
                    put(f"{tgt}.gate_up.weight", torch.cat([t, u], dim=0))
                elif name.endswith("down_proj.weight"):
                    put(f"{tgt}.down.weight", t)
            elif ".mlp." in name or ".shared_expert." in name:
                li = int(name.split(".")[2])
                dense = li in (model.cfg.mlp_only_layers or [])
                inter = (model.cfg.intermediate_size if dense
                         else model.cfg.shared_expert_intermediate_size)
                a0 = _TP_STATE["rank"] * inter // _TP_STATE["world"]
                a1 = (_TP_STATE["rank"] + 1) * inter // _TP_STATE["world"]
                if name.endswith("mlp.gate.weight"):
                    put(f"layers.{li}.mlp.router.weight", t)
                elif name.endswith("gate_proj.weight"):
                    if dense:
                        tgt = f"layers.{li}.mlp"
                    else:
                        tgt = f"layers.{li}.mlp.shared"
                    u = sf.get_tensor(name.replace("gate_proj", "up_proj"))
                    if u.shape[0] != a1 - a0:  # presplit stored up unsliced
                        u = u[a0:a1]
                    put(f"{tgt}.gate_up.weight",
                        torch.cat([t[:a1 - a0], u], dim=0))
                elif name.endswith("down_proj.weight"):
                    tgt = f"layers.{li}.mlp" if dense \
                        else f"layers.{li}.mlp.shared"
                    put(f"{tgt}.down.weight", t)
            elif name.endswith("input_layernorm.weight"):
                put(f"layers.{int(name.split('.')[2])}.norm1.weight", t)
            elif name.endswith("post_attention_layernorm.weight"):
                put(f"layers.{int(name.split('.')[2])}.norm2.weight", t)
            del t
