"""PTQR retraining of the Qwen3.8-27B target for the native-int8 MI100 stack.

Phase 1 of the native-int8 program (docs/recipes + AGENTS.md). The student
forward replicates the DEPLOYED quantization of the production recipe exactly,
with annealed dither ("PTQR") on the rounding decisions so the training forward
converges to deployment as tau -> 0 (late training is bit-identical to serving):

* weights     int8 RTN, G128 groups along the input dim, fp16 group scales
              (GPTQ GS128 symmetric grid; retraining re-derives q AND scales)
* activations per-token dynamic symmetric int8, fp32 scale,
              q = clamp(floor(x/s + 0.5), -127, 127)  [round-half-up]
              -- replicates vllm act_quant_rn.py exactly
* KV cache    int8_block_g{G} per (token, kv_head) vector: fp16 group scales
              from max|g|/127 (min 1e-6), round-half-away-from-zero,
              clamp [-128, 127]  -- replicates reshape_and_cache_g8 exactly
* mamba/GDN   state fp32 (deployed fp32 mamba cache); conv + norms untouched
* teacher     frozen full-BF16 reference (the only baseline; no Q8 anything)

Dither (the PTQR mechanism, mapped to a single-candidate stack): each rounding
decision gets centered uniform noise scaled by tau; tau anneals geometrically to
0 over training, so exploration of neighbouring grid points early in training
gives way to the exact deployed rounding late. Straight-through estimators
carry gradients through every quantizer.

Runs under torchrun TP4 (one rank per MI100) with the SDGraft TP adapters
(common/tp_ssm.py + common/tp_attention.py) imported from the sibling checkout;
flash attention via the curvedinf Triton-AMD fork (FLASH_ATTENTION_TRITON_AMD_ENABLE),
and the FLA chunk_gated_delta_rule backward fault on gfx908 avoided through the
T-chunked torch fallback those adapters install.

Usage (training venv, 4 GPUs):
  FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  /home/curved/.venvs/sdgraft-train/bin/python -m torchrun --nproc_per_node=4 \
    scripts/ptqr_train_target.py --model /home/curved/models/Qwen3.8-27B-bf16-ref \
    --data_dir /home/curved/SDGraft/data --data_name qwen38_longctx/train_tokens.pt \
    --seq_len 4096 --max_steps 200
Smoke (tiny model, 1 GPU):  scripts/ptqr_train_target.py --tiny --max_steps 2
"""

from __future__ import annotations

import argparse
import gc
import itertools
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

SDGRAFT_ROOT = os.environ.get("SDGRAFT_ROOT", "/home/curved/SDGraft")


# ---------------------------------------------------------------------------
# Deployed-quantization fake-quant primitives (exact kernel semantics)
# ---------------------------------------------------------------------------

_QUANT_MAX_W = 127.0  # weight grid: symmetric 127 levels
_ACT_MAX = 127.0      # act grid: symmetric, clamp +/-127 (act_quant_rn.py)


def _dither(shape, device, dtype, tau: float, gen: torch.Generator | None):
    """Centered uniform dither in [-tau/2, tau/2] added to rounded values."""
    if tau <= 0.0:
        return None
    u = torch.rand(shape, device=device, generator=gen)
    return (u - 0.5) * tau


def act_quant_fake(x: torch.Tensor, tau: float = 0.0,
                   gen: torch.Generator | None = None) -> torch.Tensor:
    """Per-token dynamic symmetric int8, round-half-up (act_quant_rn.py).

    x: [M, K] fp16/bf16/fp32 -> dequantized fake-quant with STE gradient.
    Scale is fp32 max|x|/127 per row (1.0 when the row is all-zero).
    """
    orig_dtype = x.dtype
    xf = x.float()
    s = xf.abs().amax(dim=-1, keepdim=True) / _ACT_MAX
    s = torch.where(s == 0.0, torch.ones_like(s), s)
    # replicate act_quant_rn.py EXACTLY: reciprocal multiply, not division
    z = xf * (1.0 / s)
    z = torch.floor(z + 0.5)
    d = _dither(z.shape, x.device, torch.float32, tau, gen)
    if d is not None:
        # explore neighbouring grid points, then re-round (deployed grid)
        z = torch.floor(z + d + 0.5)
    z = z.clamp(-_ACT_MAX, _ACT_MAX)
    out = z * s
    return x + (out.to(orig_dtype) - x).detach()  # STE


def kv_block_quant_fake(t: torch.Tensor, group: int, tau: float = 0.0,
                        gen: torch.Generator | None = None) -> torch.Tensor:
    """int8_block_g{G} fake-quant for [B, S, H, D] or [B, H, S, D] K/V tensors.

    Per (token, head) vector, D is split into D//G groups; scale =
    fp16(max|g|/127 clamped to 1e-6), round-half-away-from-zero, clamp
    [-128, 127] — the reshape_and_cache_g8 writer, bit for bit.
    """
    orig_dtype = t.dtype
    B, S, H, D = t.shape
    assert D % group == 0, f"head_size {D} not divisible by KV group {group}"
    tf = t.float().reshape(B, S, H, D // group, group)
    amax = tf.abs().amax(dim=-1)
    s16 = (amax / _QUANT_MAX_W).clamp_min(1e-6)
    s16 = s16.to(torch.float16).float()  # deployed scales are stored fp16
    # replicate reshape_and_cache_g8 EXACTLY: reciprocal multiply
    z = tf * (1.0 / s16).unsqueeze(-1)
    sign = torch.sign(z)
    az = z.abs()
    az = torch.floor(az + 0.5)
    d = _dither(az.shape, t.device, torch.float32, tau, gen)
    if d is not None:
        az = torch.floor(az + d + 0.5)
    az = az.clamp(0.0, 128.0)
    z = sign * az
    z = z.clamp(-128.0, 127.0)
    out = z * s16.unsqueeze(-1)
    out = out.reshape(B, S, H, D)
    return t + (out.to(orig_dtype) - t).detach()  # STE


class _PTQRLinFn(torch.autograd.Function):
    """Memory-safe W8A8 fake-quant linear.

    F.linear's autograd node SAVES its weight operand, so a graph-built
    quantized weight retains a full weight-sized tensor per linear for the
    whole step (~11 GiB across the stack — measured OOM driver). This Function
    never retains the quantized tensors: forward computes them graph-free, and
    backward recomputes them cheaply, applying straight-through estimators
    (d q / d w ≡ 1, d q / d s = integer payload q) — the standard QAT pattern.
    """

    _dbg_done = False

    @staticmethod
    def forward(ctx, x, w, scale, group: int, tau: float, scale_fp16: bool = True):
        ctx.save_for_backward(x, w, scale)
        ctx.group = group
        ctx.tau = tau
        ctx.scale_fp16 = scale_fp16
        with torch.no_grad():
            wq = _weight_quant_compute(w, scale, group, tau, scale_fp16=scale_fp16)
            xq = act_quant_fake(x, tau)
            return F.linear(xq, wq)

    @staticmethod
    def backward(ctx, dy):
        x, w, scale = ctx.saved_tensors
        group, tau = ctx.group, ctx.tau
        s_fp16 = ctx.scale_fp16
        dbg = os.environ.get("PTQR_MEM_DEBUG") and not _PTQRLinFn._dbg_done
        with torch.no_grad():
            wq = _weight_quant_compute(w, scale, group, tau, scale_fp16=s_fp16)
            xq = act_quant_fake(x, tau)
            if dbg:
                _PTQRLinFn._dbg_done = True
                for nm, t in (("dy", dy), ("wq", wq), ("xq", xq),
                              ("x", x), ("w", w), ("scale", scale)):
                    fin = torch.isfinite(t).all().item()
                    print(f"  [bkdbg] {nm}: finite={fin} "
                          f"absmax={t.abs().max().item() if t.numel() else 0}",
                          flush=True)
            grad_x = dy @ wq if ctx.needs_input_grad[0] else None  # y = x W^T -> dx = dy W
            gw = gs = None
            if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
                # Row-chunked so the lm_head shard's fp32 grad block never
                # exceeds one row slice (full-size fp32 spikes OOM the ~3 GiB
                # headroom — measured).
                out_f, in_f = w.shape
                dy_flat = dy.reshape(-1, dy.shape[-1])
                x_flat = xq.reshape(-1, in_f)
                R = 2048
                gw = torch.empty(w.shape, dtype=w.dtype, device=w.device)
                gs = torch.empty(scale.shape, device=scale.device)
                for r0 in range(0, out_f, R):
                    r1 = min(r0 + R, out_f)
                    gw_r = dy_flat[:, r0:r1].t() @ x_flat  # [rows, in] fp32, STE on w
                    if ctx.needs_input_grad[1]:
                        gw[r0:r1] = gw_r.to(w.dtype)
                    if ctx.needs_input_grad[2]:
                        wc = w[r0:r1].float().reshape(r1 - r0, in_f // group, group)
                        s_e = scale[r0:r1].to(torch.float16).float() if s_fp16 \
                            else scale[r0:r1].float()
                        z = wc / s_e.unsqueeze(-1)
                        q = (torch.sign(z) * torch.floor(z.abs() + 0.5)
                             .clamp(0.0, _QUANT_MAX_W)).reshape(r1 - r0, in_f)
                        gs[r0:r1] = (gw_r.reshape(r1 - r0, in_f // group, group) * q.reshape(r1 - r0, in_f // group, group)).sum(dim=-1)
        return grad_x, gw, gs, None, None, None


def _weight_integer(w: torch.Tensor, scale: torch.Tensor, group: int) -> torch.Tensor:
    """Integer payload q (pre fp16-scale multiply), row-chunked, no grad."""
    out_f, in_f = w.shape
    q = torch.empty(w.shape, dtype=torch.float32, device=w.device)
    R = 2048
    for r0 in range(0, out_f, R):
        r1 = min(r0 + R, out_f)
        wc = w[r0:r1].float().reshape(r1 - r0, in_f // group, group)
        s16 = scale[r0:r1].to(torch.float16).float()
        z = wc / s16.unsqueeze(-1)
        sign = torch.sign(z)
        az = torch.floor(z.abs() + 0.5).clamp(0.0, _QUANT_MAX_W)
        q[r0:r1] = (sign * az).reshape(r1 - r0, in_f)
    return q


def _weight_quant_compute(w: torch.Tensor, scale: torch.Tensor, group: int,
                          tau: float = 0.0,
                          gen: torch.Generator | None = None,
                          scale_fp16: bool = True) -> torch.Tensor:
    """G128 int8 weight fake-quant with learned per-group scales (no grad).

    w: [out, in] master (bf16); scale: [out, in//group] fp32 parameter.
    Deployed grid: q in [-127, 127]; fp16 group scales for the W8A8 GEMM
    (gptq GS128 contract), fp32 row scales for the per-channel LM head
    (_quantize_lm_head_w8a8_ contract).

    Row-chunked so the fp32 working set never exceeds one row block.
    """
    orig_dtype = w.dtype
    out_f, in_f = w.shape
    out = torch.empty_like(w)
    R = 2048
    for r0 in range(0, out_f, R):
        r1 = min(r0 + R, out_f)
        wc = w[r0:r1].float().reshape(r1 - r0, in_f // group, group)
        s_e = scale[r0:r1].to(torch.float16).float() if scale_fp16 \
            else scale[r0:r1].float()
        z = wc / s_e.unsqueeze(-1)
        sign = torch.sign(z)
        az = torch.floor(z.abs() + 0.5)
        d = _dither(az.shape, w.device, torch.float32, tau, gen)
        if d is not None:
            az = torch.floor(az + d + 0.5)
        z = sign * az.clamp(0.0, _QUANT_MAX_W)
        z = z.clamp(-_QUANT_MAX_W, _QUANT_MAX_W)
        out[r0:r1] = (z * s_e.unsqueeze(-1)).reshape(r1 - r0, in_f).to(orig_dtype)
    return out


# ---------------------------------------------------------------------------
# PTQR linear: deployed W8A8 GEMM in the training loop
# ---------------------------------------------------------------------------

class PTQRLinear(nn.Module):
    """Linear whose forward is the deployed W8A8 INT8 GEMM (fake-quant).

    Holds the bf16 master weight and a trainable fp32 per-G128-group scale.
    Both the input activations and the weight pass through their deployed
    quantizers; the matmul runs in bf16 with fp32 accumulation (the aiter
    kernel accumulates int8 products in int32 then scales — same values,
    different accumulation order, second-order difference only).
    """

    def __init__(self, weight: torch.Tensor, group: int = 128,
                 scale_fp16: bool = True):
        super().__init__()
        out_f, in_f = weight.shape
        assert in_f % group == 0, f"in_features {in_f} not divisible by {group}"
        self.weight = nn.Parameter(weight.detach().clone())
        with torch.no_grad():
            wf = weight.detach().float().reshape(out_f, in_f // group, group)
            amax = wf.abs().amax(dim=-1)
            self.scale = nn.Parameter((amax / _QUANT_MAX_W).clamp_min(1e-8))
        self.group = group
        # W8A8 GEMM weights use fp16 group scales (gptq GS128 contract); the
        # untied LM head uses the CK per-channel contract (fp32 row scales,
        # see _quantize_lm_head_w8a8_ in vocab_parallel_embedding.py).
        self.scale_fp16 = scale_fp16
        self.tau = 0.0
        self.quant_input = True
        self.row_reduce_group = None   # set for row-parallel use (MLP down_proj)
        self.tp_vocab_start = 0        # vocab-shard base for lm_head use
        # additive TP sharding API used by SDGraft common/tp_*.py
        self._row_ranges: list[tuple[int, int]] | None = None
        self._col_range: tuple[int, int] | None = None

    # -- TP additive API (mirrors MixedQuantLinear.shard_rows/shard_cols) -----
    def shard_rows(self, row_ranges: list[tuple[int, int]]) -> None:
        idx = torch.cat([torch.arange(a, b) for a, b in row_ranges])
        dev = self.weight.device
        self.weight = nn.Parameter(self.weight.detach().index_select(0, idx.to(dev)))
        self.scale = nn.Parameter(self.scale.detach().index_select(0, idx.to(dev)))
        self._row_ranges = row_ranges

    def shard_cols(self, col_range: tuple[int, int]) -> None:
        c0, c1 = col_range
        assert (c1 - c0) % self.group == 0, "column shard must keep G128 groups whole"
        self.weight = nn.Parameter(self.weight.detach()[:, c0:c1].contiguous())
        g0, g1 = c0 // self.group, c1 // self.group
        self.scale = nn.Parameter(self.scale.detach()[:, g0:g1].contiguous())
        self._col_range = col_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = _PTQRLinFn.apply(x, self.weight, self.scale, self.group, self.tau,
                               self.scale_fp16)
        if self.row_reduce_group is not None:
            # Row-parallel partial (own column slice of the input): sum the
            # partials across ranks, autograd-exact (_RowParallelSum).
            from common.tp_ssm import _RowParallelSum
            out = _RowParallelSum.apply(out, self.row_reduce_group)
        return out


def replace_linears_with_ptqr(model: nn.Module, group: int = 128,
                              skip: set[str] | None = None) -> dict[str, PTQRLinear]:
    """Swap every nn.Linear (except skipped names) for a PTQRLinear.

    The untied LM head gets the DEPLOYED per-channel contract (one scale per
    output row, fp32 — _quantize_lm_head_w8a8_ in the serving fork), not the
    G128 W8A8 group contract.
    """
    skip = skip or set()
    replaced: dict[str, PTQRLinear] = {}
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if full in skip or not isinstance(child, nn.Linear):
                continue
            is_head = child_name == "lm_head"
            pq = PTQRLinear(child.weight,
                            group=child.weight.shape[1] if is_head else group,
                            scale_fp16=not is_head)
            pq.quant_input = True
            setattr(module, child_name, pq)
            replaced[full] = pq
    return replaced


def fake_quant_embedding_(emb: nn.Embedding) -> None:
    """Apply the deployed int8 embedding conversion in place (frozen).

    Mirrors _quantize_embedding_int8_ (vocab_parallel_embedding.py): per-row
    fp16 scale, round, clamp [-128, 127]; the weight becomes the dequantized
    values the serving gather returns. The Phase-0 BF16 reference was itself
    served through this path, so student AND teacher both see it — exact on
    both sides.
    """
    with torch.no_grad():
        R = 8192  # chunked: the full fp32 upcast is ~15 GiB of transients
        w = emb.weight.data
        n = w.shape[0]
        for r0 in range(0, n, R):
            r1 = min(r0 + R, n)
            wf = w[r0:r1].float()
            scale = (wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0).to(w.dtype)
            q = (wf / scale.float()).round().clamp(-128, 127)
            w[r0:r1] = (q * scale.float()).to(w.dtype)


# ---------------------------------------------------------------------------
# KV fake-quant hook (attention layers): deployed cache write+read in-loop
# ---------------------------------------------------------------------------

def attach_kv_fake_quant(model: nn.Module, group: int) -> int:
    """Patch every attention module's forward to fake-quant K/V post-rotary.

    The stock Qwen3_5 attention forward computes q, k, v projections, applies
    q/k norm + rotary, then calls the attention interface. We wrap the module
    forward and intercept the interface call via a patched attention function
    registered under a dedicated name, so the fake-quant lands exactly where
    the serving stack writes the KV cache (post-rotary, pre-attention).
    """
    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    attn_cls = m.Qwen3_5Attention
    state = {"tau": 0.0, "group": group}

    def _quant_kv_fwd(module, query, key, value, attention_mask, dropout=0.0,
                      scaling=None, **kwargs):
        # query/key/value arrive as [B, H, S, D] from the stock forward
        key = kv_block_quant_fake(key.transpose(1, 2), state["group"], state["tau"]).transpose(1, 2)
        value = kv_block_quant_fake(value.transpose(1, 2), state["group"], state["tau"]).transpose(1, 2)
        base = getattr(module, "_ptqr_base_attn_fn")
        return base(module, query, key, value, attention_mask,
                    dropout=dropout, scaling=scaling, **kwargs)

    name = "ptqr_kv_fake"
    base_impl = ALL_ATTENTION_FUNCTIONS.get_interface(
        os.environ.get("PTQR_ATTN_IMPL", "rocm_triton"), m.eager_attention_forward)
    ALL_ATTENTION_FUNCTIONS.register(name, _quant_kv_fwd)

    n = 0
    for name_mod, mod in model.named_modules():
        if isinstance(mod, attn_cls):
            mod._ptqr_base_attn_fn = base_impl
            mod.config._attn_implementation = name
            mod._ptqr_kv_state = state
            n += 1
    return n


# ---------------------------------------------------------------------------
# MLP + lm_head TP sharding (attention/GDN are handled by the SDGraft adapters)
# ---------------------------------------------------------------------------

def _vocab_ranges(vocab: int, world: int) -> list[tuple[int, int]]:
    """Even-as-possible vocab partition (V need not divide world)."""
    base, rem = divmod(vocab, world)
    ranges, start = [], 0
    for r in range(world):
        n = base + (1 if r < rem else 0)
        ranges.append((start, start + n))
        start += n
    return ranges


def shard_mlp_and_heads_one(model, world: int, rank: int, group=None):
    """Column/row-parallel MLP + vocab-parallel lm_head on ONE model.

    Runs BEFORE PTQR replacement (all projections are still plain nn.Linear),
    slicing the mmap-backed weights so only each rank's shard materializes:

    * mlp.gate_proj / up_proj: column-parallel on the intermediate dim.
    * mlp.down_proj: row-parallel; partials are all-reduced by a forward hook
      (a student's hooks vanish at PTQR replacement — re-established then as
      PTQRLinear.row_reduce_group by the caller).
    * lm_head: vocab-sharded; the loss is the exact cross-rank-LSE form
      (tp_vocab_parallel_losses). tp_vocab_start is set on the new head.

    Returns (intermediate_size, vocab_range_of_this_rank).
    """
    from common.tp_ssm import _RowParallelSum

    def _shard_linear_rows(lin: nn.Linear, rows: tuple[int, int]):
        r0, r1 = rows
        new = nn.Linear(lin.in_features, r1 - r0, bias=False,
                        dtype=lin.weight.dtype, device=lin.weight.device)
        with torch.no_grad():
            new.weight.copy_(lin.weight[r0:r1])
        return new

    core = model.model
    inter = None
    for name, layer in core.named_modules():
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "gate_proj"):
            continue
        inter = mlp.gate_proj.weight.shape[0]
        q, rem = divmod(inter, world)
        assert rem == 0, f"intermediate {inter} not divisible by TP{world}"
        a, b = rank * q, (rank + 1) * q
        mlp.gate_proj = _shard_linear_rows(mlp.gate_proj, (a, b))
        mlp.up_proj = _shard_linear_rows(mlp.up_proj, (a, b))
        new_down = nn.Linear(b - a, mlp.down_proj.weight.shape[0], bias=False,
                             dtype=mlp.down_proj.weight.dtype,
                             device=mlp.down_proj.weight.device)
        with torch.no_grad():
            new_down.weight.copy_(mlp.down_proj.weight[:, a:b])
        group_ref = group

        def _reduce_hook(mod, inp, out):
            return (_RowParallelSum.apply(out, group_ref)
                    if group_ref is not None else out)
        new_down.register_forward_hook(_reduce_hook)
        mlp.down_proj = new_down
    vocab = model.lm_head.weight.shape[0]
    vr = _vocab_ranges(vocab, world)[rank]
    model.lm_head = _shard_linear_rows(model.lm_head, vr)
    model.lm_head.tp_vocab_start = vr[0]
    return inter, vr


def _mem(tag: str, rank: int):
    if os.environ.get("PTQR_MEM_DEBUG") and torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 2**30
        r = torch.cuda.memory_reserved() / 2**30
        print(f"[rank {rank}][mem] {tag}: alloc {a:.2f} GiB, reserved {r:.2f} GiB",
              flush=True)


def tp_vocab_parallel_losses(student, teacher, x, y, temperature: float,
                             chunk: int, distill_weight: float):
    """Vocab-parallel chunked CE+KLD (exact cross-rank logsumexp).

    Ported from SDGraft _tp_vocab_parallel_losses for our on-device bf16
    teacher: student/teacher lm_heads are vocab-sharded identically; CE picks
    the target's owning shard; KL splits over shards; the global LSE is exact
    via per-rank max + all-reduced shifted exp-sum (differentiable for the
    student side). Per-rank value partials are summed once at the end.
    """
    import torch.distributed as dist

    world = dist.get_world_size() if dist.is_initialized() else 1
    _mem("loss enter", 0)
    hidden = student.model(input_ids=x).last_hidden_state
    _mem("student hidden done", 0)
    with torch.no_grad():
        t_hidden = teacher.model(input_ids=x).last_hidden_state
    _mem("teacher hidden done", 0)

    H = hidden.shape[-1]
    s_flat = hidden.reshape(-1, H)
    t_flat = t_hidden.reshape(-1, H)
    y_flat = y.reshape(-1)
    n_tok = s_flat.shape[0]
    T = temperature
    s_head, t_head = student.lm_head, teacher.lm_head
    s_v0 = int(getattr(s_head, "tp_vocab_start", 0))
    assert s_v0 == int(getattr(t_head, "tp_vocab_start", 0))
    inv_world = 1.0 / world

    def _global_lse(logits, differentiable: bool):
        m_local = logits.detach().amax(dim=-1)
        m_all = m_local.clone().contiguous()
        dist.all_reduce(m_all, op=dist.ReduceOp.MAX)
        e_local = (logits - m_all.unsqueeze(-1)).exp().sum(-1)
        if differentiable:
            # NOTE: tp_utils.tp_all_reduce_sum is a no-op unless its global TP
            # state was enabled (we never do that here) — call the autograd
            # Function directly so the student LSE is always the exact global.
            from common.tp_utils import _AllReduceSum
            e_sum = _AllReduceSum.apply(e_local)
        else:
            e_sum = e_local.clone().contiguous()
            dist.all_reduce(e_sum, op=dist.ReduceOp.SUM)
        return m_all + e_sum.log()

    h_grad = torch.zeros_like(s_flat)
    lm_total, kl_total = 0.0, 0.0
    n_chunks = (n_tok + chunk - 1) // chunk
    for ci, i in enumerate(range(0, n_tok, chunk)):
        if ci % 8 == 0:
            print(f"  [tp-loss] chunk {ci}/{n_chunks}", flush=True)
        h_c = s_flat[i: i + chunk].detach().requires_grad_(True)
        logits_c = s_head(h_c).float()  # [chunk, local vocab]
        _mem(f"chunk {ci} logits", 0)
        lse = _global_lse(logits_c, differentiable=True)

        y_local = y_flat[i: i + chunk] - s_v0
        in_shard = (y_local >= 0) & (y_local < logits_c.shape[-1])
        y_safe = y_local.clamp(0, logits_c.shape[-1] - 1)
        picked = logits_c.gather(1, y_safe.unsqueeze(1)).squeeze(1)
        ce_sum_c = (lse - torch.where(in_shard, picked, torch.zeros_like(picked))).sum()

        with torch.no_grad():
            t_logits = t_head(t_flat[i: i + chunk]).float()
            t_lse = _global_lse(t_logits, differentiable=False)
            t_logp = t_logits - t_lse.unsqueeze(-1)
            t_prob = t_logp.exp()
        s_logp = logits_c - lse.unsqueeze(-1)
        kl_c = (t_prob * (t_logp - s_logp)).sum() * (T * T)

        ((ce_sum_c + distill_weight * kl_c) / n_tok).backward()
        with torch.no_grad():
            h_grad[i: i + chunk] = h_c.grad if h_c.grad is not None else torch.zeros_like(h_c)
        ce_val_c = (lse * inv_world - torch.where(in_shard, picked, torch.zeros_like(picked))).sum()
        lm_total += ce_val_c.item() / n_tok
        kl_total += kl_c.item() / n_tok
        del logits_c, h_c

    torch.autograd.set_multithreading_enabled(False)
    hidden.backward(h_grad.reshape(hidden.shape))
    torch.autograd.set_multithreading_enabled(True)
    part = torch.tensor([lm_total, kl_total], device=hidden.device)
    dist.all_reduce(part, op=dist.ReduceOp.SUM)
    return part[0].item(), part[1].item()


@torch.no_grad()
def eval_kld(student, teacher, val_iter, steps: int, chunk: int, temperature: float):
    """Held-out torch KLD vs the BF16 teacher at tau=0 — the judge metric.

    Chunked over tokens; vocab-parallel when a process group exists (exact
    cross-rank LSE, non-differentiable). Returns exact global means on all
    ranks.
    """
    import torch.distributed as dist
    world = dist.get_world_size() if dist.is_initialized() else 1
    student.eval()
    lm_sum, kl_sum = 0.0, 0.0
    for i, (x, y) in enumerate(itertools.islice(val_iter, steps)):
        x, y = x.cuda(), y.cuda()
        s_h = student.model(input_ids=x).last_hidden_state
        t_h = teacher.model(input_ids=x).last_hidden_state
        H = s_h.shape[-1]
        s_flat, t_flat = s_h.reshape(-1, H), t_h.reshape(-1, H)
        y_flat = y.reshape(-1)
        n_tok = s_flat.shape[0]
        kl_step, lm_step = 0.0, 0.0
        s_v0 = int(getattr(student.lm_head, "tp_vocab_start", 0))
        for j in range(0, n_tok, chunk):
            s_lc = student.lm_head(s_flat[j: j + chunk]).float()
            t_lc = teacher.lm_head(t_flat[j: j + chunk]).float()

            def _lse(lc):
                m = lc.detach().amax(dim=-1)
                if world > 1:
                    m_all = m.clone().contiguous()
                    dist.all_reduce(m_all, op=dist.ReduceOp.MAX)
                    e = (lc - m_all.unsqueeze(-1)).exp().sum(-1).clone().contiguous()
                    dist.all_reduce(e, op=dist.ReduceOp.SUM)
                    return m_all + e.log()
                return m + (lc - m.unsqueeze(-1)).exp().sum(-1).log()

            s_lse, t_lse = _lse(s_lc), _lse(t_lc)
            s_logp = s_lc - s_lse.unsqueeze(-1)
            t_logp = t_lc - t_lse.unsqueeze(-1)
            t_prob = t_logp.exp()
            kl_step += (t_prob * (t_logp - s_logp)).sum().item()
            y_local = y_flat[j: j + chunk] - s_v0
            in_shard = (y_local >= 0) & (y_local < s_lc.shape[-1])
            y_safe = y_local.clamp(0, s_lc.shape[-1] - 1)
            picked = s_lc.gather(1, y_safe.unsqueeze(1)).squeeze(1)
            # every rank owns lse/world of the denominator; only the target's
            # owner carries the -logit term (exact after the cross-rank sum)
            lm_step += (s_lse / world - torch.where(in_shard, picked, torch.zeros_like(picked))).sum().item()
            del s_lc, t_lc, s_logp, t_logp, t_prob
        kl_sum += kl_step / n_tok
        lm_sum += lm_step / n_tok
        del s_h, t_h, s_flat, t_flat
    if world > 1:
        part = torch.tensor([lm_sum, kl_sum], device="cuda")
        dist.all_reduce(part, op=dist.ReduceOp.SUM)
        lm_sum, kl_sum = part[0].item(), part[1].item()
    student.train()
    return lm_sum / max(1, steps), kl_sum / max(1, steps)


# ---------------------------------------------------------------------------
# Chunked KLD+CE loss (ported from SDGraft train_dynamic_quant.py)
# ---------------------------------------------------------------------------

def chunked_forward_loss(student, teacher, x, y, temperature: float,
                         chunk: int, distill_weight: float):
    """Token-chunked student/teacher forward + CE/KL loss (see SDGraft).

    Peak memory is one chunk's [chunk, V] softmax set; per-chunk detached-head
    backwards accumulate h_grad, then one final model backward runs. Exact:
    both losses are token sums.
    """
    student_base, student_head = student.model, student.lm_head
    teacher_base, teacher_head = teacher.model, teacher.lm_head

    hidden = student_base(input_ids=x).last_hidden_state
    with torch.no_grad():
        t_hidden = teacher_base(input_ids=x).last_hidden_state

    H = hidden.shape[-1]
    s_flat = hidden.reshape(-1, H)
    t_flat = t_hidden.reshape(-1, H)
    y_flat = y.reshape(-1)
    n_tok = s_flat.shape[0]

    head_dev = s_flat.device
    h_grad = torch.zeros_like(s_flat)
    n_chunks = (n_tok + chunk - 1) // chunk
    T = temperature
    lm_total, kl_total = 0.0, 0.0
    for ci, i in enumerate(range(0, n_tok, chunk)):
        if ci % 8 == 0:
            print(f"  [chunked-loss] chunk {ci}/{n_chunks}", flush=True)
        h_c = s_flat[i: i + chunk].detach().requires_grad_(True)
        logits_c = student_head(h_c)
        y_c = y_flat[i: i + chunk]
        lm_c = F.cross_entropy(logits_c, y_c, reduction="sum")
        student_log_probs = F.log_softmax(logits_c / T, dim=-1)
        del logits_c
        with torch.no_grad():
            t_logits_c = teacher_head(t_flat[i: i + chunk].to(teacher_head.weight.device)).to(head_dev)
            teacher_probs = F.softmax(t_logits_c / T, dim=-1)
            del t_logits_c
        kl_c = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum() * (T * T)
        del student_log_probs, teacher_probs
        ((lm_c + distill_weight * kl_c) / n_tok).backward()
        with torch.no_grad():
            g = h_c.grad if h_c.grad is not None else torch.zeros_like(h_c)
            h_grad[i: i + chunk] = g
        lm_total += lm_c.item() / n_tok
        kl_total += kl_c.item() / n_tok
        del lm_c, kl_c, h_c
    torch.autograd.set_multithreading_enabled(False)
    hidden.backward(h_grad.reshape(hidden.shape))
    torch.autograd.set_multithreading_enabled(True)
    return lm_total, kl_total


# ---------------------------------------------------------------------------
# Model construction (stripped-checkpoint form: Qwen3_5TextModel + head)
# ---------------------------------------------------------------------------

class LMWithHead(nn.Module):
    """Qwen3_5TextModel + separate lm_head, matching stripped checkpoint keys."""

    def __init__(self, text_model: nn.Module, vocab_size: int):
        super().__init__()
        self.model = text_model
        self.config = text_model.config
        hidden = text_model.config.hidden_size
        dtype = next(text_model.parameters()).dtype
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False).to(dtype)

    def forward(self, input_ids=None, **kw):
        out = self.model(input_ids=input_ids, **kw)
        return SimpleNamespace(logits=self.lm_head(out.last_hidden_state))


def _tiny_override(cfg):
    """Shrink a Qwen3.5-family config for the --tiny wiring smoke test."""
    cfg.num_hidden_layers = 4
    cfg.hidden_size = 512
    cfg.intermediate_size = 1024
    if hasattr(cfg, "num_attention_heads"):
        cfg.num_attention_heads = 4
    if hasattr(cfg, "num_key_value_heads"):
        cfg.num_key_value_heads = 4  # divisible by TP4 for the wiring test
    if hasattr(cfg, "head_dim"):
        cfg.head_dim = 128
    if hasattr(cfg, "vocab_size"):
        cfg.vocab_size = 4096
    layer = getattr(cfg, "layer_types", None)
    text = getattr(cfg, "text_config", None)
    return cfg


def build_lm_model(args, dtype, attn_impl):
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import Qwen3_5TextModel

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", cfg)
    text_cfg.use_cache = False
    if getattr(args, "n_layers", 0):
        text_cfg.num_hidden_layers = args.n_layers  # bisection: real dims, fewer layers
    if args.tiny:
        _tiny_override(text_cfg)
    if attn_impl is not None:
        text_cfg._attn_implementation = attn_impl
    with torch.device("meta"):
        text_model = Qwen3_5TextModel(text_cfg)
        model = LMWithHead(text_model, text_cfg.vocab_size)

    from safetensors.torch import load_file
    if not args.tiny and getattr(args, "base_checkpoint", None) and Path(args.base_checkpoint).exists():
        # File-backed mmap load: pages are shared across the 4 TP ranks and
        # faulted lazily, so host anon stays bounded (the eager safetensors
        # path OOM-kills at 4x52 GB). The dict must stay referenced — the
        # assign-loaded parameters point into its storages.
        ckpt = torch.load(args.base_checkpoint, weights_only=True,
                          map_location="cpu", mmap=True)
        load_sd = ckpt["model_state_dict"]
        missing, unexpected = model.load_state_dict(load_sd, strict=False, assign=True)
        loaded = len(load_sd) - len(unexpected)
        print(f"  [load] {loaded} tensors loaded (mmap), {len(missing)} missing, "
              f"{len(unexpected)} unexpected", flush=True)
        if missing:
            raise RuntimeError(f"missing keys from base checkpoint: {missing[:5]}...")
        # Non-persistent buffers (rotary inv_freq / original_inv_freq) are not
        # in the state dict and stayed meta under the meta-device init; rebuild
        # them properly from the live config.
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5TextRotaryEmbedding,
        )
        for name, mod in model.named_modules():
            if isinstance(mod, Qwen3_5TextRotaryEmbedding):
                fresh = Qwen3_5TextRotaryEmbedding(mod.config, device="cpu")
                mod.inv_freq = fresh.inv_freq
                if getattr(mod, "original_inv_freq", None) is not None and fresh.original_inv_freq is not None:
                    mod.original_inv_freq = fresh.original_inv_freq
        model._base_ckpt_ref = ckpt  # keep the mmap alive
        return model.to(dtype)
    load_sd: dict[str, torch.Tensor] = {}
    if not args.tiny:
        for f in sorted(Path(args.model).glob("*.safetensors")):
            load_sd.update(load_file(str(f)))
        # The published reference is the full multimodal checkpoint: the LM
        # lives under model.language_model.*, plus a vision tower (skip) and
        # an mtp head (skip). lm_head.weight is top-level.
        if any(k.startswith("model.language_model.") for k in load_sd):
            head_w = load_sd.get("lm_head.weight")
            load_sd = {"model." + k[len("model.language_model."):]: v
                       for k, v in load_sd.items()
                       if k.startswith("model.language_model.")}
            if head_w is not None:
                load_sd["lm_head.weight"] = head_w
        elif any(k.startswith("layers.") for k in load_sd):
            head_w = load_sd.get("lm_head.weight")
            load_sd = {"model." + k: v for k, v in load_sd.items()
                       if k != "lm_head.weight"}
            if head_w is not None:
                load_sd["lm_head.weight"] = head_w
        missing, unexpected = model.load_state_dict(load_sd, strict=False, assign=True)
        loaded = len(load_sd) - len(unexpected)
        print(f"  [load] {loaded} tensors loaded, {len(missing)} missing, "
              f"{len(unexpected)} unexpected", flush=True)
        if missing:
            raise RuntimeError(f"missing keys from reference checkpoint: {missing[:5]}...")
        del load_sd
        gc.collect()
    else:
        with torch.no_grad():
            for name, module in model.named_modules():
                for pname, p in list(module.named_parameters(recurse=False)):
                    if p.is_meta:
                        module._parameters[pname] = torch.empty_like(p, device="cpu").normal_(0, 0.02)
                for bname, b in list(module.named_buffers(recurse=False)):
                    if b is not None and b.is_meta:
                        module._buffers[bname] = torch.zeros_like(b, device="cpu")
    return model.to(dtype)


# ---------------------------------------------------------------------------
# Manual gradient checkpointing (this HF impl never calls _gradient_
# checkpointing_func — without wrapping, all 64 layers' activations stay
# live, ~5-6 GiB at 4k tokens) + hook-based per-tensor weight SGD (full .grad
# buffers for every weight would add another ~11 GiB/rank).
# ---------------------------------------------------------------------------

class _CkptLayer(nn.Module):
    """Wrap a decoder layer in torch.utils.checkpoint (non-reentrant)."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                self.layer, *args, use_reentrant=False, **kwargs)
        return self.layer(*args, **kwargs)


def wrap_decoder_checkpointing(model) -> int:
    core = model.model if hasattr(model, "model") else model
    layers = getattr(core, "layers", None)
    if layers is None or not isinstance(layers, nn.ModuleList):
        return 0
    for i, layer in enumerate(layers):
        if not isinstance(layer, _CkptLayer):
            layers[i] = _CkptLayer(layer)
    return len(layers)


def attach_weight_sgd_hooks(replaced: dict, lr: float, scale_lr: float) -> int:
    """Per-tensor SGD on PTQR masters + scales, applied and freed the moment
    each grad lands (post-accumulate hook). Keeps peak grad memory at ONE
    tensor instead of the full 11 GiB set. (Adafactor was tried first and its
    initial update NaN'd a scale tensor on real data — measured; plain SGD is
    also exactly composable with the per-chunk loss backwards, being linear
    in the grad.) Scales are clamped to the deployed fp16-scale domain and
    kept strictly positive."""
    import torch as _t

    n = 0
    for m in replaced.values():
        m.scale._ptqr_master_ref = m.weight  # for the feasibility projection
        for p, p_lr, is_scale in ((m.weight, lr, False), (m.scale, scale_lr, True)):

            def _hook(p=p, p_lr=p_lr, is_scale=is_scale):
                if p.grad is None:
                    return
                g = p.grad
                if not _t.isfinite(g).all():
                    p.grad = None
                    return
                with _t.no_grad():
                    gn = g.norm()
                    if gn > 1.0:  # per-tensor clip (no global clip under hooks)
                        g = g * (1.0 / gn)
                    p.data.add_(g.to(p.dtype), alpha=-p_lr)
                    if is_scale:
                        p.data.clamp_(1e-7, 65500.0)  # fp16-representable, positive
                        # RTN feasibility: a group's scale must cover its
                        # master's amax (|w| <= 127*s) — independent SGD on w
                        # and s drifts past this (measured saturation), which
                        # the deployed grid cannot represent.
                        w_ref = p._ptqr_master_ref
                        if w_ref is not None:
                            amax = w_ref.detach().float().reshape(
                                w_ref.shape[0], -1, p.shape[1]).abs().amax(dim=-1)
                            p.data.copy_(torch.maximum(p.data, amax / 127.0))
                p.grad = None
            p.register_post_accumulate_grad_hook(lambda *a, h=_hook: h())
            p.requires_grad_(True)
            n += 1
    return n


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def anneal_tau(step: int, max_steps: int, start: float, end: float) -> float:
    """Geometric anneal with a hard 0 for the final 10% (exact deployment)."""
    if max_steps <= 1:
        return 0.0
    hard_start = int(max_steps * 0.9)
    if step >= hard_start:
        return 0.0
    frac = step / max(1, hard_start - 1)
    return start * (end / start) ** frac


def build_dataloader(data_path: str, seq_len: int, batch_size: int,
                     tiny: bool = False, tiny_vocab: int = 4096):
    if tiny:
        tokens = torch.randint(0, tiny_vocab, (seq_len * 64 + 1,), dtype=torch.long)
    else:
        tokens = torch.load(data_path, weights_only=True)

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return max(0, len(tokens) - seq_len)

        def __getitem__(self, idx):
            chunk = tokens[idx: idx + seq_len + 1]
            return chunk[:-1].long(), chunk[1:].long()

    return torch.utils.data.DataLoader(_DS(), batch_size=batch_size,
                                       shuffle=True, drop_last=True, num_workers=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/curved/models/Qwen3.8-27B-bf16-ref")
    p.add_argument("--base_checkpoint",
                   default="/home/curved/models/Qwen3.8-27B-bf16-ref-lm.pt")
    p.add_argument("--data_dir", default="/home/curved/SDGraft/data")
    p.add_argument("--data_name", default="qwen38_longctx/train_tokens.pt")
    p.add_argument("--val_name", default="qwen38_longctx/val_tokens.pt")
    p.add_argument("--out_dir", default="/home/curved/models/ptqr_out")
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--scale_lr_mult", type=float, default=1.0)
    p.add_argument("--distill_weight", type=float, default=2.0)
    p.add_argument("--distill_temperature", type=float, default=1.0)
    p.add_argument("--temp_start", type=float, default=1.0)
    p.add_argument("--temp_end", type=float, default=0.001)
    p.add_argument("--weight_group", type=int, default=128)
    p.add_argument("--kv_group", type=int, default=128)
    p.add_argument("--logits_chunk", type=int, default=256)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=10)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--tiny", action="store_true", help="tiny-model wiring smoke test")
    p.add_argument("--n_layers", type=int, default=0,
                   help="override num_hidden_layers (bisection; real dims kept)")
    p.add_argument("--no_ptqr", action="store_true",
                   help="infrastructure control: unquantized student (teacher-vs-teacher); "
                        "isolates TP/attn/checkpoint bugs from quant-stack bugs")
    p.add_argument("--attn_impl", default="rocm_triton")
    args = p.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")  # RCCL underneath on ROCm
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # --- SDGraft adapters (TP sharding, flash attn, FLA fallback) -----------
    sys.path.insert(0, SDGRAFT_ROOT)
    from common.tp_ssm import apply_ssm_tp, install_t_chunked_fallback
    from common.tp_attention import apply_tp_attention, register_rocm_triton_tp
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    install_t_chunked_fallback()
    register_rocm_triton_tp()

    # --- host-RAM-staggered build (chain lock): each rank materializes its
    # student+teacher shards ALONE (peak ~24 GB anon per rank; 4 concurrent
    # builds OOM-kill the 61 GB host — measured twice). mmap pages are shared
    # and evictable; the anon slices are not, hence the serialization.
    build_lock = None
    if world > 1:
        import time as _time
        ppid = os.getppid()  # torchrun agent: identical on all ranks, run-scoped
        done_flag = f"/tmp/ptqr_build_{ppid}_{rank}.done"
        if rank > 0:
            prev_flag = f"/tmp/ptqr_build_{ppid}_{rank - 1}.done"
            while not os.path.exists(prev_flag):
                _time.sleep(2)
        print(f"[rank {rank}] build slot acquired", flush=True)

    def _trim():
        gc.collect()
        import ctypes
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    group = None
    if world > 1:
        import torch.distributed as dist
        group = dist.group.WORLD

    print(f"[rank {rank}] building student (mmap base ckpt) ...", flush=True)
    student = build_lm_model(args, dtype, args.attn_impl)

    # --- TP sharding BEFORE PTQR replacement ---------------------------------
    # Slicing the mmap-backed weights materializes ONLY each rank's shard per
    # module, so no rank ever holds a full anon copy of the model.
    apply_ssm_tp(student.model, tp_size=world if world > 1 else None)
    apply_tp_attention(student.model, tp_size=world if world > 1 else None)
    inter = vocab_range = None
    if world > 1:
        inter, vocab_range = shard_mlp_and_heads_one(student, world, rank, group=group)
        print(f"[rank {rank}] MLP sharded on intermediate {inter}; "
              f"lm_head vocab shard {vocab_range}", flush=True)

    print(f"[rank {rank}] replacing linears with PTQR (G{args.weight_group}) ...", flush=True)
    if args.no_ptqr:
        replaced = {}
        n_kv = 0
        print(f"[rank {rank}] --no_ptqr: student left unquantized (control run)",
              flush=True)
    else:
        replaced = replace_linears_with_ptqr(student, group=args.weight_group)
        n_kv = attach_kv_fake_quant(student, args.kv_group)
    print(f"[rank {rank}] {len(replaced)} PTQR linears, {n_kv} attention layers "
          f"with KV fake-quant g{args.kv_group}", flush=True)

    # Row-parallel reduce for the student's MLP down_projs (their plain-Linear
    # hooks vanished with replacement) + lm_head vocab base for the loss.
    if world > 1:
        for name, mod in student.named_modules():
            if isinstance(mod, PTQRLinear) and name.endswith("mlp.down_proj"):
                mod.row_reduce_group = group
        student.lm_head.tp_vocab_start = vocab_range[0]

    # Only the PTQR masters + group scales train; everything else (embeddings,
    # norms, biases, conv1d, A_log/dt_bias) is frozen at the reference values.
    for p_ in student.parameters():
        p_.requires_grad_(False)
    for m_ in replaced.values():
        m_.weight.requires_grad_(True)
        m_.scale.requires_grad_(True)

    student.to(device)
    fake_quant_embedding_(student.model.embed_tokens)
    _trim()
    _mem("student on GPU", rank)
    print(f"[rank {rank}] student on GPU; building teacher ...", flush=True)

    teacher = build_lm_model(args, dtype, args.attn_impl)
    apply_ssm_tp(teacher.model, tp_size=world if world > 1 else None)
    apply_tp_attention(teacher.model, tp_size=world if world > 1 else None)
    if world > 1:
        shard_mlp_and_heads_one(teacher, world, rank, group=group)
    # Teacher shares the (frozen, identical) embedding table — saves ~2 GiB
    # per rank and is exact: neither side ever updates it. The student's is
    # already on GPU.
    teacher.model.embed_tokens = student.model.embed_tokens
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    teacher.eval()
    teacher.to(device)
    _trim()
    _mem("teacher on GPU", rank)

    if world > 1:
        # Release reserved-but-unallocated VRAM so NCCL's first-collective
        # calloc (~6 MB outside the caching allocator) cannot fail against a
        # ~31 GiB reserved pool (measured: ncclUnhandledCudaError at barrier).
        torch.cuda.empty_cache()
        open(done_flag, "w").write("x")
        torch.distributed.barrier()

    # This HF impl never calls _gradient_checkpointing_func, so wrap the
    # decoder layers ourselves (full retention is ~5-6 GiB at 4k tokens).
    if args.gradient_checkpointing:
        n_ckpt = wrap_decoder_checkpointing(student)
        print(f"[rank {rank}] decoder layers checkpoint-wrapped: {n_ckpt}", flush=True)

    # --- optimizer -----------------------------------------------------------
    # Weights AND scales: per-tensor SGD via post-accumulate hooks — each
    # grad is applied (clipped) and freed as it lands, so peak grad memory is
    # one tensor, not the full ~11 GiB set. (Adafactor NaN'd scales on its
    # first real update — measured, see probe logs.)
    n_hook = attach_weight_sgd_hooks(
        replaced, args.lr, args.lr * args.scale_lr_mult) if replaced else 0
    s_params = [m.scale for m in replaced.values()]
    opt = None
    print(f"[rank {rank}] optimizer: {n_hook} per-tensor SGD hooks "
          f"(w lr {args.lr:g}, s lr {args.lr * args.scale_lr_mult:g})", flush=True)

    train_dl = build_dataloader(str(Path(args.data_dir) / args.data_name),
                                args.seq_len, args.batch_size, tiny=args.tiny)
    val_dl = build_dataloader(str(Path(args.data_dir) / args.val_name),
                              args.seq_len, 1, tiny=args.tiny)
    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    kv_state_refs = [m._ptqr_kv_state for m in student.model.modules()
                     if hasattr(m, "_ptqr_kv_state")]
    student.train()
    if args.no_ptqr:
        # Control run: nothing requires grad, so skip training entirely —
        # one held-out eval of student-vs-teacher (identical unquantized
        # models) measures the infra's true CE / zero-KLD sanity.
        if rank == 0 and os.environ.get("PTQR_MEM_DEBUG"):

            def _probe(name):
                def h(mod, inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    if torch.is_tensor(hs):
                        print(f"[probe] {name}: |h| mean {hs.float().abs().mean().item():.4f}",
                              flush=True)
                return h
            student.model.embed_tokens.register_forward_hook(_probe("embed"))
            for i, layer in enumerate(student.model.layers):
                layer.register_forward_hook(_probe(f"L{i:02d}"))
            student.lm_head.register_forward_hook(_probe("lm_head"))
        ev_lm, ev_kl = eval_kld(student, teacher, val_dl, max(4, args.eval_steps),
                                args.logits_chunk, args.distill_temperature)
        print(f"[control] unquantized student vs teacher: "
              f"CE {ev_lm:.4f}, KLD/tok {ev_kl:.6f} (expect ~0)", flush=True)
        if world > 1:
            torch.distributed.destroy_process_group()
        print("[done]", flush=True)
        return
    step = 0
    for x, y in itertools.islice(train_dl, args.max_steps):
        # Identical RNG state on every rank: the dither draws (act/weight/KV
        # quant exploration) then match across TP ranks, so the noisy forward
        # is the same objective everywhere (SDGraft's TP routing-seed rule).
        torch.manual_seed(1234 + step)
        tau = anneal_tau(step, args.max_steps, args.temp_start, args.temp_end)
        for m in replaced.values():
            m.tau = tau
        for st in kv_state_refs:
            st["tau"] = tau

        x, y = x.to(device), y.to(device)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        if world > 1:
            lm_loss, kl_loss = tp_vocab_parallel_losses(
                student, teacher, x, y, args.distill_temperature,
                args.logits_chunk, args.distill_weight)
        else:
            lm_loss, kl_loss = chunked_forward_loss(
                student, teacher, x, y, args.distill_temperature,
                args.logits_chunk, args.distill_weight)
        # the loss functions ran backward internally; grads are in place.
        if opt is not None:
            torch.nn.utils.clip_grad_norm_(s_params, args.grad_clip)
            opt.step()
            bad = [i for i, p in enumerate(s_params) if not torch.isfinite(p).all()]
            if bad or (step % 10 == 0 and rank == 0):
                import math as _math
                allf = torch.cat([p.reshape(-1).float() for p in s_params])
                print(f"  [scale-health] step {step}: nonfinite scales {len(bad)}, "
                      f"scale min {allf.min().item():.3e} max {allf.max().item():.3e}",
                      flush=True)

        if rank == 0:
            print(f"step {step} | tau {tau:.4f} | CE {lm_loss:.4f} | "
                  f"KLD {kl_loss:.6f} | KLD/token {kl_loss:.6f}", flush=True)

        if args.eval_every and step > 0 and step % args.eval_every == 0:
            # collective (vocab-parallel LSE): every rank must enter together
            for m in replaced.values():
                m.tau = 0.0
            for st in kv_state_refs:
                st["tau"] = 0.0
            ev_lm, ev_kl = eval_kld(student, teacher, val_dl, args.eval_steps,
                                    args.logits_chunk, args.distill_temperature)
            if rank == 0:
                print(f"[eval] step {step} | val CE {ev_lm:.4f} | val KLD/tok {ev_kl:.6f}",
                      flush=True)
            for m in replaced.values():
                m.tau = tau
            for st in kv_state_refs:
                st["tau"] = tau

        if args.save_every and step > 0 and step % args.save_every == 0:
            # every rank saves ITS shard (the exporter stitches all four)
            save_ptqr_checkpoint(student, replaced, out_dir, step, args, rank, world)
        step += 1

    if rank == 0:
        save_ptqr_checkpoint(student, replaced, out_dir, args.max_steps, args, rank, world)
    if world > 1:
        torch.distributed.destroy_process_group()
    print("[done]", flush=True)


def save_ptqr_checkpoint(model, replaced, out_dir: Path, step: int, args,
                         rank: int = 0, world: int = 1):
    """Save this rank's shard of masters + scales (stripped-key form).

    Under TP each rank holds only its slices; the export step (Phase 3)
    stitches using the deterministic shard geometry (attention/GDN adapters +
    the MLP/vocab ranges recomputed from config). The vocab/intermediate
    ranges are recorded here so stitching never has to guess.
    """
    sd = model.state_dict()
    remap = { (k[len("model."):] if k.startswith("model.") else k): v
              for k, v in sd.items() }
    vocab = model.lm_head.weight.shape[0]
    ckpt = {"model_state_dict": remap, "step": step, "rank": rank, "world": world,
            "vocab_range": _vocab_ranges(vocab, world)[rank] if world > 1 else (0, vocab),
            "args": {a: getattr(args, a) for a in
                     ("weight_group", "kv_group", "seq_len", "max_steps",
                      "lr", "distill_weight", "temp_start", "temp_end")}}
    suffix = f"_rank{rank}" if world > 1 else ""
    path = out_dir / f"ptqr_target_step{step}{suffix}.pt"
    torch.save(ckpt, path)
    print(f"  [save] {path}", flush=True)
    # scales live inside PTQRLinear modules; also emit them keyed by tensor name
    scales = {}
    for name, mod in model.named_modules():
        if isinstance(mod, PTQRLinear):
            scales[name + ".scale"] = mod.scale.detach().cpu()
    torch.save(scales, out_dir / f"ptqr_target_scales_step{step}{suffix}.pt")


if __name__ == "__main__":
    main()
