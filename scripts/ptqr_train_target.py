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
                             chunk: int, distill_weight: float,
                             pos_mask=None, t_hidden=None):
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
    if t_hidden is None:
        with torch.no_grad():
            t_hidden = teacher.model(input_ids=x).last_hidden_state
    _mem("teacher hidden done", 0)
    # collapse eval/backward fragmentation BEFORE the chunk loop (the peak
    # phase); without this the caching allocator held ~1 GiB of split blocks
    # and chunk-0 backward OOM'd with 0 bytes free (measured)
    torch.cuda.empty_cache()

    H = hidden.shape[-1]
    s_flat = hidden.reshape(-1, H)
    t_flat = t_hidden.reshape(-1, H)
    y_flat = y.reshape(-1)
    n_tok = s_flat.shape[0]
    # rollout mode: loss only on student-generated (committed) positions
    mask_flat = None
    n_norm = float(n_tok)
    if pos_mask is not None:
        mask_flat = pos_mask.reshape(-1).to(s_flat.device)
        n_norm = max(1.0, mask_flat.sum().item())
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
        ce_rows = lse - torch.where(in_shard, picked, torch.zeros_like(picked))
        if mask_flat is not None:
            ce_rows = ce_rows * mask_flat[i: i + chunk]
        ce_sum_c = ce_rows.sum()

        with torch.no_grad():
            t_logits = t_head(t_flat[i: i + chunk]).float()
            t_lse = _global_lse(t_logits, differentiable=False)
            t_logp = t_logits - t_lse.unsqueeze(-1)
            t_prob = t_logp.exp()
        s_logp = logits_c - lse.unsqueeze(-1)
        kl_rows = (t_prob * (t_logp - s_logp)).sum(-1) * (T * T)
        if mask_flat is not None:
            kl_rows = kl_rows * mask_flat[i: i + chunk]
        kl_c = kl_rows.sum()

        ((ce_sum_c + distill_weight * kl_c) / n_norm).backward()
        with torch.no_grad():
            h_grad[i: i + chunk] = h_c.grad if h_c.grad is not None else torch.zeros_like(h_c)
        ce_val_c = (lse * inv_world - torch.where(in_shard, picked, torch.zeros_like(picked)))
        if mask_flat is not None:
            ce_val_c = ce_val_c * mask_flat[i: i + chunk]
        lm_total += ce_val_c.sum().item() / n_norm
        kl_total += kl_c.item() / n_norm
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
    for i, item in enumerate(itertools.islice(val_iter, steps)):
        if len(item) == 3:  # rollout sample: (x, y, pos_mask)
            x, y, mask = item
            x, y, mask = x.cuda(), y.cuda(), mask.cuda()
        else:
            x, y = item
            mask = None
            x, y = x.cuda(), y.cuda()
        s_h = student.model(input_ids=x).last_hidden_state
        t_h = teacher.model(input_ids=x).last_hidden_state
        H = s_h.shape[-1]
        s_flat, t_flat = s_h.reshape(-1, H), t_h.reshape(-1, H)
        y_flat = y.reshape(-1)
        mask_flat = mask.reshape(-1) if mask is not None else None
        n_tok = s_flat.shape[0]
        nrm = mask_flat.sum().item() if mask_flat is not None else n_tok
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
            kl_rows = (t_prob * (t_logp - s_logp)).sum(-1)
            y_local = y_flat[j: j + chunk] - s_v0
            in_shard = (y_local >= 0) & (y_local < s_lc.shape[-1])
            y_safe = y_local.clamp(0, s_lc.shape[-1] - 1)
            picked = s_lc.gather(1, y_safe.unsqueeze(1)).squeeze(1)
            # every rank owns lse/world of the denominator; only the target's
            # owner carries the -logit term (exact after the cross-rank sum)
            lm_rows = s_lse / world - torch.where(in_shard, picked, torch.zeros_like(picked))
            if mask_flat is not None:
                m_c = mask_flat[j: j + chunk]
                kl_rows = kl_rows * m_c
                lm_rows = lm_rows * m_c
            kl_step += kl_rows.sum().item()
            lm_step += lm_rows.sum().item()
            del s_lc, t_lc, s_logp, t_logp, t_prob
        kl_sum += kl_step / max(1.0, nrm)
        lm_sum += lm_step / max(1.0, nrm)
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
                         chunk: int, distill_weight: float, pos_mask=None):
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
    mask_flat = None
    n_norm = float(n_tok)
    if pos_mask is not None:
        mask_flat = pos_mask.reshape(-1).to(s_flat.device)
        n_norm = max(1.0, mask_flat.sum().item())

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
        ce_rows = F.cross_entropy(logits_c, y_c, reduction="none")
        student_log_probs = F.log_softmax(logits_c / T, dim=-1)
        del logits_c
        with torch.no_grad():
            t_logits_c = teacher_head(t_flat[i: i + chunk].to(teacher_head.weight.device)).to(head_dev)
            teacher_probs = F.softmax(t_logits_c / T, dim=-1)
            del t_logits_c
        kl_rows = F.kl_div(student_log_probs, teacher_probs,
                           reduction="none").sum(-1) * (T * T)
        if mask_flat is not None:
            m_c = mask_flat[i: i + chunk]
            ce_rows = ce_rows * m_c
            kl_rows = kl_rows * m_c
        lm_c, kl_c = ce_rows.sum(), kl_rows.sum()
        del student_log_probs, teacher_probs
        ((lm_c + distill_weight * kl_c) / n_norm).backward()
        with torch.no_grad():
            g = h_c.grad if h_c.grad is not None else torch.zeros_like(h_c)
            h_grad[i: i + chunk] = g
        lm_total += lm_c.item() / n_norm
        kl_total += kl_c.item() / n_norm
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


def _sr_sgd_step_(p, g, lr: float) -> None:
    """One stochastic-rounding SGD step on a bf16 master (chunked).

    Plain add_ rounds sub-ULP updates away (lr*g ~ 1e-7 vs ULP ~ 4e-5),
    turning SGD into a rounding-driven random walk — the measured rising-KLD
    mechanism. SR keeps the expected update exact at zero extra memory.
    Chunked: whole-tensor fp32 passes are 1.19 GiB on the lm_head shard
    (measured OOM inside the chunk-0 backward). 8M-element chunks keep the
    fp32 transient at ~32 MiB; smaller chunks are kernel-launch-bound (R=2048
    measured ~20 min/step across the stack).
    """
    import torch as _t
    with _t.no_grad():
        R = 8_000_000
        flat = p.data.reshape(-1)
        gflat = g.reshape(-1)
        for r0 in range(0, flat.numel(), R):
            r1 = min(r0 + R, flat.numel())
            x = flat[r0:r1].float() - lr * gflat[r0:r1].float()
            sign = _t.where(x < 0, -1.0, 1.0)
            ax = x.abs().clamp_min(1e-38)
            spacing = _t.pow(2.0, _t.floor(_t.log2(ax)) - 7)
            rr = ax / spacing
            frac = rr - _t.floor(rr)
            stepped = _t.floor(rr) + (_t.rand_like(frac) < frac).float()
            flat[r0:r1] = (sign * stepped * spacing).to(p.dtype)


def _apply_param_update_(p, g, lr: float, is_scale: bool) -> None:
    import torch as _t
    if not _t.isfinite(g).all():
        return
    with _t.no_grad():
        gn = g.norm()
        if gn > 1.0:  # per-tensor clip (no global clip under hooks)
            g = g * (1.0 / gn)
        if p.dtype == _t.bfloat16:
            _sr_sgd_step_(p, g, lr)
        else:
            p.data.add_(g.to(p.dtype), alpha=-lr)
        if is_scale:
            p.data.clamp_(1e-7, 65500.0)  # fp16-representable, positive


def attach_weight_sgd_hooks(replaced: dict, lr: float, scale_lr: float,
                             deferred: set | None = None) -> int:
    """Per-tensor SGD on PTQR masters + scales, applied and freed the moment
    each grad lands (post-accumulate hook). Keeps peak grad memory at ONE
    tensor instead of the full 11 GiB set. (Adafactor was tried first and its
    initial update NaN'd a scale tensor on real data — measured; plain SGD is
    also exactly composable with the per-chunk loss backwards, being linear
    in the grad.)

    Params named in `deferred` never apply in-hook: the lm_head receives 16
    partial grads per step (one per loss chunk) and applying per chunk made
    SR 16x as expensive (a step took 20 min — measured); those params
    accumulate and are applied once by apply_deferred_updates() after the
    loss (SGD is linear in the grad, so this is exact)."""
    import torch as _t

    deferred = deferred or set()
    n = 0
    for m in replaced.values():
        for p, p_lr, is_scale in ((m.weight, lr, False), (m.scale, scale_lr, True)):
            p._ptqr_lr = p_lr
            p._ptqr_is_scale = is_scale
            p.requires_grad_(True)
            if getattr(p, "_ptqr_deferred", False):
                n += 1
                continue

            def _hook(p=p, p_lr=p_lr, is_scale=is_scale):
                if p.grad is None:
                    return
                _apply_param_update_(p, p.grad, p_lr, is_scale)
                p.grad = None
            p.register_post_accumulate_grad_hook(lambda *a, h=_hook: h())
            p.requires_grad_(True)
            n += 1
    return n


def apply_deferred_updates(model) -> int:
    """Apply + clear the accumulated grads of deferred (lm_head) params."""
    n = 0
    for p in model.parameters():
        if getattr(p, "_ptqr_deferred", False) and p.grad is not None:
            _apply_param_update_(p, p.grad, p._ptqr_lr, p._ptqr_is_scale)
            p.grad = None
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


class _RolloutDS(torch.utils.data.Dataset):
    """DAgger-style rollout transcripts: (ctx tail + committed) sequences.

    Loss lives only on the committed (student-generated) region; the prompt is
    truncated to the last --ctx_window tokens (checkpoint-boundary memory makes
    full 20k+ contexts infeasible next to two models; the Path B ranking showed
    per-state int8 noise is context-length-independent, so the truncated-window
    states carry the same corrective signal).
    """

    def __init__(self, items, ctx_window: int):
        self.items = items          # list[(prompt_ids(list), committed(list), tag)]
        self.ctx = ctx_window

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        prompt, committed, _tag = self.items[i]
        ctx = list(prompt[-self.ctx:])
        seq = ctx + list(committed)
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        mask = torch.zeros_like(y)
        mask[len(ctx) - 1:] = 1     # y[j] predicts seq[j+1]; train iff committed
        return x, y, mask


def build_rollout_loaders(rollout_dir: str, ctx_window: int, val_n: int):
    import glob as _glob
    items = []
    for f in sorted(_glob.glob(os.path.join(rollout_dir, "*_committed.pt"))):
        tag = os.path.basename(f)[: -len("_committed.pt")]
        ids_f = os.path.join(rollout_dir, f"{tag}_ids.pt")
        if not os.path.exists(ids_f):
            print(f"  [rollout] skip {tag}: no ids file", flush=True)
            continue
        prompt = torch.load(ids_f, weights_only=False)["prompt_ids"]
        committed = torch.load(f, weights_only=False)["committed_ids"]
        if len(committed) < 64:
            print(f"  [rollout] skip {tag}: only {len(committed)} committed", flush=True)
            continue
        items.append((list(prompt), list(committed), tag))
    if len(items) < val_n + 1:
        raise RuntimeError(f"only {len(items)} usable rollout legs in {rollout_dir}")
    val_items = items[-val_n:] if val_n else []
    train_items = items[: len(items) - (val_n if val_n else 0)]
    train_dl = torch.utils.data.DataLoader(_RolloutDS(train_items, ctx_window),
                                           batch_size=1, shuffle=False, num_workers=0)
    val_dl = torch.utils.data.DataLoader(_RolloutDS(val_items, ctx_window),
                                         batch_size=1, shuffle=False, num_workers=0)
    print(f"  [rollout] {len(train_items)} train legs, {len(val_items)} val legs "
          f"(ctx_window {ctx_window})", flush=True)
    return train_dl, val_dl


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
    p.add_argument("--rollout_dir", default=None,
                   help="DAgger rollouts: dir of {tag}_ids.pt + {tag}_committed.pt "
                        "legs; replaces the corpus train set (loss masked to "
                        "the committed region; corpus KLD still eval'd)")
    p.add_argument("--ctx_window", type=int, default=4096,
                   help="rollout prompt truncation window (memory-bound)")
    p.add_argument("--rollout_val_n", type=int, default=4,
                   help="last N legs held out for rollout-val KLD")
    p.add_argument("--init_ckpt_dir", default=None,
                   help="continue from a prior run's per-rank shards "
                        "(requires --init_step)")
    p.add_argument("--init_step", type=int, default=60)
    p.add_argument("--teacher_body_offload", action="store_true",
                   help="cycle the teacher's decoder body GPU<->host per step "
                        "(frees ~14 GiB during the student backward; the "
                        "teacher lm_head stays resident; ~1.4 s/step "
                        "transfer overhead — needed for rollout-length "
                        "sequences next to two 27B bodies on 32 GiB)")
    args = p.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        import torch.distributed as dist
        import datetime as _dt
        # the serialized chain build + overlay can leave a rank inside a
        # barrier for >10 min (rank 3 swap-squeezes ~20 min behind rank 0 —
        # measured watchdog abort at the default 600 s)
        dist.init_process_group(backend="nccl",
                                timeout=_dt.timedelta(hours=2))
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

    def _pd(tag: str) -> None:
        try:
            with open("/proc/self/smaps_rollup") as fh:
                d = {}
                for line in fh:
                    if line.startswith(("Rss", "Private_Dirty", "Swap")):
                        k, v = line.split(":")
                        d[k] = int(v.split()[0]) / 2**20
            print(f"[rank {rank}][mem] {tag}: rss {d.get('Rss', 0):.1f} GiB "
                  f"priv-dirty {d.get('Private_Dirty', 0):.1f} GiB "
                  f"swapped {d.get('Swap', 0):.1f} GiB", flush=True)
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

    # rung-2 continuation happens after student.to(device) (see below) — the
    # serialized build slot is host-RAM bound (measured wedge at rank 2 when
    # the 15.5 GiB init mmap was faulted pre-.to and its storages retained).

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
    _pd("student on GPU")
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
    _pd("teacher on GPU")

    if world > 1:
        # Release reserved-but-unallocated VRAM so NCCL's first-collective
        # calloc (~6 MB outside the caching allocator) cannot fail against a
        # ~31 GiB reserved pool (measured: ncclUnhandledCudaError at barrier).
        torch.cuda.empty_cache()
        open(done_flag, "w").write("x")
        torch.distributed.barrier()

    # rung-2 continuation: overlay a prior run's per-rank masters+scales onto
    # the GPU-resident student. Runs AFTER all ranks finished building (the
    # build window is host-RAM critical — overlaying inside it segfaulted
    # rank 2/3 under pressure; measured twice). Serialized by a second chain:
    # each overlay's transient mmap faults stay file-backed and evictable,
    # and only one rank faults at a time.
    if args.init_ckpt_dir:
        import re as _re
        if world > 1:
            import time as _t2
            if rank > 0:
                while not os.path.exists(f"/tmp/ptqr_ov_{ppid}_{rank - 1}.done"):
                    _t2.sleep(2)
        _pd("pre-overlay")
        init_path = (Path(args.init_ckpt_dir) /
                     f"ptqr_target_step{args.init_step}_rank{rank}.pt")
        if not init_path.exists() and world == 1:
            init_path = (Path(args.init_ckpt_dir) /
                         f"ptqr_target_step{args.init_step}.pt")
        ck = torch.load(init_path, map_location="cpu", mmap=True,
                        weights_only=True)
        live = dict(student.named_parameters())
        live.update(dict(student.named_buffers()))
        # Pageable->GPU copies stage through torch's PINNED host cache, which
        # never shrinks (no host_empty_cache in this build) — a direct
        # .to(device) per tensor retained the full 15.5 GiB shard per rank
        # (measured +14.6 GiB priv-dirty, host-exhaustion segfaults). Route
        # every copy through ONE reused pinned scratch so the cache stays
        # scratch-sized regardless of checkpoint size.
        SCR_BYTES = 256 << 20
        scr_by_dtype: dict[torch.dtype, torch.Tensor] = {}

        def _h2d(p, v):
            dt = v.dtype
            s = scr_by_dtype.get(dt)
            if s is None:
                s = torch.empty(SCR_BYTES // dt.itemsize, dtype=dt,
                                pin_memory=True)
                scr_by_dtype[dt] = s
            fv, fp = v.reshape(-1), p.reshape(-1)
            nel = fv.numel()
            step = s.numel()
            for off in range(0, nel, step):
                n = min(step, nel - off)
                s[:n].copy_(fv[off:off + n])
                fp[off:off + n].copy_(s[:n])

        n_load, n_bad = 0, []
        with torch.no_grad():
            for k, v in ck["model_state_dict"].items():
                tk = k if k.startswith("lm_head") else "model." + k
                # R10 shards were saved with the gradient-checkpoint wrapper
                # installed: layers.N.layer.* -> layers.N.*
                tk = _re.sub(r"^(model\.layers\.\d+)\.layer\.", r"\1.", tk)
                p = live.get(tk)
                if p is None:
                    n_bad.append(tk)
                    continue
                if v.dtype == p.dtype and v.is_contiguous() and p.is_contiguous():
                    _h2d(p, v)
                else:
                    p.copy_(v.to(p.device, p.dtype))
                n_load += 1
        del live, ck, scr_by_dtype
        _trim()
        _pd("post-overlay")
        if world > 1:
            open(f"/tmp/ptqr_ov_{ppid}_{rank}.done", "w").write("x")
            torch.distributed.barrier()
        fatal = [k for k in n_bad
                 if k.endswith((".weight", ".scale"))
                 and (k.startswith("model.layers") or k.startswith("lm_head"))]
        print(f"[rank {rank}] init overlay from {init_path.name}: "
              f"{n_load} tensors copied, {len(n_bad)} unmatched", flush=True)
        if fatal:
            raise RuntimeError(f"init ckpt trainable tensors unmatched: {fatal[:6]}")

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
    # The lm_head's params take 16 partial grads per step (one per loss
    # chunk): defer them to a single post-loss apply (exact; SGD is linear).
    if replaced:  # only exists in PTQR mode (lm_head is a PTQRLinear)
        for p in (student.lm_head.weight, student.lm_head.scale):
            p._ptqr_deferred = True
    n_hook = attach_weight_sgd_hooks(
        replaced, args.lr, args.lr * args.scale_lr_mult) if replaced else 0
    s_params = [m.scale for m in replaced.values()]
    opt = None
    print(f"[rank {rank}] optimizer: {n_hook} per-tensor SGD hooks "
          f"(w lr {args.lr:g}, s lr {args.lr * args.scale_lr_mult:g}; "
          f"lm_head deferred)", flush=True)

    train_dl = build_dataloader(str(Path(args.data_dir) / args.data_name),
                                args.seq_len, args.batch_size, tiny=args.tiny)
    val_dl = build_dataloader(str(Path(args.data_dir) / args.val_name),
                              args.seq_len, 1, tiny=args.tiny)
    # deterministic eval: identical batches every eval (no shuffle) so KLD
    # points are directly comparable across steps and runs
    val_dl.shuffle = False
    rollout_train_dl = rollout_val_dl = None
    if args.rollout_dir:
        rollout_train_dl, rollout_val_dl = build_rollout_loaders(
            args.rollout_dir, args.ctx_window, args.rollout_val_n)
    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    kv_state_refs = [m._ptqr_kv_state for m in student.model.modules()
                     if hasattr(m, "_ptqr_kv_state")]

    # teacher-body offload helpers: the body layers swap GPU<->file-backed
    # mmap views of a per-rank scratch file. A plain GPU->host .to('cpu')
    # materializes 14 GiB of ANON per rank (4x = 56 GiB — host OOM-killed a
    # run); the teacher is frozen, so file pages are exact, evictable, and
    # ~free. The shared embedding table and the teacher lm_head never swap.
    t_params = dict(teacher.named_parameters())
    _T_SKIP = ("embed_tokens.", "lm_head.")
    _t_body_names = [n for n in t_params
                     if not any(s in n for s in _T_SKIP)]
    _t_mck = None
    if args.teacher_body_offload:
        scratch_path = (Path(args.out_dir) /
                        f"teacher_body_scratch_rank{rank}.pt")
        if world > 1:
            import time as _t3
            if rank > 0:
                while not os.path.exists(
                        f"{scratch_path.parent}/tbs_{rank - 1}.done"):
                    _t3.sleep(2)
        if not scratch_path.exists():
            _sd = {n: t_params[n].detach().cpu() for n in _t_body_names}
            torch.save(_sd, scratch_path)
            del _sd
            _trim()
        _t_mck = torch.load(scratch_path, map_location="cpu", mmap=True,
                            weights_only=True)
        assert set(_t_mck.keys()) == set(_t_body_names)
        if world > 1:
            open(f"{scratch_path.parent}/tbs_{rank}.done", "w").write("x")
        print(f"[rank {rank}] teacher body scratch: {scratch_path.name} "
              f"({len(_t_body_names)} tensors)", flush=True)

    def _teacher_body_off():
        if _t_mck is None:
            return
        for n in _t_body_names:
            t_params[n].data = _t_mck[n]     # drop GPU refs -> freed
        torch.cuda.empty_cache()

    _pin_scratch: dict[torch.dtype, torch.Tensor] = {}

    def _h2d_file(t):
        # mmap view -> GPU through ONE reused pinned scratch per dtype (a
        # direct .to(device) stages through the never-shrinking pinned host
        # cache — the +14.6 GiB priv-dirty mechanism, measured)
        dt = t.dtype
        s = _pin_scratch.get(dt)
        if s is None:
            s = torch.empty((64 << 20) // dt.itemsize, dtype=dt,
                            pin_memory=True)
            _pin_scratch[dt] = s
        g = torch.empty(t.shape, dtype=dt, device=device)
        fv, gv = t.reshape(-1), g.reshape(-1)
        for off in range(0, fv.numel(), s.numel()):
            n2 = min(s.numel(), fv.numel() - off)
            s[:n2].copy_(fv[off:off + n2])
            gv[off:off + n2].copy_(s[:n2])
        return g

    def _teacher_body_on():
        if _t_mck is None:
            return
        for n in _t_body_names:
            t_params[n].data = _h2d_file(_t_mck[n])

    def _teacher_forward(x):
        _teacher_body_on()
        with torch.no_grad():
            t_h = teacher.model(input_ids=x).last_hidden_state
        _teacher_body_off()
        return t_h

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

    def _cycle(dl):
        while True:
            for b in dl:
                yield b

    # rollout baseline BEFORE any update (rung-2 comparability)
    if args.rollout_dir and rollout_val_dl is not None:
        for m in replaced.values():
            m.tau = 0.0
        for st in kv_state_refs:
            st["tau"] = 0.0
        _teacher_body_on()
        _rlm, _rkl = eval_kld(student, teacher, rollout_val_dl,
                              args.rollout_val_n, args.logits_chunk,
                              args.distill_temperature)
        _teacher_body_off()
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"[eval-rollout] step 0 (init) | rollout KLD/tok {_rkl:.6f} "
                  f"| CE {_rlm:.4f}", flush=True)

    for batch in itertools.islice(_cycle(rollout_train_dl if args.rollout_dir
                                         else train_dl), args.max_steps):
        # Identical RNG state on every rank: the dither draws (act/weight/KV
        # quant exploration) then match across TP ranks, so the noisy forward
        # is the same objective everywhere (SDGraft's TP routing-seed rule).
        torch.manual_seed(1234 + step)
        tau = anneal_tau(step, args.max_steps, args.temp_start, args.temp_end)
        for m in replaced.values():
            m.tau = tau
        for st in kv_state_refs:
            st["tau"] = tau

        if len(batch) == 3:
            x, y, pos_mask = batch
            x, y = x.to(device), y.to(device)
            pos_mask = pos_mask.to(device)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            pos_mask = None
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        t_hidden = _teacher_forward(x) if args.teacher_body_offload else None
        if world > 1:
            lm_loss, kl_loss = tp_vocab_parallel_losses(
                student, teacher, x, y, args.distill_temperature,
                args.logits_chunk, args.distill_weight, pos_mask=pos_mask,
                t_hidden=t_hidden)
        else:
            lm_loss, kl_loss = chunked_forward_loss(
                student, teacher, x, y, args.distill_temperature,
                args.logits_chunk, args.distill_weight, pos_mask=pos_mask)
        # the loss functions ran backward internally; grads are in place.
        apply_deferred_updates(student)  # lm_head's 16 partials, one apply
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
            _teacher_body_on()
            ev_lm, ev_kl = eval_kld(student, teacher, val_dl, args.eval_steps,
                                    args.logits_chunk, args.distill_temperature)
            if args.rollout_dir and rollout_val_dl is not None:
                _rlm, _rkl = eval_kld(student, teacher, rollout_val_dl,
                                      args.rollout_val_n, args.logits_chunk,
                                      args.distill_temperature)
            _teacher_body_off()
            torch.cuda.empty_cache()
            if rank == 0:
                print(f"[eval] step {step} | val CE {ev_lm:.4f} | val KLD/tok {ev_kl:.6f}",
                      flush=True)
                if args.rollout_dir and rollout_val_dl is not None:
                    print(f"[eval-rollout] step {step} | rollout KLD/tok {_rkl:.6f} "
                          f"| CE {_rlm:.4f}", flush=True)
            for m in replaced.values():
                m.tau = tau
            for st in kv_state_refs:
                st["tau"] = tau

        if args.save_every and step > 0 and step % args.save_every == 0:
            # every rank saves ITS shard (the exporter stitches all four)
            save_ptqr_checkpoint(student, replaced, out_dir, step, args, rank, world)
        step += 1

    # final save on every rank (the interval save is step>0-based and short
    # runs never reach it — measured: a 5-step run saved rank 0 only)
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
    # keep only the LATEST shard per rank — 4-rank saves are ~30 GB per step
    # and historical steps filled the 936 GB disk mid-run (measured iostream
    # failure); retraining is deterministic enough to rely on the latest.
    for old in out_dir.glob(f"ptqr_target_step*{suffix}.pt"):
        if old != path:
            old.unlink(missing_ok=True)
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
