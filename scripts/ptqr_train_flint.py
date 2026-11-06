"""PTQR retraining of RecursiveSI/flint (target) for the native-int8 MI100 stack.

Thin trainer over the generic machinery in scripts/ptqr_train_target.py
(PTQRLinear / replace_linears_with_ptqr / SR-SGD hooks / tau anneal /
token dataloader / DAgger rollouts) with the Flint reference model from
scripts/ptqr_flint_model.py (Flint has no transformers modeling code; the
teacher/student tree is FlintRef).

Student deployment contract mirrored exactly:
* every Linear (qkv/o, dense+shared MLP, routed experts, router EXCLUDED,
  lm_head INCLUDED per VLLM_GFX908_INT8_LM_HEAD) -> PTQRLinear
  (int8 RTN G128 weights + per-token int8 activations, annealed dither)
* embedding int8 fake-quant (fake_quant_embedding_)
* KV cache int8_block_g{G} per (token, kv_head): k/v quantized post-rope
  with the exact reshape_and_cache_g8 semantics (kv_block_quant_fake)
* norms, per-head QK RMSNorm, attention sinks, router: untouched (bf16;
  the measured exceptions for this stack)
* teacher: frozen full-BF16 FlintRef

Single-process trainer (teacher+student fit on 4x MI100 via expert/head
sharding in the TP port; --tiny runs the wiring smoke on CPU/GPU0).
Usage:
  .venv/bin/python scripts/ptqr_train_flint.py --tiny --max_steps 2
  (real runs: torchrun TP4 -- data from SDGraft-style token files or
   Flint-chat-templated agentic corpora; see flint-int8-recipe.md)
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ptqr_flint_model import (  # noqa: E402
    FlintRef,
    flint_tiny_config,
    load_flint_config,
)
from ptqr_train_target import (  # noqa: E402
    PTQRLinear,
    anneal_tau,
    apply_deferred_updates,
    attach_weight_sgd_hooks,
    build_dataloader,
    fake_quant_embedding_,
    kv_block_quant_fake,
    replace_linears_with_ptqr,
)

# Modules never quantized (bf16 exceptions per Qwen/Bolt findings + norms):
# router (gating stays dense). QK RMSNorm weights and sinks are parameters,
# not linears, so they are naturally untouched.
_SKIP_PATTERNS = (".mlp.router",)


def _collect_skip_names(model: torch.nn.Module) -> set[str]:
    """Names of linears that stay bf16 (router/gating dense; QK norms are not
    linears; sinks are parameters)."""
    skip = set()
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            full = f"{name}.{child_name}" if name else child_name
            if any(p in full for p in _SKIP_PATTERNS):
                skip.add(full)
    return skip


def build_student(teacher: FlintRef, group: int, kv_group: int) -> FlintRef:
    student = copy.deepcopy(teacher)
    replaced = replace_linears_with_ptqr(
        student, group=group, skip=_collect_skip_names(student)
    )
    fake_quant_embedding_(student.embed)
    for m in student.modules():
        if m.__class__.__name__ == "AttentionRef":
            m.kv_fake_quant_group = kv_group
    return student, replaced


def _apply_kv_fake(att, k: torch.Tensor) -> torch.Tensor:
    g = getattr(att, "kv_fake_quant_group", None)
    if g is None:
        return k
    return kv_block_quant_fake(k, group=g)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/curved/models/flint-bf16")
    p.add_argument("--data_dir", default="/home/curved/SDGraft/data")
    p.add_argument("--data_name", default="flint_longctx/train_tokens.pt")
    p.add_argument("--val_name", default="flint_longctx/val_tokens.pt")
    p.add_argument("--out_dir", default="/home/curved/models/ptqr_flint_out")
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
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=10)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--n_experts_keep", type=int, default=0,
                   help="smoke only: keep the first N experts per MoE layer "
                        "(0 = all). Validates real-checkpoint wiring without "
                        "the full 37.5B expert mass.")
    p.add_argument("--no_ptqr", action="store_true",
                   help="unquantized student control (teacher-vs-teacher)")
    args = p.parse_args()

    torch.manual_seed(0)
    device = "cuda:0" if torch.cuda.is_available() and not args.tiny else "cpu"

    # ---- TP4 launch (torchrun): shard teacher from the streamed checkpoint
    # and shard the PTQR-swapped student in place.
    import os as _os
    world = int(_os.environ.get("WORLD_SIZE", "1"))
    rank = int(_os.environ.get("RANK", "0"))
    local_rank = int(_os.environ.get("LOCAL_RANK", "0"))
    tp_group = None
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl" if device != "cpu" else "gloo")
        tp_group = dist.group.WORLD
        device = f"cuda:{local_rank}" if device != "cpu" else device
        torch.cuda.set_device(local_rank) if device != "cpu" else None

    if args.tiny:
        cfg = flint_tiny_config()
        cfg.attention_bias = False
        teacher = FlintRef(cfg, device=device).to(device)
        for prm in teacher.parameters():
            torch.nn.init.normal_(prm, std=0.02)
    else:
        cfg = load_flint_config(args.model)
        if args.n_experts_keep:
            cfg.num_experts = args.n_experts_keep
            cfg.num_experts_per_tok = min(cfg.num_experts_per_tok,
                                          args.n_experts_keep)
        teacher = FlintRef(cfg, device="cpu")
        from ptqr_flint_model import load_flint_shards, shard_flint_tp
        if world > 1:
            # stream (name-map + expert EP assign); build the student from
            # the UNSHARDED teacher, then shard each separately
            load_flint_shards(teacher, args.model, world, rank)
            teacher.to(device)
        else:
            from ptqr_flint_model import load_flint_state
            load_flint_state(teacher, args.model)
            teacher.to(device)
    teacher.eval()
    for prm in teacher.parameters():
        prm.requires_grad_(False)

    from ptqr_flint_model import shard_flint_tp as _shard
    if args.no_ptqr:
        student, replaced = copy.deepcopy(teacher), {}
    else:
        student, replaced = build_student(teacher, args.weight_group,
                                          args.kv_group)
    if world > 1:
        _shard(teacher, world, rank, group=tp_group)
        if not args.no_ptqr:
            _shard(student, world, rank, group=tp_group)
    student.to(device)

    n_lin = len(replaced)
    print(f"student: {n_lin} PTQRLinear surfaces, kv_group={args.kv_group}, "
          f"device={device}")

    if replaced:
        attach_weight_sgd_hooks(replaced, lr=args.lr,
                                scale_lr=args.lr * args.scale_lr_mult)

    data_path = os.path.join(args.data_dir, args.data_name)
    val_path = os.path.join(args.data_dir, args.val_name)
    train_iter = build_dataloader(data_path, args.seq_len, args.batch_size,
                                  tiny=args.tiny, tiny_vocab=teacher.cfg.vocab_size)
    val_iter = build_dataloader(val_path, args.seq_len, args.batch_size,
                                tiny=args.tiny, tiny_vocab=teacher.cfg.vocab_size) \
        if os.path.exists(val_path) or args.tiny else None

    def one_batch(x, y, tau):
        if replaced:
            for m in replaced.values():
                m.tau = tau
        logits_s = student(x)
        with torch.no_grad():
            logits_t = teacher(x)
        ce = F.cross_entropy(
            logits_s.float().reshape(-1, logits_s.shape[-1]), y.reshape(-1))
        T = args.distill_temperature
        lp_s = F.log_softmax(logits_s.float() / T, dim=-1)
        p_t = F.softmax(logits_t.float() / T, dim=-1)
        kd = F.kl_div(
            lp_s.reshape(-1, lp_s.shape[-1]), p_t.reshape(-1, p_t.shape[-1]),
            reduction="batchmean") * (T * T)
        loss = ce + args.distill_weight * kd
        return loss, ce.detach(), kd.detach()

    step = 0
    while step < args.max_steps:
        for x, y in train_iter:
            if step >= args.max_steps:
                break
            tau = anneal_tau(step, args.max_steps, args.temp_start,
                             args.temp_end)
            loss, ce, kd = one_batch(x.to(device), y.to(device), tau)
            loss.backward()
            if replaced:
                apply_deferred_updates(student)
            if step % 10 == 0:
                print(f"step {step:4d} tau {tau:.4f} loss {loss.item():.4f} "
                      f"ce {ce.item():.4f} kd {kd.item():.4f}", flush=True)
            step += 1
            student.zero_grad(set_to_none=True)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({"step": step, "cfg": vars(teacher.cfg) if not args.tiny else {}},
               os.path.join(args.out_dir, "train_meta.pt"))
    print("PTQR flint trainer run complete ->", args.out_dir)


if __name__ == "__main__":
    main()
