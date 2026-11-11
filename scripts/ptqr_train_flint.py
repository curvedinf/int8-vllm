"""PTQR retraining of RecursiveSI/flint (target) for the native-int8 MI100 stack.

Thin trainer over the generic machinery in scripts/ptqr_train_target.py
(PTQRLinear / replace_linears_with_ptqr / SR-SGD hooks / tau anneal /
token dataloader / DAgger rollouts) with the Flint reference model from
scripts/ptqr_flint_model.py (Flint has no transformers modeling code; the
teacher/student tree is FlintRef).

Student deployment contract mirrored exactly:
* every Linear (qkv/o, dense+shared MLP, routed experts, router and
  lm_head INCLUDED - both serve int8 and PTQR recovers their cold-quant
  KLD) -> PTQRLinear
  (int8 RTN G128 weights + per-token int8 activations, annealed dither)
* embedding int8 fake-quant (fake_quant_embedding_)
* KV cache int8_block_g{G} per (token, kv_head): k/v quantized post-rope
  with the exact reshape_and_cache_g8 semantics (kv_block_quant_fake)
* norms, per-head QK RMSNorm, attention sinks: untouched (bf16; the
  measured exceptions for this stack)
* teacher: frozen full-BF16 FlintRef

Single-process trainer (teacher+student fit on 4x MI100 via expert/head
sharding in the TP port; --tiny runs the wiring smoke on CPU/GPU0).
Usage:
  .venv/bin/python scripts/ptqr_train_flint.py --tiny --max_steps 2
  (real runs: torchrun TP4 -- data from SDGraft-style token files or
   Flint-chat-templated agentic corpora; see flint-int8-recipe.md)
"""

from __future__ import annotations

import os as _os0
with open(f"/proc/{_os0.getpid()}/status") as _f0:
    _r0 = [l.split()[1] for l in _f0 if l.startswith("VmRSS")][0]
print(f"[BOOT pid={_os0.getpid()} rank={_os0.environ.get('RANK','?')}] "
      f"rss={_r0}kB", flush=True)

import argparse
import faulthandler as _fh
import signal as _sig
try:
    _fh.register(_sig.SIGUSR1, all_threads=True)
except (AttributeError, ValueError):
    pass
import copy
import itertools
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_DBG_LOGITS_DONE = False

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
    save_ptqr_checkpoint,
)

# Modules never quantized. The router is INCLUDED: serving quantizes the
# gate to int8 GS128 and the retrain trains it on that grid (PTQR recovers
# the cold-quant KLD). QK RMSNorm weights and sinks are parameters, not
# linears, so they are naturally untouched.
_SKIP_PATTERNS = ()


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
    for prm in student.parameters():
        prm.requires_grad_(True)  # teacher may already be frozen
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


@torch.no_grad()
def eval_kld_flint(student, teacher, val_src, steps: int | None = None,
                   chunk: int = 256):
    """Held-out KLD vs the BF16 teacher at tau=0 (the judge metric).

    val_src: iterable of (x, y) with a LIVE teacher, or the KD-cache list
    [((x, y), logits_path)] with precomputed fp16 teacher logits.
    Chunked over tokens (vocab 150144 makes a full [S, V] fp32 pair
    ~2.4 GB; chunks keep the transient at ~0.6 GB).
    """
    taus = [(m, m.tau) for m in student.modules()
            if m.__class__.__name__ == "PTQRLinear"]
    for m, _ in taus:
        m.tau = 0.0
    was_training = student.training
    student.eval()
    kl_sum = 0.0
    lm_sum = 0.0
    n_tok = 0
    items = itertools.islice(val_src, steps) if steps is not None else val_src
    for item in items:
        if isinstance(item[0], (tuple, list)):   # cached: ((x, y), path)
            (x, y), ltp = item
            x, y = x.cuda(), y.cuda()
            logits_t = torch.load(ltp, mmap=True, weights_only=True)["logits"] \
                .to(x.device).float()
        else:                              # live teacher
            x, y = item
            x, y = x.cuda(), y.cuda()
            logits_t = teacher(x).float()
        logits_s = student(x).float()
        y_flat = y.reshape(-1)
        s_flat = logits_s.reshape(-1, logits_s.shape[-1])
        t_flat = logits_t.reshape(-1, logits_t.shape[-1])
        for j in range(0, y_flat.numel(), chunk):
            sl = slice(j, j + chunk)
            lp_s = F.log_softmax(s_flat[sl], dim=-1)
            p_t = F.softmax(t_flat[sl], dim=-1)
            kl_sum += F.kl_div(lp_s, p_t, reduction="sum").item()
            lm_sum += F.cross_entropy(
                t_flat[sl], y_flat[sl], reduction="sum").item()
        n_tok += y_flat.numel()
        del logits_s, logits_t, s_flat, t_flat
    if was_training:
        student.train()
    for m, t in taus:
        m.tau = t
    if n_tok == 0:
        return float("nan"), float("nan")
    return kl_sum / n_tok, lm_sum / n_tok


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
    p.add_argument("--rank_shards", default="",                   help="dir of presplit per-rank safetensors (rankN.safetensors); "
                        "student build reads only this rank's file")
    p.add_argument("--teacher_stream", action="store_true",
                   help="layer-streamed mmap teacher (full-expert runs; "
                        "no resident teacher anywhere)")
    p.add_argument("--init_from", default="",
                   help="previous rung's out_dir: resume masters+scales "
                        "from its latest saved per-rank shard")
    p.add_argument("--kd_cache", type=int, default=20,
                   help="stage-replay: sweep the streamed teacher once over "
                        "this many train batches and replay their logits "
                        "(0 = live teacher per step — ~320 GB/step of IO)")
    p.add_argument("--data_seed", type=int, default=None,
                   help="seed the dataloader shuffle for reproducible "
                        "windows (A/B comparisons)")
    p.add_argument("--n_experts_keep", type=int, default=0,
                   help="smoke only: keep the first N experts per MoE layer "
                        "(0 = all). Validates real-checkpoint wiring without "
                        "the full 37.5B expert mass.")
    p.add_argument("--grad_checkpoint", action="store_true",
                   help="recompute layers in backward (memory for seq 2048 "
                        "training; ~30%% slower)")
    p.add_argument("--no_ptqr", action="store_true",
                   help="unquantized student control (teacher-vs-teacher)")
    args = p.parse_args()

    def _rssm(tag):
        with open(f"/proc/{os.getpid()}/status") as _f:
            for _l in _f:
                if _l.startswith("VmRSS"):
                    print(f"[mem r{os.environ.get('RANK', 's')}] {tag}: "
                          f"{int(_l.split()[1]) // 1048576} GB", flush=True)
                    return

    _rssm("post-parse")
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
        import datetime as _dt

        import torch.distributed as dist
        _rssm("pre-dist")
        # The streamed teacher sweep is IO-bound and rank-skewed; the
        # default 10-min NCCL watchdog fired on the logits all-gather.
        dist.init_process_group(
            "nccl" if device != "cpu" else "gloo",
            timeout=_dt.timedelta(hours=2),
        )
        tp_group = dist.group.WORLD
        device = f"cuda:{local_rank}" if device != "cpu" else device
        torch.cuda.set_device(local_rank) if device != "cpu" else None
        _rssm("post-dist")

    prm_frozen = False
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
        if world > 1 and args.teacher_stream:
            from ptqr_flint_model import StreamedFlintTeacher
            # presplit rank shards when available: per-rank IO is its own
            # 20GB file, not the full 80GB checkpoint per pass
            tdir = args.rank_shards if args.rank_shards else args.model
            teacher = StreamedFlintTeacher(
                cfg, tdir, world, rank,
                device=device if device != "cpu" else "cpu")
            teacher.eval()
            # streaming teacher bypasses the resident path entirely
            prm_frozen = True
        else:
            teacher = FlintRef(cfg, device="cpu")
            prm_frozen = False
        from ptqr_flint_model import load_flint_shards, shard_flint_tp
        if prm_frozen is False:
            if world > 1:
                # stream (name-map + expert EP assign); build the student
                # from the UNSHARDED teacher, then shard each separately
                load_flint_shards(teacher, args.model, world, rank)
                teacher.to(device)
            else:
                from ptqr_flint_model import load_flint_state
                load_flint_state(teacher, args.model)
                teacher.to(device)
    teacher.eval()
    if prm_frozen is False:
        for prm in teacher.parameters():
            prm.requires_grad_(False)

    from ptqr_flint_model import shard_flint_tp as _shard
    if args.teacher_stream and world > 1:
        # student needs a resident source. A full sharded skeleton is
        # ~20GB bf16/rank; four concurrent builds OOM the 61GB host (the
        # SIGKILL class), so builds run staggered two ranks at a time and
        # land the finished student on the GPU before the next pair.
        from ptqr_flint_model import load_flint_shards
        import gc as _gc
        wave = rank  # one rank at a time: a single 20GB skeleton + overhead
        if wave > 0:
            dist.barrier(group=tp_group)
        # in-place PTQR swap: deepcopy of a 23GB skeleton OOM-kills the
        # host, so the skeleton IS the student (the streamed teacher is a
        # separate mmap-backed object; no copy needed)
        # meta-skeleton: an unsharded cpu skeleton materializes ~80GB of
        # zero pages when the slicer reads them (the 54GB OOM class); on
        # meta the shard math is shape-only, and to_empty materializes
        # just the ~20GB shard.
        # torch.device context: FlintRef's internal nn.Linears otherwise
        # default to CPU + kaiming init, materializing all 80GB of pages —
        # THE 54GB OOM (SIGUSR1 dump named nn.Linear.reset_parameters)
        _prev_dt = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("meta"):
                student = FlintRef(cfg, device="meta")
        finally:
            torch.set_default_dtype(_prev_dt)
        _shard(student, world, rank, group=tp_group)  # shapes shrink to shard
        # materialize the shard directly on the GPU; the host never holds
        # the student (the entire host-OOM class dies here)
        print(f"[gpu r{rank}] pre-toempty allocated="
              f"{torch.cuda.memory_allocated()/2**30:.1f}G reserved="
              f"{torch.cuda.memory_reserved()/2**30:.1f}G", flush=True)
        student.to_empty(device=device)
        print(f"[gpu r{rank}] post-toempty allocated="
              f"{torch.cuda.memory_allocated()/2**30:.1f}G", flush=True)
        if student.rope is not None:  # plain attrs stay meta after to_empty
            from ptqr_flint_model import Rope as _Rope
            nr = _Rope(cfg.head_dim, cfg.max_position_embeddings,
                       cfg.rope_theta, device)
            student.rope = nr
            _nr = set(cfg.no_rope_layer_indices or [])
            for _i, _layer in enumerate(student.layers):
                _layer.attn.rope = None if _i in _nr else nr
        if args.rank_shards:
            from ptqr_flint_model import load_flint_rank_shard
            load_flint_rank_shard(
                student, os.path.join(args.rank_shards,
                                     f"rank{rank}.safetensors"))
        else:
            load_flint_shards(student, args.model, world, rank)
        from ptqr_train_target import (
            fake_quant_embedding_ as _fqe,
            replace_linears_with_ptqr as _swap,
        )
        replaced = _swap(student, group=args.weight_group,
                         skip=_collect_skip_names(student), mem_probe=True)
        _fqe(student.embed)
        for m in student.modules():
            if m.__class__.__name__ == "AttentionRef":
                m.kv_fake_quant_group = args.kv_group
        if args.init_from:
            # chained rungs: overwrite masters+scales with the previous
            # rung's saved shard (identical geometry — sharding is
            # deterministic); embedding rides along already fake-quantized
            import glob as _glob
            prev = sorted(_glob.glob(os.path.join(
                args.init_from, f"ptqr_target_step*_rank{rank}.pt")))
            assert prev, f"--init_from: no shard for rank {rank} in {args.init_from}"
            ck = torch.load(prev[-1], weights_only=True,
                            map_location="cpu", mmap=True)
            sd = ck["model_state_dict"]
            mine = dict(student.state_dict())
            with torch.no_grad():
                for k, v in sd.items():
                    assert k in mine and mine[k].shape == v.shape, \
                        f"init_from {k}: {mine.get(k)} vs {tuple(v.shape)}"
                    mine[k].copy_(v.to(mine[k].dtype))
            print(f"[init r{rank}] {os.path.basename(prev[-1])} "
                  f"({len(sd)} tensors, step {ck.get('step')})", flush=True)
        _rssm("post-swap")
        _rssm("post-todevice")  # already on device since to_empty
        _gc.collect()
        if wave == 0:
            dist.barrier(group=tp_group)
    elif args.no_ptqr:
        student, replaced = copy.deepcopy(teacher), {}
    else:
        student, replaced = build_student(teacher, args.weight_group,
                                          args.kv_group)
        if world > 1:
            _shard(teacher, world, rank, group=tp_group)
            _shard(student, world, rank, group=tp_group)
    student.to(device)
    student._grad_ckpt = bool(args.grad_checkpoint)

    n_lin = len(replaced)
    print(f"student: {n_lin} PTQRLinear surfaces, kv_group={args.kv_group}, "
          f"device={device}")

    if replaced:
        attach_weight_sgd_hooks(replaced, lr=args.lr,
                                scale_lr=args.lr * args.scale_lr_mult)

    if os.environ.get("PTQR_SHAPE_AUDIT"):
        # full-expert forward shape-bug hunt: dump every linear-ish module's
        # weight shape so a wrong-shaped master is visible pre-forward
        r = dist.get_rank() if dist.is_initialized() else 0
        with open(f"/tmp/ptqr_audit_r{r}.txt", "w") as f:
            f.write(f"cfg: {vars(student.cfg)}\n")
            f.write(f"n_lin: {n_lin}\n")
            for n, m in student.named_modules():
                cn = m.__class__.__name__
                if cn == "PTQRLinear":
                    f.write(f"{n} PTQR w={tuple(m.weight.shape)} "
                            f"s={tuple(m.scale.shape)} g={m.group}\n")
                elif hasattr(m, "weight"):
                    f.write(f"{n} {cn} w={tuple(m.weight.shape)}\n")

    data_path = os.path.join(args.data_dir, args.data_name)
    val_path = os.path.join(args.data_dir, args.val_name)
    if args.data_seed is not None:
        torch.manual_seed(args.data_seed)   # deterministic shuffle windows
    train_iter = build_dataloader(data_path, args.seq_len, args.batch_size,
                                  tiny=args.tiny, tiny_vocab=teacher.cfg.vocab_size)
    val_iter = build_dataloader(val_path, args.seq_len, args.batch_size,
                                tiny=args.tiny, tiny_vocab=teacher.cfg.vocab_size) \
        if os.path.exists(val_path) or args.tiny else None

    # --- KD-logit cache (stage-replay): the streamed teacher runs ONCE per
    # rung in a layer-major sweep (each layer's weights stage once for ALL
    # cached batches — one 20GB/rank pass total); training then replays the
    # cached bf16-teacher logits. Live per-step streaming costs a full
    # checkpoint pass per batch per step (~320 GB/step aggregate — measured
    # crawl at ~1 MB/s). Set --kd_cache 0 for the live path.
    kd_dir = os.path.join(args.out_dir, "kdcache")
    kd_train = kd_val = None
    if args.kd_cache > 0 and prm_frozen:
        import itertools as _it

        def _precache(it, n, split):
            """ONE sweep over the first n batches (each chunk re-stages every
            layer — chunking multiplies the shard IO; the sweep offloads
            fp16 logits per batch so a single pass is memory-safe). Rank 0
            spills to files (one shared page-cache copy for all ranks —
            in-RAM copies would be 4x ~12 GB on this 61 GB host).
            Returns [((x, y), path), ...]."""
            batches = list(_it.islice(it, n))
            if not batches:
                return []
            lts = teacher.sweep([x for x, _ in batches])
            outs = []
            for j in range(len(batches)):
                p = os.path.join(kd_dir, f"{split}_{j}.pt")
                if rank == 0:
                    # y rides with the logits: the dataloader SHUFFLES
                    # unseeded, so a later re-iteration cannot reconstruct
                    # which window the sweep saw (the g2x NLL probes were
                    # invalidated by exactly this)
                    torch.save({"logits": lts[j],
                                "y": batches[j][1].clone()}, p)
                outs.append((batches[j], p))
            del lts
            return outs

        if rank == 0:
            os.makedirs(kd_dir, exist_ok=True)
        if world > 1:
            dist.barrier(group=tp_group)
        t0 = time.time()
        for r in range(world - 1, -1, -1):   # reverse: rank0 warms LAST
            if rank == r:
                teacher.warm_shard()
            dist.barrier(group=tp_group)
        print(f"[kd r{rank}] shard warmed in {time.time() - t0:.0f}s",
              flush=True)
        kd_train = _precache(train_iter, args.kd_cache, "train")
        kd_val = _precache(val_iter, args.eval_steps, "val") \
            if val_iter is not None else []
        print(f"[kd r{rank}] cached {len(kd_train)} train + {len(kd_val)} val "
              f"teacher sweeps in {time.time() - t0:.0f}s", flush=True)

    def _kd_load(path):
        obj = torch.load(path, mmap=True, weights_only=True)
        # _precache writes {"logits": ..., "y": ...}; older caches held the
        # bare logits tensor.
        return obj["logits"] if isinstance(obj, dict) else obj

    def one_batch(x, y, tau, logits_t=None):
        if replaced:
            for m in replaced.values():
                m.tau = tau
        # chunked lm_head + loss: the full-vocab fp32 logits path peaked
        # ~5 GB of transients at seq 2048 and OOM'd the backward
        # (hipblasCreate ALLOC_FAILED, g19/g20); chunks keep one slice
        # live at a time (the Qwen trainer's partial-grad pattern — the
        # exact-sum form matches the old batchmean semantics)
        h = student.forward_hidden(x).reshape(-1, student.cfg.hidden_size)
        if logits_t is None:
            with torch.no_grad():
                logits_t = teacher(x)
        # Keep the teacher logits in their stored dtype: .float() on the
        # whole tensor costs 2x (16.3 GiB at seq 16384) and OOMed the
        # long-context rungs. Each loss chunk casts its own slice.
        t_flat = logits_t.reshape(-1, logits_t.shape[-1])
        y_flat = y.reshape(-1)
        n = y_flat.numel()
        T = args.distill_temperature
        ce_sum = torch.zeros((), device=h.device)
        kd_sum = torch.zeros((), device=h.device)
        global _DBG_LOGITS_DONE
        for j in range(0, n, args.logits_chunk):
            sl = slice(j, j + args.logits_chunk)
            lc = student.head(h[sl]).float()
            if (_os0.environ.get("PTQR_DEBUG_LOGITS") and not _DBG_LOGITS_DONE
                    and j == 0):
                with torch.no_grad():
                    _s = lc.reshape(-1, lc.shape[-1])
                    _t = t_flat[sl].float().reshape(-1, t_flat.shape[-1])
                    _top1 = (_s.argmax(-1) == _t.argmax(-1)).float().mean()
                    _corr = torch.corrcoef(torch.stack([_s.flatten(),
                                                        _t.flatten()]))[0, 1]
                    _ss = _s.sort(dim=-1).values.flatten()
                    _ts = _t.sort(dim=-1).values.flatten()
                    _corrs = torch.corrcoef(torch.stack([_ss, _ts]))[0, 1]
                print(f"[dbg-logits] top1agree {_top1.item():.4f} "
                      f"corr {_corr.item():.4f} sorted_corr {_corrs.item():.4f}",
                      flush=True)
                _DBG_LOGITS_DONE = True
            ce_sum = ce_sum + F.cross_entropy(lc, y_flat[sl],
                                              reduction="sum")
            lp = F.log_softmax(lc / T, dim=-1)
            pt = F.softmax(t_flat[sl].float() / T, dim=-1)
            kd_sum = kd_sum + F.kl_div(lp, pt, reduction="sum")
        loss = (ce_sum + args.distill_weight * kd_sum) / n
        return loss, (ce_sum / n).detach(), (kd_sum / n).detach()

    step = 0
    last_saved = -1
    while step < args.max_steps:
        if kd_train is not None:
            (x, y), ltp = kd_train[step % len(kd_train)]
            tau = anneal_tau(step, args.max_steps, args.temp_start,
                             args.temp_end)
            loss, ce, kd = one_batch(x.to(device), y.to(device), tau,
                                     _kd_load(ltp).to(device))
            loss.backward()
            if replaced:
                apply_deferred_updates(student)
            if step % 10 == 0:
                print(f"step {step:4d} tau {tau:.4f} loss {loss.item():.4f} "
                      f"ce {ce.item():.4f} kd {kd.item():.4f}", flush=True)
            if kd_val and args.eval_every and step % args.eval_every == 0:
                kld, lm = eval_kld_flint(student, teacher, kd_val)
                if rank == 0:
                    print(f"eval step {step:4d} kld {kld:.6f} "
                          f"teacher_nll {lm:.4f}", flush=True)
            # step 0 satisfies the modulo, and a 76 GB save there only
            # delays the run (measured 2026-11-11); the final save covers
            # the end of the rung.
            if args.save_every and step > 0 and step % args.save_every == 0:
                save_ptqr_checkpoint(student, replaced, Path(args.out_dir),
                                     step, args, rank=rank, world=world,
                                     vocab_size=teacher.cfg.vocab_size)
                last_saved = step
            step += 1
            student.zero_grad(set_to_none=True)
            continue
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
            if val_iter is not None and args.eval_every and \
                    step % args.eval_every == 0:
                kld, lm = eval_kld_flint(student, teacher, val_iter,
                                         args.eval_steps)
                if rank == 0:
                    print(f"eval step {step:4d} kld {kld:.6f} "
                          f"teacher_nll {lm:.4f}", flush=True)
            # step 0 satisfies the modulo, and a 76 GB save there only
            # delays the run (measured 2026-11-11); the final save covers
            # the end of the rung.
            if args.save_every and step > 0 and step % args.save_every == 0:
                save_ptqr_checkpoint(student, replaced, Path(args.out_dir),
                                     step, args, rank=rank, world=world,
                                     vocab_size=teacher.cfg.vocab_size)
                last_saved = step
            step += 1
            student.zero_grad(set_to_none=True)

    os.makedirs(args.out_dir, exist_ok=True)
    if replaced and last_saved != step:
        save_ptqr_checkpoint(student, replaced, Path(args.out_dir), step,
                             args, rank=rank, world=world,
                             vocab_size=teacher.cfg.vocab_size)
    torch.save({"step": step, "cfg": vars(teacher.cfg) if not args.tiny else {}},
               os.path.join(args.out_dir, "train_meta.pt"))
    print("PTQR flint trainer run complete ->", args.out_dir)


if __name__ == "__main__":
    main()
