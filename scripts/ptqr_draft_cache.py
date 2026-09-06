#!/usr/bin/env python3
"""Stage-A cache: freeze target exit states for draft PTQR training.

Runs the bf16 target TP4 (sdgraft-train venv) over val/train sequences and
dumps, per sequence: token ids + the 5 aux exit hidden states [C, 5, H] fp16.
The draft trainer then consumes the cache standalone on one GPU.

Usage:
  FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python -m torch.distributed.run \
    --nproc_per_node=4 scripts/ptqr_draft_cache.py --n_seqs 400 --seq_len 1024
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/curved/SDGraft")
import argparse as _ap


class _A:
    model = "/home/curved/models/Qwen3.8-27B-bf16-ref"
    base_checkpoint = "/home/curved/models/Qwen3.8-27B-bf16-ref-lm.pt"
    tiny = False
    attn_impl = "rocm_triton"
    n_layers = 0


from ptqr_train_target import build_lm_model, shard_mlp_and_heads_one  # noqa: E402
from common.tp_ssm import apply_ssm_tp, install_t_chunked_fallback  # noqa: E402
from common.tp_attention import apply_tp_attention, register_rocm_triton_tp  # noqa: E402

EXITS = [5, 19, 33, 47, 61]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_seqs", type=int, default=400)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--split", default="val", choices=("val", "train"))
    p.add_argument("--out", default="/home/curved/models/ptqr_draft_cache")
    args = p.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    sys.path.insert(0, "/home/curved/SDGraft")
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    install_t_chunked_fallback()
    register_rocm_triton_tp()

    target = build_lm_model(_A(), torch.bfloat16, "rocm_triton")
    apply_ssm_tp(target.model, tp_size=world if world > 1 else None)
    apply_tp_attention(target.model, tp_size=world if world > 1 else None)
    if world > 1:
        shard_mlp_and_heads_one(target, world, rank, group=None)
    target.lm_head = torch.nn.Identity()  # exit hiddens only; frees VRAM
    target.to(dev).eval()

    caps = {L: [] for L in EXITS}
    handles = []
    core = target.model
    for L in EXITS:
        def mk(layer):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                caps[layer].append(h.detach()[0].to(torch.float16).cpu())
            return hook
        handles.append(core.layers[L].register_forward_hook(mk(L)))

    tokens = torch.load(
        "/home/curved/SDGraft/data/qwen38_longctx/val_tokens.pt"
        if args.split == "val" else
        "/home/curved/SDGraft/data/qwen38_longctx/train_tokens.pt",
        weights_only=True)

    out_dir = Path(args.out)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_seqs):
        seq = tokens[i * (args.seq_len + 1):(i + 1) * (args.seq_len + 1)].long()
        if seq.numel() < args.seq_len + 1:
            break
        for L in EXITS:
            caps[L].clear()
        with torch.no_grad():
            target(input_ids=seq[:-1][None].to(dev))
        if rank != 0:
            continue
        states = torch.stack([caps[L][0] for L in EXITS], dim=1)  # [C, 5, H]
        torch.save({"tokens": seq, "states": states},
                   out_dir / f"seq{i:05d}.pt")
        if i % 20 == 0:
            print(f"seq {i}: states {tuple(states.shape)} "
                  f"rms {states.float().pow(2).mean().sqrt():.2f}", flush=True)

    for h in handles:
        h.remove()
    if world > 1:
        torch.distributed.destroy_process_group()
    print("CACHE DONE", flush=True)


if __name__ == "__main__":
    main()
