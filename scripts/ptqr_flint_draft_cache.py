#!/usr/bin/env python3
"""Stage-A cache for the FLINT DFlash2 draft PTQR training.

Runs the bf16 Flint target (streamed, TP4, presplit shards — ONE
layer-major pass for ALL sequences) over flint_longctx sequences and
dumps per sequence: token ids + the 5 target exit states [C, 5, H] fp16
at dflash_config.target_layer_ids. Mirrors scripts/ptqr_draft_cache.py's
output contract (consumed by the draft trainer unchanged).

Usage:
  PYTHONPATH=scripts:. .venv/bin/python -m torch.distributed.run \
    --nproc_per_node=4 scripts/ptqr_flint_draft_cache.py --n_seqs 243
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from ptqr_flint_model import StreamedFlintTeacher, load_flint_config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/curved/models/flint-bf16")
    p.add_argument("--rank_shards",
                   default="/home/curved/models/flint-bf16-rankshards")
    p.add_argument("--n_seqs", type=int, default=243)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--split", default="val", choices=("val", "train"))
    p.add_argument("--chunk", type=int, default=64,
                   help="sequences per layer-major sweep (activations "
                        "~8 MB each; each sweep re-reads the shard once)")
    p.add_argument("--out", default="/home/curved/models/flint_draft_cache")
    args = p.parse_args()

    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    cfg = load_flint_config(args.model)
    # exit layers live in the DRAFT's dflash_config (the target config
    # does not carry them)
    import json

    dfl = json.load(open("/home/curved/models/flint-dflash2-bf16/"
                         "config.json"))["dflash_config"]
    exits = dfl["target_layer_ids"]

    teacher = StreamedFlintTeacher(cfg, args.rank_shards, world, rank,
                                   device=dev)
    tokens = torch.load(
        f"/home/curved/SDGraft/data/flint_longctx/"
        f"{'val' if args.split == 'val' else 'train'}_tokens.pt",
        weights_only=True)

    out_dir = Path(args.out)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    while done < args.n_seqs:
        n = min(args.chunk, args.n_seqs - done)
        seqs = []
        for i in range(done, done + n):
            seq = tokens[i * (args.seq_len + 1):
                         (i + 1) * (args.seq_len + 1)].long()
            if seq.numel() < args.seq_len + 1:
                seqs = None
                break
            seqs.append(seq)
        if not seqs:
            break
        states = teacher.sweep_states([s[:-1][None].to(dev) for s in seqs],
                                      exits)
        if rank == 0:
            for j, s in enumerate(seqs):
                torch.save({"tokens": s.cpu(), "states": states[j]},
                           out_dir / f"seq{done + j:05d}.pt")
            print(f"[cache] {done + n} seqs done "
                  f"(rms {states[-1].float().pow(2).mean().sqrt():.2f})",
                  flush=True)
        done += n
    dist.destroy_process_group()
    if rank == 0:
        print("CACHE DONE", flush=True)


if __name__ == "__main__":
    main()
