#!/usr/bin/env python3
"""Draft-forward fidelity gate: my dense DFlash2 forward vs real target states.

Runs the frozen bf16 target (TP4) over corpus sequences, captures hidden
states at the 5 draft exit layers (target layers 5/19/33/47/61), then feeds
them + the anchor token to scripts/ptqr_train_draft.DraftModel and measures
top-1 next-token agreement (vs the true next token and vs the target's argmax).

If my forward is faithful, agreement should be in the deployed draft's
acceptance ballpark (top-1 ~0.6-0.9); a broken forward lands near chance.

Usage:
  FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE torchrun --nproc_per_node=4 \
    scripts/ptqr_draft_fidelity.py --n_seqs 8 --seq_len 1024
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/curved/SDGraft")

from ptqr_train_draft import CFG, DraftModel, load_draft, target_embed_table  # noqa: E402

EXITS = (5, 19, 33, 47, 61)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_seqs", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--probe_every", type=int, default=64,
                   help="probe positions per sequence (spread)")
    args = p.parse_args()

    rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))
    world = int(__import__("os").environ.get("WORLD_SIZE", "1"))
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    from common.tp_ssm import apply_ssm_tp
    from common.tp_attention import apply_tp_attention, register_rocm_triton_tp
    import ptqr_train_target as T
    import os
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    register_rocm_triton_tp()

    class A:  # minimal args for build_lm_model
        model = "/home/curved/models/Qwen3.8-27B-bf16-ref"
        base_checkpoint = "/home/curved/models/Qwen3.8-27B-bf16-ref-lm.pt"
        tiny = False
        n_layers = 0
        attn_impl = "rocm_triton"
    target = T.build_lm_model(A(), torch.bfloat16, "rocm_triton")
    apply_ssm_tp(target.model, tp_size=world)
    apply_tp_attention(target.model, tp_size=world)
    target.to(dev).eval()

    # capture exit-layer hidden states (post-layer output, pre-final-norm)
    caps: dict[int, list[torch.Tensor]] = {L: [] for L in EXITS}
    handles = []
    core = target.model
    for L in EXITS:
        def mk(layer):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                caps[layer].append(h.detach()[0].float().cpu())  # [T, H] rank0 only
            return hook
        handles.append(core.layers[L].register_forward_hook(mk(L)))

    tokens = torch.load("/home/curved/SDGraft/data/qwen38_longctx/val_tokens.pt",
                        weights_only=True)
    seqs = [tokens[i * args.seq_len:(i + 1) * args.seq_len + 1].long()
            for i in range(args.n_seqs)]

    # rank0 also builds the draft (small)
    draft = None
    if rank == 0:
        base = torch.load("/home/curved/models/Qwen3.8-27B-bf16-ref-lm.pt",
                          weights_only=True, mmap=True)["model_state_dict"]
        lm_head = base["lm_head.weight"].to(torch.bfloat16)
        draft = load_draft(DraftModel(target_embed_table(base), lm_head)) \
            .to(dev).to(torch.bfloat16).eval()

    n_probe = 0
    n_top1_true = 0
    n_top1_target = 0
    for si, seq in enumerate(seqs):
        for L in EXITS:
            caps[L].clear()
        x = seq[:-1].to(dev)
        with torch.no_grad():
            logits = target(input_ids=x[None]).logits
        if rank != 0:
            continue
        # note: with TP the hook fired on every rank; caps on rank0 hold
        # rank0's full replica? No — layers are sharded; out[0] is the
        # all-reduced hidden (replicated). Verified: o_proj partials are
        # summed, so the layer output IS replicated on every rank.
        states = [caps[L][0] for L in EXITS]          # [T, H] each
        tgt_argmax = logits[0].argmax(-1).cpu()        # [T]
        Tn = x.shape[0]
        for pos in range(64, Tn - 1, args.probe_every):
            # anchor token at pos, predict pos+1 using context states [0, pos]
            anchor = seq[pos].to(dev)
            ctx = [s[:pos + 1, None, :].to(dev).to(torch.bfloat16) for s in states]
            toks = torch.full((1 + CFG["ns"],), CFG["mask_token"],
                              dtype=torch.long, device=dev)
            toks[0] = anchor
            with torch.no_grad():
                dl, _ = draft(toks, ctx)
            pred = dl[0].argmax(-1).cpu()
            truth = seq[pos + 1]
            n_probe += 1
            n_top1_true += int(pred == truth)
            n_top1_target += int(pred == tgt_argmax[pos])
        if rank == 0:
            print(f"seq {si}: probes so far {n_probe} "
                  f"top1-true {n_top1_true / max(1, n_probe):.3f} "
                  f"top1-target {n_top1_target / max(1, n_probe):.3f}", flush=True)

    for h in handles:
        h.remove()
    if rank == 0:
        print(f"FIDELITY: probes {n_probe} | top1 vs TRUE {n_top1_true/max(1,n_probe):.3f} | "
              f"top1 vs TARGET-ARGMAX {n_top1_target/max(1,n_probe):.3f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
