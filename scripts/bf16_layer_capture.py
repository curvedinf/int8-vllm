#!/usr/bin/env python3
"""bf16 reference layer capture for the G1b Path B per-layer ranking.

Replays the d30 transcript (prompt + committed ids) through the frozen bf16
reference (HF Qwen3_5TextModel, TP4, same scaffolding as ptqr_draft_fidelity)
and records, for every layer L and every position t, the projection onto the
FIXED random matrix R — identical to the engine LAYERPROBE
(torch.randn(hidden, 16, generator=cpu seed 1234), fp32):

  mlp[t]  = (mlp_out[t]  .float() @ R)   vLLM post-layer split stream
            (engine probe recorded exactly this quantity)
  resid[t] = (layer_out[t].float() @ R)   HF residual stream (diagnostic)

Rank0 saves {ids, mlp{L: [T,16] fp32}, resid{...}} to <out>/capture.pt.
Every rank all-reduce-checks its projections against rank0 (validates that
the TP4 down_proj all-reduce makes mlp outputs replicated).

Usage:
  PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  .venv/bin/python -m torch.distributed.run --nproc_per_node=4 \
    scripts/bf16_layer_capture.py --out logs/garble/rank_bf16 [--chunk 0]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/curved/SDGraft")

IDS_PT = "/home/curved/vllm-gfx908/logs/garble/d30_origgptq_temp1_ids.pt"
COMMITTED_PT = "/home/curved/vllm-gfx908/logs/garble/d30_origgptq_temp1_committed.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="logs/garble/rank_bf16")
    p.add_argument("--chunk", type=int, default=0,
                   help="if >0, forward in chunks of this many tokens with a "
                        "carried cache (fallback if single-pass OOMs)")
    args = p.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    from common.tp_ssm import apply_ssm_tp
    from common.tp_attention import apply_tp_attention, register_rocm_triton_tp
    import ptqr_train_target as T

    register_rocm_triton_tp()

    class A:
        model = "/home/curved/models/Qwen3.8-27B-bf16-ref"
        base_checkpoint = "/home/curved/models/Qwen3.8-27B-bf16-ref-lm.pt"
        tiny = False
        n_layers = 0
        attn_impl = "rocm_triton"

    target = T.build_lm_model(A(), torch.bfloat16, "rocm_triton")
    apply_ssm_tp(target.model, tp_size=world)
    apply_tp_attention(target.model, tp_size=world)
    T.shard_mlp_and_heads_one(target, world, rank, group=dist.group.WORLD)
    target.lm_head = torch.nn.Identity()  # not needed; frees vocab head
    target.to(dev).eval()
    n_layers = target.model.config.num_hidden_layers
    hidden = target.model.config.hidden_size

    # R bit-identical to the engine probe (cpu generator, seed 1234)
    g = torch.Generator(device="cpu").manual_seed(1234)
    R = torch.randn(hidden, 16, generator=g).to(dev, torch.float32)

    ids_doc = torch.load(IDS_PT, map_location="cpu", weights_only=False)
    committed = torch.load(COMMITTED_PT, map_location="cpu",
                           weights_only=False)["committed_ids"]
    ids = torch.tensor(list(ids_doc["prompt_ids"]) + list(committed),
                       dtype=torch.long)
    T_total = ids.numel()
    print(f"[r{rank}] transcript {T_total} tokens, {n_layers} layers, "
          f"hidden {hidden}", flush=True)

    mlp_proj: dict[int, torch.Tensor] = {}
    resid_proj: dict[int, torch.Tensor] = {}

    def mk_mlp(L):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            mlp_proj[L] = h.detach()[0].float() @ R  # [T,16] on GPU
        return hook

    def mk_layer(L):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            resid_proj[L] = h.detach()[0].float() @ R
        return hook

    handles = []
    for L in range(n_layers):
        handles.append(target.model.layers[L].mlp.register_forward_hook(mk_mlp(L)))
        handles.append(target.model.layers[L].register_forward_hook(mk_layer(L)))

    cache = None
    if args.chunk > 0:
        from transformers import DynamicCache
        cache = DynamicCache()

    with torch.no_grad():
        if args.chunk <= 0:
            target(input_ids=ids[None].to(dev))
        else:
            for s in range(0, T_total, args.chunk):
                e = min(s + args.chunk, T_total)
                target(input_ids=ids[s:e][None].to(dev),
                       past_key_values=cache, use_cache=True,
                       cache_position=torch.arange(s, e, device=dev))
                print(f"[r{rank}] chunk {s}-{e} done", flush=True)

    for h in handles:
        h.remove()

    # TP validation: every rank's projections must equal rank0's (all-reduce
    # makes mlp outputs replicated; layer outputs likewise). Report the worst
    # deviation across all layers.
    worst = torch.zeros((), device=dev)
    for L in range(n_layers):
        ref = mlp_proj[L].clone()
        dist.broadcast(ref, src=0)
        worst = torch.maximum(worst, (mlp_proj[L] - ref).abs().max())
        ref = resid_proj[L].clone()
        dist.broadcast(ref, src=0)
        worst = torch.maximum(worst, (resid_proj[L] - ref).abs().max())
    print(f"[r{rank}] TP4 replication max|diff| vs rank0 = {worst.item():.3e}",
          flush=True)

    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        out = {
            "ids": ids,
            "R_col_norms": R.norm(dim=0).cpu(),
            "mlp": {L: mlp_proj[L].cpu() for L in range(n_layers)},
            "resid": {L: resid_proj[L].cpu() for L in range(n_layers)},
        }
        torch.save(out, os.path.join(args.out, "capture.pt"))
        print(f"[r0] saved {os.path.join(args.out, 'capture.pt')}", flush=True)
        print(f"[r0] peak VRAM {torch.cuda.max_memory_allocated()/2**30:.1f} GiB",
              flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
