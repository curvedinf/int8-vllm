#!/usr/bin/env python3
"""End-to-end stitch test for ptqr_export_flint (tiny cfg, CPU, TP4 gloo).

Each rank builds the SAME deterministic tiny FlintRef (fixed seed), shards it
TP4, PTQR-swaps it, and saves via save_ptqr_checkpoint. Rank 0 also writes the
unsharded reference under HF names. The export is then run as a subprocess and
its --verify report is asserted: stitched masters must equal the reference
bit-exactly (no training steps ran, masters are clones of the source weights).

Usage: .venv/bin/python -m torch.distributed.run --nproc_per_node=4 \
           scripts/test_flint_export_stitch.py
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ptqr_flint_model import (  # noqa: E402
    FlintRef,
    flint_tiny_config,
    shard_flint_tp,
)
from ptqr_train_target import (  # noqa: E402
    replace_linears_with_ptqr,
    save_ptqr_checkpoint,
)

STEP = 7
TAG = "flint_stitch_test"


class _A:  # save_ptqr_checkpoint reads these args attrs
    weight_group = 32
    kv_group = 32
    seq_len = 64
    max_steps = 0
    lr = 0.0
    distill_weight = 2.0
    temp_start = 1.0
    temp_end = 0.001


def _build_full():
    torch.manual_seed(1234)
    cfg = flint_tiny_config()
    cfg.attention_bias = False
    return cfg, FlintRef(cfg)


def main():
    import torch.distributed as dist

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    ckpt_dir = Path(f"/tmp/{TAG}/ckpt")
    src_dir = Path(f"/tmp/{TAG}/src")
    out_dir = Path(f"/tmp/{TAG}/out")

    cfg, model = _build_full()
    if rank == 0:
        # reference under HF names (mirrors load_flint_state's inverse map)
        ref = {}
        sd = model.state_dict()
        ref["model.embed_tokens.weight"] = sd["embed.weight"]
        ref["lm_head.weight"] = sd["lm_head.weight"]
        ref["model.norm.weight"] = sd["final_norm.weight"]
        for n in range(cfg.num_hidden_layers):
            p = f"layers.{n}"
            h = f"model.layers.{n}"
            ref[f"{h}.input_layernorm.weight"] = sd[f"{p}.norm1.weight"]
            ref[f"{h}.post_attention_layernorm.weight"] = sd[f"{p}.norm2.weight"]
            ref[f"{h}.self_attn.q_proj.weight"] = \
                sd[f"{p}.attn.qkv.weight"][: cfg.num_attention_heads *
                                            cfg.head_dim]
            ref[f"{h}.self_attn.k_proj.weight"] = \
                sd[f"{p}.attn.qkv.weight"][cfg.num_attention_heads *
                                           cfg.head_dim:][: cfg.num_key_value_heads * cfg.head_dim]
            ref[f"{h}.self_attn.v_proj.weight"] = \
                sd[f"{p}.attn.qkv.weight"][cfg.num_key_value_heads *
                                           cfg.head_dim + cfg.num_attention_heads * cfg.head_dim:]
            ref[f"{h}.self_attn.o_proj.weight"] = sd[f"{p}.attn.o.weight"]
            ref[f"{h}.self_attn.q_norm.weight"] = sd[f"{p}.attn.q_norm.weight"]
            ref[f"{h}.self_attn.k_norm.weight"] = sd[f"{p}.attn.k_norm.weight"]
            ref[f"{h}.self_attn.softmax_offset"] = sd[f"{p}.attn.sink"]
            if n in cfg.mlp_only_layers:
                gu = sd[f"{p}.mlp.gate_up.weight"]
                g = gu[: gu.shape[0] // 2]
                u = gu[gu.shape[0] // 2:]
                ref[f"{h}.mlp.gate_proj.weight"] = g
                ref[f"{h}.mlp.up_proj.weight"] = u
                ref[f"{h}.mlp.down_proj.weight"] = sd[f"{p}.mlp.down.weight"]
            else:
                ref[f"{h}.mlp.gate.weight"] = sd[f"{p}.mlp.router.weight"]
                for pre, dst in ((f"{p}.mlp.shared.",
                                  f"{h}.mlp.shared_expert."),):
                    gu = sd[pre + "gate_up.weight"]
                    ref[dst + "gate_proj.weight"] = gu[: gu.shape[0] // 2]
                    ref[dst + "up_proj.weight"] = gu[gu.shape[0] // 2:]
                    ref[dst + "down_proj.weight"] = sd[pre + "down.weight"]
                for e in range(cfg.num_experts):
                    gu = sd[f"{p}.mlp.experts.{e}.gate_up.weight"]
                    ref[f"{h}.mlp.experts.{e}.gate_proj.weight"] = \
                        gu[: gu.shape[0] // 2]
                    ref[f"{h}.mlp.experts.{e}.up_proj.weight"] = \
                        gu[gu.shape[0] // 2:]
                    ref[f"{h}.mlp.experts.{e}.down_proj.weight"] = \
                        sd[f"{p}.mlp.experts.{e}.down.weight"]
        src_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / "config.json").write_text(json.dumps({
            "num_hidden_layers": cfg.num_hidden_layers,
            "vocab_size": cfg.vocab_size,
            "hidden_size": cfg.hidden_size,
            "head_dim": cfg.head_dim,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "num_experts": cfg.num_experts,
            "mlp_only_layers": cfg.mlp_only_layers,
        }))
        from safetensors.torch import save_file

        save_file(ref, str(src_dir / "ref.safetensors"))

    dist.barrier()

    # shard + swap + save (same deterministic full model on every rank)
    cfg2, model2 = _build_full()
    shard_flint_tp(model2, world, rank, group=None)
    skip = set()
    for name, module in model2.named_modules():
        for child_name, child in module.named_children():
            full = f"{name}.{child_name}" if name else child_name
            if ".mlp.router" in full:
                skip.add(full)
    replaced = replace_linears_with_ptqr(model2, group=32, skip=skip)
    save_ptqr_checkpoint(model2, replaced, ckpt_dir, STEP, _A(),
                         rank=rank, world=world, vocab_size=cfg2.vocab_size)

    dist.barrier()
    if rank == 0:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).parent) + \
            (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        r = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "ptqr_export_flint.py"),
             "--ckpt_dir", str(ckpt_dir), "--step", str(STEP),
             "--world", str(world), "--src", str(src_dir),
             "--out", str(out_dir), "--group", "32", "--verify"],
            capture_output=True, text=True, env=env)
        print(r.stdout[-3000:])
        if r.returncode != 0:
            print(r.stderr[-3000:])
            print("FAIL: exporter crashed")
            dist.destroy_process_group()
            sys.exit(1)
        diffs = [float(ln.split()[0]) for ln in r.stdout.splitlines()
                 if ln.startswith("  ") and "max|diff|" not in ln
                 and ln.strip()[:1].isdigit()]
        if not diffs:
            print("FAIL: no verify diffs parsed")
            dist.destroy_process_group()
            sys.exit(1)
        worst = max(diffs)
        n_bad = sum(1 for d in diffs if d > 1e-6)
        idx = json.loads((out_dir / "model.safetensors.index.json").read_text())
        n_qw = sum(1 for k in idx["weight_map"] if k.endswith(".qweight"))
        ok = worst <= 1e-6 and n_bad == 0 and n_qw > 0
        print(f"{'PASS' if ok else 'FAIL'}: {len(diffs)} stitched tensors, "
              f"worst max|diff|={worst:.2e}, {n_qw} packed qweight tensors")
        # config/tokenizer copies absent in synthetic src — fine for the test
        dist.destroy_process_group()
        sys.exit(0 if ok else 1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
