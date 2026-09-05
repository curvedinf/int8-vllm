#!/usr/bin/env python3
"""Wrapper-export a PTQR training checkpoint to servable GPTQ GS128.

The proven vehicle (see ledger PTQR_P1_export_vehicle): copy the DEPLOYED
checkpoint directory verbatim (its file layout, quantize_config, g_idx and
bf16 tensors all load correctly through the fork's AITER W8A8 loader), then
overwrite the 400 quantized payloads with the trained checkpoint's
values (stitched from the 4 TP shards) and optionally the trained bf16
in_proj_a/b + lm_head.

Usage:
  python scripts/ptqr_wrapper_export.py --ckpt_dir /home/curved/models/ptqr_r8 \
      --step 15 --out /home/curved/models/Qwen3.8-27B-PTQR-R8S15
"""
import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
from ptqr_export_gptq import (  # noqa: E402
    _is_row_parallel,
    quant_pack,
    stitch,
    stitch_scales,
)

DEPLOYED = "/home/curved/models/Qwen3.8-27B-GPTQ-8bit-gs128"
G = 128


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("--bf16_surfaces", type=int, default=1,
                   help="also overwrite in_proj_a/b + lm_head with trained "
                        "bf16 values (in_proj dequantized from its grid)")
    args = p.parse_args()

    shards = [torch.load(f"{args.ckpt_dir}/ptqr_target_step{args.step}_rank{r}.pt",
                         weights_only=True, mmap=True,
                         map_location="cpu")["model_state_dict"]
              for r in range(args.world)]
    names = {k[:-len(".weight")] for s in shards for k in s if k.endswith(".weight")}
    scale_names = {k[:-len(".scale")] for s in shards for k in s if k.endswith(".scale")}

    # precompute trained payloads lazily: name -> (qw, qz, sc_fp16) or bf16 w
    def trained(name):
        full = stitch(name, shards)
        if name not in scale_names:
            return None, None, None, full.to(torch.bfloat16)
        sc = stitch_scales(name, shards)  # stitched: dim1 = FULL group count
        if full.shape[1] != G * sc.shape[1]:
            return None, None, None, full.to(torch.bfloat16)
        qw, qz, sc16 = quant_pack(full, sc, G)
        leaf = name.split(".")[-1]
        if leaf in ("in_proj_a", "in_proj_b"):
            # deployed wrapper keeps these bf16: dequant the trained grid
            q8 = torch.stack([(qw >> (8 * s) & 0xFF) for s in range(4)], 1) \
                .reshape(qw.shape[0] * 4, qw.shape[1]).float()
            deq = torch.zeros_like(q8)
            for g in range(sc16.shape[0]):
                r0, r1 = g * G, (g + 1) * G
                deq[r0:r1] = (q8[r0:r1] - 128.0) * sc16[g].float().unsqueeze(0)
            return None, None, None, deq.t().contiguous().to(torch.bfloat16)
        return qw, qz, sc16, None

    os.makedirs(args.out, exist_ok=True)
    n_q, n_bf = 0, 0
    for fp in sorted(glob.glob(f"{DEPLOYED}/model-*.safetensors")):
        shard = load_file(fp)
        for k in list(shard):
            if k.endswith(".qweight"):
                base = k[:-len(".qweight")]
                norm = base[len("model.language_model."):]
                tree = next((n for n in names
                             if n.replace(".layer.", ".", 1) == norm), None)
                if tree is None:
                    continue
                qw, qz, sc16, _ = trained(tree)
                if qw is not None:
                    shard[k] = qw
                    shard[base + ".qzeros"] = qz
                    shard[base + ".scales"] = sc16
                    n_q += 1
            elif args.bf16_surfaces and k.endswith(".weight"):
                base = k[:-len(".weight")]
                leaf = base.split(".")[-1]
                if leaf in ("in_proj_a", "in_proj_b") or k == "lm_head.weight":
                    norm = base[len("model.language_model."):] if base.startswith("model.language_model.") else base
                    tree = next((n for n in names
                                 if n.replace(".layer.", ".", 1) == norm), None)
                    if tree is not None:
                        _, _, _, bf = trained(tree)
                        shard[k] = bf
                        n_bf += 1
        save_file(shard, f"{args.out}/{fp.split('/')[-1]}")
    for f in os.listdir(DEPLOYED):
        if not f.endswith(".safetensors"):
            shutil.copy(f"{DEPLOYED}/{f}", f"{args.out}/{f}")
    print(f"wrapper-export -> {args.out}: {n_q} trained quant payloads, "
          f"{n_bf} trained bf16 surfaces", flush=True)


if __name__ == "__main__":
    main()
