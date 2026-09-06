#!/usr/bin/env python3
"""Wrapper-export the trained PTQR draft to servable GPTQ GS128.

Same proven vehicle as the target (ledger PTQR_P1_export_vehicle): copy the
deployed int8 DFlash2 checkpoint verbatim (its layout, quantize_config and
g_idx all load through the fork's AITER W8A8 loader), then overwrite:
  - the 35 qweight linears (q/k/v/o + gate/up/down x 5 layers) with
    quant_pack(trained weight, trained scale, G128)
  - the 12 PTQR surfaces the checkpoint keeps bf16 (conv kernel_projection
    x10, fc, selector hidden_projection) with the DEQUANTIZED trained grid
    (deployed wrapper keeps these bf16; same pattern as target in_proj_a/b)

Untouched: embed/lm_head come from the target at serve time, codebooks,
base_kernels, norms (training only updated PTQR weights), g_idx.

Usage:
  .venv/bin/python scripts/ptqr_draft_export.py \
      --ckpt /home/curved/models/ptqr_draft_r1/draft_final.pt \
      --out /home/curved/models/dflash2-ptqr-r1
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
from ptqr_export_gptq import quant_pack  # noqa: E402

TEMPLATE = Path.home() / ".cache/huggingface/dflash2-int8/Qwen3.8-27B-DFlash2-GPTQ-8bit"
G = 128


def dequant_grid(full: torch.Tensor, scale: torch.Tensor, group: int):
    """Values ON the trained grid (what PTQRLinear at tau=0 computes)."""
    out_f, in_f = full.shape
    wf = full.float().reshape(out_f, in_f // group, group)
    amax = wf.abs().amax(dim=-1)
    s16 = torch.maximum(scale.float(), amax / 127.0).to(torch.float16).float()
    q = torch.sign(wf / s16.unsqueeze(-1)) * \
        torch.floor((wf / s16.unsqueeze(-1)).abs() + 0.5).clamp(0, 127.0)
    return (q.reshape(out_f, in_f) * s16.repeat_interleave(group, dim=1)
            ).to(torch.bfloat16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ck = torch.load(args.ckpt, weights_only=True, map_location="cpu")
    # trainer DraftModel keeps selector parts at top level; the deployed
    # checkpoint nests them under candidate_selector.
    alias = {"hidden_projection": "candidate_selector.hidden_projection"}
    raw_scales = {k[:-len(".scale")] for k in ck if k.endswith(".scale")}
    rev = {alias.get(s, s): s for s in raw_scales}  # checkpoint name -> ckpt
    print(f"ckpt: {len(ck)} tensors, {len(raw_scales)} PTQR linears", flush=True)

    if Path(args.out).exists():
        shutil.rmtree(args.out)
    shutil.copytree(TEMPLATE, args.out)

    fp = Path(args.out) / "model.safetensors"
    sd = load_file(fp)
    n_q = n_bf = 0
    for k in list(sd):
        if k.endswith(".qweight"):
            base = k[:-len(".qweight")]
            if base not in rev:
                print(f"  KEEP deployed payload: {base}", flush=True)
                continue
            full, sc = ck[rev[base] + ".weight"], ck[rev[base] + ".scale"]
            assert full.dim() == 2 and full.shape[1] == G * sc.shape[1], \
                (base, full.shape, sc.shape)
            qw, qz, s16 = quant_pack(full, sc, G)
            sd[k] = qw
            sd[base + ".qzeros"] = qz
            sd[base + ".scales"] = s16
            n_q += 1
        elif k.endswith(".weight"):
            base = k[:-len(".weight")]
            if base in rev:
                sd[k] = dequant_grid(ck[rev[base] + ".weight"],
                                     ck[rev[base] + ".scale"], G)
                n_bf += 1
    save_file(sd, str(fp))
    print(f"draft wrapper-export -> {args.out}: {n_q} trained quant payloads "
          f"(expect 35), {n_bf} trained dequant-grid bf16 surfaces (expect 12)",
          flush=True)


if __name__ == "__main__":
    main()
