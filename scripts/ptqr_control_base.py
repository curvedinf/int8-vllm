#!/usr/bin/env python3
"""Build a control (trusted-eval) base .pt from PTQR trainer rank shards.

Reconstructs full tensors from the 4 TP shards, grid-quantizes every PTQR
linear exactly like the export path (feasibility-projected scales, fp16
cast), and emits the flat-name bf16 state dict the trainer's --no_ptqr
control harness consumes. This is the weight-quality gate that exposed the
trained-rung damage (see ledger PTQR_control_CE_anchors).

Usage:
  python scripts/ptqr_control_base.py --ckpt_dir .../ptqr_r9 --step 15 \
      --out /home/curved/models/r9s15_ctl.pt
"""
import argparse
from pathlib import Path

import torch

G = 128
ROWPAR = ("o_proj", "out_proj", "down_proj")
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent))
from ptqr_export_gptq import stitch_qkv  # noqa: E402  (segmented [q|k|v] fix)


def flat(k: str) -> str:
    return k.replace(".layer.", ".", 1)


def replicas(vals) -> bool:
    v0 = vals[0]
    return all(v.shape == v0.shape and torch.equal(v, v0) for v in vals[1:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    shards = [torch.load(f"{args.ckpt_dir}/ptqr_target_step{args.step}_rank{r}.pt",
                         weights_only=True, mmap=True,
                         map_location="cpu")["model_state_dict"]
              for r in range(args.world)]
    out, n_q = {}, 0
    for k in shards[0]:
        if k.endswith(".scale"):
            continue
        leaf = k.split(".")[-1] if not k.endswith(".weight") else k.split(".")[-2]
        tgt = "model." + flat(k)
        vals = [s[k] for s in shards]
        gdn_param = (not k.endswith(".weight")) or leaf == "conv1d"
        if gdn_param:
            out[tgt] = vals[0] if replicas(vals) else torch.cat(vals, 0)
            continue
        name = k[:-len(".weight")]
        if replicas(vals):
            out[tgt] = vals[0]
            continue
        if leaf == "in_proj_qkv":
            w = stitch_qkv(name, shards).float()
        else:
            dim = 1 if leaf in ROWPAR else 0
            w = torch.cat(vals, dim=dim).float()
        sc = torch.cat([s[name + ".scale"] for s in shards], dim=dim).float() \
            if all((name + ".scale") in s for s in shards) else None
        if sc is None or w.dim() != 2 or w.shape[1] != G * sc.shape[1]:
            out[tgt] = w.to(torch.bfloat16)  # per-channel lm_head etc.
            continue
        wf = w.reshape(w.shape[0], -1, G)
        s_eff = torch.maximum(sc, wf.abs().amax(-1) / 127.0).to(torch.float16).float()
        q = torch.clamp(torch.round(wf / s_eff.unsqueeze(-1)), -127, 127)
        out[tgt] = (q * s_eff.unsqueeze(-1)).reshape(w.shape).to(torch.bfloat16)
        n_q += 1
        del w, wf
    if "lm_head.weight" in shards[0]:
        out["lm_head.weight"] = torch.cat([s["lm_head.weight"] for s in shards], 0)
    torch.save({"model_state_dict": out}, args.out)
    print(f"control base -> {args.out}: {len(out)} tensors, {n_q} grid-quantized")


if __name__ == "__main__":
    main()
