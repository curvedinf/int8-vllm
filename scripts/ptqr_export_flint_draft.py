#!/usr/bin/env python3
"""Export the trained flint DFlash2 PTQR draft to a servable GPTQ GS128 dir.

The flint draft ships only as bf16 (no deployed int8 wrapper to clone), so
this builds the checkpoint from the bf16 draft layout, replacing the 47
PTQR surfaces:
  - 35 attention/MLP linears (q/k/v/o + gate/up/down x 5 layers) -> GPTQ
    qweight/qzeros/scales in the same gptqmodel layout the target export
    uses (`ptqr_export_gptq.quant_pack`);
  - 12 surfaces the draft keeps bf16 (conv kernel_projection x10, fc,
    candidate_selector.hidden_projection) -> dequantized bf16 values on
    the trained grid (tau=0 semantics), like the qwen draft wrapper.
Untouched: norms, qk norms, conv base kernels, codebooks, hidden_norm.
embed/lm_head are not part of the draft checkpoint (served from the
target).

Usage:
  .venv/bin/python scripts/ptqr_export_flint_draft.py \
    --ckpt ~/models/flint_ptqr_draft_r1/draft_final.pt \
    --src  ~/models/flint-dflash2-bf16 \
    --out  ~/models/flint-dflash2-ptqr-r1-int8
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
from ptqr_export_gptq import quant_pack  # noqa: E402

G = 128


def dequant_grid(w: torch.Tensor, s: torch.Tensor, group: int = G):
    out_f, in_f = w.shape
    wf = w.float().reshape(out_f, in_f // group, group)
    amax = wf.abs().amax(dim=-1)
    s = torch.maximum(s.float(), amax / 127.0)
    z = wf / s.unsqueeze(-1)
    q = (torch.sign(z) * torch.floor(z.abs() + 0.5).clamp(0, 127.0)).clamp(
        -127, 127)
    return (q * s.unsqueeze(-1)).reshape(out_f, in_f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",
                   default=os.path.expanduser(
                       "~/models/flint_ptqr_draft_r1/draft_final.pt"))
    p.add_argument("--src",
                   default=os.path.expanduser("~/models/flint-dflash2-bf16"))
    p.add_argument("--out",
                   default=os.path.expanduser(
                       "~/models/flint-dflash2-ptqr-r1-int8"))
    p.add_argument("--group", type=int, default=G)
    args = p.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    src = Path(args.src)
    base = load_file(str(src / "model.safetensors"))
    cfg = json.loads((src / "config.json").read_text())
    n_layers = int(cfg["num_hidden_layers"])

    quant = []
    bf16_grid = []
    for n in range(n_layers):
        for proj in ("self_attn.q_proj", "self_attn.k_proj",
                     "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            quant.append(f"layers.{n}.{proj}")
        for conv in ("attention_conv", "mlp_conv"):
            bf16_grid.append(f"layers.{n}.{conv}.kernel_projection")
    bf16_grid += ["fc", "candidate_selector.hidden_projection"]
    # trainer-side names for the selector surfaces
    trainer_name = {
        "candidate_selector.hidden_projection": "hidden_projection",
    }

    out = dict(base)  # untouched tensors keep their published dtypes
    for name in quant:
        w = sd[f"{name}.weight"]
        s = sd[f"{name}.scale"]
        qw, qz, sc = quant_pack(w, s, args.group)
        out[f"{name}.qweight"] = qw
        out[f"{name}.qzeros"] = qz
        out[f"{name}.scales"] = sc
        out.pop(f"{name}.weight", None)
        assert tuple(w.shape)[1] % args.group == 0, name

    for name in bf16_grid:
        tname = trainer_name.get(name, name)
        w = sd[f"{tname}.weight"]
        s = sd[f"{tname}.scale"]
        out[f"{name}.weight"] = dequant_grid(
            w, s, args.group).to(torch.bfloat16).contiguous()

    # sanity: every base key is either replaced by GPTQ or kept
    replaced = {f"{n}.weight" for n in quant}
    missing = [k for k in base if k not in out and k not in replaced]
    assert not missing, f"lost tensors: {missing[:5]}"
    for n in quant:
        for suffix in (".qweight", ".qzeros", ".scales"):
            assert f"{n}{suffix}" in out, f"missing {n}{suffix}"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "model.safetensors"))
    cfg["quantization_config"] = {
        "quant_method": "gptq", "bits": 8, "group_size": args.group,
        "desc_act": False, "sym": True, "checkpoint_format": "gptq",
        "lm_head": False, "format": "gptq",
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "quantize_config.json").write_text(json.dumps(
        cfg["quantization_config"], indent=2))
    print(f"exported {len(quant)} GPTQ linears + {len(bf16_grid)} bf16-grid "
          f"surfaces -> {out_dir} ({len(out)} tensors)", flush=True)


if __name__ == "__main__":
    main()
