#!/usr/bin/env python3
"""Export PTQR rank-shard checkpoints to the GPTQ GS128 format vLLM loads.

Stitches the per-rank trainer shards back into full tensors (deterministic
geometry: row-sharded = concat dim0; row-PARALLEL modules o_proj/out_proj/
down_proj = concat dim1), quantizes each PTQR master onto its deployed grid,
and packs to the gptqmodel checkpoint format byte layout verified against the
deployed curveinf checkpoint:

  qweight int32 [in/4, out]  — 4x uint8 stored as (q+128), LSB-first along
                                consecutive input dims
  qzeros  int32 [in/G, out/4] — constant 0x7F7F7F7F (zero-point 128, stored -1)
  scales  fp16 [in/G, out]     — group scales, per output channel
  (sym=true, desc_act=false, group_size=128)

The untied LM head and embeddings stay bf16 in the checkpoint (the serving
loader applies its own per-channel / per-row int8 conversion at load).

Usage (any torch venv):
  python scripts/ptqr_export_gptq.py \
    --ckpt_dir /home/curved/models/ptqr_out --step 300 --world 4 \
    --base /home/curved/models/Qwen3.8-27B-bf16-ref-lm.pt \
    --out /home/curved/models/Qwen3.8-27B-PTQR-INT8-GS128
"""
import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

ROW_PARALLEL = (".o_proj.", ".out_proj.", ".mlp.down_proj.")


def load_rank_shards(ckpt_dir: Path, step: int, world: int):
    shards = []
    for r in range(world):
        p = ckpt_dir / f"ptqr_target_step{step}_rank{r}.pt"
        if not p.exists():
            raise FileNotFoundError(p)
        shards.append(torch.load(p, weights_only=True, map_location="cpu"))
    return shards


def stitch(name: str, shards) -> torch.Tensor:
    """Rebuild the full tensor from per-rank slices."""
    if any(k in f".{name}." for k in ROW_PARALLEL) or \
            any(name.endswith(k.rstrip(".")) for k in ROW_PARALLEL):
        dim = 1
    else:
        dim = 0
    parts = [s["model_state_dict"][name + ".weight"] if (name + ".weight") in
             s["model_state_dict"] else s["model_state_dict"][name]
             for s in shards]
    if all(torch.equal(parts[0], p) for p in parts[1:]) and parts[0].shape[dim] % 4 != 0:
        # replicated (norms, biases, replicated buffers): shapes identical and
        # not shard-shaped — take rank 0
        return parts[0]
    full = torch.cat(parts, dim=dim)
    return full


def stitch_scales(name: str, shards) -> torch.Tensor:
    parts = [s["model_state_dict"][name + ".scale"] for s in shards]
    if all(torch.equal(parts[0], p) for p in parts[1:]) and parts[0].dim() == 1:
        return parts[0]
    # scale layout [out, in/G]: column-sharded modules concat dim 1
    dim = 1 if any(k in name for k in ROW_PARALLEL) else 0
    return torch.cat(parts, dim=dim)


def quant_pack(w: torch.Tensor, scale: torch.Tensor, group: int):
    """Final quantization + gptqmodel-format packing.

    w [out, in] master; scale [out, in/G] trained (fp32). Returns
    (qweight int32, qzeros int32, scales fp16) in checkpoint layout.
    """
    out_f, in_f = w.shape
    wf = w.float().reshape(out_f, in_f // group, group)
    s16 = scale.to(torch.float16).float()
    z = wf / s16.unsqueeze(-1)
    q = (torch.sign(z) * torch.floor(z.abs() + 0.5).clamp(0, 127.0)).clamp(-127, 127)
    stored = (q.reshape(out_f, in_f) + 128.0).to(torch.uint8)          # [out, in]
    # qweight: [in/4, out], LSB-first along consecutive INPUT dims
    st = stored.t().contiguous()                                       # [in, out]
    qw = (st[0::4].long() | (st[1::4].long() << 8) |
          (st[2::4].long() << 16) | (st[3::4].long() << 24)).to(torch.int32)
    scales = s16.t().contiguous()                                      # [in/G, out]
    qz = torch.full((in_f // group, out_f // 4), 2139062143, dtype=torch.int32)
    return qw, qz, scales


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="/home/curved/models/ptqr_out")
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--base", default="/home/curved/models/Qwen3.8-27B-bf16-ref-lm.pt")
    p.add_argument("--out", default="/home/curved/models/Qwen3.8-27B-PTQR-INT8-GS128")
    p.add_argument("--group", type=int, default=128)
    args = p.parse_args()

    shards = load_rank_shards(Path(args.ckpt_dir), args.step, args.world)
    base = torch.load(args.base, weights_only=True, mmap=True)["model_state_dict"]

    names = set()
    scale_names = set()
    for s in shards:
        for k in s["model_state_dict"]:
            if k.endswith(".weight"):
                names.add(k[: -len(".weight")])
            elif k.endswith(".scale"):
                scale_names.add(k[: -len(".scale")])
    print(f"{len(names)} weight tensors, {len(scale_names)} PTQR scale sets")

    out_sd: dict[str, torch.Tensor] = {}
    n_quant = 0
    for name in sorted(names):
        full = stitch(name, shards)
        key = "model.language_model." + name[len("model."):] if name.startswith("model.") \
            else name
        if name in scale_names:
            scale = stitch_scales(name, shards)
            if full.shape[1] != args.group * scale.shape[1]:
                # per-channel LM head stays bf16 (loader requantizes)
                out_sd[key + ".weight"] = full.to(torch.bfloat16)
                continue
            qw, qz, sc = quant_pack(full, scale, args.group)
            out_sd[key + ".qweight"] = qw
            out_sd[key + ".qzeros"] = qz
            out_sd[key + ".scales"] = sc
            n_quant += 1
        else:
            # non-Linear tensors come from the pristine base checkpoint (the
            # trainer froze them; the embed must be pre-int8-conversion since
            # the loader applies its own)
            out_sd[key + ".weight"] = base[name + ".weight"].to(torch.bfloat16)
    # any base tensors not present in shards (buffers etc.)
    for k, v in base.items():
        if k.endswith(".weight") and k[:-len(".weight")] not in names:
            out_sd.setdefault("model.language_model." + k[len("model."):]
                              if k.startswith("model.") else k, v.to(torch.bfloat16))
    print(f"{n_quant} tensors packed to GPTQ GS128; "
          f"{len(out_sd) - 3 * n_quant} kept bf16")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # single-file shards of ~5 GB
    n_files = math.ceil(sum(t.numel() * t.element_size() for t in out_sd.values()) / 4.5e9)
    items = sorted(out_sd.items(), key=lambda kv: kv[0])
    per = math.ceil(len(items) / n_files)
    index = {"metadata": {"total_size": 0}}
    for i in range(n_files):
        chunk = dict(items[i * per:(i + 1) * per])
        save_file(chunk, str(out_dir / f"model-{i+1:05d}-of-{n_files:05d}.safetensors"))
        for k in chunk:
            index[k] = f"model-{i+1:05d}-of-{n_files:05d}.safetensors"
        index["metadata"]["total_size"] += sum(
            t.numel() * t.element_size() for t in chunk.values())
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    qcfg = {
        "bits": 8, "group_size": args.group, "desc_act": False, "sym": True,
        "lm_head": False, "method": "gptq", "quant_method": "gptq",
        "checkpoint_format": "gptq", "pack_dtype": "int32",
        "meta": {"quantizer": ["ptqr-retrain:scripts/ptqr_export_gptq.py"]},
    }
    (out_dir / "quantize_config.json").write_text(json.dumps(qcfg, indent=2))
    print(f"exported -> {out_dir}")


if __name__ == "__main__":
    main()
