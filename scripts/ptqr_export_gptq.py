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

ROW_PARALLEL_SUFFIX = ("o_proj", "out_proj", "down_proj")  # column-sharded (concat dim 1)


def _is_row_parallel(name: str) -> bool:
    return name.split(".")[-1] in ROW_PARALLEL_SUFFIX


def load_rank_shards(ckpt_dir: Path, step: int, world: int):
    shards = []
    for r in range(world):
        p = ckpt_dir / f"ptqr_target_step{step}_rank{r}.pt"
        if not p.exists():
            raise FileNotFoundError(p)
        # mmap keeps the shards file-backed (page cache) — the eager load
        # OOM-kills the 61 GB host while training runs (measured exit 137)
        shards.append(torch.load(p, weights_only=True, map_location="cpu",
                                 mmap=True)["model_state_dict"])
    return shards


# GDN in_proj_qkv: each rank's slice is [q_seg | k_seg | v_seg] CONCATENATED
# in range-list order (SSMShardSpec.qkv_rows), NOT contiguous rows of the full
# tensor. Segment sizes per rank at TP4: q=k=k_per*head_k (512), v=v_per*head_v
# (1536). Plain dim-0 concat interleaves q/k/v blocks across ranks — the
# scrambling bug that produced CE 13.6 on UNTRAINED weights (ledger
# PTQR_lr0_stitch_bug).
QKV_SEGMENT_LEAFS = ("in_proj_qkv",)


def _is_qkv_segmented(name: str) -> bool:
    return name.split(".")[-1] in QKV_SEGMENT_LEAFS


def stitch_qkv(name: str, shards) -> torch.Tensor:
    """Scatter each rank's [q|k|v] slice back into the segmented full tensor.

    Geometry from the Qwen3.8-27B GDN config (SSMShardSpec): 16 k-heads,
    48 v-heads, head dims 128 — full rows = 2048 q + 2048 k + 6144 v; each
    TP4 rank holds 512 q + 512 k + 1536 v concatenated in that order.
    """
    import torch as _t
    per = [s[name + ".weight"] for s in shards]
    out_f = sum(p.shape[0] for p in per)
    in_f = per[0].shape[1]
    kd, head = 2048, 128
    vd = out_f - 2 * kd
    assert out_f == 10240 and vd == 6144, f"unexpected qkv geometry {out_f}"
    w = len(per)
    k_per, v_per = kd // w, vd // w
    full = _t.empty(out_f, in_f, dtype=per[0].dtype)
    for r, p in enumerate(per):
        assert p.shape[0] == k_per * head * 2 + v_per * head
        q_seg = p[: k_per * head]
        k_seg = p[k_per * head: 2 * k_per * head]
        v_seg = p[2 * k_per * head:]
        full[r * k_per * head: (r + 1) * k_per * head] = q_seg
        full[kd + r * k_per * head: kd + (r + 1) * k_per * head] = k_seg
        full[2 * kd + r * v_per * head:
             2 * kd + (r + 1) * v_per * head] = v_seg
    return full


def stitch(name: str, shards) -> torch.Tensor:
    """Rebuild the full tensor from per-rank slices."""
    if _is_qkv_segmented(name):
        return stitch_qkv(name, shards)
    dim = 1 if _is_row_parallel(name) else 0
    parts = [s[name + ".weight"] if (name + ".weight") in s
             else s[name] for s in shards]
    if all(torch.equal(parts[0], p) for p in parts[1:]) and dim == 0 \
            and parts[0].shape[dim] < 1000:
        # replicated (norms, biases, small replicated buffers): identical
        # small tensors — take rank 0
        return parts[0]
    return torch.cat(parts, dim=dim)


def stitch_scales(name: str, shards) -> torch.Tensor:
    if _is_qkv_segmented(name):
        # scales follow the same [q|k|v] row segmentation as the weights
        per = [s[name + ".scale"] for s in shards]
        import torch as _t
        full = _t.empty(sum(p.shape[0] for p in per), per[0].shape[1],
                        dtype=per[0].dtype)
        # groups of 128 input dims; rows = output rows (segmented)
        seg_rows = [p.shape[0] // 3 for p in per]  # not equal q/k/v! fall back
        # q rows == k rows < v rows: split 1:1:2 by row count
        total_r = per[0].shape[0]
        qr = total_r // 4          # q rows per rank (= k rows)
        vr = total_r - 2 * qr      # v rows per rank
        kd_rows = sum(qr for _ in per)
        for r, p in enumerate(per):
            full[r * qr: (r + 1) * qr] = p[:qr]
            full[kd_rows + r * qr: kd_rows + (r + 1) * qr] = p[qr: 2 * qr]
            full[2 * kd_rows + r * vr: 2 * kd_rows + (r + 1) * vr] = p[2 * qr:]
        return full
    parts = [s[name + ".scale"] for s in shards]
    if all(torch.equal(parts[0], p) for p in parts[1:]) and parts[0].dim() == 1:
        return parts[0]
    # scale layout [out, in/G]: column-sharded modules concat dim 1
    dim = 1 if _is_row_parallel(name) else 0
    return torch.cat(parts, dim=dim)


def quant_pack(w: torch.Tensor, scale: torch.Tensor, group: int):
    """Final quantization + gptqmodel-format packing.

    w [out, in] master; scale [out, in/G] trained (fp32). The trained scale
    can drift below a group's feasibility bound (independent SGD on w and s —
    measured 0.018% saturated elements); project it back up so RTN is exact
    everywhere instead of clamping outliers.

    Returns (qweight int32, qzeros int32, scales fp16) in checkpoint layout.
    """
    out_f, in_f = w.shape
    wf = w.float().reshape(out_f, in_f // group, group)
    amax = wf.abs().amax(dim=-1)
    scale = torch.maximum(scale.float(), amax / 127.0)
    s16 = scale.to(torch.float16).float()
    z = wf / s16.unsqueeze(-1)
    q = (torch.sign(z) * torch.floor(z.abs() + 0.5).clamp(0, 127.0)).clamp(-127, 127)
    stored = (q.reshape(out_f, in_f) + 128.0).to(torch.uint8)          # [out, in]
    # qweight: [in/4, out], LSB-first along consecutive INPUT dims
    st = stored.t().contiguous()                                       # [in, out]
    qw = (st[0::4].long() | (st[1::4].long() << 8) |
          (st[2::4].long() << 16) | (st[3::4].long() << 24)).to(torch.int32)
    scales = s16.to(torch.float16).t().contiguous()                     # [in/G, out] FP16 STORAGE (the loader parameter is fp16; fp32 bytes = garbage)
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
    p.add_argument("--aux_src",
                   default="/home/curved/models/Qwen3.8-27B-GPTQ-8bit-gs128",
                   help="dir to copy config/tokenizer/generation files from")
    args = p.parse_args()

    shards = load_rank_shards(Path(args.ckpt_dir), args.step, args.world)
    base = torch.load(args.base, weights_only=True, mmap=True)["model_state_dict"]

    names = set()
    scale_names = set()
    for s in shards:
        for k in s:
            if k.endswith(".weight"):
                names.add(k[: -len(".weight")])
            elif k.endswith(".scale"):
                scale_names.add(k[: -len(".scale")])
    print(f"{len(names)} weight tensors, {len(scale_names)} PTQR scale sets",
          flush=True)

    def build(name: str) -> tuple[str, torch.Tensor]:
        """Build the export tensor for one weight name (streamed, freed).

        Trainer shard names are the stripped model-tree form; transformers
        5.9 nests decoder submodules under `.layer.` while the published
        checkpoints do not — normalize that segment away. lm_head is
        top-level in the deployed format.
        """
        norm = name.replace(".layer.", ".", 1) if ".layer." in name else name
        if norm == "lm_head":
            key = "lm_head"
        else:
            key = "model.language_model." + norm
        if name in scale_names:
            full = stitch(name, shards)
            scale = stitch_scales(name, shards)
            if full.shape[1] != args.group * scale.shape[1]:
                return key + ".weight", full.to(torch.bfloat16)
            qw, qz, sc = quant_pack(full, scale, args.group)
            del full, scale
            return key, (qw, qz, sc)  # tuple expands to 3 keys
        return key + ".weight", base["model." + norm + ".weight"].to(torch.bfloat16)

    quant_keys = [n for n in sorted(names) if n in scale_names]
    bf16_keys = [n for n in sorted(names) if n not in scale_names]
    # extra base tensors not in shards: ALL of them — A_log/dt_bias/biases do
    # not end in .weight and were silently dropped in the first export,
    # leaving GDN decay/delta at config-init (garbage output — measured)
    stripped_seen = set()
    for n in names:
        stripped_seen.add(n.replace(".layer.", ".", 1) if ".layer." in n else n)
    extra = [k for k in base if k[len("model."):] not in stripped_seen
             and not k.endswith(".weight")]

    n_quant = 0
    total_bytes = 0
    file_plan: list[list[str]] = []  # per-file list of (kind, name)
    # ~4.5 GiB per file; quantized tensor ~= out*in bytes, bf16 ~= 2*out*in
    def tensor_bytes(name):
        if name in scale_names and name not in bf16_keys:
            # quantized: 1 byte/elem + scales; approximate with the weight
            q = shards[0][name + ".weight"]
            if isinstance(q, torch.Tensor):
                return q.numel() * 4  # upper bound on packed+scales
        q = shards[0][name + ".weight"]
        return q.numel() * q.element_size()

    cur, cur_b = [], 0
    for n in quant_keys + bf16_keys:
        b = tensor_bytes(n)
        if cur_b + b > 4.5e9 and cur:
            file_plan.append(cur)
            cur, cur_b = [], 0
        cur.append(n)
        cur_b += b
    if cur:
        file_plan.append(cur)

    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_files = len(file_plan)
    for i, names_i in enumerate(file_plan):
        chunk: dict[str, torch.Tensor] = {}
        for n in names_i:
            key, val = build(n)
            if isinstance(val, tuple):
                chunk[key + ".qweight"] = val[0]
                chunk[key + ".qzeros"] = val[1]
                chunk[key + ".scales"] = val[2]
                n_quant += 1
            else:
                chunk[key] = val
        fn = f"model-{i+1:05d}-of-{n_files:05d}.safetensors"
        save_file(chunk, str(out_dir / fn))
        for k in chunk:
            index["weight_map"][k] = fn
        index["metadata"]["total_size"] += sum(
            t.numel() * t.element_size() for t in chunk.values())
        del chunk
        print(f"  wrote {fn}", flush=True)
    # base-only tensors (buffers, embed if absent, etc.)
    if extra:
        ref_idx = Path(args.aux_src) / "model.safetensors.index.json"
        ref_keys = set()
        if ref_idx.exists():
            ref_keys = set(json.loads(ref_idx.read_text())["weight_map"])
        extra = [k for k in extra if
                 ("model.language_model." + k[len("model."):]
                  if k.startswith("model.") else k) in ref_keys]
        chunk = {}
        for k in extra:
            kk = ("lm_head" if k == "model.lm_head.weight"
                  else "model.language_model." + k[len("model."):])
            chunk[kk] = base[k].to(torch.bfloat16)
        fn = f"model-{n_files+1:05d}-of-{n_files+1:05d}.safetensors"
        # renumber: simplest is a single extra file; fix index filenames
        for i in range(n_files):
            old = f"model-{i+1:05d}-of-{n_files:05d}.safetensors"
            new = f"model-{i+1:05d}-of-{n_files+1:05d}.safetensors"
            (out_dir / old).rename(out_dir / new)
            for k, v in list(index["weight_map"].items()):
                if v == old:
                    index["weight_map"][k] = new
        save_file(chunk, str(out_dir / fn))
        for k in chunk:
            index["weight_map"][k] = fn
        index["metadata"]["total_size"] += sum(
            t.numel() * t.element_size() for t in chunk.values())
        print(f"  wrote {fn} ({len(chunk)} base-only tensors)", flush=True)
        n_files += 1
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    print(f"{n_quant} tensors packed to GPTQ GS128; "
          f"{len(index) - 3 * n_quant - 1} kept bf16", flush=True)
    qcfg = {
        "bits": 8, "group_size": args.group, "desc_act": False, "sym": True,
        "lm_head": False, "method": "gptq", "quant_method": "gptq",
        "checkpoint_format": "gptq", "pack_dtype": "int32",
        "meta": {"quantizer": ["ptqr-retrain:scripts/ptqr_export_gptq.py"]},
    }
    (out_dir / "quantize_config.json").write_text(json.dumps(qcfg, indent=2))
    # aux files the loader needs (same architecture/tokenizer as source)
    src_cfg = Path(args.aux_src) if args.aux_src else Path(args.base).parent
    for fname in ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "generation_config.json", "chat_template.jinja",
                  "processor_config.json"):
        f = src_cfg / fname
        if f.exists():
            (out_dir / fname).write_bytes(f.read_bytes())
    print("aux files copied (config/tokenizer/generation)", flush=True)
    print(f"exported -> {out_dir}")


if __name__ == "__main__":
    main()
