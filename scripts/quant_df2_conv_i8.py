#!/usr/bin/env python3
"""E4: quantize the DFlash2 draft `*.base_kernel` conv tensors (bf16
[2, taps, hidden]) to symmetric int8 with per-[side,tap] fp32 scales,
stored alongside the bf16 originals in a new draft checkpoint dir.

The base kernel is only ever ADDED to the projected delta (never
matmul'd), so DFlashGroupedConv._convolve dequantizes the int8 planes on
add when VLLM_GFX908_DF2_CONV_I8=1 and skips the bf16 param entirely.

Output: ~/.cache/huggingface/dflash2-int8/Qwen3.8-27B-DFlash2-GPTQ-8bit-convq8
"""
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

SRC = os.path.expanduser(
    "~/.cache/huggingface/dflash2-int8/Qwen3.8-27B-DFlash2-GPTQ-8bit"
)
DST = os.path.expanduser(
    "~/.cache/huggingface/dflash2-int8/Qwen3.8-27B-DFlash2-GPTQ-8bit-convq8"
)

shutil.copytree(SRC, DST, dirs_exist_ok=True)

st = os.path.join(DST, "model.safetensors")
tensors = dict(load_file(st))

worst = 0.0
new_keys = []
for name in sorted(tensors):
    if not name.endswith(".base_kernel"):
        continue
    w = tensors[name].float()  # [sides, taps, hidden]
    sides, taps, hidden = w.shape
    planes = w.reshape(sides * taps, hidden)
    # symmetric per-[side,tap] absmax: taps/sides differ in scale
    amax = planes.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scales = (amax / 127.0).to(torch.float32).reshape(sides, taps)
    q = (planes / amax * 127.0).round().clamp(-127, 127).to(torch.int8)
    dq = q.float() * scales.reshape(-1, 1)
    err = ((dq - planes).norm() / planes.norm()).item()
    worst = max(worst, err)
    tensors[f"{name}.i8"] = q.reshape(sides, taps, hidden).contiguous()
    tensors[f"{name}.scale"] = scales.contiguous()
    new_keys += [f"{name}.i8", f"{name}.scale"]
    print(
        f"{name}: rel-L2 {err * 100:.4f}%  "
        f"scales [{scales.min():.3e}, {scales.max():.3e}]"
    )

assert new_keys, "no *.base_kernel tensors found"
save_file(tensors, st, metadata={"format": "pt"})

# keep the loader's weight_map in sync so the .i8/.scale names are iterated
idx_path = os.path.join(DST, "model.safetensors.index.json")
if os.path.exists(idx_path):
    with open(idx_path) as f:
        idx = json.load(f)
    for key in new_keys:
        idx["weight_map"][key] = "model.safetensors"
    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=2)

print(f"worst rel-L2 {worst * 100:.4f}%  ({len(new_keys) // 2} tensors)  -> {DST}")
assert worst < 0.01, "quant error too large"
