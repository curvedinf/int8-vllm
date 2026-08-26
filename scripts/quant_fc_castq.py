#!/usr/bin/env python3
"""E1: cast-quantize the DFlash2 draft `fc.weight` (bf16 [5120out, 25600in])
into GPTQ GS128 format, producing a new draft checkpoint dir.

No GPTQ calibration: symmetric per-group (gs=128 along K) absmax cast.
Runtime chain dequantizes with these scales, then the standard CK
per-channel requant applies — the same end-state every other GS128 draft
linear reaches. Loader needs no changes (auto-discovered from dtypes).

Output: $DFLASH2_FC_Q8_DIR (default: $DFLASH2_INT8_DIR with -fcq8 suffix)
"""
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

SRC = os.environ.get(
    "DFLASH2_INT8_DIR",
    os.path.expanduser(
        "~/.cache/int8-vllm/dflash2-int8/Qwen3.8-27B-DFlash2-GPTQ-8bit"
    ),
)
DST = os.environ.get("DFLASH2_FC_Q8_DIR", SRC + "-fcq8")
GS = 128

shutil.copytree(SRC, DST, dirs_exist_ok=True)

st = os.path.join(DST, "model.safetensors")
tensors = dict(load_file(st))
W = tensors.pop("fc.weight").float()  # [N=5120, K=25600]
N, K = W.shape
assert (N, K) == (5120, 25600), (N, K)

# per-group symmetric cast: q = round(w / (absmax_g/127)), byte = q + 128
Wg = W.view(N, K // GS, GS)
amax = Wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)  # [N, K/GS, 1]
scales = (amax / 127.0).to(torch.float16)  # [N, K/GS]
q = (Wg / amax * 127.0).round().clamp(-127, 127).to(torch.int8)  # [N,K/GS,GS]
q = q.reshape(N, K)

# GPTQ layout: qweight int32 [K/4, N], LSB-first bytes along K, uint8b128 bias
qk = q.t().contiguous()  # [K, N]
u8 = (qk.to(torch.int16) + 128).to(torch.uint8).view(torch.uint8)  # bytes
u8 = u8.reshape(K // 4, 4, N).permute(0, 2, 1).contiguous()  # [K/4, N, 4]
qweight = (
    u8[..., 0].to(torch.int32)
    | (u8[..., 1].to(torch.int32) << 8)
    | (u8[..., 2].to(torch.int32) << 16)
    | (u8[..., 3].to(torch.int32) << 24)
).contiguous()

# scales fp16 [K/GS, N]; qzeros inert 0x7F7F7F7F [K/GS, N/4]; g_idx k//GS
scales_t = scales.reshape(N, K // GS).t().contiguous()  # [K/GS, N]
qzeros = torch.full(
    (K // GS, N // 4), 0x7F7F7F7F, dtype=torch.int32
)
g_idx = (torch.arange(K, dtype=torch.int32) // GS).contiguous()

tensors["fc.qweight"] = qweight
tensors["fc.qzeros"] = qzeros
tensors["fc.scales"] = scales_t
tensors["fc.g_idx"] = g_idx
save_file(tensors, st, metadata={"format": "pt"})

# dequant sanity vs original
sc2 = scales.reshape(N, K // GS, 1).float()
w_dq = (q.view(N, K // GS, GS).float() * sc2).reshape(N, K)
err = (w_dq - W).norm() / W.norm()
print(f"fc cast-quant: weight rel-L2 {err.item()*100:.3f}%  -> {DST}")
assert err < 0.02, "cast error too large"
