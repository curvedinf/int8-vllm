#!/usr/bin/env python3
"""Visibility matrix: which window positions does each verify row actually
attend to? One-hot probe: set K/V at ONE window position to an extreme marker
at a time; row j's output ≈ marker V iff row j sees that position.
Prints the 14x14 visibility matrix (X = sees). Correct = lower-triangular
(row j sees window positions <= j) when the anchor sees only the prefix.
"""
import sys

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "../aiter")
from aiter.ops.triton.unified_attention import unified_attention  # noqa
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (  # noqa
    triton_reshape_and_cache_flash_per_token_head_quant as write_kv,
)
from vllm.v1.kv_cache_interface import KVQuantMode  # noqa

torch.manual_seed(13)
DEV = "cuda"
NKV, HQ, HS, BS = 4, 24, 256, 32
SEQLEN = 4096
PAD = 4
CONTENT = 2 * (HS + PAD)
NBLOCKS = (SEQLEN + 64) // BS + 2
NS = 13
QL = NS + 1
P = SEQLEN  # window starts right after the prefix

cache = torch.zeros(NBLOCKS, NKV, BS, CONTENT, dtype=torch.int8, device=DEV)
f32 = torch.tensor([], dtype=torch.float32, device=DEV).set_(cache.untyped_storage())
C = CONTENT // 4
k_sc = torch.as_strided(f32, (NBLOCKS, BS, NKV), (NKV * BS * C, C, BS * C),
                        storage_offset=HS // 4)
v_sc = torch.as_strided(f32, (NBLOCKS, BS, NKV), (NKV * BS * C, C, BS * C),
                        storage_offset=(HS + PAD + HS) // 4)
split = cache.transpose(1, 2)
k_view = split[..., :HS]
v_view = split[..., HS + PAD:HS + PAD + HS]

# write the plain prefix once (all normal)
k_pre = torch.randn(P, NKV, HS, dtype=torch.float16, device=DEV) * 0.3
v_pre = torch.randn(P, NKV, HS, dtype=torch.float16, device=DEV) * 0.3
write_kv(k_pre, v_pre, k_view, v_view, k_sc, v_sc,
         torch.arange(P, dtype=torch.int64, device=DEV),
         kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD)

bt = torch.arange(NBLOCKS, dtype=torch.int32, device=DEV).unsqueeze(0)
q_ver = torch.randn(QL, HQ, HS, dtype=torch.bfloat16, device=DEV) * 0.3

def run_window():
    out = torch.empty(QL, HQ, HS, dtype=torch.bfloat16, device=DEV)
    cu_q = torch.tensor([0, QL], dtype=torch.int32, device=DEV)
    kv_len = torch.tensor([P + QL], dtype=torch.int32, device=DEV)
    unified_attention(
        q=q_ver, k=k_view, v=v_view, out=out,
        cu_seqlens_q=cu_q, max_seqlen_q=QL, seqused_k=kv_len,
        max_seqlen_k=P + QL, softmax_scale=HS ** -0.5, causal=True,
        window_size=(-1, -1), block_table=bt, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
        k_scale_cache=k_sc, v_scale_cache=v_sc,
    )
    return out

print("visibility matrix (row j sees window position p?): correct = lower-triangular")
print("     " + "".join(f"{p:>3}" for p in range(QL)))
vis = torch.zeros(QL, QL, dtype=torch.int32)
for p in range(QL):
    # marker at window position p: extreme K channel0 (dominant logit) + marker V
    k_m = torch.zeros(1, NKV, HS, dtype=torch.float16, device=DEV)
    k_m[0, :, 0] = 300.0
    v_m = torch.zeros(1, NKV, HS, dtype=torch.float16, device=DEV)
    v_m[0, :, 0] = 100.0 + p  # marker value identifies the position
    write_kv(k_m, v_m, k_view, v_view, k_sc, v_sc,
             torch.tensor([P + p], dtype=torch.int64, device=DEV),
             kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD)
    out = run_window()
    # row j sees p iff its output channel0 moved toward the marker
    vis[:, p] = (out[:, :, 0].abs() > 50).any(dim=1).int()  # dominated by marker
    # clear the marker back to normal
    k_r = torch.randn(1, NKV, HS, dtype=torch.float16, device=DEV) * 0.3
    v_r = torch.randn(1, NKV, HS, dtype=torch.float16, device=DEV) * 0.3
    write_kv(k_r, v_r, k_view, v_view, k_sc, v_sc,
             torch.tensor([P + p], dtype=torch.int64, device=DEV),
             kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD)
for j in range(QL):
    print(f"row {j:2d} " + "  ".join("X" if vis[j, p] else "." for p in range(QL)))
# also: does the anchor row see the PREFIX correctly? (baseline check)
out0 = run_window()
print("\nrow outputs channel0 (no markers):", [round(x, 2) for x in out0[:, 0, 0].tolist()])
