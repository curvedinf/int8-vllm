#!/usr/bin/env python3
"""Per-row causal-mask test for the AITER unified-attention verify window.

The earlier qlen test (test_ua_qlen_causal.py) only checked row 0 (the anchor
row, which cannot see the window). The garble anatomy shows the target
"agreeing" (p=1.0) with garbage draft tokens that sit LATER in the verify
window than the row being tested — possible only if the in-window causal mask
leaks. This test puts hot/garbage K/V at chosen LATE window positions and
checks every row's output against a per-row reference that attends only to
prefix + window positions <= the row's own position.

PASS = per-row verify output matches the causal reference for all 14 rows.
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

# normal K/V for the prefix
k_all = torch.randn(SEQLEN + 64, NKV, HS, dtype=torch.float16, device=DEV)
v_all = torch.randn(SEQLEN + 64, NKV, HS, dtype=torch.float16, device=DEV)
P = SEQLEN
# GARBAGE: extreme K (hot attention logit) + extreme V at a LATE window position
GARB = P + 10
k_all[GARB] = 0.0
k_all[GARB, :, 0] = 300.0   # huge first-channel K -> dominates any dot product
v_all[GARB] = 0.0
v_all[GARB, :, 0] = 1e4     # huge V signature

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
slot_mapping = torch.arange(SEQLEN + 64, dtype=torch.int64, device=DEV)
write_kv(k_all, v_all, k_view, v_view, k_sc, v_sc, slot_mapping,
         kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD)

# dequantized fp64 reference cache
Kd = (k_view.float().transpose(0, 1).reshape(NBLOCKS * BS, NKV, HS)
      * k_sc.transpose(0, 1).reshape(NBLOCKS * BS, NKV, 1)).double()
Vd = (v_view.float().transpose(0, 1).reshape(NBLOCKS * BS, NKV, HS)
      * v_sc.transpose(0, 1).reshape(NBLOCKS * BS, NKV, 1)).double()

bt = torch.arange(NBLOCKS, dtype=torch.int32, device=DEV).unsqueeze(0)

# verify-style call: 14 rows, positions P..P+13, window kv up to P+13
q_ver = torch.randn(QL, HQ, HS, dtype=torch.bfloat16, device=DEV)
out_ver = torch.empty(QL, HQ, HS, dtype=torch.bfloat16, device=DEV)
cu_q = torch.tensor([0, QL], dtype=torch.int32, device=DEV)
kv_len = torch.tensor([P + QL], dtype=torch.int32, device=DEV)
unified_attention(
    q=q_ver, k=k_view, v=v_view, out=out_ver,
    cu_seqlens_q=cu_q, max_seqlen_q=QL, seqused_k=kv_len,
    max_seqlen_k=P + QL, softmax_scale=HS ** -0.5, causal=True,
    window_size=(-1, -1), block_table=bt, softcap=0.0,
    q_descale=None, k_descale=None, v_descale=None,
    k_scale_cache=k_sc, v_scale_cache=v_sc,
)

# per-row fp64 reference: row j (position P+j) attends to prefix + window <= P+j
gqa = HQ // NKV
print(f"garbage at window position {GARB} (row index {GARB - P}); cache through P+{QL-1}")
print("row | max|verify - causal_ref|  verdict")
bad = 0
Kdc = Kd.cpu(); Vdc = Vd.cpu()
for j in range(QL):
    pos = P + j
    qi = q_ver[j].double().cpu() * (HS ** -0.5)
    ref = torch.empty(HQ, HS, dtype=torch.float64)
    for hq in range(HQ):
        hkv = hq // gqa
        logits = Kdc[: pos + 1, hkv] @ qi[hq]
        p = torch.softmax(logits, dim=-1)
        ref[hq] = p @ Vdc[: pos + 1, hkv]
    err = (out_ver[j].double().cpu() - ref).abs().max().item()
    ok = err < 0.05
    bad += not ok
    print(f"{j:3d} | {err:18.5f}  {'ok' if ok else 'LEAK'}")
print("VERDICT:", "MASK OK" if bad == 0 else f"MASK LEAKS in {bad}/{QL} rows")
