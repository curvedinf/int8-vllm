#!/usr/bin/env python3
"""UA read differential with a PRODUCTION-WRITTEN cache.

1. Build the packed int8-PTH cache by writing reference K/V through the
   production kernel triton_reshape_and_cache_flash_per_token_head_quant
   (no hand-packed layout to get wrong).
2. Read back per-row outputs of aiter unified_attention for a spec-shaped
   multi-row query (QLEN rows at positions P..P+QLEN-1, cu=[0,QLEN],
   seqused_k=P+QLEN).
3. Compare against an exact torch reference that dequantizes the cache
   (int8 * scale) using the scale views the backend itself constructs.

Self-check: QLEN=1 (decode) must PASS before trusting QLEN=14.
"""
import os
import sys

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "../aiter")
from aiter.ops.triton.unified_attention import unified_attention  # noqa
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (  # noqa
    triton_reshape_and_cache_flash_per_token_head_quant as write_kv,
)
from vllm.v1.kv_cache_interface import KVQuantMode  # noqa

torch.manual_seed(11)
DEV = "cuda"
NKV, HQ, HS, BS = 2, 4, 64, 32
NBLOCKS = 8
PAD = 4
CONTENT = 2 * (HS + PAD)
P = 5
QLEN = int(os.environ.get("QLEN", "14"))
SEQLEN = P + QLEN
EXTRA = 16  # garbage rows beyond the frontier

cache = torch.zeros(NBLOCKS, NKV, BS, CONTENT, dtype=torch.int8,
                    device=DEV)
f32 = torch.tensor([], dtype=torch.float32, device=DEV).set_(
    cache.untyped_storage())
C = CONTENT // 4
k_sc = torch.as_strided(f32, (NBLOCKS, BS, NKV),
                        (NKV * BS * C, C, BS * C), storage_offset=HS // 4)
v_sc = torch.as_strided(f32, (NBLOCKS, BS, NKV),
                        (NKV * BS * C, C, BS * C),
                        storage_offset=(HS + PAD + HS) // 4)

n_rows = SEQLEN + EXTRA
k_all = torch.randn(n_rows, NKV, HS, dtype=torch.float16, device=DEV) * 2
v_all = torch.randn(n_rows, NKV, HS, dtype=torch.float16, device=DEV) * 2
slot_mapping = torch.arange(n_rows, dtype=torch.int64, device=DEV)

# production write (writes all rows incl. garbage tail)
split = cache.transpose(1, 2)
k_view = split[..., :HS]
v_view = split[..., HS + PAD:HS + PAD + HS]
write_kv(k_all, v_all, k_view, v_view, k_sc, v_sc, slot_mapping,
         kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD)

# torch reference: dequantize the cache exactly as written
Kd = torch.zeros(SEQLEN, NKV, HS, dtype=torch.float32, device=DEV)
Vd = torch.zeros(SEQLEN, NKV, HS, dtype=torch.float32, device=DEV)
for pos in range(SEQLEN):
    b, s = pos // BS, pos % BS
    Kd[pos] = k_view[b, s].float() * k_sc[b, s].unsqueeze(-1)
    Vd[pos] = v_view[b, s].float() * v_sc[b, s].unsqueeze(-1)

q = torch.randn(QLEN, HQ, HS, dtype=torch.bfloat16, device=DEV)
out = torch.empty(QLEN, HQ, HS, dtype=torch.bfloat16, device=DEV)
bt = torch.zeros(1, NBLOCKS, dtype=torch.int32, device=DEV)
cu_q = torch.tensor([0, QLEN], dtype=torch.int32, device=DEV)
kv_len = torch.tensor([SEQLEN], dtype=torch.int32, device=DEV)
unified_attention(
    q=q, k=k_view, v=v_view, out=out,
    cu_seqlens_q=cu_q, max_seqlen_q=QLEN,
    seqused_k=kv_len, max_seqlen_k=SEQLEN,
    softmax_scale=HS**-0.5, causal=True,
    window_size=(-1, -1), block_table=bt, softcap=0.0,
    q_descale=None, k_descale=None, v_descale=None,
    k_scale_cache=k_sc, v_scale_cache=v_sc,
)

ref = torch.empty(QLEN, HQ, HS, dtype=torch.float32, device=DEV)
gqa = HQ // NKV
for i in range(QLEN):
    pos = P + i
    for hq in range(HQ):
        hkv = hq // gqa
        qi = q[i, hq].float() * (HS**-0.5)
        logits = Kd[:pos + 1, hkv] @ qi
        p = torch.softmax(logits, dim=-1)
        ref[i, hq] = p @ Vd[:pos + 1, hkv]

err = (out.float() - ref).abs()
for i in range(QLEN):
    print(f"  row {i:2d} (pos {P+i}): max_err={err[i].max().item():.4f}")
print("OVERALL:", "PASS" if err.max().item() < 0.1 else "FAIL")
