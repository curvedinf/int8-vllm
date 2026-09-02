#!/usr/bin/env python3
"""CAUSAL test: does the AITER unified-attention kernel's output for the SAME
query row differ between q_len=1 (decode) and the anchor row of a q_len=14
verify window, at LONG context on a real-layout int8-PTH cache?

The KLD gate showed the verify path's logits diverge from the decode path's
from token 0 (near-tie flips). Every other compute component is exonerated
(GDN kernels equivalent, W8A8 GEMM M-invariant, sampler exact). This is the
last untested surface: the attention read at q_len=1 vs q_len=14 over the
same cache.

Design: build a 40k-token int8-PTH cache via the production write kernel
(realistic K/V with a hot-scale tail, like real text). For a query at
position P=40000, compute the attention output (a) alone (q_len=1) and
(b) as row 0 of a q_len=14 window (rows P..P+13, garbage K/V beyond P —
the anchor row attends only to positions <= P in both cases). Compare.
PASS if they match within the int8 noise floor; a systematic excess =
the verify path's attention deviates from the decode path's.
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
NKV, HQ, HS, BS = 2, 4, 64, 32       # heads small; the point is q_len handling
SEQLEN = 40000
PAD = 4
CONTENT = 2 * (HS + PAD)
NBLOCKS = (SEQLEN + 64) // BS + 2

# Realistic K/V: a heavy tail of hot positions (attention sinks) so the
# scale distribution is non-trivial, like real long-context cache.
k_all = torch.randn(SEQLEN, NKV, HS, dtype=torch.float16, device=DEV)
v_all = torch.randn(SEQLEN, NKV, HS, dtype=torch.float16, device=DEV)
hot = torch.rand(SEQLEN, device=DEV) < 0.01
k_all[hot] *= 30.0
v_all[hot] *= 30.0

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
slot_mapping = torch.arange(SEQLEN, dtype=torch.int64, device=DEV)
write_kv(k_all, v_all, k_view, v_view, k_sc, v_sc, slot_mapping,
         kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD)

# dequantized reference (fp64) for the query row
Kd = (k_view.float().transpose(0, 1).reshape(NBLOCKS * BS, NKV, HS)
      * k_sc.transpose(0, 1).reshape(NBLOCKS * BS, NKV, 1)).double()
Vd = (v_view.float().transpose(0, 1).reshape(NBLOCKS * BS, NKV, HS)
      * v_sc.transpose(0, 1).reshape(NBLOCKS * BS, NKV, 1)).double()

P = 40000
bt = torch.arange(NBLOCKS, dtype=torch.int32, device=DEV).unsqueeze(0)


def attn(q_rows, seq_k):
    QL = q_rows.shape[0]
    out = torch.empty(QL, HQ, HS, dtype=torch.bfloat16, device=DEV)
    cu_q = torch.tensor([0, QL], dtype=torch.int32, device=DEV)
    kv_len = torch.tensor([seq_k], dtype=torch.int32, device=DEV)
    unified_attention(
        q=q_rows, k=k_view, v=v_view, out=out,
        cu_seqlens_q=cu_q, max_seqlen_q=QL, seqused_k=kv_len,
        max_seqlen_k=seq_k, softmax_scale=HS ** -0.5, causal=True,
        window_size=(-1, -1), block_table=bt, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
        k_scale_cache=k_sc, v_scale_cache=v_sc,
    )
    return out


# decode-style: single query at position P (seqused_k = P+1)
q_dec = torch.randn(1, HQ, HS, dtype=torch.bfloat16, device=DEV)
out_dec = attn(q_dec, P + 1)

# verify-style: 14 rows, row 0 at position P (same query), garbage queries after
q_ver = torch.randn(14, HQ, HS, dtype=torch.bfloat16, device=DEV)
q_ver[0] = q_dec[0]
out_ver = attn(q_ver, P + 14)

# fp64 reference for row 0 (attends to positions <= P)
qi = q_dec[0].double() * (HS ** -0.5)
ref0 = torch.empty(HQ, HS, dtype=torch.float64, device=DEV)
gqa = HQ // NKV
for hq in range(HQ):
    hkv = hq // gqa
    logits = Kd[: P + 1, hkv] @ qi[hq]
    p = torch.softmax(logits, dim=-1)
    ref0[hq] = p @ Vd[: P + 1, hkv]

err_dec = (out_dec[0].double() - ref0).abs()
err_ver = (out_ver[0].double() - ref0).abs()
diff = (out_ver[0].double() - out_dec[0].double()).abs()
print(f"row0 vs fp64 ref: decode max={err_dec.max().item():.5f}  verify max={err_ver.max().item():.5f}")
print(f"verify row0 vs decode row0: max abs diff = {diff.max().item():.6f}")
rel = diff / (out_dec[0].double().abs() + 1e-6)
print(f"  relative: mean={rel.mean().item():.2e} max={rel.max().item():.2e}")
print("VERDICT:", "MATCH (within noise)" if diff.max().item() < 0.05 else "DIVERGE (real)")

# sweep a few positions to be sure it's not one unlucky position
print("\npos sweep (verify row0 vs decode row0 max abs diff):")
for PP in (1000, 5000, 20000, 39999):
    qd = torch.randn(1, HQ, HS, dtype=torch.bfloat16, device=DEV)
    od = attn(qd, PP + 1)
    qv = torch.randn(14, HQ, HS, dtype=torch.bfloat16, device=DEV)
    qv[0] = qd[0]
    ov = attn(qv, PP + 14)
    dd = (ov[0].double() - od[0].double()).abs().max().item()
    print(f"  pos {PP}: {dd:.6f}")
