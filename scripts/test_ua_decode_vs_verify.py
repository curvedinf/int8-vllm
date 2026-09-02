#!/usr/bin/env python3
"""Decisive per-row mask test: verify-window row j vs a q_len=1 decode call
at the same position over the SAME cache (with extreme K/V at one LATE window
position). Any mask leak shows as a huge divergence (the extreme V dominates
whatever row actually sees it).

Production dims: HQ=24, NKV=4, HS=256, NS=13. Small prefix (4096).
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
P = 4096
PAD = 4
CONTENT = 2 * (HS + PAD)
NBLOCKS = (P + 64) // BS + 2
NS = 13
QL = NS + 1

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

# normal prefix
k_pre = torch.randn(P, NKV, HS, dtype=torch.float16, device=DEV) * 0.3
v_pre = torch.randn(P, NKV, HS, dtype=torch.float16, device=DEV) * 0.3
write_kv(k_pre, v_pre, k_view, v_view, k_sc, v_sc,
         torch.arange(P, dtype=torch.int64, device=DEV),
         kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD)
# window tokens: normal except ONE garbage position (P+10)
k_win = torch.randn(QL, NKV, HS, dtype=torch.float16, device=DEV) * 0.3
v_win = torch.randn(QL, NKV, HS, dtype=torch.float16, device=DEV) * 0.3
GARB = 10
k_win[GARB] = 0.0
k_win[GARB, :, 0] = 300.0
v_win[GARB] = 0.0
v_win[GARB, :, 0] = 1e4
write_kv(k_win, v_win, k_view, v_view, k_sc, v_sc,
         torch.arange(P, P + QL, dtype=torch.int64, device=DEV),
         kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD)

bt = torch.arange(NBLOCKS, dtype=torch.int32, device=DEV).unsqueeze(0)
q_all = torch.randn(QL, HQ, HS, dtype=torch.bfloat16, device=DEV) * 0.3


def call(q_rows, kv_len):
    L = q_rows.shape[0]
    out = torch.empty(L, HQ, HS, dtype=torch.bfloat16, device=DEV)
    cu_q = torch.tensor([0, L], dtype=torch.int32, device=DEV)
    kv = torch.tensor([kv_len], dtype=torch.int32, device=DEV)
    unified_attention(
        q=q_rows, k=k_view, v=v_view, out=out,
        cu_seqlens_q=cu_q, max_seqlen_q=L, seqused_k=kv,
        max_seqlen_k=kv_len, softmax_scale=HS ** -0.5, causal=True,
        window_size=(-1, -1), block_table=bt, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
        k_scale_cache=k_sc, v_scale_cache=v_sc,
    )
    return out


# verify window: all 14 rows at once, kv through P+13
out_ver = call(q_all, P + QL)
print(f"garbage at window row {GARB} (position {P + GARB})")
print("row | max|verify - decode(same pos)|  verdict")
bad = 0
for j in range(QL):
    # decode reference: single query at position P+j, kv through P+j
    out_dec = call(q_all[j:j + 1], P + j + 1)
    err = (out_ver[j] - out_dec[0]).abs().max().item()
    ok = err < 0.05
    bad += not ok
    print(f"{j:3d} | {err:18.5f}  {'ok' if ok else 'LEAK'}")
print("VERDICT:", "MASK OK" if bad == 0 else f"MASK LEAKS in {bad}/{QL} rows")
