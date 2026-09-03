#!/usr/bin/env python3
"""Unit tests for the int8_block_g{G} KV path (writer + read dequant).

1. Writer roundtrip: dequant(key_cache, scales) must match the original K/V
   within half a quantum per group (+ one f16 scale-step for the clamped
   group-max element).
2. Read kernel: unified_attention with g8 scales matches a torch reference
   (dequant to bf16 exactly like the kernel, fp32 softmax).

Run: PYTHONPATH="$PWD:$PWD/../aiter" .venv/bin/python scripts/test_int8_block_kv.py
"""
import sys

import torch

from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    reshape_and_cache_g8,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

DEV = "cuda"
HS, NKV, BLOCK = 128, 2, 32
GROUPS_OK = (4, 8, 16, 32, 64, 128)


def test_dtype_strings():
    for g in GROUPS_OK:
        m = get_kv_quant_mode(f"int8_block_g{g}")
        assert m.is_int8_block and m.int8_block_group == g
    assert not get_kv_quant_mode("int8_per_token_head").is_int8_block
    assert get_kv_quant_mode("int8_block_g7") == KVQuantMode.NONE
    print("dtype strings OK")


def test_writer_roundtrip():
    ntok = 256
    for G in GROUPS_OK:
        groups = HS // G
        for trial in range(4):
            torch.manual_seed(trial * 100 + G)
            key = torch.randn(ntok, NKV, HS, device=DEV, dtype=torch.bfloat16)
            kd = torch.zeros(8, BLOCK, NKV, HS, dtype=torch.int8, device=DEV)
            vd = torch.zeros_like(kd)
            kv_ = torch.zeros(8, BLOCK, NKV, groups, dtype=torch.float16, device=DEV)
            vv = torch.zeros_like(kv_)
            slot = torch.arange(ntok, device=DEV, dtype=torch.int64)
            reshape_and_cache_g8(key, key.clone(), kd, vd, kv_, vv, slot, group=G)
            torch.cuda.synchronize()
            for t in range(ntok):
                b, o = int(slot[t] // BLOCK), int(slot[t] % BLOCK)
                for h in range(NKV):
                    sc = kv_[b, o, h].float()
                    dq = kd[b, o, h].float() * sc.repeat_interleave(G)
                    bound = (
                        0.5 * sc.max().item()
                        + 127 * sc.max().item() * 2**-12
                        + 1e-6
                    )
                    e = (dq - key[t, h].float()).abs().max().item()
                    assert e <= bound, (G, t, h, e, bound)
    print("writer roundtrip OK for all G")


def test_read_kernel():
    nq = 4
    s = HS**-0.5
    for G in GROUPS_OK:
        groups = HS // G
        for (ctx, seed) in ((8, 1), (40, 3), (200, 5)):
            torch.manual_seed(seed * 1000 + G)
            nb = (ctx + nq + BLOCK - 1) // BLOCK
            key = torch.randn(ctx + nq, NKV, HS, device=DEV, dtype=torch.bfloat16)
            val = torch.randn(ctx + nq, NKV, HS, device=DEV, dtype=torch.bfloat16)
            kd = torch.zeros(nb, BLOCK, NKV, HS, dtype=torch.int8, device=DEV)
            vd = torch.zeros_like(kd)
            kv_ = torch.zeros(nb, BLOCK, NKV, groups, dtype=torch.float16, device=DEV)
            vv = torch.zeros_like(kv_)
            slot = torch.arange(ctx + nq, device=DEV, dtype=torch.int64)
            reshape_and_cache_g8(key, val, kd, vd, kv_, vv, slot, group=G)
            q = torch.randn(nq, NKV, HS, device=DEV, dtype=torch.bfloat16)
            out = torch.zeros(nq, NKV, HS, device=DEV, dtype=torch.bfloat16)
            cu = torch.tensor([0, nq], device=DEV, dtype=torch.int32)
            seq = torch.tensor([ctx + nq], device=DEV, dtype=torch.int32)
            bt = torch.arange(nb, device=DEV, dtype=torch.int32)[None, :]
            unified_attention(
                q=q, k=kd, v=vd, out=out, cu_seqlens_q=cu, max_seqlen_q=nq,
                seqused_k=seq, max_seqlen_k=ctx + nq, softmax_scale=s,
                causal=True, window_size=(-1, -1), block_table=bt, softcap=0.0,
                q_descale=None, k_descale=None, v_descale=None,
                kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
                k_scale_cache=torch.ones(nb, BLOCK, NKV, device=DEV),
                v_scale_cache=torch.ones(nb, BLOCK, NKV, device=DEV),
                g8_k_scale=kv_, g8_v_scale=vv,
            )
            torch.cuda.synchronize()
            kq_ = (
                kd.reshape(-1, NKV, HS).float()
                * kv_.reshape(-1, NKV, groups).float().repeat_interleave(G, -1)
            ).to(torch.bfloat16).float()
            vq_ = (
                vd.reshape(-1, NKV, HS).float()
                * vv.reshape(-1, NKV, groups).float().repeat_interleave(G, -1)
            ).to(torch.bfloat16).float()
            for h in range(NKV):
                for i in range(nq):
                    qq = q[i, h].float()
                    attn = (kq_[: ctx + i + 1, h] @ qq) * s
                    ref = torch.softmax(attn, -1) @ vq_[: ctx + i + 1, h]
                    got = out[i, h].float()
                    rel = ((ref - got).abs().max() / ref.abs().max()).item()
                    assert rel < 5e-2, (G, ctx, h, i, rel)
    print("read kernel OK for all G")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a GPU"
    test_dtype_strings()
    test_writer_roundtrip()
    test_read_kernel()
    print("ALL PASS")
