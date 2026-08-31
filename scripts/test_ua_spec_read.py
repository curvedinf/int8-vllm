#!/usr/bin/env python3
"""Numerical differential: AITER unified_attention multi-row spec query
(qlen=14, anchor at P) over an int8 per-token-head KV cache, vs a torch
reference (dequantize + exact causal attention).

The spec verify round reads with cu_seqlens_q=[0,14] and seqused_k=P+14 —
rows attend causally: anchor row i=0 attends 0..P, draft row i attends
0..P+i. Stale bytes beyond P+13 must never be read. Any per-row deviation
of the kernel from the reference = the garble's read-side origin.
"""
import sys

import torch

sys.path.insert(0, "../aiter")
from aiter.ops.triton.unified_attention import unified_attention  # noqa

torch.manual_seed(11)
DEV = "cuda"
NKV, HQ, HS, BS = 2, 4, 64, 32
NBLOCKS = 4
PAD = 4
CONTENT = 2 * (HS + PAD)
P = 5          # anchor position (window P..P+13)
QLEN = int(__import__('os').environ.get('QLEN', '14'))
SEQLEN = P + QLEN   # 19
VOCAB_GARBAGE = True


def quant_ref(x):
    # x: (n, NKV, HS) fp16 -> int8 + per-token-head f32 scale
    sc = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127.0
    q = torch.clamp((x.float() / sc).round(), -127, 127).to(torch.int8)
    return q, sc.squeeze(-1)  # (n,NKV,HS) int8, (n,NKV) f32


def build_cache(k_all, v_all):
    """k_all/v_all: (SEQLEN+16, NKV, HS) fp16 — first SEQLEN are 'real',
    the rest are garbage that must never be read."""
    kq, ksc = quant_ref(k_all)
    vq, vsc = quant_ref(v_all)
    cache = torch.zeros(NBLOCKS, NKV, BS, CONTENT, dtype=torch.int8,
                        device=DEV)
    f32 = torch.tensor([], dtype=torch.float32, device=DEV).set_(
        cache.untyped_storage())
    n_kv_rows = kq.shape[0]
    for pos in range(n_kv_rows):
        b, s = pos // BS, pos % BS
        for h in range(NKV):
            base = ((b * NKV + h) * BS + s) * CONTENT
            cache[b, h, s, :HS] = kq[pos, h]
            cache[b, h, s, HS + PAD:HS + PAD + HS] = vq[pos, h]
            f32[(base + HS) // 4] = ksc[pos, h]
            f32[(base + HS + PAD + HS) // 4] = vsc[pos, h]
    return cache, f32, kq, ksc, vq, vsc


def reference(k_all, v_all, q_rows):
    """Exact fp32 causal attention: row i (i=0..QLEN-1) at position P+i
    attends keys/values 0..P+i."""
    outs = []
    k_deq = (k_all[:SEQLEN].float())  # unquantized ground truth is wrong to
    # use — instead dequantize the cache for exactness:
    outs = None
    return outs


def dequant_cache(cache, f32):
    """Reconstruct fp K/V (SEQLEN, NKV, HS) from the packed cache."""
    K = torch.zeros(SEQLEN, NKV, HS, dtype=torch.float32, device=DEV)
    V = torch.zeros(SEQLEN, NKV, HS, dtype=torch.float32, device=DEV)
    for pos in range(SEQLEN):
        b, s = pos // BS, pos % BS
        for h in range(NKV):
            base = ((b * NKV + h) * BS + s) * CONTENT
            K[pos, h] = cache[b, h, s, :HS].float() * f32[(base + HS) // 4]
            V[pos, h] = cache[b, h, s, HS + PAD:HS + PAD + HS].float() * \
                f32[(base + HS + PAD + HS) // 4]
    return K, V


def main():
    n_rows = SEQLEN + 16
    k_all = torch.randn(n_rows, NKV, HS, dtype=torch.float16, device=DEV) * 2
    v_all = torch.randn(n_rows, NKV, HS, dtype=torch.float16, device=DEV) * 2
    q = torch.randn(QLEN, HQ, KS := HS, dtype=torch.bfloat16, device=DEV)
    cache, f32, *_ = build_cache(k_all, v_all)

    # strided int8 cache views like production (nkv-major rows)
    cache_u8 = cache.view(torch.uint8)  # storage (NB, NKV, BS, CONTENT)
    # production views: _split_kv_cache -> transpose(1,2).split
    split = cache_u8.transpose(1, 2)
    k_view = split[..., :HS]
    v_view = split[..., HS + PAD:HS + PAD + HS]
    C = CONTENT // 4
    # production-shaped strided f32 scale views: (NBLOCKS, BS, NKV)
    k_sc = torch.as_strided(f32, (NBLOCKS, BS, NKV),
                            (NKV * BS * C, C, BS * C), storage_offset=HS // 4)
    v_sc = torch.as_strided(f32, (NBLOCKS, BS, NKV),
                            (NKV * BS * C, C, BS * C),
                            storage_offset=(HS + PAD + HS) // 4)
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

    # torch reference from the DEQUANTIZED cache (exactness vs cache bytes)
    K, V = dequant_cache(cache, f32)
    ref = torch.empty(QLEN, HQ, HS, dtype=torch.float32, device=DEV)
    scale = HS**-0.5
    for i in range(QLEN):
        pos = P + i
        kctx = K[:pos + 1]  # (pos+1, NKV, HS)
        vctx = V[:pos + 1]
        for hq in range(HQ):
            hkv = hq % NKV if HQ >= NKV else 0
            if HQ % NKV == 0:
                hkv = hq // (HQ // NKV)
            qi = q[i, hq].float() * scale
            logits = (kctx[:, hkv] @ qi)
            p = torch.softmax(logits, dim=-1)
            ref[i, hq] = p @ vctx[:, hkv]

    err = (out.float() - ref).abs()
    print("per-row max abs err (bf16 tolerance ~0.1):")
    for i in range(QLEN):
        print(f"  row {i:2d} (pos {P+i}): {err[i].max().item():.4f}")
    print("OVERALL:", "PASS" if err.max().item() < 0.1 else "FAIL")


if __name__ == "__main__":
    main()
