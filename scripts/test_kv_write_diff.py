#!/usr/bin/env python3
"""Anchor-row KV-write differential for the AITER unified-attention int8-PTH path.

Writes the SAME committed token sequence two ways into identical packed
int8-PTH caches and compares bytes + f32 inline scales at every COMMITTED
position:

  A (plain):   one unified_attention call per committed token (1-row query).
  B (spec):    full 14-row verify rounds — the round writes draft K/V at all
               14 positions (garbage beyond the accepted prefix), then the
               NEXT round's window starts at the anchor position P0+n and
               rewrites from there. Committed positions in B must end up
               byte-identical to A, i.e. the anchor row + accepted rows must
               fully overwrite whatever the drafts left.

Cache layout (per _ensure_scale_caches): logical
(num_blocks, nkv, block_size, 2*(hs+pad)) int8, pad=4 f32-bytes; per
(head, slot): [K(hs) | K_scale(4B) | V(hs) | V_scale(4B)].
"""
import sys

import torch

sys.path.insert(0, "../aiter")
from aiter.ops.triton.unified_attention import unified_attention  # noqa: E402

torch.manual_seed(3)
DEV = "cuda"
NKV, HQ, HS = 2, 4, 64
BLOCK, NBLOCKS = 32, 4
PAD = 4
CONTENT = 2 * (HS + PAD)
VOCAB = 50


def new_cache():
    return torch.zeros(NBLOCKS, NKV, BLOCK, CONTENT, dtype=torch.int8,
                       device=DEV)


def scale_f32_view(cache, half):  # half 0=K, 1=V
    raw = cache.untyped_storage()
    f32 = torch.tensor([], dtype=torch.float32, device=DEV).set_(raw)
    # scale for (b,h,s) at int8 offset ((b*NKV+h)*BLOCK+s)*CONTENT + half*(HS+PAD) + HS
    # in f32 units: offset//4 (all aligned)
    return f32  # use byte_offset helper below


def scale_idx(b, h, s, half):
    el = ((b * NKV + h) * BLOCK + s) * CONTENT + half * (HS + PAD) + HS
    assert el % 4 == 0
    return el // 4


def get_scales(cache):
    f32 = torch.tensor([], dtype=torch.float32, device=DEV).set_(
        cache.untyped_storage())
    return f32


def call(cache, f32, k_new, v_new, cu_q, kv_len, bt, q):
    unified_attention(
        q=q,
        k=cache,  # placeholder; real k below
        v=cache,
        out=torch.empty(len(q), HQ, HS, dtype=torch.bfloat16, device=DEV),
        cu_seqlens_q=cu_q,
        max_seqlen_q=int(cu_q[-1].item()) - int(cu_q[0].item()),
        seqused_k=kv_len,
        max_seqlen_k=int(kv_len.max().item()),
        softmax_scale=HS**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=bt,
        softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
        k_scale_cache=f32, v_scale_cache=f32,
    )


def run():
    # block table: one request using block 0 for positions 0..31
    bt = torch.zeros(1, NBLOCKS, dtype=torch.int32, device=DEV)
    # committed sequence of tokens' K/V at positions P0..P0+L-1
    P0, L = 5, 40
    kv_all = torch.randn(P0 + L, NKV, HS, dtype=torch.float16, device=DEV)
    q_dummy = torch.randn(1, HQ, HS, dtype=torch.bfloat16, device=DEV)

    # ---- A: plain 1-row calls
    cacheA = new_cache()
    f32A = get_scales(cacheA)
    # seed prefix P0..P0+? : plain run writes all P0+L positions one-by-one
    from aiter.ops.triton.unified_attention import unified_attention as ua
    for pos in range(P0 + L):
        cu = torch.tensor([0, 1], dtype=torch.int32, device=DEV)
        kv_len = torch.tensor([pos + 1], dtype=torch.int32, device=DEV)
        q1 = torch.randn(1, HQ, HS, dtype=torch.bfloat16, device=DEV)
        ua(q=q1, k=kv_all[pos].unsqueeze(0).contiguous(),
           v=kv_all[pos].unsqueeze(0).contiguous(),
           out=torch.empty(1, HQ, HS, dtype=torch.bfloat16, device=DEV),
           cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=kv_len,
           max_seqlen_k=pos + 1, softmax_scale=HS**-0.5, causal=True,
           window_size=(-1, -1), block_table=bt, softcap=0.0,
           q_descale=None, k_descale=None, v_descale=None,
           k_scale_cache=f32A, v_scale_cache=f32A)
    return cacheA, f32A


if __name__ == "__main__":
    print("scaffold — fill in spec-arm next")
