#!/usr/bin/env python3
"""Numerical differential: aiter unified-attention 2D kernel vs naive
reference on the EXACT SW-verify production shape.

Shape: sliding_window=2048, q_len=14 (spec verify), int8 per-token-head KV,
block 64, seqused_k with rollback variance (context+14), null blocks below
the window (freed SW pages). Compares kernel output vs float reference for
many seeds and several context lengths, including window-edge crossings.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiter.ops.triton.attention.unified_attention import unified_attention

DEV = "cuda"
DT = torch.float16
BS = 64          # block size
NKV = 2
NH = 8           # query heads (GQA 4:1)
HS = 128
SW = 2048        # sliding window
QM = 127.0


def build(seq_len, num_blocks_total, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    k_full = torch.randn(seq_len, NKV, HS, generator=g).to(DT).to(DEV)
    v_full = torch.randn(seq_len, NKV, HS, generator=g).to(DT).to(DEV)
    q = torch.randn(14, NH, HS, generator=g).to(DT).to(DEV)

    nb = (seq_len + BS - 1) // BS
    blocks = torch.randperm(num_blocks_total, generator=g)[:nb].to(torch.int32).to(DEV)
    block_table = blocks.unsqueeze(0)

    # int8-PTH packed cache: [num_blocks, nkv, block, 2*(hs+4)] int8 with
    # inline fp32 scale after each half. Build via views like the impl.
    cache = torch.zeros(num_blocks_total, NKV, BS, 2 * (HS + 4),
                        dtype=torch.int8, device=DEV)
    raw = cache.untyped_storage()
    f32 = torch.tensor([], dtype=torch.float32, device=DEV).set_(raw)
    pad = 4  # fp32-in-int8 elements
    # K data at element offset 0..hs, K scale (4 int8 slots) at hs..hs+4
    # V data at hs+4..2hs+4, V scale at 2hs+4..2hs+8 within each half pair
    kq = torch.zeros(num_blocks_total, NKV, BS, HS, dtype=torch.int8, device=DEV)
    vq = torch.zeros(num_blocks_total, NKV, BS, HS, dtype=torch.int8, device=DEV)
    ks = torch.ones(num_blocks_total, BS, NKV, dtype=torch.float32, device=DEV)
    vs = torch.ones(num_blocks_total, BS, NKV, dtype=torch.float32, device=DEV)
    for pos in range(seq_len):
        b = int(blocks[pos // BS]); off = pos % BS
        for h in range(NKV):
            ksc = k_full[pos, h].abs().max().item() / QM
            vsc = v_full[pos, h].abs().max().item() / QM
            ksc = max(ksc, 1e-6); vsc = max(vsc, 1e-6)
            kq[b, h, off] = torch.clamp((k_full[pos, h] / ksc).round(),
                                        -QM, QM).to(torch.int8)
            vq[b, h, off] = torch.clamp((v_full[pos, h] / vsc).round(),
                                        -QM, QM).to(torch.int8)
            ks[b, off, h] = ksc
            vs[b, off, h] = vsc
    # pack data + scales into cache bytes
    cache_view = cache.view(num_blocks_total, NKV, BS, 2, HS + 4)
    cache_view[:, :, :, 0, :HS] = kq
    cache_view[:, :, :, 1, :HS] = vq
    # scales as fp32 bytes
    ks_bytes = ks.cpu().numpy().tobytes()
    ks_b = torch.frombuffer(bytearray(ks_bytes), dtype=torch.int8).reshape(
        num_blocks_total, BS, NKV, 4).to(DEV)
    vs_bytes = vs.cpu().numpy().tobytes()
    vs_b = torch.frombuffer(bytearray(vs_bytes), dtype=torch.int8).reshape(
        num_blocks_total, BS, NKV, 4).to(DEV)
    cache_view[:, :, :, 0, HS:HS+4] = ks_b.permute(0, 2, 1, 3)
    cache_view[:, :, :, 1, HS:HS+4] = vs_b.permute(0, 2, 1, 3)

    k_cache = cache_view[:, :, :, 0].transpose(1, 2)  # [nb, bs, nkv, hs+4]
    v_cache = cache_view[:, :, :, 1].transpose(1, 2)
    return q, k_full, v_full, block_table, ks, vs, k_cache, v_cache, blocks


def ref(q, k_full, v_full, seq_len, ctx):
    # queries are the last Q positions: absolute positions ctx..ctx+13
    outs = []
    scale = HS ** -0.5
    gqa = NH // NKV
    for qi in range(14):
        pos = ctx + qi
        lo = max(0, pos - SW + 1)
        k = k_full[lo:pos + 1].float()   # [win, nkv, hs]
        v = v_full[lo:pos + 1].float()
        o = torch.zeros(NH, HS, device=q.device)
        for h in range(NH):
            kvh = h // gqa
            s = (q[qi, h].float() @ k[:, kvh].T) * scale
            p = torch.softmax(s, dim=-1)
            o[h] = p @ v[:, kvh]
        outs.append(o)
    return torch.stack(outs)  # [14, NH, HS]


def run(seq_len, seed, tag):
    ctx = seq_len - 14
    q, k_full, v_full, bt, ks, vs, kc, vc, blocks = build(seq_len, (seq_len + BS - 1) // BS + 8, seed)
    seqused = torch.tensor([seq_len], dtype=torch.int32, device=DEV)
    cu_q = torch.tensor([0, 14], dtype=torch.int32, device=DEV)
    out = torch.zeros(14, NH, HS, dtype=DT, device=DEV)
    unified_attention(
        q=q, k=kc, v=vc, out=out,
        cu_seqlens_q=cu_q, max_seqlen_q=14,
        seqused_k=seqused, max_seqlen_k=seq_len,
        softmax_scale=HS ** -0.5, causal=True,
        window_size=(SW - 1, 0), block_table=bt, softcap=0,
        q_descale=None, k_descale=None, v_descale=None, sinks=None,
        k_scale_cache=ks, v_scale_cache=vs,
    )
    r = ref(q, k_full, v_full, seq_len, ctx).to(DT)
    diff = (out.float() - r.float()).abs()
    rel = diff.max() / r.float().abs().max().clamp(min=1e-6)
    print(f"{tag}: max_diff={diff.max():.4f} rel={rel:.4f} "
          f"mean={diff.mean():.5f}")
    return rel.item()


ok = True
for seed in range(3):
    for sl in (1000, 2048, 2050, 4000, 40000):
        rel = run(sl, seed, f"seq={sl} seed={seed}")
        if rel > 0.05:
            ok = False
print("OVERALL:", "PASS" if ok else "FAIL (kernel diverges from reference)")
