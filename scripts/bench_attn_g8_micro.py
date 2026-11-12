#!/usr/bin/env python3
"""Offline timing harness for the DFlash2 draft attention on the REAL
aligned int8_block_g128 KV layout.

The draft's five layers are non-causal sliding-window (window 2048) and take
the 2D kernel. In serving the draft attention call costs ~3.5 ms at the C12
batch (12 seqs x 8 query tokens, 2 KV / 8 Q heads per rank), while the
per-token-head harness measured 230 us/layer. This builds the packed g8
cache (144-byte aligned rows, data + inline fp16 group scale per half) and
calls the same fork wrapper the backend uses, so the kernel can be
optimised without booting.

  HIP_VISIBLE_DEVICES=0 .venv/bin/python scripts/bench_attn_g8_micro.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from vllm.v1.attention.ops.triton_unified_attention import (  # noqa: E402
    unified_attention,
)
from vllm.v1.kv_cache_interface import KVQuantMode  # noqa: E402

NUM_SEQS = 12
CTX = 2400
Q_PER_SEQ = 8
Q_HEADS = 8          # per TP4 rank
KV_HEADS = 2         # per TP4 rank
HEAD_DIM = 128
BLOCK_SIZE = 128
WINDOW = 2048
HS = HEAD_DIM
PADDED_HS = ((HS + 2) + 15) // 16 * 16   # aligned row, 144
CONTENT = 2 * PADDED_HS


def quant(x):
    absmax = x.abs().amax(dim=-1, keepdim=True)
    s = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
    q = (x / s).clamp(-128, 127).round().to(torch.int8)
    return q, s.squeeze(-1).to(torch.float16)


def build():
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.randn((NUM_SEQS * Q_PER_SEQ, Q_HEADS, HEAD_DIM),
                    dtype=torch.float16, device=dev) / 8
    max_blocks = (CTX + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = NUM_SEQS * max_blocks + 1
    kv = torch.zeros((num_blocks, KV_HEADS, BLOCK_SIZE, CONTENT),
                     dtype=torch.int8, device=dev)
    # Fill every slot with quantized K/V + inline fp16 group scale.
    k = torch.randn((num_blocks, KV_HEADS, BLOCK_SIZE, HEAD_DIM),
                    dtype=torch.float16, device=dev) / 8
    v = torch.randn((num_blocks, KV_HEADS, BLOCK_SIZE, HEAD_DIM),
                    dtype=torch.float16, device=dev) / 8
    kq, ks = quant(k)
    vq, vs = quant(v)
    kv[..., :HS] = kq.view(num_blocks, KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    kv[..., PADDED_HS:PADDED_HS + HS] = vq.view(
        num_blocks, KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    kv.view(torch.float16)[..., HS // 2] = ks
    kv.view(torch.float16)[..., PADDED_HS // 2 + HS // 2] = vs

    block_table = torch.arange(num_blocks - 1, dtype=torch.int32,
                               device=dev).view(NUM_SEQS, max_blocks)
    seqused_k = torch.full((NUM_SEQS,), CTX, dtype=torch.int32, device=dev)
    q_lens = torch.full((NUM_SEQS,), Q_PER_SEQ, dtype=torch.int32, device=dev)
    cu_q = torch.cat([torch.zeros(1, dtype=torch.int32, device=dev),
                      torch.cumsum(q_lens, dim=0)]).to(torch.int32)
    out = torch.zeros_like(q)

    # Same half/scale views the backend carves.
    k_half, v_half = kv.transpose(1, 2).split(PADDED_HS, dim=-1)
    base = torch.tensor([], dtype=torch.float16, device=dev).set_(
        kv.untyped_storage())
    st = kv.stride()
    block_f16, head_f16, slot_f16 = st[0] // 2, st[1] // 2, st[2] // 2
    k_scale = torch.as_strided(
        base, size=(num_blocks, BLOCK_SIZE, KV_HEADS, 1),
        stride=(block_f16, slot_f16, head_f16, 1),
        storage_offset=HS // 2)
    v_scale = torch.as_strided(
        base, size=(num_blocks, BLOCK_SIZE, KV_HEADS, 1),
        stride=(block_f16, slot_f16, head_f16, 1),
        storage_offset=PADDED_HS // 2 + HS // 2)
    return dict(q=q, k=k_half[..., :HS], v=v_half[..., :HS], out=out,
                cu_q=cu_q, seqused_k=seqused_k, block_table=block_table,
                k_scale=k_scale, v_scale=v_scale)


def call(t):
    unified_attention(
        q=t["q"], k=t["k"], v=t["v"], out=t["out"],
        cu_seqlens_q=t["cu_q"], max_seqlen_q=Q_PER_SEQ,
        seqused_k=t["seqused_k"], max_seqlen_k=CTX,
        softmax_scale=HEAD_DIM ** -0.5, causal=False,
        window_size=(WINDOW, WINDOW),
        block_table=t["block_table"], softcap=0,
        kv_quant_mode=KVQuantMode.INT8_BLOCK_G128,
        q_descale=None, k_descale=None, v_descale=None,
        k_scale_cache=None, v_scale_cache=None,
        g8_k_scale=t["k_scale"], g8_v_scale=t["v_scale"],
    )
    return t["out"]


def bench(fn, iters=20, reps=3):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    best = float("inf")
    for _ in range(reps):
        g.replay()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters * 1000)
    return best


def main():
    t = build()
    us = bench(lambda: call(t))
    print(f"draft g8 attention: {us:8.1f} us/layer "
          f"({us * 5:8.1f} us for 5 layers) | ctx={CTX} window={WINDOW} "
          f"kvh={KV_HEADS} qh={Q_HEADS} q/seq={Q_PER_SEQ}")


if __name__ == "__main__":
    main()
