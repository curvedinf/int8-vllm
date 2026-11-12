#!/usr/bin/env python3
"""Offline timing harness for the TARGET's unified attention on the real
aligned int8_block_g128 layout, at the C12 production geometry.

Target per TP4 rank: 12 seqs x 8 query tokens, 8 Q heads / 1 KV head,
head_dim 128, causal, 3D split-K path (the serving default with
VLLM_G128_ATTN3D=1). Measures a full-attention layer (window -1, the 14
layers) and a sliding layer (window 513, the 35 layers) separately.

  HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD .venv/bin/python \
    scripts/bench_attn_target_g8.py
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
KV_HEADS = 1         # per TP4 rank (GQA 8:1)
HEAD_DIM = 128
BLOCK_SIZE = 128
HS = HEAD_DIM
PADDED_HS = ((HS + 2) + 15) // 16 * 16
CONTENT = 2 * PADDED_HS
SPLITS = int(os.environ.get("ATTN_SPLITS", "64"))


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
    k = torch.randn((num_blocks, KV_HEADS, BLOCK_SIZE, HEAD_DIM),
                    dtype=torch.float16, device=dev) / 8
    v = torch.randn((num_blocks, KV_HEADS, BLOCK_SIZE, HEAD_DIM),
                    dtype=torch.float16, device=dev) / 8
    kq, ks = quant(k)
    vq, vs = quant(v)
    kv[..., :HS] = kq
    kv[..., PADDED_HS:PADDED_HS + HS] = vq
    kv.view(torch.float16)[..., HS // 2] = ks
    kv.view(torch.float16)[..., PADDED_HS // 2 + HS // 2] = vs
    block_table = torch.arange(num_blocks - 1, dtype=torch.int32,
                               device=dev).view(NUM_SEQS, max_blocks)
    seqused_k = torch.full((NUM_SEQS,), CTX, dtype=torch.int32, device=dev)
    q_lens = torch.full((NUM_SEQS,), Q_PER_SEQ, dtype=torch.int32, device=dev)
    cu_q = torch.cat([torch.zeros(1, dtype=torch.int32, device=dev),
                      torch.cumsum(q_lens, dim=0)]).to(torch.int32)
    out = torch.zeros_like(q)
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
    segm = (
        torch.empty((256, Q_HEADS, SPLITS, HEAD_DIM), dtype=torch.float32,
                    device=dev),
        torch.empty((256, Q_HEADS, SPLITS), dtype=torch.float32, device=dev),
        torch.empty((256, Q_HEADS, SPLITS), dtype=torch.float32, device=dev),
    )
    return dict(q=q, k=k_half[..., :HS], v=v_half[..., :HS], out=out,
                cu_q=cu_q, seqused_k=seqused_k, block_table=block_table,
                k_scale=k_scale, v_scale=v_scale, segm=segm)


def call(t, window):
    s_out, s_max, s_sum = t["segm"]
    unified_attention(
        q=t["q"], k=t["k"], v=t["v"], out=t["out"],
        cu_seqlens_q=t["cu_q"], max_seqlen_q=Q_PER_SEQ,
        seqused_k=t["seqused_k"], max_seqlen_k=CTX,
        softmax_scale=HEAD_DIM ** -0.5, causal=True,
        window_size=(window, window),
        block_table=t["block_table"], softcap=0,
        kv_quant_mode=KVQuantMode.INT8_BLOCK_G128,
        q_descale=None, k_descale=None, v_descale=None,
        k_scale_cache=None, v_scale_cache=None,
        g8_k_scale=t["k_scale"], g8_v_scale=t["v_scale"],
        num_par_softmax_segments=SPLITS,
        softmax_segm_output=s_out, softmax_segm_max=s_max,
        softmax_segm_expsum=s_sum, seq_threshold_3D=256,
        max_flash_decoding_splits=SPLITS,
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
    os.environ.setdefault("VLLM_UA_3D_MAXQ", "8")
    t = build()
    full = bench(lambda: call(t, -1))
    slide = bench(lambda: call(t, 513))
    print(f"target full layer (14): {full:7.1f} us/layer -> {full * 14:7.1f} us")
    print(f"target slide layer (35): {slide:7.1f} us/layer -> {slide * 35:7.1f} us")
    print(f"target attention total: {full * 14 + slide * 35:7.1f} us/step")


if __name__ == "__main__":
    main()
