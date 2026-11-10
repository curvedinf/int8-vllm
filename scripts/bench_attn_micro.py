#!/usr/bin/env python3
"""Offline timing harness for the target's unified attention at Flint shapes.

Production geometry (TP4, C12, NS=7): 12 sequences, ~2048-token context,
8 query tokens per sequence (the speculative verify block), 8 query heads
and 1 KV head per rank, head_dim 128, int8 KV. This harness builds that
call directly against `aiter.ops.triton.attention.unified_attention` and
times it inside a CUDA graph, so kernel ideas can be iterated without
booting the server (which costs ~4.5 min per try).

The deployed KV dtype is int8_block_g128, whose cache is a packed
(num_blocks, nkv, block_size, 2*(hs+pad)) tensor with the group scales
carved out of the padding region; this harness uses the simpler per-token
head scale buffers instead. Use it for STRUCTURAL ideas (tile size, warps,
segment count) and confirm any win in serving - the dequant differs, and
serving measured block_g128 as faster than per-token-head.

Usage: HIP_VISIBLE_DEVICES=0 .venv/bin/python scripts/bench_attn_micro.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from aiter.ops.triton.attention.unified_attention import unified_attention  # noqa: E402

NUM_SEQS = 12
CTX = 2048
Q_PER_SEQ = 8
Q_HEADS = 8
KV_HEADS = 1
HEAD_DIM = 128
BLOCK_SIZE = 128
DTYPE = torch.float16


def quant_per_token_head(x):
    absmax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
    q = (x / scale).clamp(-128, 127).round().to(torch.int8)
    return q, scale.squeeze(-1)


def build(ctx=CTX, num_seqs=NUM_SEQS, q_per_seq=Q_PER_SEQ, kv_heads=None,
          window=-1):
    kv_heads = KV_HEADS if kv_heads is None else kv_heads
    dev = "cuda"
    torch.manual_seed(0)
    q = torch.randn((num_seqs * q_per_seq, Q_HEADS, HEAD_DIM),
                    dtype=DTYPE, device=dev) / 8
    max_blocks = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = num_seqs * max_blocks + 1
    k_cache = torch.zeros((num_blocks, BLOCK_SIZE, kv_heads, HEAD_DIM),
                          dtype=torch.int8, device=dev)
    v_cache = torch.zeros_like(k_cache)
    k_scale = torch.ones((num_blocks, BLOCK_SIZE, kv_heads), device=dev)
    v_scale = torch.ones_like(k_scale)
    block_table = torch.zeros((num_seqs, max_blocks), dtype=torch.int32,
                              device=dev)
    key = torch.randn((num_seqs, ctx, kv_heads, HEAD_DIM), dtype=DTYPE,
                      device=dev) / 8
    value = torch.randn((num_seqs, ctx, kv_heads, HEAD_DIM), dtype=DTYPE,
                        device=dev) / 8
    for s in range(num_seqs):
        kq, ks = quant_per_token_head(key[s])
        vq, vs = quant_per_token_head(value[s])
        for b in range(max_blocks):
            blk = s * max_blocks + b
            block_table[s, b] = blk
            lo = b * BLOCK_SIZE
            k_cache[blk] = kq[lo:lo + BLOCK_SIZE]
            v_cache[blk] = vq[lo:lo + BLOCK_SIZE]
            k_scale[blk] = ks[lo:lo + BLOCK_SIZE]
            v_scale[blk] = vs[lo:lo + BLOCK_SIZE]
    out = torch.zeros_like(q)
    q_lens = torch.full((num_seqs,), q_per_seq, dtype=torch.int32, device=dev)
    cu_q = torch.cat([torch.zeros(1, dtype=torch.int32, device=dev),
                      torch.cumsum(q_lens, dim=0)]).to(torch.int32)
    seqused_k = torch.full((num_seqs,), ctx, dtype=torch.int32, device=dev)
    return dict(q=q, k=k_cache, v=v_cache, out=out, cu_q=cu_q,
                seqused_k=seqused_k, block_table=block_table,
                k_scale=k_scale, v_scale=v_scale,
                max_q=q_per_seq, max_k=ctx, window=window)


def call(t):
    unified_attention(
        q=t["q"], k=t["k"], v=t["v"], out=t["out"],
        cu_seqlens_q=t["cu_q"], max_seqlen_q=t["max_q"],
        seqused_k=t["seqused_k"], max_seqlen_k=t["max_k"],
        softmax_scale=HEAD_DIM ** -0.5, causal=True,
        window_size=(t["window"], t["window"]),
        block_table=t["block_table"], softcap=0,
        q_descale=None, k_descale=None, v_descale=None,
        k_scale_cache=t["k_scale"], v_scale_cache=t["v_scale"],
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
    print("target-path unified attention, per-token-head int8 KV", flush=True)
    ctxs = [int(c) for c in os.environ.get("ATTN_CTX", "512,2048").split(",")]
    cases = [(1, -1, "target-like kvh1 full"),
             (2, -1, "kvh2 full"),
             (2, 2048, "draft-like kvh2 window2048"),
             (1, 2048, "kvh1 window2048")]
    seqs = int(os.environ.get("ATTN_SEQS", NUM_SEQS))
    qps = int(os.environ.get("ATTN_QPS", Q_PER_SEQ))
    for ctx in ctxs:
        for kvh, win, tag in cases:
            t = build(ctx=ctx, num_seqs=seqs, q_per_seq=qps,
                      kv_heads=kvh, window=win)
            try:
                us = bench(lambda: call(t))
                print(f"ctx={ctx:5d} {tag:28s}: {us:8.1f} us/layer",
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"ctx={ctx} {tag} FAIL {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:100]}", flush=True)
                continue


if __name__ == "__main__":
    main()
