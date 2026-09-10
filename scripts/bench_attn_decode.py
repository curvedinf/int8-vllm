#!/usr/bin/env python3
"""Standalone decode-attention timing: aiter unified_attention vs vLLM
triton_unified_attention at engine-realistic shapes and varying context.

Shapes: Qwen3.8-27B TP4 full-attn layers: 6 q-heads, 1 kv-head, head_dim
256 per rank, int8-block-g128 KV (and fp16 KV variant).
"""
import sys
import torch

sys.path.insert(0, "/home/curved/aiter")
sys.path.insert(0, "/home/curved/vllm-gfx908")

from aiter.ops.triton.attention.unified_attention import unified_attention

dev = "cuda"
HQ, HKV, D = 6, 1, 256
BLOCK = 512  # paged block size (rows per block in cache)


def build(ctx, dtype_kv=torch.int8):
    nblocks = (ctx + BLOCK - 1) // BLOCK
    q = torch.randn(1, HQ, D, device=dev, dtype=torch.bfloat16)
    if dtype_kv == torch.int8:
        kcache = torch.randint(-127, 127, (nblocks, BLOCK, HKV, D),
                               device=dev, dtype=torch.int8)
        vcache = torch.randint(-127, 127, (nblocks, BLOCK, HKV, D),
                               device=dev, dtype=torch.int8)
        ks = torch.rand(nblocks, BLOCK, HKV, D // 128, 128,
                        device=dev, dtype=torch.float16)
        vs = torch.rand(nblocks, BLOCK, HKV, D // 128, 128,
                        device=dev, dtype=torch.float16)
        return q, kcache, vcache, ks, vs
    kcache = torch.randn(nblocks, BLOCK, HKV, D, device=dev,
                         dtype=torch.bfloat16)
    vcache = torch.randn(nblocks, BLOCK, HKV, D, device=dev,
                         dtype=torch.bfloat16)
    return q, kcache, vcache, None, None


def bench(ctx, dtype_kv=torch.int8, iters=100):
    q, k, v, ks, vs = build(ctx, dtype_kv)
    out = torch.empty_like(q)
    cu_q = torch.tensor([0, 1], device=dev, dtype=torch.int32)
    seqused = torch.tensor([ctx], device=dev, dtype=torch.int32)
    # simple contiguous block table: block i = token block i
    bt = torch.arange(nblocks_(ctx), device=dev, dtype=torch.int32).view(1, -1)

    def run():
        unified_attention(
            q, k, v, out,
            cu_seqlens_q=cu_q, max_seqlen_q=1,
            seqused_k=seqused, max_seqlen_k=ctx,
            softmax_scale=D ** -0.5, causal=True,
            window_size=(-1, -1),
            block_table=bt,
            softcap=0.0,
            q_descale=None,
            k_descale=ks, v_descale=vs,
        )

    try:
        for _ in range(10):
            run()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            run()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / iters
    except Exception as ex:
        return float("nan")


def nblocks_(ctx):
    return (ctx + BLOCK - 1) // BLOCK


if __name__ == "__main__":
    dt = torch.int8
    print(f"aiter unified_attention decode, HQ={HQ} HKV={HKV} D={D} int8-g128:")
    for ctx in (2000, 8000, 20000, 60000):
        ms = bench(ctx, dt)
        print(f"  ctx {ctx:6d}: {ms:8.3f} ms/layer-call", flush=True)
    print("per-step for 16 attn layers (target):")
    for ctx in (2000, 8000, 20000):
        ms = bench(ctx, dt)
        print(f"  ctx {ctx:6d}: {16*ms:8.1f} ms", flush=True)
