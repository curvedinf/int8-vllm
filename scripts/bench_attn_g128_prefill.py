#!/usr/bin/env python3
"""gfx908 G128 prefill-attention sweep (GOALOPT).

Replicates the solo 32k chunked-prefill attention geometry of the recipe:
1 sequence, q = G128_PREFILL_Q (2048) new tokens, growing context, packed
int8_block_g128 KV with inline fp16 group scales, head 256, GQA 6:1,
block 1664, scattered block table. Sweeps the 2D prefill kernel's launch
geometry via the VLLM_UA_PREFILL_* envs and reports per-call us for each
chunk position plus the summed solo-32k-prefill attention cost.
"""
import itertools
import os
import sys

import torch

sys.path.insert(0, "/home/curved/aiter")
sys.path.insert(0, "/home/curved/vllm-gfx908")

from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

dev = "cuda"
torch.manual_seed(0)

G = 128
BLOCK = 1664
NQ, NKV, D = 6, 1, 256
QTOK = int(os.environ.get("G128_PREFILL_Q", "2048"))
CTX_MAX = int(os.environ.get("G128_CTX", "32000"))
CHUNK = QTOK
NB = ((CTX_MAX + CHUNK - 1) // CHUNK) * ((CTX_MAX + BLOCK - 1) // BLOCK) + 128

PAD = D + 2 * (D // G)
CONTENT = 2 * PAD

packed = torch.randint(
    -127, 127, (NB, NKV, BLOCK, CONTENT), device=dev, dtype=torch.int8
)
k_data = packed.transpose(1, 2)[..., :D]
v_data = packed.transpose(1, 2)[..., PAD:PAD + D]
raw = packed.untyped_storage()
base_f16 = torch.tensor([], dtype=torch.float16, device=dev).set_(raw)


def f16u(n):
    return n // 2


block_f16 = f16u(packed.stride(0))
head_f16 = f16u(packed.stride(1))
slot_f16 = f16u(packed.stride(2))
g8_k = torch.as_strided(
    base_f16, (NB, BLOCK, NKV, D // G),
    (block_f16, slot_f16, head_f16, 1),
    storage_offset=f16u(D),
)
g8_v = torch.as_strided(
    base_f16, (NB, BLOCK, NKV, D // G),
    (block_f16, slot_f16, head_f16, 1),
    storage_offset=f16u(PAD + D),
)
g8_k.copy_(torch.rand_like(g8_k) * 0.005 + 0.01)
g8_v.copy_(torch.rand_like(g8_v) * 0.005 + 0.01)

q = torch.randn(QTOK, NQ, D, device=dev, dtype=torch.bfloat16)
out = torch.empty_like(q)
cu_q = torch.tensor([0, QTOK], device=dev, dtype=torch.int32)
scale = D ** -0.5

chunks = []
for ctx in range(CHUNK, CTX_MAX + 1, CHUNK):
    nblk = (ctx + BLOCK - 1) // BLOCK
    bt = torch.arange(nblk, device=dev, dtype=torch.int32).view(1, -1)
    seqused = torch.full((1,), ctx, device=dev, dtype=torch.int32)
    chunks.append((ctx, bt.contiguous(), seqused))


def run(bt, seqused, ctx):
    unified_attention(
        q, k_data, v_data, out,
        cu_seqlens_q=cu_q, max_seqlen_q=QTOK,
        seqused_k=seqused, max_seqlen_k=ctx,
        softmax_scale=scale, causal=True,
        alibi_slopes=None, window_size=(-1, -1),
        block_table=bt, softcap=0.0,
        kv_quant_mode=KVQuantMode.INT8_BLOCK_G128,
        q_descale=None, k_descale=None, v_descale=None,
        sinks=None, output_scale=None,
        k_scale_cache=None, v_scale_cache=None,
        g8_k_scale=g8_k, g8_v_scale=g8_v,
    )


def bench_chunk(bt, seqused, ctx, iters=10, warmup=3):
    for _ in range(warmup):
        run(bt, seqused, ctx)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        run(bt, seqused, ctx)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


def sweep_config():
    parts = []
    for var in ("VLLM_UA_PREFILL_TILE", "VLLM_UA_PREFILL_BLOCKM",
                "VLLM_UA_PREFILL_WARPS", "VLLM_UA_PREFILL_STAGES"):
        v = os.environ.get(var)
        if v:
            parts.append(f"{var.split('_')[-1].lower()}={v}")
    return " ".join(parts) if parts else "default"


if __name__ == "__main__":
    ctx_probe = [2048, 8192, 16384, 24576, 32000]
    sel = [c for c in chunks if c[0] in ctx_probe]
    total_us = 0.0
    times = []
    for ctx, bt, seqused in sel:
        t = bench_chunk(bt, seqused, ctx)
        times.append((ctx, t))
        print(f"  ctx={ctx:6d}: {t:9.1f} us/call", flush=True)
    # full solo-prefill sum over every chunk
    for ctx, bt, seqused in chunks:
        total_us += bench_chunk(bt, seqused, ctx, iters=4, warmup=2)
    print(f"CONFIG [{sweep_config()}] probe_sum="
          f"{sum(t for _, t in times) / 1000:.2f} ms "
          f"solo32k_sum={total_us / 1000:.1f} ms", flush=True)
