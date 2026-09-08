#!/usr/bin/env python3
"""In-vitro T-shape A/B of the production unified attention on int8-g128 KV.

One cache, identical queries: run the AITER unified attention with T=1
(decode), T=14 (verify), and T=2048 (prefill chunk) query shapes over the
same int8-block-g128 KV and compare row 0. T=1 and T=2048 agree with the
clean comparator in-situ; if T=14's row 0 differs, that read path is the
G1b deep component.

Usage: HIP_VISIBLE_DEVICES=0 PYTHONPATH=... python scripts/test_attn_tshape.py
"""
import sys

import torch

sys.path.insert(0, "/home/curved/vllm-gfx908")
sys.path.insert(0, "/home/curved/aiter")
from aiter.ops.triton.attention.unified_attention import unified_attention  # noqa

dev = "cuda"
torch.manual_seed(0)

NUM_Q_HEADS = 8   # per-rank (16/TP2-equivalent shrink for speed)
NUM_KV_HEADS = 2
HEAD_DIM = 128
CTX = 8192
BLOCK = 64
NUM_BLOCKS = CTX // BLOCK + 64
SCALE = HEAD_DIM ** -0.5
GS = 128

g = torch.Generator(device=dev).manual_seed(21)
# int8 KV cache pages + per-token-head scales
k_cache = torch.zeros(NUM_BLOCKS, BLOCK, NUM_KV_HEADS, HEAD_DIM,
                      dtype=torch.int8, device=dev)
v_cache = torch.zeros_like(k_cache)
k_scale = torch.rand(NUM_BLOCKS, BLOCK, NUM_KV_HEADS,
                     dtype=torch.float32, device=dev, generator=g) * 0.02 + 0.005
v_scale = torch.rand_like(k_scale) * 0.02 + 0.005
# fill CTX tokens with realistic int8 values
kq = torch.randint(-100, 101, (NUM_KV_HEADS, CTX, HEAD_DIM), generator=g,
                   device=dev, dtype=torch.int8)
vq = torch.randint(-100, 101, (NUM_KV_HEADS, CTX, HEAD_DIM), generator=g,
                   device=dev, dtype=torch.int8)
for pos in range(CTX):
    blk, off = pos // BLOCK, pos % BLOCK
    k_cache[blk, off] = kq[:, pos]
    v_cache[blk, off] = vq[:, pos]

# queries at the context tail (bf16 like production)
q_tail = torch.randn(NUM_Q_HEADS, 14, HEAD_DIM, generator=g, device=dev,
                     dtype=torch.bfloat16) * 0.5


def run(T, q_rows, ctx_len):
    """q_rows: [H, T, D] flattened to [T, H, D] tokens, one request."""
    q = q_rows.permute(1, 0, 2).contiguous()  # [T, H, D]
    out = torch.empty_like(q)
    cu = torch.arange(0, T + 1, dtype=torch.int32, device=dev) * 0
    cu = torch.zeros(T + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.arange(1, T + 1, dtype=torch.int32, device=dev)
    seqused = torch.tensor([ctx_len], dtype=torch.int32, device=dev)
    bt = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=dev)[:CTX // BLOCK + 8]
    bt = bt.reshape(1, -1).contiguous()
    unified_attention(
        q=q, k=k_cache, v=v_cache, out=out,
        cu_seqlens_q=cu, max_seqlen_q=T,
        seqused_k=seqused, max_seqlen_k=CTX,
        softmax_scale=SCALE, causal=True,
        window_size=(-1, -1), block_table=bt, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
        k_scale_cache=k_scale, v_scale_cache=v_scale,
    )
    torch.cuda.synchronize()
    return out


CTX_LEN = CTX - 14  # rows attend the same full context
out1 = run(1, q_tail[:, :1, :], CTX_LEN + 1)
out14 = run(14, q_tail, CTX_LEN + 14)
# prefill-shaped read: treat all 14 rows as a chunk at the tail with full ctx
out2048 = None

d14v1 = (out14[0].float() - out1[0].float()).abs()
print(f"row0 |Δ| T=14 vs T=1:  max={d14v1.max().item():.6f} mean={d14v1.mean().item():.8f}")
print(f"  out scale: {out1.float().abs().mean().item():.4f}")
# also compare each T=14 row against the T=1 run of the same query row
outs1 = []
for t in range(14):
    o = run(1, q_tail[:, t:t+1, :], CTX_LEN + 1 + t)
    outs1.append(o[0])
outs1 = torch.stack(outs1)  # [14, H, D]
dtot = (out14.float() - outs1.float()).abs()
print(f"all rows T=14 vs per-row T=1: max={dtot.max().item():.6f} "
      f"mean={dtot.mean().item():.8f}")
per_row = dtot.mean(dim=(1, 2))
print("per-row mean|Δ|:", [f"{v:.6f}" for v in per_row.tolist()])

# --- T=2048 (prefill shape): row 0 at the same position/context as the T=1 read
q_big = torch.randn(NUM_Q_HEADS, 2048, HEAD_DIM, generator=g, device=dev,
                    dtype=torch.bfloat16) * 0.5
# put q_tail row0 at the START of the 2048 block; rows attend causally so
# row 0 attends only seqused (the shared context) — same as its T=1 read.
q_big[:, 0] = q_tail[:, 0]
out2048 = run(2048, q_big, CTX_LEN + 2048)
d_2048v1 = (out2048[0].float() - out1[0].float()).abs()
print(f"row0 |Δ| T=2048 vs T=1: max={d_2048v1.max().item():.6f} mean={d_2048v1.mean().item():.8f}")
d_2048v14 = (out2048[0].float() - out14[0].float()).abs()
print(f"row0 |Δ| T=2048 vs T=14: max={d_2048v14.max().item():.6f} mean={d_2048v14.mean().item():.8f}")
