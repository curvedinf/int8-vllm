#!/usr/bin/env python3
"""Standalone repro: triton_unified_attention noncausal+SWA at DFlash2 draft shapes.

Observed serving metadata (SPEC-DBGB): 8 seqs x 8 queries, seq_len=17,
window=(2047,0), causal=False, draft 8Q/2KV heads per rank, head_dim 128,
fp16 KV. Reference = dense bidirectional attention over the 17 keys (the
2048 window covers everything at these lengths).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm.v1.attention.ops.triton_unified_attention import unified_attention

torch.manual_seed(0)
dev = "cuda"

NUM_SEQS, Q_PER_SEQ, SEQ_LEN = 8, 8, 17
NQ, NKV, HD = 8, 2, 128
BLOCK, NUM_BLOCKS = 32, 64
NT = NUM_SEQS * Q_PER_SEQ

q = torch.randn(NT, NQ, HD, dtype=torch.float16, device=dev)
k_cache = torch.randn(NUM_BLOCKS, BLOCK, NKV, HD, dtype=torch.float16, device=dev)
v_cache = torch.randn(NUM_BLOCKS, BLOCK, NKV, HD, dtype=torch.float16, device=dev)

# seq i occupies logical positions [0, 17) -> blocks 2i and 2i+1 (17 > 32? no, 17<=32 -> 1 block)
block_table = torch.zeros(NUM_SEQS, 16, dtype=torch.int32, device=dev)
for i in range(NUM_SEQS):
    block_table[i, 0] = i  # seq i uses block i, 17 tokens fit one 32 block

seqused_k = torch.full((NUM_SEQS,), SEQ_LEN, dtype=torch.int32, device=dev)
cu_q = torch.arange(0, NT + 1, Q_PER_SEQ, dtype=torch.int32, device=dev)

out = torch.full((NT, NQ, HD), float("nan"), dtype=torch.float16, device=dev)

unified_attention(
    q=q,
    k=k_cache,
    v=v_cache,
    out=out,
    cu_seqlens_q=cu_q,
    max_seqlen_q=Q_PER_SEQ,
    seqused_k=seqused_k,
    max_seqlen_k=SEQ_LEN,
    softmax_scale=HD**-0.5,
    causal=False,
    window_size=(2047, 0),
    block_table=block_table,
    softcap=0.0,
    q_descale=None,
    k_descale=None,
    v_descale=None,
)
torch.cuda.synchronize()

# Dense reference: bidirectional over all 17 keys of the same block.
kt = k_cache[:NUM_SEQS, :SEQ_LEN].float()  # [S, L, NKV, HD]
vt = v_cache[:NUM_SEQS, :SEQ_LEN].float()
ref = torch.empty(NT, NQ, HD, dtype=torch.float32, device=dev)
for i in range(NUM_SEQS):
    qi = q[i * Q_PER_SEQ : (i + 1) * Q_PER_SEQ].float()  # [8q, NQ, HD]
    ki, vi = kt[i].repeat_interleave(NQ // NKV, dim=1), vt[i].repeat_interleave(
        NQ // NKV, dim=1
    )
    scores = torch.einsum("qhd,khd->hqk", qi, ki) * HD**-0.5
    p = torch.softmax(scores, dim=-1)
    ref[i * Q_PER_SEQ : (i + 1) * Q_PER_SEQ] = torch.einsum(
        "hqk,khd->qhd", p, vi
    )

diff = (out.float() - ref).abs()
print(f"out nan={int(out.isnan().sum())}/{out.numel()} absmax={out.abs().max():.4f}")
print(f"ref absmax={ref.abs().max():.4f}")
print(f"maxdiff={diff.max():.4f} meandiff={diff.mean():.6f}")
zero_rows = (out.abs().amax(dim=-1) == 0).sum()
print(f"all-zero query-head rows: {zero_rows}/{NT * NQ}")
