#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long-context mixed-load attention profile: G128 decode/verify kernel
wall vs context length, against the HBM bandwidth floor.

Serving geometry (6 seqs x 7 verify rows, 6 q-heads : 1 kv-head, D=256,
int8 block-g128 with inline fp16 group scales), matching what the
production recipe actually executes per decode step per full-attn layer.
"""
import math
import sys
import time

import torch

sys.path.insert(0, "/home/curved/aiter")
sys.path.insert(0, "/home/curved/int8-vllm")

from vllm.v1.attention.ops.triton_unified_attention import unified_attention  # noqa: E402
from vllm.v1.kv_cache_interface import KVQuantMode  # noqa: E402

dev = "cuda"
torch.manual_seed(0)
G, BLOCK = 128, 1664
NQ, NKV, D = 6, 1, 256
SEQS, QTOK = 6, 7

# 3D split-K buffers (production uses 64 splits over 256 rows)
SPLITS = 64
SEGM_ROWS = 512
_seg_out = torch.empty((SEGM_ROWS, NQ, SPLITS, D), dtype=torch.float32, device=dev)
_seg_max = torch.empty((SEGM_ROWS, NQ, SPLITS), dtype=torch.float32, device=dev)
_seg_sum = torch.empty((SEGM_ROWS, NQ, SPLITS), dtype=torch.float32, device=dev)
SPLIT_KW = dict(
    num_par_softmax_segments=SPLITS,
    softmax_segm_output=_seg_out,
    softmax_segm_max=_seg_max,
    softmax_segm_expsum=_seg_sum,
    seq_threshold_3D=256,
    max_flash_decoding_splits=SPLITS,
)

PAD = D + 2 * (D // G)
CONTENT = 2 * PAD

# MI100 HBM ~1229 GB/s; effective read efficiency on scattered pages ~80%
HBM = 1229e9 * 0.8

print(f"{'ctx':>7} {'us/call':>9} {'floor_us':>9} {'ratio':>6} "
      f"{'MB_read':>8} {'GB/s':>7}")

for CTX in (16_000, 32_000, 64_000, 128_000, 200_000):
    NB = SEQS * (-(-CTX // BLOCK)) + 8
    blocks_per_seq = -(-CTX // BLOCK)
    bt = torch.arange(SEQS * blocks_per_seq, device=dev,
                      dtype=torch.int32).reshape(SEQS, blocks_per_seq)
    packed = torch.randint(
        -127, 127, (NB, NKV, BLOCK, CONTENT), device=dev, dtype=torch.int8
    )
    raw = packed.untyped_storage()
    base_f16 = torch.tensor([], dtype=torch.float16, device=dev).set_(raw)

    def f16u(n):
        return n // 2

    block_f16 = f16u(packed.stride(0))
    head_f16 = f16u(packed.stride(1))
    slot_f16 = f16u(packed.stride(2))
    g8 = torch.as_strided(
        base_f16, (NB, BLOCK, NKV, D // G),
        (block_f16, slot_f16, head_f16, 1),
        storage_offset=f16u(D),
    )

    k_data = packed.transpose(1, 2)[..., :D].contiguous().transpose(1, 2)
    v_data = packed.transpose(1, 2)[..., PAD:PAD + D].contiguous().transpose(1, 2)
    # keep (NB, BLOCK, NKV, D) logical layout the kernel reads
    k_data = packed.transpose(1, 2)[..., :D]
    v_data = packed.transpose(1, 2)[..., PAD:PAD + D]

    q = torch.randn(SEQS * QTOK, NQ, D, device=dev, dtype=torch.bfloat16) * 0.1
    out = torch.empty_like(q)
    cu = torch.arange(0, SEQS * QTOK + 1, QTOK, device=dev, dtype=torch.int32)
    seqused = torch.full((SEQS,), CTX, device=dev, dtype=torch.int32)
    bt = torch.arange(SEQS * blocks_per_seq, device=dev,
                      dtype=torch.int32).reshape(SEQS, blocks_per_seq)

    def call():
        unified_attention(
            q, k_data, v_data, out,
            cu_seqlens_q=cu, max_seqlen_q=QTOK,
            seqused_k=seqused, max_seqlen_k=CTX,
            softmax_scale=0.03125, causal=True,
            alibi_slopes=None, window_size=(-1, -1),
            block_table=bt, softcap=0.0,
            kv_quant_mode=KVQuantMode.INT8_BLOCK_G128,
            q_descale=None, k_descale=None, v_descale=None,
            sinks=None, output_scale=None,
            k_scale_cache=None, v_scale_cache=None,
            g8_k_scale=g8, g8_v_scale=g8,
            **SPLIT_KW,
        )

    for _ in range(3):
        call()
    torch.cuda.synchronize()
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        call()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / n * 1e6

    kv_bytes = SEQS * CTX * CONTENT  # int8 KV read per call
    mb = kv_bytes / 1e6
    floor_us = kv_bytes / HBM * 1e6
    gbps = kv_bytes / (us / 1e6) / 1e9
    print(f"{CTX:>7} {us:>9.0f} {floor_us:>9.0f} {us/floor_us:>6.1f} "
          f"{mb:>8.0f} {gbps:>7.0f}")
    del packed, q, out, bt, g8
    torch.cuda.empty_cache()
