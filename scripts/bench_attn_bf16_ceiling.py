#!/usr/bin/env python3
"""Ceiling probe: same geometry as bench_attn_g128 but bf16 KV, no quant.

Splits 'g8 machinery is the drag' from 'core kernel/grid is the drag':
identical shapes, block table scatter, 3D split-K, and segm buffers —
only the KV dtype/quant path differs (bf16 = 105MB/call vs int8 62MB).
"""
import os, sys, torch

sys.path.insert(0, "/home/curved/aiter")
sys.path.insert(0, "/home/curved/vllm-gfx908")

from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

dev = "cuda"
torch.manual_seed(0)
BLOCK = 64
NQ, NKV, D = 6, 1, 256
SEQS, CTX, QTOK = 6, 20000, 7
NB = (SEQS * (-(-CTX // BLOCK))) + 64

k_bf = torch.randn(NB, BLOCK, NKV, D, device=dev, dtype=torch.bfloat16)
v_bf = torch.randn(NB, BLOCK, NKV, D, device=dev, dtype=torch.bfloat16)
q = torch.randn(SEQS * QTOK, NQ, D, device=dev, dtype=torch.bfloat16)
out = torch.empty_like(q)
cu_q = torch.arange(0, SEQS * QTOK + 1, QTOK, device=dev, dtype=torch.int32)
seqused = torch.full((SEQS,), CTX, device=dev, dtype=torch.int32)
perm = torch.randperm(NB - 8, device=dev)[: (SEQS * (-(-CTX // BLOCK)))].to(torch.int32)
bt = perm.view(SEQS, -1).contiguous()
scale = D ** -0.5
SPLITS = 64
segm_out = torch.empty((512, NQ, SPLITS, D), dtype=torch.float32, device=dev)
segm_max = torch.empty((512, NQ, SPLITS), dtype=torch.float32, device=dev)
segm_sum = torch.empty((512, NQ, SPLITS), dtype=torch.float32, device=dev)


def run_3d():
    unified_attention(
        q, k_bf, v_bf, out,
        cu_seqlens_q=cu_q, max_seqlen_q=QTOK,
        seqused_k=seqused, max_seqlen_k=CTX,
        softmax_scale=scale, causal=True,
        alibi_slopes=None, window_size=(-1, -1),
        block_table=bt, softcap=0.0,
        kv_quant_mode=KVQuantMode.NONE,
        q_descale=None, k_descale=None, v_descale=None,
        sinks=None, output_scale=None,
        k_scale_cache=None, v_scale_cache=None,
        num_par_softmax_segments=SPLITS,
        softmax_segm_output=segm_out,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_sum,
        seq_threshold_3D=SEQS,
        max_flash_decoding_splits=SPLITS,
    )


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


if __name__ == "__main__":
    os.environ["VLLM_UA_3D_MAXQ"] = "8"
    try:
        t = bench(run_3d)
        gb = SEQS * CTX * 2 * D * 2 / 1e9
        print(f"bf16 no-quant 3D split-K: {t:9.1f} us/call "
              f"({gb:.0f} MB KV -> {gb / (t / 1e6):.0f} GB/s)")
    finally:
        os.environ.pop("VLLM_UA_3D_MAXQ", None)
