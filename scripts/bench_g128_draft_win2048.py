#!/usr/bin/env python3
"""Validate + bench the draft G128 Gluon core at the live window (2048).

GOALOPT: capture-time guard debug showed live draft calls arriving with
SLIDING_WINDOW=2048 (window_size 2047), which the previous 2049-only
guard silently dropped to the generic 3D kernel. This bench replicates
the exact live geometry - int8_block_g128 packed KV, head 128, one
fp16 group scale per 128 dims, 2 kv-heads, GQA 4:1, noncausal SWA -
and compares the draft core against the generic kernel at window_size
(2047, 0), plus a torch reference for window-boundary correctness.
"""
import os
import sys

import torch

sys.path.insert(0, "/home/curved/aiter")
sys.path.insert(0, "/home/curved/vllm-gfx908")

from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

os.environ.setdefault("VLLM_UA_3D_MAXQ", "8")

dev = "cuda"
torch.manual_seed(0)

G = 128
BLOCK = 1664
NQ, NKV, D = 8, 2, 128
SEQS, QTOK = 6, 7
CTX = int(os.environ.get("DRAFT_CTX", "32000"))
PAD = D + 2 * (D // G)
CONTENT = 2 * PAD
NB = SEQS * (-(-CTX // BLOCK)) + 64

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
    (block_f16, slot_f16, head_f16, 1), storage_offset=f16u(D),
)
g8_v = torch.as_strided(
    base_f16, (NB, BLOCK, NKV, D // G),
    (block_f16, slot_f16, head_f16, 1), storage_offset=f16u(PAD + D),
)
g8_k.copy_(torch.rand_like(g8_k) * 0.005 + 0.01)
g8_v.copy_(torch.rand_like(g8_v) * 0.005 + 0.01)

q = torch.randn(SEQS * QTOK, NQ, D, device=dev, dtype=torch.bfloat16)
out = torch.empty_like(q)
cu_q = torch.arange(0, SEQS * QTOK + 1, QTOK, device=dev, dtype=torch.int32)
seqused = torch.full((SEQS,), CTX, device=dev, dtype=torch.int32)
perm = torch.randperm(NB - 8, device=dev)[: SEQS * (-(-CTX // BLOCK))].to(torch.int32)
bt = perm.view(SEQS, -1).contiguous()
scale = D ** -0.5
SPLITS = 32
segm_out = torch.empty((SEQS * QTOK, NQ, SPLITS, D), dtype=torch.float32, device=dev)
segm_max = torch.empty((SEQS * QTOK, NQ, SPLITS), dtype=torch.float32, device=dev)
segm_sum = torch.empty((SEQS * QTOK, NQ, SPLITS), dtype=torch.float32, device=dev)


def run(env_on: bool):
    if env_on:
        os.environ["VLLM_G128_DRAFT_GLUON"] = "1"
    else:
        os.environ.pop("VLLM_G128_DRAFT_GLUON", None)
    unified_attention(
        q, k_data, v_data, out,
        cu_seqlens_q=cu_q, max_seqlen_q=QTOK,
        seqused_k=seqused, max_seqlen_k=CTX,
        softmax_scale=scale, causal=False,
        alibi_slopes=None, window_size=(2047, 0),
        block_table=bt, softcap=0.0,
        kv_quant_mode=KVQuantMode.INT8_BLOCK_G128,
        q_descale=None, k_descale=None, v_descale=None,
        sinks=None, output_scale=None,
        k_scale_cache=None, v_scale_cache=None,
        g8_k_scale=g8_k, g8_v_scale=g8_v,
        num_par_softmax_segments=SPLITS,
        softmax_segm_output=segm_out,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_sum,
        seq_threshold_3D=SEQS,
        max_flash_decoding_splits=SPLITS,
    )
    torch.cuda.synchronize()
    return out.clone()


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
    ref = run(False)
    got = run(True)
    d = (got.float() - ref.float()).abs()
    rel = d / ref.float().abs().clamp_min(1e-3)
    print(f"ctx={CTX} draft-gluon vs generic (win 2048, noncausal): "
          f"max_abs={d.max().item():.3e} mean_abs={d.mean().item():.3e} "
          f"mean_rel={rel.mean().item():.3e} "
          f"gt10m={(d > 0.01).sum().item()}/{d.numel()}")

    def go():
        run(True)

    def go_ref():
        run(False)

    t_new = bench(go)
    t_ref = bench(go_ref)
    print(f"per-call us: generic={t_ref:.1f} draft-gluon={t_new:.1f} "
          f"speedup={t_ref / t_new:.2f}x")
    os.environ.pop("VLLM_G128_DRAFT_GLUON", None)
