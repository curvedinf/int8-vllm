#!/usr/bin/env python3
"""Validate + bench the gfx908 G128 prefill Gluon core (GOALOPT).

Runs the generic 2D unified-attention path as the reference, then the
VLLM_G128_PREFILL_GLUON=1 dispatch on identical inputs; reports per-call
time and output deltas across chunk geometries plus a ragged multi-seq
mapping case.
"""
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
CHUNK = 2048
CTX_MAX = 32768
NB = 4 * ((CTX_MAX + BLOCK - 1) // BLOCK) + 128

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
    (block_f16, slot_f16, head_f16, 1), storage_offset=f16u(D),
)
g8_v = torch.as_strided(
    base_f16, (NB, BLOCK, NKV, D // G),
    (block_f16, slot_f16, head_f16, 1), storage_offset=f16u(PAD + D),
)
g8_k.copy_(torch.rand_like(g8_k) * 0.005 + 0.01)
g8_v.copy_(torch.rand_like(g8_v) * 0.005 + 0.01)

scale = D ** -0.5
bt_full = torch.arange(
    (CTX_MAX + BLOCK - 1) // BLOCK, device=dev, dtype=torch.int32
).view(1, -1).contiguous()


def run(q, out, cu_q, seqused, ctx, bt):
    unified_attention(
        q, k_data, v_data, out,
        cu_seqlens_q=cu_q, max_seqlen_q=int(cu_q[-1].item()) - int(cu_q[0].item()) if cu_q.numel() == 2 else int(
            torch.diff(cu_q).max().item()),
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


def bench(fn, iters=10, warmup=3):
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


def compare(tag, q, out, cu_q, seqused, ctx, bt):
    os.environ.pop("VLLM_G128_PREFILL_GLUON", None)
    run(q, out, cu_q, seqused, ctx, bt)
    torch.cuda.synchronize()
    ref = out.clone()
    t_ref = bench(lambda: run(q, out, cu_q, seqused, ctx, bt))
    os.environ["VLLM_G128_PREFILL_GLUON"] = "1"
    run(q, out, cu_q, seqused, ctx, bt)
    torch.cuda.synchronize()
    got = out.clone()
    t_new = bench(lambda: run(q, out, cu_q, seqused, ctx, bt))
    os.environ.pop("VLLM_G128_PREFILL_GLUON", None)
    d = (got.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-3)
    rel = d / denom
    print(
        f"{tag}: generic={t_ref:8.1f}us gluon={t_new:8.1f}us "
        f"speedup={t_ref / t_new:5.2f}x max_abs={d.max():.3e} "
        f"mean_abs={d.mean():.3e} "
        f"p999_abs={d.flatten().kthvalue(int(d.numel() * 0.999)).values:.3e} "
        f"mean_rel={rel.mean():.2e}",
        flush=True,
    )


if __name__ == "__main__":
    for ctx in (2048, 8192, 16384, 24576, 32768):
        q = torch.randn(CHUNK, NQ, D, device=dev, dtype=torch.bfloat16)
        out = torch.empty_like(q)
        cu_q = torch.tensor([0, CHUNK], device=dev, dtype=torch.int32)
        seqused = torch.full((1,), ctx, device=dev, dtype=torch.int32)
        compare(f"chunk ctx={ctx:6d}", q, out, cu_q, seqused, ctx, bt_full)

    # Ragged: prefill 2048 + prefill 1000 (non-multiple of 64) + tiny 7-row
    # decode-tag-along sequence, each with its own context.
    q_lens = [2048, 1000, 7]
    ctxs = [32768, 8192, 4096]
    total = sum(q_lens)
    q = torch.randn(total, NQ, D, device=dev, dtype=torch.bfloat16)
    out = torch.full_like(q, float("nan"))
    cu = [0]
    for ql in q_lens:
        cu.append(cu[-1] + ql)
    cu_q = torch.tensor(cu, device=dev, dtype=torch.int32)
    seqused = torch.tensor(ctxs, device=dev, dtype=torch.int32)
    bt_rows = []
    off = 0
    for c in ctxs:
        n = (c + BLOCK - 1) // BLOCK
        bt_rows.append(torch.arange(off, off + n, device=dev, dtype=torch.int32))
        off += n
    bt = torch.nn.utils.rnn.pad_sequence(
        bt_rows, batch_first=True
    ).contiguous()
    os.environ.pop("VLLM_G128_PREFILL_GLUON", None)
    run(q, out, cu_q, seqused, max(ctxs), bt)
    torch.cuda.synchronize()
    ref = out.clone()
    os.environ["VLLM_G128_PREFILL_GLUON"] = "1"
    out.fill_(float("nan"))
    run(q, out, cu_q, seqused, max(ctxs), bt)
    torch.cuda.synchronize()
    d = (out.float() - ref.float()).abs()
    nan_rows = int(out.view(total, -1).isnan().any(dim=-1).sum())
    print(
        f"ragged: max_abs={d.max():.3e} mean_abs={d.mean():.3e} "
        f"nan_rows={nan_rows}/{total}",
        flush=True,
    )
    os.environ.pop("VLLM_G128_PREFILL_GLUON", None)
