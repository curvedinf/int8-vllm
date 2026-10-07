#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness + speed gate for the gfx908_g128_i8dot decode core."""
import os
import sys
import time

import torch

sys.path.insert(0, "/home/curved/aiter")
sys.path.insert(0, "/home/curved/int8-vllm")

from vllm.v1.attention.ops.triton_unified_attention import reduce_segments  # noqa: F401
from vllm.v1.attention.ops.gfx908_g128_i8dot import _g128_i8dot_kernel  # noqa: E402

dev = "cuda"
torch.manual_seed(0)
G, BLOCK = 128, 1664
NQ, D = 6, 256
SEQS, QTOK = 6, 7
DG = D // 2
PAD = D + 2 * (D // G)
CONTENT = 2 * PAD

SPLITS = 64


def build(CTX):
    NB = SEQS * (-(-CTX // BLOCK)) + 8
    packed = torch.randint(-127, 127, (NB, 1, BLOCK, CONTENT),
                           device=dev, dtype=torch.int8)
    # The fp16 group scales live INLINE in each 520-byte row
    # (bytes 256-259 = K scales, 516-519 = V scales). Random int8 bytes
    # reinterpreted as fp16 give inf/NaN, so write sane scales instead.
    # realistic magnitudes: int8 data ±127, group scales ~0.02-0.04
    sk0 = torch.tensor([0.0234], dtype=torch.float16, device=dev).view(torch.int8)
    sk1 = torch.tensor([0.0401], dtype=torch.float16, device=dev).view(torch.int8)
    sv0 = torch.tensor([0.0189], dtype=torch.float16, device=dev).view(torch.int8)
    sv1 = torch.tensor([0.0312], dtype=torch.float16, device=dev).view(torch.int8)
    pb = packed.view(-1, CONTENT)
    for off, s in ((D, sk0), (D + 2, sk1), (PAD + D, sv0), (PAD + D + 2, sv1)):
        pb[:, off:off + 2] = s
    k = packed.transpose(1, 2)[..., :D]
    v = packed.transpose(1, 2)[..., PAD:PAD + D]
    raw = packed.untyped_storage()
    bf = torch.tensor([], dtype=torch.float16, device=dev).set_(raw)
    f = lambda n: n // 2
    g8 = torch.as_strided(
        bf, (NB, BLOCK, 1, D // G),
        (f(packed.stride(0)), f(packed.stride(2)), f(packed.stride(1)), 1),
        storage_offset=f(D),
    )
    g8v = torch.as_strided(
        bf, (NB, BLOCK, 1, D // G),
        (f(packed.stride(0)), f(packed.stride(2)), f(packed.stride(1)), 1),
        storage_offset=f(PAD + D),
    )
    q = (torch.randn(SEQS * QTOK, NQ, D, device=dev, dtype=torch.bfloat16) * 0.5)
    return NB, packed, k, v, g8, g8v, q


def reference(packed, g8, g8v, q, CTX):
    """Dequant fp32 reference softmax attention."""
    outs = []
    for s in range(SEQS):
        qs = q[s * QTOK:(s + 1) * QTOK].float()  # (7, 6, 256)
        nblk = -(-CTX // BLOCK)
        Kd = torch.empty(CTX, D, device=dev)
        Vd = torch.empty(CTX, D, device=dev)
        for b in range(nblk):
            lo, hi = b * BLOCK, min((b + 1) * BLOCK, CTX)
            row = packed[s * nblk + b, 0, :hi - lo]  # (T, 520)
            kq = row[:, :D].float()
            vq = row[:, PAD:PAD + D].float()
            kk = g8[s * nblk + b, :hi - lo, 0]  # (T, 2) fp16 (K scales)
            vv = g8v[s * nblk + b, :hi - lo, 0]  # (T, 2) fp16 (V scales)
            Kd[lo:hi, :DG] = kq[:, :DG] * kk[:, 0:1].float()
            Kd[lo:hi, DG:] = kq[:, DG:] * kk[:, 1:2].float()
            Vd[lo:hi, :DG] = vq[:, :DG] * vv[:, 0:1].float()
            Vd[lo:hi, DG:] = vq[:, DG:] * vv[:, 1:2].float()
        for t in range(QTOK):
            ctx_end = CTX - QTOK + t + 1  # causal over verify rows
            sc = torch.einsum("hd,td->ht", qs[t], Kd[:ctx_end]) * 0.03125
            p = torch.softmax(sc, dim=-1)
            o = p @ Vd[:ctx_end]  # (6, 256)
            outs.append(o)
    return torch.stack(outs)  # (42, 6, 256)


def run_kernel(packed, k, v, g8, g8v, q, CTX):
    NB = packed.shape[0]
    nblk_per_seq = -(-CTX // BLOCK)
    bt = torch.arange(SEQS * nblk_per_seq, device=dev,
                      dtype=torch.int32).reshape(SEQS, nblk_per_seq)
    cu = torch.arange(0, SEQS * QTOK + 1, QTOK, device=dev, dtype=torch.int32)
    seqused = torch.full((SEQS,), CTX, device=dev, dtype=torch.int32)

    total_q = SEQS * QTOK
    segm_out = torch.full((512, NQ, SPLITS, D), float('nan'), dtype=torch.float32, device=dev)
    segm_max = torch.full((512, NQ, SPLITS), 123.0, dtype=torch.float32, device=dev)
    segm_sum = torch.full((512, NQ, SPLITS), 5.0, dtype=torch.float32, device=dev)
    out = torch.empty((total_q, NQ, D), dtype=torch.bfloat16, device=dev)

    nblocks = int(cu[-1].item()) // 10 + SEQS
    grid = (nblocks, SPLITS)
    TILE = int(os.environ.get("I8_TILE", "64"))
    STAGES = int(os.environ.get("I8_STAGES", "1"))
    _g128_i8dot_kernel[grid](
        q, k, v, g8, g8v, bt, seqused, cu,
        segm_out, segm_max, segm_sum,
        0.03125,
        num_query_heads=NQ, nq_per_kv=NQ, num_seqs=SEQS,
        block_size=BLOCK, splits=SPLITS,
        bt_stride=bt.stride(0),
        q_stride0=q.stride(0), q_stride1=q.stride(1),
        k_stride0=k.stride(0), k_stride1=k.stride(1), k_stride2=k.stride(2),
        v_stride0=v.stride(0), v_stride1=v.stride(1), v_stride2=v.stride(2),
        s_stride0=g8.stride(0), s_stride1=g8.stride(1), s_stride2=g8.stride(2),
        ROWS=64, TILE=TILE, D=D, DG=DG,
        num_warps=4,
        num_stages=STAGES,
    )

    reduce_segments[(total_q, NQ)](
        output_ptr=out,
        segm_output_ptr=segm_out,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_sum,
        seq_lens_ptr=seqused,
        num_seqs=SEQS,
        num_query_heads=NQ,
        out_scale_inv=1.0,
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        block_table_stride=bt.stride(0),
        TILE_SIZE=32,
        HEAD_SIZE=D,
        HEAD_SIZE_PADDED=D,
        query_start_len_ptr=cu,
        BLOCK_Q=7,
        NUM_SEGMENTS_PER_SEQ=SPLITS,
        USE_FP8=False,
    )
    return out


if __name__ == "__main__":
    CTX = 20000
    NB, packed, k, v, g8, g8v, q = build(CTX)
    print("building fp32 reference (slow)...")
    ref = reference(packed, g8, g8v, q, CTX)
    got = run_kernel(packed, k, v, g8, g8v, q, CTX).float()
    rel = (got - ref).abs() / (ref.abs() + 1e-3)
    print(f"shape {tuple(got.shape)} | max_abs {(got-ref).abs().max():.4f} "
          f"| mean_rel {rel.mean():.4f} | p99_rel {rel.flatten().kthvalue(int(rel.numel()*0.99)).values:.4f}")
    top_ref = ref.argmax(-1)
    top_got = got.argmax(-1)
    print(f"top-1 match: {(top_ref == top_got).float().mean():.4f}")

    # speed sweep
    print(f"\n{'ctx':>7} {'us/call':>9} {'GB/s':>7}")
    HBM = 1229e9 * 0.8
    for CTX in (16_000, 32_000, 64_000, 128_000, 200_000):
        NB, packed, k, v, g8, g8v, q = build(CTX)
        for _ in range(3):
            run_kernel(packed, k, v, g8, g8v, q, CTX)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 20
        for _ in range(n):
            run_kernel(packed, k, v, g8, g8v, q, CTX)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / n * 1e6
        kv_bytes = SEQS * CTX * CONTENT
        print(f"{CTX:>7} {us:>9.0f} {kv_bytes/(us/1e6)/1e9:>7.0f}")
        del packed, k, v, g8, g8v, q
        torch.cuda.empty_cache()
