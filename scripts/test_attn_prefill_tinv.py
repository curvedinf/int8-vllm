#!/usr/bin/env python3
"""Prefill T-invariance of the AITER unified attention kernel.

Engine probes (G1_PARTIAL_CHUNK_ROOT_CAUSE) showed the last partial
scheduler chunk's row values depend on its token count. GEMM (M-tiling) and
the FLA GDN prefill op are both bitwise T-invariant in the 1672..2048 range,
so the attention read is the remaining candidate. This test: prefix of N
keys in pages + T query rows with their own keys appended; run T_part vs
T_full on identical content; the shared prefix rows must be BITWISE equal.

Usage: HIP_VISIBLE_DEVICES=0 PYTHONPATH=pwd:../aiter .venv/bin/python \
  scripts/test_attn_prefill_tinv.py
"""
import argparse
import sys

import torch

sys.path.insert(0, "/home/curved/vllm-gfx908")
sys.path.insert(0, "/home/curved/aiter")
from aiter.ops.triton.attention.unified_attention import unified_attention  # noqa

p = argparse.ArgumentParser()
p.add_argument("--ctx", type=int, default=20000)
p.add_argument("--heads", type=int, default=6)
p.add_argument("--kvh", type=int, default=1)
p.add_argument("--hd", type=int, default=256)
p.add_argument("--block", type=int, default=64)
p.add_argument("--tries", type=int, default=3)
args = p.parse_args()

H, KVH, D, BLOCK = args.heads, args.kvh, args.hd, args.block
N = args.ctx
T_FULL = 2048
SCALE = D ** -0.5
dev = "cuda"
NPAGES = (N + T_FULL) // BLOCK + 4

kc = torch.zeros(NPAGES, BLOCK, KVH, D, dtype=torch.float16, device=dev)
vc = torch.zeros_like(kc)


def build(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    kall = (torch.randn(KVH, N + T_FULL, D, generator=g, device=dev) * 0.5
            ).to(torch.float16)
    vall = (torch.randn(KVH, N + T_FULL, D, generator=g, device=dev) * 0.5
            ).to(torch.float16)
    for pos in range(N + T_FULL):
        kc[pos // BLOCK, pos % BLOCK] = kall[:, pos]
        vc[pos // BLOCK, pos % BLOCK] = vall[:, pos]
    q = (torch.randn(H, T_FULL, D, generator=g, device=dev) * 0.35
         ).to(torch.bfloat16)
    return q


def go(q_hdt, T):
    q = q_hdt[:, :T].permute(1, 0, 2).contiguous()      # [T, H, D]
    out = torch.empty_like(q)
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    bt = torch.arange((N + T) // BLOCK + 2, device=dev,
                      dtype=torch.int32).reshape(1, -1).contiguous()
    unified_attention(
        q=q, k=kc, v=vc, out=out,
        cu_seqlens_q=cu, max_seqlen_q=T,
        seqused_k=torch.tensor([N + T], dtype=torch.int32, device=dev),
        max_seqlen_k=N + T,
        softmax_scale=SCALE, causal=True,
        window_size=(-1, -1), block_table=bt, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
    )
    torch.cuda.synchronize()
    return out.contiguous()


for trial in range(args.tries):
    q = build(2000 + trial)
    out_full = go(q, T_FULL)
    for T_part in (1672, 1984, 2047):
        out_part = go(q, T_part)
        a, b = out_full[:T_part].float(), out_part.float()
        d = (a - b).abs()
        eq = torch.equal(out_full[:T_part], out_part)
        nz = (d > 0).float().mean().item()
        print(f"trial {trial} T={T_part:5d}: bitwise={eq} "
              f"frac_diff={nz:.5f} max={d.max():.3e} mean={d.mean():.3e}",
              flush=True)
