#!/usr/bin/env python3
"""T-shape verify-read test v2 — decisive harness for the G1b open surface.

  1. EXACT fp32 reference (dequantized causal attention in torch) scores
     BOTH the T=1 (decode) and T=14 (verify) reads.
  2. PEAKED attention: queries aligned with a few dominant keys.
  3. MATCHED causal geometry: batch rows pre-written; T=14 row t attends
     [0, N+t], its per-row T=1 counterpart attends the identical key set.
  4. g128 scale layout: fp16 [blocks, block, kvh, D/128, 128].

Usage: HIP_VISIBLE_DEVICES=0 PYTHONPATH=pwd:../aiter .venv/bin/python \
         scripts/test_attn_tshape_v2.py [--ctx 2048] [--tries 4]
"""
import argparse
import statistics as st
import sys

import torch

sys.path.insert(0, "/home/curved/vllm-gfx908")
sys.path.insert(0, "/home/curved/aiter")
from aiter.ops.triton.attention.unified_attention import unified_attention  # noqa

p = argparse.ArgumentParser()
p.add_argument("--ctx", type=int, default=2048)
p.add_argument("--heads", type=int, default=6)
p.add_argument("--kvh", type=int, default=1)
p.add_argument("--hd", type=int, default=256)
p.add_argument("--tries", type=int, default=4)
args = p.parse_args()

H, KVH, D = args.heads, args.kvh, args.hd
CTX, BLOCK, GS = args.ctx, 64, 128
NG, NB = D // GS, CTX // BLOCK + 8
T, N = 14, CTX
SCALE = D ** -0.5
dev = "cuda"

kc = torch.zeros(NB, BLOCK, KVH, D, dtype=torch.int8, device=dev)
vc = torch.zeros_like(kc)
ks = torch.zeros(NB, BLOCK, KVH, NG, GS, dtype=torch.float16, device=dev)
vs = torch.zeros_like(ks)


def build(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    ks.copy_((torch.rand(NB, BLOCK, KVH, NG, GS, generator=g,
                         device=dev, dtype=torch.float16).float() * 0.008
              + 0.004).to(torch.float16))
    vs.copy_((torch.rand(NB, BLOCK, KVH, NG, GS, generator=g,
                         device=dev, dtype=torch.float16).float() * 0.008
              + 0.004).to(torch.float16))
    kq = (torch.randn(KVH, N + T, D, generator=g, device=dev,
                      dtype=torch.float32) * 28).to(torch.int8)
    vq = (torch.randn(KVH, N + T, D, generator=g, device=dev,
                      dtype=torch.float32) * 28).to(torch.int8)
    q = torch.randn(H, T, D, generator=g, device=dev,
                    dtype=torch.bfloat16) * 0.35
    for h in range(H):
        for pos in torch.randint(0, N, (5,), generator=g, device=dev).tolist():
            qf = q[h, 0].float()
            kq[:, pos] = (qf / max(qf.abs().max().item(), 1e-6) * 90.0
                          ).to(torch.int8).clamp(-127, 127)
    for pos in range(N + T):
        kc[pos // BLOCK, pos % BLOCK] = kq[:, pos]
        vc[pos // BLOCK, pos % BLOCK] = vq[:, pos]
    return q


def go(q_hdt, seqv):
    Tq = q_hdt.shape[1]
    q = q_hdt.permute(1, 0, 2).contiguous()
    out = torch.empty_like(q)
    cu = torch.zeros(2, dtype=torch.int32, device=dev)
    cu[1] = Tq  # ONE request of Tq rows (verify batch), not Tq tiny requests
    bt = torch.arange(CTX // BLOCK + 2, device=dev,
                      dtype=torch.int32).reshape(1, -1).contiguous()
    unified_attention(
        q=q, k=kc, v=vc, out=out,
        cu_seqlens_q=cu, max_seqlen_q=Tq,
        seqused_k=torch.tensor([seqv], dtype=torch.int32, device=dev),
        max_seqlen_k=seqv,
        softmax_scale=SCALE, causal=True,
        window_size=(-1, -1), block_table=bt, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
        k_scale_cache=ks, v_scale_cache=vs,
    )
    torch.cuda.synchronize()
    return out.contiguous()  # [T, H, D]


def deq(pos, which):
    blk, off = pos // BLOCK, pos % BLOCK
    i8 = kc[blk, off] if which == "k" else vc[blk, off]
    sc = ks[blk, off] if which == "k" else vs[blk, off]
    return (i8.float().view(KVH, NG, GS) * sc.float()).view(KVH, D)


def exact(q_sel, rows):
    """q_sel: [H, len(rows), D] row-selected; returns [len(rows), H, D]."""
    outs = torch.zeros(len(rows), H, D, device=dev, dtype=torch.float32)
    for ri, t in enumerate(rows):
        L = N + t + 1
        K = torch.stack([deq(pp, "k") for pp in range(L)])
        V = torch.stack([deq(pp, "v") for pp in range(L)])
        for h in range(H):
            kvh = h % KVH
            logits = (q_sel[h, ri].float() @ K[:, kvh].T) * SCALE
            outs[ri, h] = torch.softmax(logits, dim=-1) @ V[:, kvh]
    return outs


stats = {"t1": [], "t14": [], "cross": []}
rows = [0, 1, 6, 13]
for trial in range(args.tries):
    q = build(1000 + trial)
    out14 = go(q, N + T)
    outs1 = torch.stack([go(q[:, t:t + 1, :], N + t + 1)[0] for t in range(T)])  # [T,H,D]

    q_sel = torch.stack([q[:, r, :] for r in rows], dim=1)   # [H, len(rows), D]
    ref = exact(q_sel, rows)
    r14 = torch.stack([out14[r] for r in rows]).float()
    r1 = torch.stack([outs1[r] for r in rows]).float()
    s = ref.abs().mean().item() + 1e-9
    e14 = ((r14 - ref).abs().mean() / s).item()
    e1 = ((r1 - ref).abs().mean() / s).item()
    cross = ((r14 - r1).abs().mean() / s).item()
    stats["t14"].append(e14); stats["t1"].append(e1); stats["cross"].append(cross)
    print(f"trial {trial}: T14-vs-exact {e14:.6f}  T1-vs-exact {e1:.6f}  "
          f"T14-vs-T1 {cross:.6f}  (scale {s:.4f})", flush=True)

print("\n==== SUMMARY", args.tries, "trials")
for k in ("t14", "t1", "cross"):
    v = stats[k]
    print(f"  {k}: {st.mean(v):.6f} +/- {(float(st.stdev(v)) if len(v) > 1 else 0.0):.6f}")
ratio = st.mean(stats["t14"]) / max(st.mean(stats["t1"]), 1e-12)
print(f"  T14/T1 error ratio: {ratio:.3f}  "
      f"({'T=14 SPECIFIC' if ratio > 3 else 'shape-noise only'})")
