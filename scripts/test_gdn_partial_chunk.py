#!/usr/bin/env python3
"""Partial-chunk vs full-chunk equivalence for the FLA GDN prefill chain.

The engine prefill splits prompts into 2048-token scheduler chunks; the last
one is usually partial. Causal-invariance probes (G1_PARTIAL_CHUNK_ROOT_CAUSE)
showed row values depend on the partial chunk's token count. This test
reproduces that at the op level: same inputs prefix, cu_seqlens over T_full
vs T_partial, compare outputs on the shared prefix rows.

Usage: HIP_VISIBLE_DEVICES=0 .venv/bin/python scripts/test_gdn_partial_chunk.py
"""
import sys

import torch

sys.path.insert(0, "/home/curved/vllm-gfx908")
from vllm.third_party.flash_linear_attention.ops.chunk import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)
from vllm.third_party.flash_linear_attention.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)

dev = "cuda"
g = torch.Generator(device=dev).manual_seed(11)

H, HV, K, V = 4, 12, 128, 128
T_FULL = 2048
DT = torch.bfloat16

q = (torch.randn(1, T_FULL, H, K, generator=g, device=dev) * 0.5).to(DT)
k = (torch.randn(1, T_FULL, H, K, generator=g, device=dev) * 0.5).to(DT)
v = (torch.randn(1, T_FULL, HV, V, generator=g, device=dev) * 0.5).to(DT)
# the engine passes UNnormalized q/k with use_qk_l2norm_in_kernel=False only
# when the FLA chain normalizes internally; when calling the op directly the
# caller must pre-normalize (l2norm_fwd semantics: fp32 norm, same dtype out).
def l2n(x):
    xf = x.float()
    xf = xf / (xf * xf).sum(-1, keepdim=True).add(1e-6).sqrt()
    return xf.to(x.dtype)
q, k = l2n(q), l2n(k)
# log-decay gates (negative, fp32), beta in (0,1)
gg = (-torch.rand(1, T_FULL, HV, generator=g, device=dev) * 0.10).float()
beta = torch.rand(1, T_FULL, HV, generator=g, device=dev).float() * 0.8 + 0.1
h0 = (torch.randn(1, HV, V, K, generator=g, device=dev) * 0.3).float()


def run(T):
    cu = torch.tensor([0, T], dtype=torch.int32)
    ci = prepare_chunk_indices(cu, 64).to(dev)
    co = prepare_chunk_offsets(cu, 64).to(dev)
    cu = cu.to(dev)
    o, hs = fla_chunk_gated_delta_rule(
        q=q[:, :T].contiguous(), k=k[:, :T].contiguous(),
        v=v[:, :T].contiguous(), g=gg[:, :T].contiguous(),
        beta=beta[:, :T].contiguous(), initial_state=h0.clone(),
        output_final_state=True, cu_seqlens=cu,
        chunk_indices=ci, chunk_offsets=co,
        use_qk_l2norm_in_kernel=False,
    )
    return o, hs


o_full, hs_full = run(T_FULL)
print(f"full: o={tuple(o_full.shape)} hs={tuple(hs_full.shape)}")

for T_part in (1672, 1673, 1984, 2047, 512):
    o_p, hs_p = run(T_part)
    a, b = o_full[0, :T_part].float(), o_p[0].float()
    d = (a - b).abs()
    denom = a.abs().mean().clamp(min=1e-6)
    # per-row max abs diff profile: head / mid / tail of the shared prefix
    n = T_part
    prof = []
    for lo, hi, tag in [(0, 64, "head"), (n // 2 - 32, n // 2 + 32, "mid"),
                        (n - 64, n, "tail")]:
        seg = d[lo:hi]
        prof.append(f"{tag} max={seg.max():.4e}")
    print(f"T={T_part:5d} ({T_part % 64:2d} mod64): overall max={d.max():.4e} "
          f"mean={d.mean():.4e} rel={d.max() / denom:.3e} | " +
          " | ".join(prof))

# --- isolation pass: which side has NaN, first NaN row ---
print("\nisolation:")
of = o_full[0].float()
print(f"full  nan={torch.isnan(of).sum().item()} inf={torch.isinf(of).sum().item()} "
      f"absmax={of[~torch.isnan(of)].abs().max().item():.3e}")
for T_part in (1984, 1672, 512):
    o_p, hs_p = run(T_part)
    op = o_p[0].float()
    nan_rows = torch.isnan(op).any(dim=-1).any(dim=-1).nonzero().flatten()
    first = int(nan_rows[0]) if len(nan_rows) else -1
    print(f"T={T_part:5d} nan={torch.isnan(op).sum().item()} first_nan_row={first} "
          f"of {T_part} | state nan={torch.isnan(hs_p.float()).sum().item()} "
          f"absmax_nonnan={op[~torch.isnan(op)].abs().max().item():.3e}")
