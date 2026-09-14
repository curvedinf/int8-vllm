#!/usr/bin/env python3
"""Chunked GDN prefill precision gate: bf16 vs fp32-mode vs fp64 recurrence.

VLLM_GDN_PREFILL_FP32=1 runs the FLA chunked chain in fp32; this test
measures both modes against an exact fp64 sequential recurrence at
production-ish shapes. Gate for the fp32 mode: max rel <= 1e-5.

Usage: HIP_VISIBLE_DEVICES=0 PYTHONPATH=pwd .venv/bin/python \
  scripts/test_gdn_prefill_fp32.py
"""
import os
import sys

os.environ["VLLM_GDN_PREFILL_FP32"] = "1"  # import-time stage clamp must see it

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
T = 2048
DT = torch.bfloat16

q = (torch.randn(1, T, H, K, generator=g, device=dev) * 0.5).to(DT)
k = (torch.randn(1, T, H, K, generator=g, device=dev) * 0.5).to(DT)
v = (torch.randn(1, T, HV, V, generator=g, device=dev) * 0.5).to(DT)


def l2n(x):
    xf = x.float()
    xf = xf / (xf * xf).sum(-1, keepdim=True).add(1e-6).sqrt()
    return xf.to(x.dtype)


q, k = l2n(q), l2n(k)
gg = (-torch.rand(1, T, HV, generator=g, device=dev) * 0.10).float()
beta = torch.rand(1, T, HV, generator=g, device=dev).float() * 0.8 + 0.1
h0 = (torch.randn(1, HV, V, K, generator=g, device=dev) * 0.3).float()


def run():
    cu = torch.tensor([0, T], dtype=torch.int32)
    ci = prepare_chunk_indices(cu, 64).to(dev)
    co = prepare_chunk_offsets(cu, 64).to(dev)
    cu = cu.to(dev)
    o, hs = fla_chunk_gated_delta_rule(
        q=q.contiguous(), k=k.contiguous(), v=v.contiguous(),
        g=gg.contiguous(), beta=beta.contiguous(),
        initial_state=h0.clone(), output_final_state=True, cu_seqlens=cu,
        chunk_indices=ci, chunk_offsets=co, use_qk_l2norm_in_kernel=False,
    )
    return o, hs


# exact fp64 sequential recurrence
def reference():
    qf = torch.nn.functional.normalize(q.double(), dim=-1)
    kf = torch.nn.functional.normalize(k.double(), dim=-1)
    vf = v.double()
    scale = K**-0.5
    o = torch.empty(1, T, HV, V, dtype=torch.float64, device=dev)
    h = h0.double().clone()
    sel = torch.arange(HV, device=dev) // (HV // H)
    for t in range(T):
        qt = qf[0, t][sel] * scale          # [HV, K]
        kt = kf[0, t][sel]                  # [HV, K]
        vt = vf[0, t]                       # [HV, V]
        gate = torch.exp(gg[0, t].double())  # [HV]
        b = beta[0, t].double()
        h = h * gate[:, None, None]
        vt = vt - (h * kt[:, None, :]).sum(-1) * b[:, None]
        h = h + vt[:, :, None] * kt[:, None, :]
        o[0, t] = (h * qt[:, None, :]).sum(-1)
    return o, h


ref_o, ref_h = reference()
ref_scale = ref_o.abs().mean().item()

os.environ.pop("VLLM_GDN_PREFILL_FP32", None)
o_bf16, hs_bf16 = run()
os.environ["VLLM_GDN_PREFILL_FP32"] = "1"
o_fp32, hs_fp32 = run()
os.environ.pop("VLLM_GDN_PREFILL_FP32", None)

for name, o, hs in (("bf16-mode", o_bf16, hs_bf16), ("fp32-mode", o_fp32, hs_fp32)):
    eo = (o.double() - ref_o).abs()
    eh = (hs.double() - ref_h).abs()
    print(f"{name}: out rel={(eo.max()/ref_scale).item():.3e} "
          f"mean={(eo.mean()/ref_scale).item():.3e} | "
          f"state rel={(eh.max()/ref_h.abs().mean()).item():.3e}")
print(f"gate: fp32-mode rel <= 1e-5 required (scale {ref_scale:.3f})")
