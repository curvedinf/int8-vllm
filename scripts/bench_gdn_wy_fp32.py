#!/usr/bin/env python3
"""FP32-upcast chunked-WY GDN prefill vs the exact serial kernel (GOALOPT).

The exact-recurrence prefill is serial over all 2048 tokens per chunk
(7.2s of a ~21s solo 32k TTFT). The chunked WY pipeline is parallel
across a chunk's tokens; its historical 4e-3 error came from bf16
intermediates. This bench runs the same pipeline with fp32-upcast
q/k/v (exact-class FMA dots on gfx908) and compares against the exact
serial kernel at production geometry.
"""
import os
import sys

os.environ["VLLM_GDN_PREFILL_FP32"] = "1"

import torch

sys.path.insert(0, "/home/curved/vllm-gfx908")

from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (  # noqa: E401
    fused_sigmoid_gating_delta_rule_update,
)

dev = "cuda"
torch.manual_seed(0)

H, HV, K, V = 4, 12, 128, 128
POOL = 8
T = int(os.environ.get("GDN_T", "2048"))

A_log = torch.randn(HV, device=dev, dtype=torch.float32) * 0.1 - 1.0
a = torch.randn(T, HV, device=dev, dtype=torch.bfloat16) * 0.3
b = torch.randn(T, HV, device=dev, dtype=torch.bfloat16)
dt_bias = torch.randn(HV, device=dev, dtype=torch.float32) * 0.1
q = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, T, HV, V, device=dev, dtype=torch.bfloat16)
cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
SLOT = 3
si = torch.full((1, T), SLOT, device=dev, dtype=torch.int32)

# Exact serial reference (production path), bf16 in / fp32 state.
pool = torch.randn(POOL, HV, V, K, device=dev, dtype=torch.float32) * 0.05
pool0 = pool.clone()
o_exact, _ = fused_sigmoid_gating_delta_rule_update(
    A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v,
    initial_state=pool, inplace_final_state=True,
    cu_seqlens=cu, ssm_state_indices=si,
    num_accepted_tokens=None, use_qk_l2norm_in_kernel=True,
)
torch.cuda.synchronize()
state_exact = pool[SLOT].clone()

# Chunked WY with fp32-upcast inputs.
from vllm.third_party.flash_linear_attention.ops.chunk import (  # noqa: E401
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)

pool2 = torch.randn(POOL, HV, V, K, device=dev, dtype=torch.float32) * 0.05
pool2.copy_(pool0)
init2 = pool2[SLOT:SLOT + 1].clone()  # [1, HV, V, K], production layout
g = None  # computed inside via a/b? chunk API takes g - check signature
# chunk_gated_delta_rule(q,k,v,g,beta,scale,initial_state,...) - g here is
# the log-decay; derive exactly like the kernel: g = -exp(A_log)*softplus(a+dtb)
softplus = lambda x: torch.where(x <= 20, torch.log1p(torch.exp(x)), x)
g_vec = -torch.exp(A_log.float()) * softplus(a.float() + dt_bias.float())
# layout for fla: (1, T, HV)
g_in = g_vec.view(1, T, HV)
beta_in = torch.sigmoid(b.float()).view(1, T, HV)
o_wy, state_wy = fla_chunk_gated_delta_rule(
    q=q.float(), k=k.float(), v=v.float(),
    g=g_in, beta=beta_in,
    scale=K ** -0.5,
    initial_state=init2,
    output_final_state=True,
    cu_seqlens=cu,
    use_qk_l2norm_in_kernel=True,
)
torch.cuda.synchronize()

d = (o_wy.float() - o_exact.float()).abs()
print(f"T={T}: fp32-WY vs exact-serial: max_abs={d.max().item():.3e} "
      f"mean_abs={d.mean().item():.3e} "
      f"p999={d.flatten().kthvalue(int(d.numel() * 0.999)).values:..3e}"
      if False else
      f"T={T}: fp32-WY vs exact-serial: max_abs={d.max().item():.3e} "
      f"mean_abs={d.mean().item():.3e}")
ds = (state_wy.float() - state_exact).abs()
print(f"final state: max_abs={ds.max().item():.3e} "
      f"scale={state_exact.abs().mean().item():.4f}")


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
    return s.elapsed_time(e) / iters


def run_serial():
    fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v,
        initial_state=pool, inplace_final_state=True,
        cu_seqlens=cu, ssm_state_indices=si,
        num_accepted_tokens=None, use_qk_l2norm_in_kernel=True,
    )


def run_wy():
    fla_chunk_gated_delta_rule(
        q=q.float(), k=k.float(), v=v.float(),
        g=g_in, beta=beta_in, scale=K ** -0.5,
        initial_state=init2, output_final_state=True,
        cu_seqlens=cu, use_qk_l2norm_in_kernel=True,
    )


t_ser = bench(run_serial)
t_wy = bench(run_wy)
print(f"per-call ms: exact-serial={t_ser:.2f} fp32-WY={t_wy:.2f} "
      f"speedup={t_ser / t_wy:.2f}x")
