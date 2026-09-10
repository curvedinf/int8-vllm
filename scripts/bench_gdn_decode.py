#!/usr/bin/env python3
"""Standalone timing of the aiter GDN decode kernels at engine-realistic shapes.

The kernels' inputs have NO context-length parameter (recurrent state is
indexed by slot), so timing across synthetic 'ctx' values tests whether any
hidden input scaling exists (e.g. via state gather). Confirms/denies GDN as
the O(ctx) per-step term.
"""
import sys
import torch

sys.path.insert(0, "/home/curved/aiter")
from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
    fused_rearrange_sigmoid_gated_delta_rule,
)

torch.manual_seed(0)
dev = "cuda"

# Per-rank GDN shapes for Qwen3.8-27B TP4 (from tp_ssm sharding):
# 16 k-heads/4 = 4 k-heads, 48 v-heads/4 = 12 v-heads, dims 128/128
HK, HV, K, V = 4, 12, 128, 128
key_dim, value_dim = HK * K, HV * V
NSLOTS = 1024  # state slots (plenty)


def bench(T=14, iters=200):
    qkv = torch.randn(T, key_dim * 2 + value_dim, device=dev, dtype=torch.bfloat16)
    a = torch.randn(T, HV, device=dev, dtype=torch.float32)
    b = torch.randn(T, HV, device=dev, dtype=torch.float32)
    A_log = torch.randn(HV, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=dev, dtype=torch.float32)
    state = torch.zeros(NSLOTS, key_dim, value_dim, device=dev, dtype=torch.float32)
    slot = torch.tensor([7], device=dev, dtype=torch.int64)
    cu = torch.tensor([0, T], device=dev, dtype=torch.long)
    out = torch.empty(T * HV * V, device=dev, dtype=torch.bfloat16)

    def run():
        fused_rearrange_sigmoid_gated_delta_rule(
            A_log=A_log, a=a, b=b, dt_bias=dt_bias, qkv=qkv,
            key_dim=key_dim, value_dim=value_dim, head_k_dim=K, head_v_dim=V,
            initial_state=state, inplace_final_state=True,
            cu_seqlens=cu, ssm_state_indices=slot,
            use_qk_l2norm_in_kernel=True, core_attn_out=out,
        )

    for _ in range(20):
        run()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        run()
    t1.record()
    torch.cuda.synchronize()
    ms = t0.elapsed_time(t1) / iters
    return ms


for T in (1, 14):
    ms = bench(T=T)
    print(f"GDN fused_rearrange_sigmoid_gdr T={T:2d}: {ms:8.3f} ms/call", flush=True)

# 48 layers x T=14 at the measured ms -> per-step GDN total
ms14 = bench(T=14)
print(f"per-step GDN total (48 layers x T=14): {48*ms14:.1f} ms")
