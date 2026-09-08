#!/usr/bin/env python3
"""In-vitro A/B: the fork's fused GDN MTP decode kernel (spec path) vs the
FLA packed decode (non-spec decode path) on identical inputs.

The two implementations must agree (same math, same tokens, same state) for
spec-verify outputs to be consistent with decode outputs. Any systematic
difference is the constant per-position error the accumulation curve shows.

Usage: HIP_VISIBLE_DEVICES=0 PYTHONPATH=... python scripts/test_gdn_mtp_equiv.py
"""
import sys

import torch

sys.path.insert(0, "/home/curved/vllm-gfx908")
from vllm import _custom_ops as ops  # noqa: E402
from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    fused_recurrent_gated_delta_rule_packed_decode,
)

dev = "cuda"
torch.manual_seed(0)
H, HV, K, V = 16, 48, 128, 128
TP = 1  # per-rank shape; equivalence is per-rank
NS = 13

g = torch.Generator(device=dev).manual_seed(3)
T = 2  # NS=1-style verify: bonus + 1 draft
mixed = torch.randn(T, 2 * H * K + HV * V, generator=g, device=dev,
                    dtype=torch.bfloat16)
a = torch.randn(T, HV, generator=g, device=dev, dtype=torch.bfloat16)
b = torch.randn(T, HV, generator=g, device=dev, dtype=torch.bfloat16)
A_log = torch.randn(HV, generator=g, device=dev, dtype=torch.float32) - 1.0
dt_bias = torch.randn(HV, generator=g, device=dev, dtype=torch.float32)
state = torch.zeros(32, HV, V, K, device=dev, dtype=torch.float32)
state[1] = torch.randn(HV, V, K, generator=g, device=dev, dtype=torch.float32) * 0.05
gate = torch.randn(T, HV, V, generator=g, device=dev, dtype=torch.bfloat16)
norm_w = torch.randn(HV * V, generator=g, device=dev, dtype=torch.bfloat16)

scale = K ** -0.5
# spec path (fused CUDA MTP kernel)
si = torch.tensor([[1, 2]], dtype=torch.int32, device=dev)
cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
na = torch.tensor([1], dtype=torch.int32, device=dev)
state_spec = state.clone()
out_spec = ops.fused_gdn_decode_post_conv_mtp(
    mixed_qkv=mixed, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
    state_indices=si, cu_seqlens=cu, num_accepted_tokens=na,
    state=state_spec, output_gate=gate, norm_weight=norm_w,
    scale=scale, norm_eps=1e-5,
)

# non-spec decode path (FLA Triton), token by token from the same state
state_dec = state.clone()
outs = []
for t in range(T):
    o = torch.empty(1, 1, HV, V, device=dev, dtype=torch.bfloat16)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed[t:t + 1].contiguous(),
        a=a[t:t + 1].contiguous(), b=b[t:t + 1].contiguous(),
        A_log=A_log, dt_bias=dt_bias, scale=scale,
        initial_state=state_dec,
        out=o,
        ssm_state_indices=torch.tensor([1], dtype=torch.int32, device=dev),
        use_qk_l2norm_in_kernel=True,
    )
    outs.append(o)
    # apply the same output gate + RMSNorm as the fused kernel does
outs = torch.cat(outs).squeeze(1)  # [T, HV, V]

# NOTE: the fused kernel applies gate*RMSNorm(out) internally; the FLA call
# returns the pre-norm core output. For comparison we replicate the epilogue.
def rms_norm_gated(core, gate, w, eps=1e-5):
    x = core.float() * gate.float()
    xf = x.reshape(T, -1)
    rms = xf.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (xf * rms * w.float()).to(torch.bfloat16).reshape(T, HV, V)

outs_post = rms_norm_gated(outs, gate, norm_w)

d = (out_spec.float() - outs_post.float()).abs()
rel = d.max() / outs_post.float().abs().max().clamp(min=1e-9)
print(f"out max|Δ|={d.max().item():.6f}  rel={rel.item():.6f}  "
      f"mean|Δ|={d.mean().item():.6f}")
sd = (state_spec[2] - state_dec[1]).abs().max().item()
print(f"state after (spec slot2 vs decode slot1): max|Δ|={sd:.6f}")
tol = 5e-3
print("EQUIVALENT" if d.max().item() < tol * outs_post.float().abs().max().item()
      else "DIVERGENT — spec fused kernel differs from decode path")
