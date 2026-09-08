#!/usr/bin/env python3
"""In-vitro A/B of the causal_conv1d spec-decode branch.

Equivalence property: processing tokens [t0, t1] through ONE spec-mode call
(num_accepted=1 => conv read offset 0, like a fresh resume) must equal TWO
sequential non-spec decode calls, token-for-token, starting from the same
conv_state. A mismatch convicts the spec branch's shift arithmetic
(the spec-only `idx_tokens + 1` source shift) as a real output error.

Usage: HIP_VISIBLE_DEVICES=0 python scripts/test_conv_spec_equiv.py
"""
import sys

import torch

sys.path.insert(0, "/home/curved/vllm-gfx908")
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (  # noqa: E402
    causal_conv1d_update,
)

torch.manual_seed(0)
dev = "cuda"
DIM, WIDTH, SEQS = 256, 4, 1
STATE_LEN = WIDTH - 1 + 13  # kw-1 + NS = 16, matches production window


def make(ref=False):
    g = torch.Generator(device=dev).manual_seed(7)
    w = torch.randn(DIM, WIDTH, generator=g, device=dev, dtype=torch.float32)
    b = torch.randn(DIM, generator=g, device=dev, dtype=torch.float32)
    state = torch.zeros(4, DIM, STATE_LEN, device=dev, dtype=torch.float32)
    state[1] = torch.randn(DIM, STATE_LEN, generator=g, device=dev)
    x2 = torch.randn(2, DIM, generator=g, device=dev, dtype=torch.float32)
    return w, b, state, x2


def run_spec():
    w, b, state, x = make()
    idx = torch.tensor([1], dtype=torch.int32, device=dev)
    na = torch.tensor([1], dtype=torch.int32, device=dev)
    qsl = torch.tensor([0, 2], dtype=torch.int32, device=dev)
    out = torch.empty_like(x)
    causal_conv1d_update(
        x, state, w, b, activation="silu",
        conv_state_indices=idx, num_accepted_tokens=na,
        query_start_loc=qsl, max_query_len=2,
        out=out,
    )
    torch.cuda.synchronize()
    return out, state[1].clone()


def run_decode():
    w, b, state, x = make()
    idx = torch.tensor([1], dtype=torch.int32, device=dev)
    outs = []
    for t in range(2):
        xt = x[t:t + 1].contiguous()
        causal_conv1d_update(
            xt, state, w, b, activation="silu",
            conv_state_indices=idx, validate_data=True,
        )
        torch.cuda.synchronize()
        outs.append(xt.clone())
    return torch.cat(outs), state[1].clone()


o_spec, s_spec = run_spec()
o_dec, s_dec = run_decode()
print("spec out :", [f"{v:.4f}" for v in o_spec[0, :4].tolist()])
print("dec  out :", [f"{v:.4f}" for v in o_dec[0, :4].tolist()])
d0 = (o_spec[0] - o_dec[0]).abs().max().item()
d1 = (o_spec[1] - o_dec[1]).abs().max().item()
sd = (s_spec - s_dec).abs().max().item()
print(f"max|Δout0|={d0:.6f}  max|Δout1|={d1:.6f}  max|Δstate|={sd:.6f}")
tol = 2e-3
print("EQUIVALENT" if max(d0, d1, sd) < tol else "MISMATCH — spec branch "
      "produces different outputs than sequential decode")
