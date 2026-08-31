#!/usr/bin/env python3
"""GDN SSM spec-rewind differential for fused_sigmoid_gating_delta_rule_update.

Mirrors the production calling convention (qwen_gdn_linear_attn.py:1444-1462):
varlen spec call with per-token ssm_state_indices rows [r_0..r_{T-1}],
inplace per-token state writes, initial state read from column A-1
(num_accepted_tokens anchor). Reference is an independent fp32 fold.

Two configurations:
  distinct  - 14 separate state rows per request (the intended layout);
              every written row must equal the fold state after its token.
  aliased   - all columns point at ONE row (the failure hypothesis):
              demonstrates last-write-wins state-ahead if it occurs.
"""
import sys

import torch

sys.path.insert(0, ".")
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)

torch.manual_seed(7)
DEV = "cuda"
DTYPE = torch.float32

H = HV = 4
K = 128
V = 256
T = 14           # verify rows = NS + 1
NUM_STATE_ROWS = 64
TOL = 2e-4


def ref_fold(q, k, v, a, b, A_log, dt_bias, h_prev, scale):
    """Independent fp32 fold. q/k: (T,H,K); v: (T,HV,V); a/b: (T,HV);
    h_prev: (HV,K,V). Returns out (T,HV,V), states (T,HV,K,V)."""
    Tn = q.shape[0]
    out = torch.empty(Tn, HV, V, device=q.device, dtype=torch.float32)
    states = torch.empty(Tn, HV, V, K, device=q.device, dtype=torch.float32)
    h = h_prev.clone()   # (HV, V, K) — kernel layout: row=v, col=k
    for t in range(Tn):
        for hv in range(HV):
            hq = q[t, hv].double()
            hk = k[t, hv].double()
            hq = hq * torch.rsqrt(hq @ hq + 1e-6)
            hk = hk * torch.rsqrt(hk @ hk + 1e-6)
            hq = hq * scale
            x = a[t, hv].double() + dt_bias[hv].double()
            sp = torch.where(
                x <= 20.0, torch.log1p(torch.exp(x)), x)
            g = -torch.exp(A_log[hv].double()) * sp
            beta = torch.sigmoid(b[t, hv].double())
            hh = h[hv].double()          # (V,K)
            hh = hh * torch.exp(g)
            vv = v[t, hv].double() - hh @ hk   # (V,)
            vv = vv * beta
            hh = hh + vv[:, None] * hk[None, :]   # (V,K)
            h[hv] = hh.float()
            out[t, hv] = (hh @ hq).float()
            states[t, hv] = hh.float()
    return out, states


def run_case(accept, aliased, tag):
    A = accept + 1   # kernel num_accepted includes the bonus token
    q = torch.randn(1, T, H, K, device=DEV, dtype=DTYPE)
    k = torch.randn(1, T, H, K, device=DEV, dtype=DTYPE)
    v = torch.randn(1, T, HV, V, device=DEV, dtype=DTYPE)
    a = torch.randn(T, HV, device=DEV, dtype=DTYPE) * 0.3
    b = torch.randn(T, HV, device=DEV, dtype=DTYPE)
    A_log = torch.randn(HV, device=DEV, dtype=DTYPE) - 1.0
    dt_bias = torch.randn(HV, device=DEV, dtype=DTYPE) * 0.1
    scale = K**-0.5

    ssm_state = torch.randn(NUM_STATE_ROWS, HV, V, K, device=DEV,
                            dtype=DTYPE) * 0.05
    base = 10
    if aliased:
        indices = torch.full((1, T), base, dtype=torch.int32, device=DEV)
    else:
        indices = torch.arange(base, base + T, dtype=torch.int32,
                               device=DEV).unsqueeze(0)
    # Seed the anchor row the kernel must resume from (state after the
    # previous round's token A-1).
    h_prev = torch.randn(HV, V, K, device=DEV, dtype=DTYPE) * 0.05
    ssm_state[base + A - 1] = h_prev.clone()
    pre_rows = ssm_state[indices[0]].clone()

    na = torch.tensor([A], dtype=torch.int32, device=DEV)
    cu = torch.tensor([0, T], dtype=torch.int32, device=DEV)

    out, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=a, b=b, dt_bias=dt_bias,
        q=q, k=k, v=v,
        initial_state=ssm_state,
        inplace_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        num_accepted_tokens=na,
        use_qk_l2norm_in_kernel=True,
    )
    assert out.shape == (1, T, HV, V), out.shape

    oref, sref = ref_fold(q[0], k[0], v[0], a, b, A_log, dt_bias,
                          h_prev, scale)
    oe = (out[0].float() - oref).abs().max().item()
    post = ssm_state[indices[0]]
    # Every column must hold the fold state after its token.
    se = (post - sref).abs().max().item()
    # Rewind correctness: the anchor row for the NEXT round (column A-1)
    # must equal the fold through token A-1 (and NOT through T-1).
    anchor_next = ssm_state[base + A - 1]
    ahead = (anchor_next - sref[T - 1]).abs().max().item() \
        if A - 1 != T - 1 else 0.0
    good = oe < TOL and se < TOL
    print(f"[{tag}] accept={accept:2d}: out_err={oe:.6f} state_err={se:.6f} "
          f"anchor_ahead_by_(T-A)={ahead:.6f}"
          f"{'' if good else '  <-- DIVERGENCE'}")
    return good


def main():
    ok = True
    for accept in (1, 5, 9, 13):
        ok &= run_case(accept, aliased=False, tag="distinct")
    # Failure-hypothesis demo: aliased columns.
    run_case(5, aliased=True, tag="aliased ")
    print("OVERALL (distinct):", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
