#!/usr/bin/env python3
"""Ground-truth probe: spec-decode GDN verify chunk vs exact recurrence.

The fused spec path (fused_sigmoid_gating_delta_rule_update) processes the
verify rows [anchor, d_1..d_K] as one in-kernel sequential chunk starting
from checkpoint slot si[num_accepted-1]. Row 0 was audited clean against a
one-timestep reference (VLLM_GDN_ROWAUDIT); rows 1..6 (the in-chunk
recurrence) have never been compared against an exact reference.

For each acceptance a in {1..7}:
  - seed checkpoint state at slot si[a-1] (random fp32, like prod mamba state)
  - run the fused kernel on the 7-row query chunk (l2norm on, as prod)
  - run an exact torch recurrence from the SAME checkpoint + inputs
  - compare every row's output and every row's stored per-row state slot

Any row >= 1 divergence = in-chunk recurrence fault (per-step, spec-only,
argmax-robust — the residual drift profile).

Usage: HIP_VISIBLE_DEVICES=0 python scripts/test_gdn_chunk_reference.py
"""
import sys

import torch

sys.path.insert(0, "/home/curved/vllm-gfx908")
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (  # noqa
    fused_sigmoid_gating_delta_rule_update,
)

dev = "cuda"
g = torch.Generator(device=dev).manual_seed(11)

# Prod-like small dims (Qwen3.8 GDN: K=V=128, GQA k:v handled via H==HV here)
H = HV = 8
K = V = 128
T = 7                    # verify rows: anchor + 6 drafts
SLOTS = 16               # ssm state cache lines (line 0 = null)

DT = torch.bfloat16      # q/k/v/a/b dtype in prod
q = (torch.randn(1, T, H, K, generator=g, device=dev) * 0.5).to(DT)
k = (torch.randn(1, T, H, K, generator=g, device=dev) * 0.5).to(DT)
v = (torch.randn(1, T, HV, V, generator=g, device=dev) * 0.5).to(DT)
a = (torch.randn(1, T, HV, generator=g, device=dev) * 0.5).to(DT)
b = (torch.randn(1, T, HV, generator=g, device=dev) * 0.5).to(DT)

A_log = torch.randn(HV, generator=g, device=dev, dtype=torch.float32) * 0.3
dt_bias = torch.randn(HV, generator=g, device=dev, dtype=torch.float32) * 0.3
beta, threshold = 1.0, 20.0
scale = K**-0.5

# Per-row state slots si[0..6] (distinct, nonzero), 2D like the engine's
si = torch.arange(1, T + 1, dtype=torch.int32, device=dev).unsqueeze(0)

# Distinct random checkpoints per slot so a wrong slot is detectable
state0 = torch.randn(SLOTS, HV, V, K, generator=g, device=dev,
                     dtype=torch.float32) * 0.3
cu = torch.tensor([0, T], dtype=torch.int32, device=dev)


def reference(ckpt, a_rows, b_rows, qs, ks, vs):
    """Exact fp32 recurrence (kernel math, per token). Returns (o, states)."""
    o = torch.zeros(T, HV, V, dtype=torch.float32, device=dev)
    states = []
    h = ckpt.clone()                                        # [HV, V, K]
    for t in range(T):
        gating = -torch.exp(A_log) * torch.nn.functional.softplus(
            beta * (a_rows[t].float() + dt_bias))
        bet = torch.sigmoid(b_rows[t].float())              # [HV]
        qt = torch.nn.functional.normalize(
            qs[t].float(), dim=-1, eps=(1e-6) ** 0.5) * scale
        kt = torch.nn.functional.normalize(
            ks[t].float(), dim=-1, eps=(1e-6) ** 0.5)
        h = h * gating[:, None, None].exp()                 # decay
        vt = vs[t].float()                                  # [HV, V]
        vhat = torch.einsum("hvk,hk->hv", h, kt)
        vnew = bet[:, None] * (vt - vhat)
        h = h + vnew[:, :, None] * kt[:, None, :]           # rank-1 update
        o[t] = torch.einsum("hvk,hk->hv", h, qt)
        states.append(h.clone())
    return o, states


print(f"H={H} K={K} V={V} T={T}  (rows: anchor + 6 drafts)")
print(f"{'a':>2} {'out0':>10} {'out_max(1..6)':>14} {'state_max':>10}")

any_fail = False
for acc in range(1, T + 1):
    ssm = state0.clone()
    na = torch.tensor([acc], dtype=torch.int32, device=dev)
    out, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=a, b=b, dt_bias=dt_bias,
        q=q.clone(), k=k.clone(), v=v.clone(),
        beta=beta, threshold=threshold, scale=scale,
        initial_state=ssm, inplace_final_state=True,
        cu_seqlens=cu, ssm_state_indices=si,
        num_accepted_tokens=na, use_qk_l2norm_in_kernel=True,
    )
    ref_o, ref_s = reference(state0[si[0, acc - 1]], a[0], b[0], q[0], k[0], v[0])
    do = (out.float().reshape(T, HV, V) - ref_o).abs()
    d0 = do[0].max().item()
    drest = do[1:].max().item() if T > 1 else 0.0
    ds = max(
        (ssm[int(si[0, t])] - ref_s[t]).abs().max().item() for t in range(T)
    )
    fail = max(d0, drest, ds) > 2e-2
    any_fail |= fail
    print(f"{acc:>2} {d0:10.6f} {drest:14.6f} {ds:10.6f}"
          f"{'   <-- MISMATCH' if fail else ''}", flush=True)

print("\nVERDICT:",
      "GDN spec chunk MISMATCH vs exact recurrence"
      if any_fail else
      "spec chunk outputs + per-row states == exact recurrence (all rows, "
      "all a) — in-chunk verify recurrence EXONERATED")
