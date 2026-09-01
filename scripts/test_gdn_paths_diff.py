#!/usr/bin/env python3
"""GDN kernel-path differential: sequential packed-decode vs spec-window fold.

Question: do the no-spec decode kernel
(fused_recurrent_gated_delta_rule_packed_decode, 1 token/step) and the spec
verify kernel (fused_sigmoid_gating_delta_rule_update, 14-token windows with
full acceptance = chunked fold) produce the SAME recurrent state and per-token
outputs over long sequences?

Path A (sequential): token-by-token packed_decode, state carried in-place in
                     one state row (mirrors _forward_core_decode_non_spec,
                     qwen_gdn_linear_attn.py:1709).
Path B (spec window): 14-token windows, num_accepted = window size, anchor
                     column convention from test_gdn_rewind.py (initial state
                     read from column A-1; per-token states written to their
                     columns; final state lands back in the anchor row).
Reference: ref_fold copied VERBATIM from scripts/test_gdn_rewind.py
           (fp64 internal math, state rounded to fp32 per token).

Gate: at T=14 all three must agree to ~1e-4 before the long run is trusted.
"""
import sys

import torch

sys.path.insert(0, ".")
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)

torch.manual_seed(7)
DEV = "cuda"
DTYPE = torch.float32

H = HV = 4
K = 128
V = 256
NUM_STATE_ROWS = 64
WINDOW = 14          # NS + 1 verify rows, full acceptance
T_LONG = 4000
SCALE = K**-0.5

CHECKPOINTS = [14, 500, 1000, 2000, 4000]   # token counts (1-based)
TOL_SHORT = 1e-4

# State rows (0 is NULL_BLOCK_ID — both kernels skip state_idx <= 0).
ROW_SEQ = 1                      # path A single carried row
ROW_ANCHOR = 46                  # path B anchor column (A-1)
ROW_SCRATCH = 33                 # path B columns 0..12 -> rows 33..45


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


def ref_fold_checkpoints(q, k, v, a, b, A_log, dt_bias, scale, checkpoints):
    """ref_fold math vectorized over heads (identical op sequence, fp64
    internals, state rounded to fp32 per token), materializing states only at
    1-based token counts. Validated against verbatim ref_fold at T=14."""
    Tn = q.shape[0]
    out = torch.empty(Tn, HV, V, device=q.device, dtype=torch.float32)
    ckpt = {}
    want = set(checkpoints)
    A_log_d = A_log.double()
    dt_bias_d = dt_bias.double()
    h = torch.zeros(HV, V, K, device=q.device, dtype=torch.float32)
    for t in range(Tn):
        hq = q[t].double()                     # (HV, K)
        hk = k[t].double()
        hq = hq * torch.rsqrt((hq * hq).sum(-1, keepdim=True) + 1e-6)
        hk = hk * torch.rsqrt((hk * hk).sum(-1, keepdim=True) + 1e-6)
        hq = hq * scale
        x = a[t].double() + dt_bias_d          # (HV,)
        sp = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
        g = -torch.exp(A_log_d) * sp
        beta = torch.sigmoid(b[t].double())    # (HV,)
        hh = h.double()                        # (HV, V, K)
        hh = hh * torch.exp(g)[:, None, None]
        vv = v[t].double() - torch.bmm(hh, hk.unsqueeze(-1)).squeeze(-1)
        vv = vv * beta[:, None]
        hh = hh + vv.unsqueeze(-1) * hk.unsqueeze(1)
        h = hh.float()                         # fp32-rounded per-token state
        out[t] = torch.bmm(hh, hq.unsqueeze(-1)).squeeze(-1).float()
        if (t + 1) in want:
            ckpt[t + 1] = h.clone()
    return out, ckpt


def make_inputs(T):
    """Realistic GDN inputs (test_gdn_rewind.py conventions):
    unit-ish q/k (kernel L2-normalizes), v in [-1,1], a softplus-plausible,
    b in [-2,2] so beta=sigmoid(b) lies comfortably in (0,1)."""
    q = torch.randn(T, H, K, device=DEV, dtype=DTYPE)
    k = torch.randn(T, H, K, device=DEV, dtype=DTYPE)
    v = torch.empty(T, HV, V, device=DEV, dtype=DTYPE).uniform_(-1.0, 1.0)
    a = torch.randn(T, HV, device=DEV, dtype=DTYPE) * 0.3
    b = torch.randn(T, HV, device=DEV, dtype=DTYPE).clamp_(-2.0, 2.0)
    A_log = torch.randn(HV, device=DEV, dtype=DTYPE) - 1.0
    dt_bias = torch.randn(HV, device=DEV, dtype=DTYPE) * 0.1
    return q, k, v, a, b, A_log, dt_bias


def run_path_a(q, k, v, a, b, A_log, dt_bias, checkpoints):
    """Sequential packed-decode kernel, one token per call, state in-place."""
    Tn = q.shape[0]
    state = torch.zeros(NUM_STATE_ROWS, HV, V, K, device=DEV, dtype=DTYPE)
    qkv_dim = 2 * H * K + HV * V
    mixed_qkv = torch.empty(Tn, qkv_dim, device=DEV, dtype=DTYPE)
    mixed_qkv[:, : H * K] = q.reshape(Tn, H * K)
    mixed_qkv[:, H * K: 2 * H * K] = k.reshape(Tn, H * K)
    mixed_qkv[:, 2 * H * K:] = v.reshape(Tn, HV * V)
    idx = torch.tensor([ROW_SEQ], dtype=torch.int32, device=DEV)
    out_buf = torch.empty(1, 1, HV, V, device=DEV, dtype=DTYPE)
    outs = torch.empty(Tn, HV, V, device=DEV, dtype=DTYPE)
    ckpt = {}
    want = set(checkpoints)
    for t in range(Tn):
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv[t: t + 1],
            a=a[t: t + 1],
            b=b[t: t + 1],
            A_log=A_log,
            dt_bias=dt_bias,
            scale=SCALE,
            initial_state=state,
            out=out_buf,
            ssm_state_indices=idx,
            use_qk_l2norm_in_kernel=True,
        )
        outs[t] = out_buf[0, 0]
        if (t + 1) in want:
            ckpt[t + 1] = state[ROW_SEQ].clone()
    return outs, ckpt, state


def run_path_b(q, k, v, a, b, A_log, dt_bias, checkpoints):
    """Spec-window kernel in WINDOW-token chunks, full acceptance.

    Column convention (test_gdn_rewind.py): indices row [c_0..c_{w-1}], the
    kernel reads the initial state from column num_accepted-1 and writes the
    state after token i to column i. With the anchor row as the LAST column,
    the previous window's final state is exactly where the next window reads
    it — zero copies, production semantics."""
    Tn = q.shape[0]
    state = torch.zeros(NUM_STATE_ROWS, HV, V, K, device=DEV, dtype=DTYPE)
    outs = torch.empty(Tn, HV, V, device=DEV, dtype=DTYPE)
    ckpt = {}
    want = set(checkpoints)
    for w0 in range(0, Tn, WINDOW):
        w = min(WINDOW, Tn - w0)
        # Columns 0..w-2 -> scratch rows, column w-1 -> anchor row.
        cols = [ROW_SCRATCH + i for i in range(w - 1)] + [ROW_ANCHOR]
        indices = torch.tensor([cols], dtype=torch.int32, device=DEV)
        na = torch.tensor([w], dtype=torch.int32, device=DEV)
        cu = torch.tensor([0, w], dtype=torch.int32, device=DEV)
        out, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a[w0: w0 + w], b=b[w0: w0 + w], dt_bias=dt_bias,
            q=q[w0: w0 + w].unsqueeze(0), k=k[w0: w0 + w].unsqueeze(0),
            v=v[w0: w0 + w].unsqueeze(0),
            initial_state=state,
            inplace_final_state=True,
            cu_seqlens=cu,
            ssm_state_indices=indices,
            num_accepted_tokens=na,
            use_qk_l2norm_in_kernel=True,
        )
        outs[w0: w0 + w] = out[0]
        for t in range(w0, w0 + w):
            if (t + 1) in want:
                ckpt[t + 1] = state[cols[t - w0]].clone()
    return outs, ckpt, state


def rel_err(x, y):
    d = (x - y).abs().max().item()
    denom = y.abs().max().item()
    return d, d / max(denom, 1e-12)


def main():
    torch.cuda.init()
    free_b, _ = torch.cuda.mem_get_info()
    print(f"device free VRAM: {free_b / 2**20:.0f} MiB")

    # ------------------------------------------------------------------
    # Phase 1: T=14 gate — A, B, and the fp64 reference must agree ~1e-4.
    # ------------------------------------------------------------------
    q, k, v, a, b, A_log, dt_bias = make_inputs(WINDOW)
    oa, ca, _ = run_path_a(q, k, v, a, b, A_log, dt_bias, [WINDOW])
    ob, cb, _ = run_path_b(q, k, v, a, b, A_log, dt_bias, [WINDOW])
    oref, sref = ref_fold(q, k, v, a, b, A_log, dt_bias,
                          torch.zeros(HV, V, K, device=DEV), SCALE)

    # Vectorized long-run reference must reproduce the verbatim fold.
    oref2, cref2 = ref_fold_checkpoints(q, k, v, a, b, A_log, dt_bias, SCALE,
                                        [WINDOW])
    d, r = rel_err(oref2, oref)
    ds, rs = rel_err(cref2[WINDOW], sref[WINDOW - 1])
    print(f"[ref self-check] vectorized vs verbatim ref_fold: "
          f"out abs={d:.3e} rel={r:.3e}  state abs={ds:.3e} rel={rs:.3e}")
    if d > 1e-5 or ds > 1e-5:
        print("REF SELF-CHECK FAILED — vectorized fold diverges from "
              "verbatim ref_fold. Harness bug, aborting.")
        sys.exit(1)

    print(f"\n[T=14 gate] max abs / rel error vs fp64 reference:")
    rows = [
        ("A out ", oa, oref), ("A state", ca[WINDOW], sref[WINDOW - 1]),
        ("B out ", ob, oref), ("B state", cb[WINDOW], sref[WINDOW - 1]),
        ("A-B out ", oa, ob), ("A-B state", ca[WINDOW], cb[WINDOW]),
    ]
    gate_ok = True
    for tag, x, y in rows:
        d, r = rel_err(x.float(), y.float())
        bad = d > TOL_SHORT and tag.startswith(("A out", "A state",
                                                "B out", "B state"))
        gate_ok &= not bad
        print(f"  {tag}: abs={d:.3e} rel={r:.3e}"
              f"{'  <-- GATE FAIL' if bad else ''}")
    if not gate_ok:
        print("\nGATE FAILED at T=14 — harness invocation is wrong, "
              "not a kernel divergence. Aborting before the long run.")
        sys.exit(1)
    print("GATE PASS at T=14")

    # ------------------------------------------------------------------
    # Phase 2: long run, T=4000.
    # ------------------------------------------------------------------
    q, k, v, a, b, A_log, dt_bias = make_inputs(T_LONG)
    oa, ca, _ = run_path_a(q, k, v, a, b, A_log, dt_bias, CHECKPOINTS)
    print("path A done")
    ob, cb, _ = run_path_b(q, k, v, a, b, A_log, dt_bias, CHECKPOINTS)
    print("path B done")
    oref, cref = ref_fold_checkpoints(q, k, v, a, b, A_log, dt_bias, SCALE,
                                      CHECKPOINTS)
    print("reference done")

    print(f"\n[state checkpoints] max abs / rel difference:")
    print(f"  {'T':>5}  {'A-B':>18}  {'A-ref':>18}  {'B-ref':>18}")
    growth = []
    for t in CHECKPOINTS:
        dab, rab = rel_err(ca[t], cb[t])
        dar, rar = rel_err(ca[t], cref[t])
        dbr, rbr = rel_err(cb[t], cref[t])
        growth.append((t, dab))
        print(f"  {t:>5}  {dab:9.3e}/{rab:9.3e}  "
              f"{dar:9.3e}/{rar:9.3e}  {dbr:9.3e}/{rbr:9.3e}")

    # Per-token output divergence (A vs B), overall and per 500-token block.
    diff = (oa - ob).abs().reshape(T_LONG, -1).amax(dim=1)   # per-token max
    print(f"\n[per-token outputs A vs B] median max-abs={diff.median():.3e}  "
          f"overall max={diff.max():.3e}")
    for b0 in range(0, T_LONG, 500):
        blk = diff[b0: b0 + 500]
        print(f"  tokens {b0:>5}-{b0 + len(blk) - 1:>5}: "
              f"median={blk.median():.3e}  max={blk.max():.3e}")
    dr, rr = rel_err(oa, oref)
    print(f"[outputs A vs ref] abs={dr:.3e} rel={rr:.3e}")
    dr, rr = rel_err(ob, oref)
    print(f"[outputs B vs ref] abs={dr:.3e} rel={rr:.3e}")

    print("\n[verdict]")
    first, last = growth[0], growth[-1]
    grew = last[1] > 10 * max(first[1], 1e-12)
    print(f"  A-B final-state abs diff: T={first[0]} -> {first[1]:.3e}, "
          f"T={last[0]} -> {last[1]:.3e} "
          f"({'GROWS with length' if grew else 'flat / fp-noise-scale'})")
    sys.exit(0)


if __name__ == "__main__":
    main()
