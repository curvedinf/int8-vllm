#!/usr/bin/env python3
"""Can the real GDN decode kernels produce NaN from finite inputs?

Production observation (bf16 target, fp32 ssm state, NS=13): spec-window
checkpoint blocks at deep draft positions (~columns 5-6 of the 14-wide
window) contain partial NaN (~150-200 elements of the first 4096) across
layers, while the running state (accepted prefix) stays clean and the
committed text is still clean — the NaN appears BEFORE any visible garble.

This script drives the REAL kernels (causal_conv1d_update spec mode +
fused_sigmoid_gating_delta_rule_update, production calling convention per
qwen_gdn_linear_attn.py:1336-1349/1448-1466) through chained spec rounds
(q=14, fixed 14-block window, num_accepted chained at A_R=6 so columns 0-5
are the committed/running prefix and columns 6-13 are the optimistic tail)
while sweeping input magnitudes, and scans EVERY checkpoint column for
NaN/inf after every round.

Phases (deterministic, fixed seeds):
  A  all-token magnitude ladder 10^0 .. 10^32 (chained state across rounds).
  B  same ladder but ONLY the rejected-draft slots (j >= A_R) are spiked;
     committed slots stay at scale 1 — isolates optimistic-tail NaN with a
     clean running state (the production signature).
  C  decay-off attractor: a ~ N(-20, 3) => g ~ 0 => exp(g) ~ 1 (no decay),
     v scale ladder 1 .. 1e8, chained; then one strong-decay round
     (a ~ +1e4 => exp(g) = 0) to probe the 0 * inf -> NaN corner, and one
     beta-extreme round (b ~ +/-1e4). Measures max finite |ssm|.
  D  control: committed slots clean, rejected-draft slots carry actual
     inf/NaN values (simulated upstream hidden-state corruption) — shows the
     propagation signature for comparison with production.

Reference frame: the ssm recurrence is
  h <- h * exp(g) + beta * (v - h @ k) k^T,   out = h @ q
with k,q L2-normalized in-kernel, g = -exp(A_log) * softplus(a + dt_bias)
(<= 0 always), beta = sigmoid(b) in (0,1).  exp(g) in (0,1] can never
amplify; overflow needs |h| ~ 1e38.
"""

import sys

import torch

sys.path.insert(0, ".")

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)

DEV = "cuda"
NS = 13
WIN = NS + 1
KW = 4
CONV_W = KW - 1 + NS          # 16, production rolling-buffer width
CDIM = 32
H = 4
HV = 4
DK = 64
DV = 64
NBLK = WIN + 2                # block ids 1..14 in use, 0 = NULL
A_R = 6                       # committed per round -> running col = A_R-1 = 5

SEED = 1234


def scan_block(t):
    nan = t.isnan().sum().item()
    inf = t.isinf().sum().item()
    fin = t[torch.isfinite(t)]
    mx = fin.abs().max().item() if fin.numel() else 0.0
    return nan, inf, mx


class Sweep:
    def __init__(self):
        g = torch.Generator(device=DEV).manual_seed(SEED)
        self.g = g
        self.conv_pool = torch.zeros(NBLK, CDIM, CONV_W, dtype=torch.bfloat16,
                                     device=DEV)
        self.ssm_pool = torch.zeros(NBLK, HV, DV, DK, dtype=torch.float32,
                                    device=DEV)
        self.row = torch.arange(1, WIN + 1, dtype=torch.int32,
                                device=DEV).unsqueeze(0)     # (1, 14)
        self.qsl = torch.tensor([0, WIN], dtype=torch.int32, device=DEV)
        self.na = torch.ones(1, dtype=torch.int32, device=DEV)
        self.conv_w = (torch.randn(CDIM, KW, generator=g, device=DEV) * 0.5
                       ).to(torch.bfloat16)
        self.conv_b = (torch.randn(CDIM, generator=g, device=DEV) * 0.5
                       ).to(torch.bfloat16)
        self.A_log = torch.randn(HV, generator=g, device=DEV) - 1.0
        self.dt_bias = torch.randn(HV, generator=g, device=DEV) * 0.1
        self.scale = DK ** -0.5
        self.max_finite_ssm = 0.0
        self.first_nan = None      # (phase, round, desc, state, col)

    def reset_state(self):
        self.conv_pool.zero_()
        self.ssm_pool.zero_()
        self.na.fill_(1)

    def rnd(self, *shape, scale=1.0, offset=0.0):
        return torch.randn(*shape, generator=self.g, device=DEV) * scale + offset

    def run_round(self, tok_scale, draft_scale=None, a_off=0.0, b_off=0.0,
                  corrupt_drafts=None, v_scale=None):
        """One q=14 spec round. tok_scale: per-token log-uniform magnitude for
        all slots (if draft_scale is None) or for committed slots j < A_R only.
        draft_scale: magnitude for rejected slots j >= A_R.
        v_scale: override magnitude for v only (q/k/a/b stay at tok_scale).
        corrupt_drafts: None | 'nan' | 'inf' — inject actual non-finite values
        into a few elements of the rejected slots."""
        u = torch.rand(WIN, generator=self.g, device=DEV) * 2 - 1  # in [-1,1]
        mag_c = 10.0 ** (torch.rand(WIN, generator=self.g, device=DEV)
                         * 0.0 + torch.log10(torch.tensor(tok_scale)))
        # per-slot magnitude multiplier with per-token jitter (x0.3..x3)
        jitter = 10.0 ** (torch.rand(WIN, 1, 1, generator=self.g,
                                      device=DEV) - 0.5)
        mag = mag_c.view(WIN, 1, 1) * jitter
        mag = mag.float()
        if draft_scale is not None:
            mag_d = (torch.tensor(draft_scale, device=DEV)
                     * 10.0 ** (torch.rand(WIN, 1, 1, generator=self.g,
                                           device=DEV) - 0.5)).float()
            mag[A_R:] = mag_d[A_R:]

        x = (self.rnd(WIN, CDIM) * mag[:, :, 0]).to(torch.bfloat16)
        q = self.rnd(WIN, H, DK) * mag
        k = self.rnd(WIN, H, DK) * mag
        v = self.rnd(WIN, HV, DV) * mag
        if v_scale is not None:
            v = self.rnd(WIN, HV, DV) * v_scale * 10.0 ** (
                torch.rand(WIN, 1, 1, generator=self.g, device=DEV) - 0.5)
        a = self.rnd(WIN, HV) * mag[:, :, 0] + a_off
        b = self.rnd(WIN, HV) * mag[:, :, 0] + b_off
        if corrupt_drafts is not None:
            val = float("nan") if corrupt_drafts.startswith("nan") \
                else float("inf")
            # ~3% of elements in the rejected slots, matching a partial
            # corruption pattern.
            targets = (v,) if corrupt_drafts.endswith("_v") else (q, k, v)
            for t in targets:
                mask = (torch.rand_like(t) < 0.03)
                mask[:A_R] = False
                t[mask] = val
            if not corrupt_drafts.endswith("_v"):
                mask = (torch.rand_like(a) < 0.03)
                mask[:A_R] = False
                a[mask] = val

        cout = causal_conv1d_update(
            x.clone(), self.conv_pool, self.conv_w, self.conv_b, "silu",
            conv_state_indices=self.row[:, 0],
            num_accepted_tokens=self.na,
            query_start_loc=self.qsl,
            max_query_len=WIN,
            validate_data=False,
        )
        sout, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log, a=a, b=b, dt_bias=self.dt_bias,
            q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
            initial_state=self.ssm_pool, inplace_final_state=True,
            cu_seqlens=self.qsl, ssm_state_indices=self.row,
            num_accepted_tokens=self.na,
            use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize()
        self.na.fill_(A_R)
        return cout, sout

    def scan(self, phase, r, desc, finite_inputs=True):
        ssm_cols = [scan_block(self.ssm_pool[j + 1]) for j in range(WIN)]
        conv_cols = [scan_block(self.conv_pool[j + 1]) for j in range(WIN)]
        ssm_nan = sum(c[0] for c in ssm_cols)
        ssm_inf = sum(c[1] for c in ssm_cols)
        conv_nan = sum(c[0] for c in conv_cols)
        conv_inf = sum(c[1] for c in conv_cols)
        run_max = ssm_cols[A_R - 1][2]
        deep_max = max(c[2] for c in ssm_cols[A_R:])
        if deep_max == deep_max:  # not nan
            self.max_finite_ssm = max(self.max_finite_ssm, deep_max, run_max)
        nan_cols = [j for j, c in enumerate(ssm_cols) if c[0] > 0]
        inf_cols = [j for j, c in enumerate(ssm_cols) if c[1] > 0]
        if (ssm_nan or conv_nan) and self.first_nan is None and finite_inputs:
            state = "ssm" if ssm_nan else "conv"
            self.first_nan = (phase, r, desc, state, nan_cols)
        flag = ""
        if ssm_nan or conv_nan or ssm_inf or conv_inf:
            flag = (f"  ssm_nan_cols={nan_cols} ssm_inf_cols={inf_cols}"
                    f" conv_nan={conv_nan} conv_inf={conv_inf}")
        print(f"  {phase} r{r:2d} {desc:34s} ssm_nan={ssm_nan:6d} "
              f"ssm_inf={ssm_inf:6d} conv_nan={conv_nan} conv_inf={conv_inf} "
              f"max|ssm| run={run_max:.3e} deep={deep_max:.3e}{flag}")
        return ssm_nan, ssm_inf, conv_nan, conv_inf

    def out_scan(self, cout, sout, phase, r):
        """NaN/inf in outputs, split committed prefix vs optimistic tail."""
        co = cout.float()
        so = sout[0].float()
        c_run = co[:A_R].isnan().sum().item() + co[:A_R].isinf().sum().item()
        c_deep = co[A_R:].isnan().sum().item() + co[A_R:].isinf().sum().item()
        s_run = so[:A_R].isnan().sum().item() + so[:A_R].isinf().sum().item()
        s_deep = so[A_R:].isnan().sum().item() + so[A_R:].isinf().sum().item()
        if c_run or c_deep or s_run or s_deep:
            print(f"    OUT {phase} r{r}: conv_out bad run={c_run} "
                  f"deep={c_deep} | ssm_out bad run={s_run} deep={s_deep}")


def main():
    sw = Sweep()
    ladder = [10.0 ** (2 * i) for i in range(17)]   # 1 .. 1e32

    print("== Phase A: all-token magnitude ladder, chained state ==")
    sw.reset_state()
    for r, s in enumerate(ladder):
        cout, sout = sw.run_round(s)
        sw.scan("A", r, f"scale~1e{2 * r}", )
        sw.out_scan(cout, sout, "A", r)

    print("== Phase B: rejected-draft slots (j>=6) spiked, committed clean ==")
    sw.reset_state()
    for r, s in enumerate(ladder):
        cout, sout = sw.run_round(1.0, draft_scale=s)
        sw.scan("B", r, f"draft scale~1e{2 * r}")
        sw.out_scan(cout, sout, "B", r)

    print("== Phase C: decay-off attractor (a~N(-20,3) => exp(g)~1) ==")
    sw.reset_state()
    for r, s in enumerate([1.0, 1e2, 1e4, 1e6, 1e8]):
        cout, sout = sw.run_round(s, a_off=-20.0)
        sw.scan("C", r, f"v scale~1e{int(torch.log10(torch.tensor(s)))} "
                        f"a_off=-20")
        sw.out_scan(cout, sout, "C", r)
    # strong-decay round right after the growth rounds: exp(g)=0, probes
    # 0 * inf -> NaN if the attractor overflowed.
    cout, sout = sw.run_round(1.0, a_off=1e4)
    sw.scan("C", 5, "a_off=+1e4 (exp(g)=0, 0*inf probe)")
    sw.out_scan(cout, sout, "C", 5)
    # beta extremes
    cout, sout = sw.run_round(1e4, b_off=-1e4)
    sw.scan("C", 6, "b_off=-1e4 (beta=0)")
    cout, sout = sw.run_round(1e4, b_off=1e4)
    sw.scan("C", 7, "b_off=+1e4 (beta=1)")

    print("== Phase D: control — actual inf/NaN injected into rejected "
          "draft slots only ==")
    for what in ("nan", "inf", "nan_v"):
        sw.reset_state()
        cout, sout = sw.run_round(1.0, corrupt_drafts=what)
        sw.scan("D", 0, f"drafts contain {what}", finite_inputs=False)
        sw.out_scan(cout, sout, "D", 0)
        # per-column partial-NaN pattern for the v-only case
        if what == "nan_v":
            for j in range(WIN):
                blk = sw.ssm_pool[j + 1]
                n = blk.isnan().sum().item()
                if n:
                    rows = blk.isnan().any(dim=2).sum().item()
                    print(f"    D(nan_v) col {j}: {n} NaN over "
                          f"{int(rows)}/{HV * DV} v-rows (block has "
                          f"{HV * DV * DK} elems)")
        # one more clean round: does the chain recover (running col clean)?
        cout, sout = sw.run_round(1.0)
        sw.scan("D", 1, "clean round after corruption", finite_inputs=False)
        sw.out_scan(cout, sout, "D", 1)

    print("== Phase E: v-only ladder into the fp32 overflow regime, "
          "decay off, beta~1, chained ==")
    sw.reset_state()
    for r, s in enumerate([1e10, 1e20, 1e25, 1e30, 1e34, 1e36, 1e37, 1e38]):
        cout, sout = sw.run_round(1.0, a_off=-20.0, b_off=5.0, v_scale=s)
        sw.scan("E", r, f"v scale~1e{int(round(torch.log10(torch.tensor(s)).item()))} "
                        f"a_off=-20 b_off=5")
        sw.out_scan(cout, sout, "E", r)

    print()
    print(f"max finite |ssm| over sweep: {sw.max_finite_ssm:.6e} "
          f"(fp32 max 3.40e+38, bf16 max 3.39e+38)")
    if sw.first_nan is None:
        print("VERDICT: no NaN/inf produced from FINITE inputs anywhere in "
              "the sweep (ladder to 1e32, decay-off attractor, beta/gate "
              "extremes).")
    else:
        phase, r, desc, state, cols = sw.first_nan
        print(f"VERDICT: first NaN from finite inputs at phase {phase} "
              f"round {r} ({desc}) in {state} state, columns {cols}")
    print("Phase D shows the corrupted-INPUT signature for comparison.")


if __name__ == "__main__":
    main()
