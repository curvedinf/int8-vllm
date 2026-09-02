#!/usr/bin/env python3
"""Decisive offline differential for the mamba "align"-mode spec-decode state
chain (conv + SSM) through boundary oscillation and query-length collapse.

Drives the REAL kernels with the exact production calling conventions:

  * preprocess:  preprocess_mamba_align_fused_kernel   (vllm/v1/worker/mamba_utils.py)
  * precopy:     precopy_mamba_align_fused_kernel      (same file; DS conv path)
  * conv fwd:    causal_conv1d_update in spec mode, conv_state_indices =
                 row[:, 0], num_accepted = post-preprocess value,
                 max_query_len = WINDOW (qwen_gdn_linear_attn.py:1336-1349)
  * ssm fwd:     fused_sigmoid_gating_delta_rule_update with the 14-wide
                 window row, initial state read from column na-1, checkpoint
                 after token j written to column j
                 (qwen_gdn_linear_attn.py:1448-1466, scripts/test_gdn_rewind.py)
  * postprocess: postprocess_mamba_fused_kernel with PRECOMPUTED_NEW_COMPUTED
                 (run_fused_postprocess_align, mamba_hybrid.py:577) — the
                 block-aligned prefix-cache save that resets num_accepted to 1
                 when src==dest (exact-boundary full-accept landings).

Reference: the same total committed token stream folded through the SAME two
kernels in pure sequential decode (T=1, fixed block row, na=1 — "none" mode),
plus independent intent models (python rolling conv window over committed
tokens; fp64 delta-rule fold) to validate the sequential baseline itself.

Engine invariants replicated:
  * num_computed_r = committed - 1 (last committed token re-fed as anchor).
  * start_r = (num_computed_r + query_len_r - 1) // MAMBA_BLOCK_SIZE.
  * preprocess: src_col = state_idx(prev), src_off = max(na_prev-1, 0);
    state_idx <- (computed_after + BS - 1)//BS - 1; na <- 1 on change.
  * precopy (on crossing, either direction): ssm bt[src_col+src_off] ->
    bt[start_r]; conv bt[src_col][src_off:] -> bt[start_r][:conv_w-src_off].
  * acceptance a_r commits x_1..x_{a_r-1} plus resampled t* = G[nc + a_r];
    rejected drafts x_{a_r}.. carry garbage values (never committed).
  * end-of-round TRUE state = sequential state through G[nc_{r+1} - 1]
    (all committed EXCEPT the re-fed anchor).

Fix-candidate evaluation (env FIX_MODE):
  none - production behavior (guard bails when na > query_len).
  f1   - engine clamp emulation: na <- min(na, query_len) after preprocess.
  f2   - engine resync emulation: na <- 1 after preprocess when na > query_len.
  f3/f4 are kernel-side and tested by editing causal_conv1d.py directly
  (triton re-JITs on source change); run with FIX_MODE=none.

A host-side content map tracks which committed-stream position every conv
buffer entry holds; on a guard-fire round the harness prints the full buffer
mapping, the entries the correct rewind needs, and what the kernel would have
written.
"""

import os
import sys

import torch

sys.path.insert(0, ".")

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.v1.worker.mamba_utils import (
    postprocess_mamba_fused_kernel,
    precopy_mamba_align_fused_kernel,
    preprocess_mamba_align_fused_kernel,
)

DEV = "cuda"
FIX_MODE = os.environ.get("FIX_MODE", "none")

# --- chain geometry (production ratios, shrunk) ---------------------------
BS = 64          # MAMBA_BLOCK_SIZE
NS = 13          # num_speculative_tokens
WIN = NS + 1     # 14-wide verify window
KW = 4           # conv kernel width
CONV_W = KW - 1 + NS   # allocated rolling-buffer width = 16 (production rule)
CDIM = 8
H = 2            # k/q heads
HV = 2           # v heads
DK = 16
DV = 16
MAX_COLS = 48
NBLK = MAX_COLS + 1          # physical ids 1..MAX_COLS (+ slot 0 = NULL)
REF_BASE = 20                # sequential-reference window: blocks 20..33

SSM_TOL = 1e-5   # same-kernel bitwise-ish
CONV_TOL = 0.0   # bf16 byte-movement must be exact
OUT_TOL = 1e-5


# --- independent intent models (validate the sequential baseline) ----------

def ref_fold(q, k, v, a, b, A_log, dt_bias, h_prev, scale):
    """fp64 delta-rule fold (from scripts/test_gdn_rewind.py).
    q/k: (T,H,K); v: (T,HV,V); a/b: (T,HV); h_prev: (HV,V,K) (row=v, col=k).
    Returns states (T,HV,V,K)."""
    Tn = q.shape[0]
    states = torch.empty(Tn, HV, DV, DK, device=q.device, dtype=torch.float64)
    h = h_prev.double().clone()
    for t in range(Tn):
        for hv in range(HV):
            hq = q[t, hv].double()
            hk = k[t, hv].double()
            hq = hq * torch.rsqrt(hq @ hq + 1e-6)
            hk = hk * torch.rsqrt(hk @ hk + 1e-6)
            hq = hq * scale
            x = a[t, hv].double() + dt_bias[hv].double()
            sp = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
            g = -torch.exp(A_log[hv].double()) * sp
            beta = torch.sigmoid(b[t, hv].double())
            hh = h[hv]                       # (V,K)
            hh = hh * torch.exp(g)
            vv = v[t, hv].double() - hh @ hk
            vv = vv * beta
            hh = hh + vv[:, None] * hk[None, :]
            h[hv] = hh
            states[t, hv] = hh
    return states


class ConvIntent:
    """Documented conv intent: rolling window over the committed stream.
    Buffer after token i = last KW-1 inputs ending at i (bf16, verbatim)."""

    def __init__(self, weight, bias):
        self.w = weight.float().cpu()        # (CDIM, KW)
        self.b = bias.float().cpu()          # (CDIM,)
        self.hist = torch.zeros(CDIM, KW - 1, dtype=torch.bfloat16)

    def step(self, x):                       # x: (CDIM,) bf16
        acc = self.b.clone()
        for j in range(KW - 1):
            acc += self.w[:, j] * self.hist[:, j].float()
        acc += self.w[:, KW - 1] * x.float().cpu()
        out = acc / (1.0 + torch.exp(-acc))  # silu
        self.hist = torch.cat(
            [self.hist[:, 1:], x.cpu().unsqueeze(1)], dim=1)
        return out.to(torch.bfloat16)


# --- scenario definitions ---------------------------------------------------
# Each round: (query_len, accepted).  nc_0 = P - 1 (anchor = G[P-1]).

def scenario_control(seed):
    # No oscillation: steady q=14 across three boundaries.  r0 lands nc'
    # exactly on boundary 64 with a full accept -> postprocess_align fires
    # (src==dest, bias=13, na<-1), exercising the aligned-save path.
    rounds = [(14, 14), (14, 9), (14, 13), (14, 14), (14, 1), (14, 14),
              (14, 14), (14, 5), (14, 14), (14, 14), (14, 14), (14, 14),
              (14, 14), (14, 14), (14, 14), (14, 14)]
    return dict(name="control", P=51, rounds=rounds)


def scenario_oscillation(seed):
    # Boundary oscillation at 64 (regress 1->0 at r2, re-cross at r4) and
    # at 192 (regress 3->2 at r20, re-cross at r21), with collapses to
    # q in {1..4} and acceptance in {1, partial, full}.
    rounds = [
        (14, 10),   # r0:  ca=64  start 0 (exact-boundary edge)
        (14, 1),    # r1:  ca=74  start 1  CROSS up
        (2, 1),     # r2:  ca=63  start 0  REGRESS 1->0
        (1, 1),     # r3:  ca=63  start 0
        (4, 3),     # r4:  ca=67  start 1  CROSS up
        (14, 14),   # r5:  ca=80  start 1
        (14, 7),    # r6:  ca=94  start 1
        (14, 1),    # r7:  ca=101 start 1
        (14, 13),   # r8:  ca=102 start 1
        (14, 14),   # r9:  ca=115 start 1
        (14, 5),    # r10: ca=129 start 2  CROSS up
        (1, 1),     # r11: ca=121 start 1  REGRESS 2->1
        (14, 14),   # r12: ca=135 start 2  CROSS up
        (14, 2),    # r13: ca=149 start 2
        (3, 2),     # r14: ca=140 start 2  (collapse, no cross, na=2<=3)
        (2, 1),     # r15: ca=141 start 2
        (14, 14),   # r16: ca=154 start 2
        (14, 14),   # r17: ca=168 start 2
        (14, 14),   # r18: ca=182 start 2
        (14, 1),    # r19: ca=196 start 3  CROSS up
        (2, 1),     # r20: ca=185 start 2  REGRESS 3->2
        (14, 5),    # r21: ca=198 start 3  CROSS up
        (14, 14),   # r22: ca=203 start 3
    ]
    return dict(name="oscillation", P=51, rounds=rounds)


def scenario_collapse_no_cross(seed):
    # Regression gate for the mid-block collapse mechanism: full accept
    # (na=14) followed by q=3 WITHOUT a boundary change -> no preprocess
    # reset, na > q.  Requires the conv guard to bound against
    # max_query_len (the allocated window width), not this round's seqlen.
    rounds = [
        (14, 14),   # r0: ca=43 start 0
        (3, 2),     # r1: ca=46 start 0  NO CROSS -> na=14 > q=3
        (14, 1),    # r2: ca=61 start 0
        (14, 14),   # r3: ca=75 start 1  CROSS
        (14, 14),   # r4: ca=89 start 1
    ]
    return dict(name="collapse_no_cross", P=30, rounds=rounds)


# --- harness ---------------------------------------------------------------

class Harness:
    def __init__(self, scen, seed):
        self.scen = scen
        P = scen["P"]
        rounds = scen["rounds"]
        self.L = P + sum(a for _, a in rounds)
        R = len(rounds)
        g = torch.Generator(device=DEV).manual_seed(seed)
        gw = torch.Generator(device=DEV).manual_seed(seed * 100003 + 7)

        def rnd(*shape, scale=1.0):
            return torch.randn(*shape, generator=g, device=DEV) * scale

        # Global committed stream inputs (per G index).
        self.gx = (rnd(self.L, CDIM) * 1.0).to(torch.bfloat16)
        self.gq = rnd(self.L, H, DK)
        self.gk = rnd(self.L, H, DK)
        self.gv = rnd(self.L, HV, DV)
        self.ga = rnd(self.L, HV, scale=0.3)
        self.gb = rnd(self.L, HV)
        # Garbage values for rejected drafts (distinct magnitude/offset).
        def rndw(*shape, scale=1.0):
            return torch.randn(*shape, generator=gw, device=DEV) * scale
        self.wx = (rndw(R, WIN, CDIM, scale=3.0) + 5.0).to(torch.bfloat16)
        self.wq = rndw(R, WIN, H, DK, scale=3.0) + 5.0
        self.wk = rndw(R, WIN, H, DK, scale=3.0) + 5.0
        self.wv = rndw(R, WIN, HV, DV, scale=3.0) + 5.0
        self.wa = rndw(R, WIN, HV, scale=3.0) + 5.0
        self.wb = rndw(R, WIN, HV, scale=3.0) + 5.0

        self.A_log = rnd(HV) - 1.0
        self.dt_bias = rnd(HV, scale=0.1)
        self.conv_w = rnd(CDIM, KW, scale=0.5).to(torch.bfloat16)
        self.conv_b = rnd(CDIM, scale=0.5).to(torch.bfloat16)
        self.scale = DK ** -0.5

        # State pools, NaN-filled.
        nan = float("nan")
        self.conv_spec = torch.full((NBLK, CDIM, CONV_W), nan,
                                    dtype=torch.bfloat16, device=DEV)
        self.ssm_spec = torch.full((NBLK, HV, DV, DK), nan,
                                   dtype=torch.float32, device=DEV)
        self.conv_ref = torch.full((NBLK, CDIM, CONV_W), nan,
                                   dtype=torch.bfloat16, device=DEV)
        self.ssm_ref = torch.full((NBLK, HV, DV, DK), nan,
                                  dtype=torch.float32, device=DEV)
        # Reference starts from a clean zero state.
        self.conv_ref[REF_BASE:REF_BASE + WIN].zero_()
        self.ssm_ref[REF_BASE:REF_BASE + WIN].zero_()

        # Block tables: distinct physical id per column (aliasing visible).
        self.bt = torch.arange(1, MAX_COLS + 1, dtype=torch.int32,
                               device=DEV).unsqueeze(0)      # (1, MAX_COLS)
        self.bt_ref = torch.arange(REF_BASE, REF_BASE + WIN, dtype=torch.int32,
                                   device=DEV).unsqueeze(0)  # (1, WIN)

        # Persistent engine bookkeeping tensors.
        self.state_idx = torch.full((1,), (P - 1) // BS, dtype=torch.int32,
                                    device=DEV)
        self.num_accepted = torch.ones(1, dtype=torch.int32, device=DEV)
        self.na_pp = torch.ones(1, dtype=torch.int32, device=DEV)  # pp snapshot
        self.num_computed = torch.zeros(1, dtype=torch.int32, device=DEV)
        self.qsl = torch.zeros(2, dtype=torch.int32, device=DEV)
        self.src_col = torch.full((1,), -1, dtype=torch.int32, device=DEV)
        self.src_off = torch.zeros(1, dtype=torch.int32, device=DEV)
        self.idx_map = torch.zeros(1, dtype=torch.int32, device=DEV)

        # Copy-kernel metadata (one layer, two state types: conv=0, ssm=1).
        i64 = dict(dtype=torch.int64, device=DEV)
        i32 = dict(dtype=torch.int32, device=DEV)
        self.bt_ptrs = torch.tensor([self.bt.data_ptr()], **i64)
        self.base_addrs = torch.tensor(
            [self.conv_spec.data_ptr(), self.ssm_spec.data_ptr()], **i64)
        self.block_strides = torch.tensor(
            [self.conv_spec.stride(0) * self.conv_spec.element_size(),
             self.ssm_spec.stride(0) * self.ssm_spec.element_size()], **i64)
        self.elem_sizes = torch.tensor(
            [self.conv_spec.element_size(), self.ssm_spec.element_size()],
            **i32)
        self.inner_sizes = torch.tensor([1, HV * DV * DK], **i64)
        self.conv_widths = torch.tensor([CONV_W, 0], **i32)
        self.group_indices = torch.tensor([0, 0], **i32)
        self.dim_row_count = torch.tensor([CDIM, 0], **i32)
        self.dim_row_stride = torch.tensor(
            [self.conv_spec.stride(1) * self.conv_spec.element_size(), 0],
            **i64)

        # Host-side conv content map: physical block id -> [label]*CONV_W,
        # mirroring the kernels' byte movement (prime/precopy/forward/pp).
        self.map_conv = {}
        self._fire_printed = False

    def _map_shift(self, src_blk, dst_blk, bias):
        """conv[dst][k] = conv[src][k+bias] for k < CONV_W-bias; tail kept."""
        smap = self.map_conv.get(src_blk, [None] * CONV_W)
        dmap = list(self.map_conv.get(dst_blk, [None] * CONV_W))
        for k in range(CONV_W - bias):
            dmap[k] = smap[k + bias]
        self.map_conv[dst_blk] = dmap

    def _map_forward(self, blk, off, q_len, a_r, nc, r):
        """Kernel rolling write: new = [old[off+1], old[off+2], x_0..x_{q-1}]
        at entries [0, KW-2+q); tail kept.  off = na_fwd - 1."""
        old = self.map_conv.get(blk, [None] * CONV_W)
        new = list(old)
        for k in range(KW - 2):
            new[k] = old[off + 1 + k] if off + 1 + k < CONV_W else "OOB"
        for j in range(q_len):
            new[KW - 2 + j] = f"G{nc + j}" if j < a_r else f"x{r}.{j}"
        self.map_conv[blk] = new

    def _print_fire_mapping(self, r, nc, q_len, na_fwd, blk):
        """Task 1: exact buffer mapping at the guard-fire round."""
        m = self.map_conv.get(blk, [None] * CONV_W)
        lo, hi = na_fwd - 1, na_fwd - 1 + KW - 1   # entries the rewind reads
        need = [f"G{nc - (KW - 1) + i}" for i in range(KW - 1)]
        print(f"  --- guard-fire buffer mapping: round {r}, block b{blk}, "
              f"na={na_fwd}, q={q_len} ---")
        print(f"  correct rewind needs history {[f'G{nc-k}' for k in range(KW-1, 0, -1)]}"
              f" at entries [{lo}, {hi})")
        for k in range(CONV_W):
            mark = " <== needed" if lo <= k < hi else ""
            print(f"    entry {k:2d}: {str(m[k]):>8s}{mark}")
        print(f"  entries [{lo}, {hi}) hold {[m[k] for k in range(lo, hi)]}"
              f" — needed {need}")
        print(f"  kernel would have written entries [0, {KW-2+q_len}) = "
              f"[old[{na_fwd}], old[{na_fwd+1}], "
              f"{[f'G{nc+j}' if j < self.scen['rounds'][r][1] else f'x{r}.{j}' for j in range(q_len)]}]")

    # -- sequential reference over the full committed stream ----------------
    def run_reference(self):
        L = self.L
        self.ref_ssm = [None] * L       # state through token i
        self.ref_conv = [None] * L      # buffer (last KW-1 inputs) after i
        self.ref_conv_out = [None] * L
        self.ref_ssm_out = [None] * L
        intent = ConvIntent(self.conv_w, self.conv_b)
        intent_conv_err = 0.0
        intent_cout_err = 0.0
        qsl1 = torch.tensor([0, 1], dtype=torch.int32, device=DEV)
        na1 = torch.ones(1, dtype=torch.int32, device=DEV)
        for i in range(L):
            x = self.gx[i].unsqueeze(0).clone()          # (1, CDIM)
            out = causal_conv1d_update(
                x, self.conv_ref, self.conv_w, self.conv_b, "silu",
                conv_state_indices=self.bt_ref[:, 0],
                num_accepted_tokens=na1,
                query_start_loc=qsl1,
                max_query_len=WIN,
                validate_data=False,
            )
            so, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log, a=self.ga[i].unsqueeze(0),
                b=self.gb[i].unsqueeze(0), dt_bias=self.dt_bias,
                q=self.gq[i].unsqueeze(0).unsqueeze(0),
                k=self.gk[i].unsqueeze(0).unsqueeze(0),
                v=self.gv[i].unsqueeze(0).unsqueeze(0),
                initial_state=self.ssm_ref, inplace_final_state=True,
                cu_seqlens=qsl1, ssm_state_indices=self.bt_ref,
                num_accepted_tokens=na1, use_qk_l2norm_in_kernel=True,
            )
            self.ref_conv_out[i] = out[0].float().cpu()
            self.ref_ssm_out[i] = so[0, 0].float().cpu()   # (HV, DV)
            self.ref_conv[i] = self.conv_ref[REF_BASE, :, :KW - 1].cpu().clone()
            self.ref_ssm[i] = self.ssm_ref[REF_BASE].cpu().clone()
            # Intent checks (sequential path vs documented semantics).
            io = intent.step(self.gx[i])
            ic_err = (self.ref_conv[i] != intent.hist).sum().item()
            intent_conv_err = max(intent_conv_err, ic_err)
            ic_o = (self.ref_conv_out[i] - io.float()).abs().max().item()
            intent_cout_err = max(intent_cout_err, ic_o)
        # fp64 fold sanity for the ssm side.
        st = ref_fold(self.gq, self.gk, self.gv, self.ga, self.gb,
                      self.A_log, self.dt_bias,
                      torch.zeros(HV, DV, DK, device=DEV), self.scale)
        fold_err = max(
            (st[i].float() - self.ref_ssm[i].to(DEV)).abs().max().item()
            for i in range(L))
        return intent_conv_err, intent_cout_err, fold_err

    # -- spec+align rounds -----------------------------------------------------
    def run(self):
        P, rounds = self.scen["P"], self.scen["rounds"]
        i_err, o_err, f_err = self.run_reference()
        print(f"  baseline intent checks: conv_state_exact_mismatch="
              f"{i_err} conv_out_bf16_err={o_err:.2e} "
              f"ssm_fp64_fold_err={f_err:.2e}")

        # Prime the spec pools: state through G[P-2] at column (P-1)//BS.
        seed_col = (P - 1) // BS
        pb = int(self.bt[0, seed_col])
        self.ssm_spec[pb] = self.ref_ssm[P - 2].to(DEV)
        self.conv_spec[pb, :, :KW - 1] = self.ref_conv[P - 2].to(DEV)
        self.map_conv[pb] = [f"G{P - KW + k}" for k in range(KW - 1)] \
            + [None] * (CONV_W - (KW - 1))

        nc = P - 1
        first_bad = None
        rows = []
        for r, (q_len, a_r) in enumerate(rounds):
            assert 1 <= a_r <= q_len
            ca = nc + q_len
            start_exp = (ca - 1) // BS

            # Round inputs: committed prefix from G, rejected drafts garbage.
            xr = self.wx[r, :q_len].clone()
            qr = self.wq[r, :q_len].clone()
            kr = self.wk[r, :q_len].clone()
            vr = self.wv[r, :q_len].clone()
            ar = self.wa[r, :q_len].clone()
            br = self.wb[r, :q_len].clone()
            for j in range(a_r):
                xr[j] = self.gx[nc + j]
                qr[j] = self.gq[nc + j]
                kr[j] = self.gk[nc + j]
                vr[j] = self.gv[nc + j]
                ar[j] = self.ga[nc + j]
                br[j] = self.gb[nc + j]

            na_pre = int(self.num_accepted[0])
            self.num_computed.fill_(nc)
            self.qsl[0] = 0
            self.qsl[1] = q_len

            # 1. preprocess (real kernel)
            preprocess_mamba_align_fused_kernel[(1,)](
                self.idx_map, self.state_idx, self.num_computed, self.qsl,
                self.num_accepted, self.src_col, self.src_off, 1,
                BLOCK_SIZE=256, MAMBA_BLOCK_SIZE=BS,
            )
            torch.cuda.synchronize()
            start = int(self.state_idx[0])
            s_col = int(self.src_col[0])
            s_off = int(self.src_off[0])
            na_fwd = int(self.num_accepted[0])
            assert start == start_exp, (r, start, start_exp)

            # Fix-candidate engine emulations (F1 clamp / F2 resync),
            # applied where an engine-side change would take effect: after
            # preprocess, before the forward.
            fix_note = ""
            if na_fwd > q_len and FIX_MODE == "f1":
                self.num_accepted.fill_(q_len)
                na_fwd = q_len
                fix_note = f"f1 clamp na->{q_len}"
            elif na_fwd > q_len and FIX_MODE == "f2":
                self.num_accepted.fill_(1)
                na_fwd = 1
                fix_note = "f2 resync na->1"

            # 2. precopy (real kernel, production grid/flags)
            precopy_mamba_align_fused_kernel[(1, 2, 16)](
                self.state_idx, self.src_col, self.src_off,
                self.bt_ptrs, self.bt.stride(0),
                self.base_addrs, self.block_strides, self.elem_sizes,
                self.inner_sizes, self.conv_widths, self.group_indices,
                self.dim_row_count, self.dim_row_stride,
                self.idx_map, 1,
                COPY_BLOCK_SIZE=1024, CONV_STATE_DIM_FIRST=True,
                HAS_IDX_MAPPING=True, TEMPORAL_TILES=16,
            )
            torch.cuda.synchronize()
            copied = s_col >= 0 and s_col != start
            copy_desc = "-"
            if copied:
                ssrc = int(self.bt[0, s_col + s_off])
                csrc = int(self.bt[0, s_col])
                dst = int(self.bt[0, start])
                copy_desc = f"ssm b{ssrc}->b{dst} conv b{csrc}->b{dst}"
                self._map_shift(csrc, dst, s_off)

            # 3. forward (real kernels, production invocation)
            row = self.bt[0, start:start + WIN].unsqueeze(0).contiguous()
            cb = int(row[0, 0])
            # Host-side mirror of the conv guard actually compiled into the
            # kernel under test:
            #   none/f1/f2: production guard  bail iff na < 1 or na > q_len
            #   f3:         OOB-only guard    bail iff na < 1 or na > WIN
            #   f4:         clamp, never bail (offset clamped to [0, WIN-1])
            if FIX_MODE == "f3":
                bailed = na_fwd > WIN or na_fwd < 1
            elif FIX_MODE == "f4":
                bailed = False
            else:
                bailed = na_fwd > q_len or na_fwd < 1
            eff_off = na_fwd - 1
            if FIX_MODE == "f4":
                eff_off = min(max(na_fwd - 1, 0), WIN - 1)
            if bailed and not self._fire_printed:
                self._print_fire_mapping(r, nc, q_len, na_fwd, cb)
                self._fire_printed = True
            cout = causal_conv1d_update(
                xr, self.conv_spec, self.conv_w, self.conv_b, "silu",
                conv_state_indices=row[:, 0],
                num_accepted_tokens=self.num_accepted,
                query_start_loc=self.qsl,
                max_query_len=WIN,
                validate_data=False,
            )
            sout, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log, a=ar, b=br, dt_bias=self.dt_bias,
                q=qr.unsqueeze(0), k=kr.unsqueeze(0), v=vr.unsqueeze(0),
                initial_state=self.ssm_spec, inplace_final_state=True,
                cu_seqlens=self.qsl, ssm_state_indices=row,
                num_accepted_tokens=self.num_accepted,
                use_qk_l2norm_in_kernel=True,
            )
            torch.cuda.synchronize()
            if not bailed:
                self._map_forward(cb, eff_off, q_len, a_r, nc, r)

            # 4. compare against sequential reference
            nc_next = nc + a_r
            errs = {}
            ssm_col_err = []
            for j in range(a_r):
                b = int(row[0, j])
                got = self.ssm_spec[b].cpu()
                exp = self.ref_ssm[nc + j]
                e = (got - exp).abs().max().item()
                nn = got.isnan().sum().item()
                ssm_col_err.append((j, b, e, nn))
            errs["ssm"] = max(e for _, _, e, _ in ssm_col_err)
            errs["ssm_nan"] = sum(nn for _, _, _, nn in ssm_col_err)
            got_c = self.conv_spec[cb, :, a_r - 1:a_r + KW - 2].cpu()
            exp_c = self.ref_conv[nc_next - 1]
            errs["conv"] = (got_c.float() - exp_c.float()).abs().max().item()
            errs["conv_nan"] = got_c.isnan().sum().item()
            o_ce = 0.0
            o_se = 0.0
            for j in range(a_r):
                o_ce = max(o_ce, (cout[j].float().cpu()
                                  - self.ref_conv_out[nc + j]).abs()
                           .max().item())
                o_se = max(o_se, (sout[0, j].float().cpu()
                                  - self.ref_ssm_out[nc + j]).abs()
                           .max().item())
            errs["cout"] = o_ce
            errs["sout"] = o_se

            bad = (errs["ssm"] > SSM_TOL or errs["conv"] > CONV_TOL
                   or errs["cout"] > 1e-2 or errs["sout"] > OUT_TOL
                   or errs["ssm_nan"] or errs["conv_nan"])
            rows.append(dict(
                r=r, nc=nc, q=q_len, ca=ca, start=start, src_col=s_col,
                src_off=s_off, na_pre=na_pre, na_fwd=na_fwd,
                copied=copy_desc, a=a_r, errs=errs,
                ssm_cols=ssm_col_err, conv_block=cb,
                map_snap=list(self.map_conv.get(cb, [None] * CONV_W)),
                bailed=bailed, fix_note=fix_note))
            if bad and first_bad is None:
                first_bad = rows[-1]

            # 5. acceptance scatter + bookkeeping
            self.num_accepted.fill_(a_r)
            nc = nc_next

            # 6. postprocess_align (real kernel): block-aligned save; resets
            #    na to 1 when src==dest (exact-boundary full-accept landing).
            self.num_computed.fill_(nc)
            self.na_pp.copy_(self.num_accepted)
            postprocess_mamba_fused_kernel[(1, 2, 16)](
                self.na_pp, self.state_idx, None, self.num_computed, None,
                self.bt_ptrs, self.bt.stride(0),
                self.base_addrs, self.block_strides, self.elem_sizes,
                self.inner_sizes, self.conv_widths, self.group_indices,
                self.dim_row_count, self.dim_row_stride,
                self.num_accepted, self.idx_map, 1,
                block_size=BS, COPY_BLOCK_SIZE=1024,
                CONV_STATE_DIM_FIRST=True, HAS_IDX_MAPPING=True,
                PRECOMPUTED_NEW_COMPUTED=True, TEMPORAL_TILES=16,
            )
            torch.cuda.synchronize()
            # Host mirror for the content map / trace.
            num_running = nc - a_r + 1
            aligned = (nc // BS) * BS
            pp_desc = "-"
            if aligned >= num_running:
                bias = aligned - num_running
                dest = aligned // BS - 1
                ssrc = int(self.bt[0, start + bias])
                dblk = int(self.bt[0, dest])
                pp_desc = (f"pp ssm b{ssrc}->b{dblk} conv b{int(self.bt[0, start])}"
                           f"->b{dblk} bias={bias}"
                           f"{' na->1' if dest == start else ''}")
                self._map_shift(int(self.bt[0, start]), dblk, bias)
            rows[-1]["pp"] = pp_desc

        ok = first_bad is None
        print(f"  [{'PASS' if ok else 'FAIL'}] {self.scen['name']}")
        hdr = ("  r |  nc  q  ca | start src+off | na(pre->fwd) | precopy |"
               " postprocess |  a | ssm_err conv_err cout_err sout_err nan")
        print(hdr)
        for d in rows:
            e = d["errs"]
            flag = ""
            if first_bad is not None and d["r"] == first_bad["r"]:
                flag = "  <-- FIRST DIVERGENCE"
            bail = " BAIL" if d["bailed"] else ""
            print(f"  {d['r']:2d}| {d['nc']:3d} {d['q']:2d} {d['ca']:3d} |"
                  f"  {d['start']:2d}   {d['src_col']:2d}+{d['src_off']:2d} |"
                  f"    {d['na_pre']:2d} -> {d['na_fwd']:2d}   |"
                  f" {d['copied'][:24]:24s}| {d.get('pp', '-')[:30]:30s}|"
                  f" {d['a']:2d} |"
                  f" {e['ssm']:.2e} {e['conv']:.2e} {e['cout']:.2e}"
                  f" {e['sout']:.2e} n{e['ssm_nan']+e['conv_nan']}"
                  f"{bail}{d['fix_note']}{flag}")
        if first_bad is not None:
            d = first_bad
            print(f"  --- first divergence at round {d['r']} ---")
            if d["bailed"]:
                print(f"  MECHANISM: conv spec-mode guard fired: "
                      f"num_accepted={d['na_fwd']} > query_len={d['q']} with "
                      f"no boundary crossing to reset it. "
                      f"causal_conv1d_update zeroed the round outputs and "
                      f"SKIPPED the conv-state write; the rolling buffer "
                      f"still holds the previous round's optimistic window "
                      f"(rejected-draft garbage), which later rounds fold "
                      f"into committed outputs.")
            bad_cols = [(j, b, e, nn) for j, b, e, nn in d["ssm_cols"]
                        if e > SSM_TOL or nn]
            for j, b, e, nn in bad_cols:
                print(f"    ssm col {j} block b{b}: err={e:.3e} nan={nn}")
            if d["errs"]["conv"] > CONV_TOL or d["errs"]["conv_nan"]:
                print(f"    conv window err={d['errs']['conv']:.3e} "
                      f"nan={d['errs']['conv_nan']} "
                      f"(buffer map at divergence: {d['map_snap']})")
        return ok


def main():
    overall = True
    for make in (scenario_control, scenario_oscillation,
                 scenario_collapse_no_cross):
        for seed in (11, 23, 37):
            scen = make(seed)
            tag = " (probe — informational, not gated)" \
                if scen.get("probe") else ""
            print(f"scenario={scen['name']} seed={seed} FIX_MODE={FIX_MODE}{tag}")
            ok = Harness(scen, seed).run()
            if not scen.get("probe"):
                overall &= ok
            print()
    print("OVERALL (gated scenarios):", "PASS" if overall else "FAIL")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
