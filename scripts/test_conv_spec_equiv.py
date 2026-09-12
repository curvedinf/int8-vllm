#!/usr/bin/env python3
"""Spec-mode causal-conv ground-truth probe (TRUE engine geometry).

2026-09-12: an earlier version of this probe convicted the spec branch
(a>=2 output mismatch). That was the probe's own error, twice over:
1. STATE_LEN=24 did not match the engine's rolling width
   L = KW - 1 + K (shape funcs pass num_spec = num_speculative_tokens),
   so the kernel's write split landed at VAL = L - N = KW - 2 (overlap
   head + x at [KW-2, L)), not at M as the probe assumed.
2. Its reference compared a steady-round spec call (window [a-1, a+KW-3])
   against a fresh-buffer sequential decode (window [0, KW-2]) — different
   protocol contexts, expected to differ for a >= 2.

Under the real geometry the protocol is exact: the read window
[a-1, a-1+KW-2] ends at col KW-2+(a-1) = x_prev[a-1] = d_{a-1}, the last
committed token before the next anchor. Verified at three levels:
- scripts/test_conv_protocol_sim.py (protocol algebra, 500 rounds, ±mig)
- brute-force window identification: row-0 output matches FIR over
  buf[:, a-1 : a+KW-3] ++ q0 for a in {1,4,7}
- this probe: full-batch outputs vs exact FIR over the true stream.

Ground truth per acceptance a: committed stream = hist ++ x_prev[:a]
(row 0 of this round's queries = the correction/anchor); every output row
must equal the causal FIR over (stream ++ queries).

NOTE: conv_state line 0 is the null block — the kernel early-returns on
it, so the probe uses line 1.

Usage: HIP_VISIBLE_DEVICES=0 python scripts/test_conv_spec_equiv.py
"""
import sys

import torch

sys.path.insert(0, "/home/curved/vllm-gfx908")
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (  # noqa
    causal_conv1d_update,
)

dev = "cuda"
torch.manual_seed(0)

DIM, KW, K = 64, 4, 6
N = 1 + K                 # draft batch rows (anchor + K drafts)
L = KW - 1 + K            # engine rolling width = 9 for KW=4, K=6

g = torch.Generator(device=dev).manual_seed(9)

weight = torch.randn(DIM, KW, generator=g, device=dev) * 0.2
bias = torch.randn(DIM, generator=g, device=dev) * 0.1
HIST = 8
hist = torch.randn(HIST, DIM, generator=g, device=dev) * 0.5
x_prev = torch.randn(N, DIM, generator=g, device=dev) * 0.5  # [anchor,d1..dK]
queries = torch.randn(N, DIM, generator=g, device=dev) * 0.5  # [c,e1..e6]


def fir(tok):  # tok: [T, dim] -> [T, dim], causal conv KW, no activation
    T = tok.shape[0]
    pad = torch.zeros(KW - 1, DIM, device=dev)
    tp = torch.cat([pad, tok])
    out = torch.zeros(T, DIM, device=dev)
    for i in range(KW):
        out += tp[i : T + i] * weight[:, i]
    return out + bias


def seed_buffer():
    """Engine steady-round layout: cols [0, KW-2) = the KW-2 stream tokens
    before x_prev[0]; cols [KW-2, L) = x_prev (x_prev[i] at col KW-2+i)."""
    buf = torch.zeros(2, DIM, L, device=dev)
    buf[1, :, : KW - 2] = hist[-(KW - 2) :].T
    buf[1, :, KW - 2 :] = x_prev.T
    return buf


def run_spec(buf_seed, a):
    buf = buf_seed.clone()                        # [2, dim, L]
    x = queries.clone()                           # [N, dim]
    idx = torch.ones(1, dtype=torch.int32, device=dev)  # line 1 (0 = null)
    qsl = torch.tensor([0, N], dtype=torch.int32, device=dev)
    na = torch.tensor([a], dtype=torch.int32, device=dev)
    causal_conv1d_update(x, buf, weight, bias, activation=None,
                         conv_state_indices=idx,
                         num_accepted_tokens=na,
                         query_start_loc=qsl,
                         max_query_len=N,
                         validate_data=False)
    return x, buf[1]


print(f"DIM={DIM} KW={KW} K={K} N={N} L={L}")
print(f"{'a':>2} {'out_max|d|':>12} {'out_mean|d|':>12}")

any_fail = False
for a in range(1, N + 1):
    spec_outs, _ = run_spec(seed_buffer(), a)
    # committed truth before this round's queries: hist ++ x_prev[:a]
    stream_true = torch.cat([hist, x_prev[:a]])
    full = fir(torch.cat([stream_true, queries]))
    ref = full[stream_true.shape[0] :]
    do = (spec_outs - ref).abs()
    fail = do.max().item() > 1e-3
    any_fail |= fail
    print(f"{a:>2} {do.max().item():12.6f} {do.mean().item():12.6f}"
          f"{'   <-- MISMATCH' if fail else ''}", flush=True)

print("\nVERDICT:",
      "SPEC CONV MISMATCH vs exact FIR"
      if any_fail else
      "spec conv outputs == exact FIR over the true committed stream "
      "(all rows, all a) — spec conv rolling protocol EXONERATED")
