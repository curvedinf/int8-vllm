#!/usr/bin/env python3
"""Reference differential for causal_conv1d_update under spec-decode rewind.

STATUS: WIP — the REFERENCE is wrong. It diverges even at accept=13 (full
accept, no rewind), which contradicts clean greedy production behavior, so
the divergence indicts the reference's buffer model, not the kernel. The
kernel comment's acceptance semantics (accept k => [h_{k+1}..hM, d1..dk],
i.e. shift-left-by-k of the drafted buffer) is not what this reference
implements. Fix the reference before trusting any result from this file.

Drives the real kernel through the exact DFlash step shape (14 query rows,
variable accepted counts with rollback) and compares the maintained conv
state and outputs against a dense reference that recomputes the rolling
buffer in plain PyTorch. A conv-state divergence under rewind would feed
corrupt queries to every later step — target-degrades-first.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)

DEV = "cuda"
DT = torch.float16
DIM = 64
WIDTH = 4            # conv kernel width
NS = 13              # speculative tokens
ROWS = NS + 1
STATE_LEN = WIDTH - 1 + NS
QM = 127  # unused; conv is fp

torch.manual_seed(0)


def reference_step(conv_state, x_rows, accept):
    """Dense reference matching the kernel's documented semantics.

    The buffer entering a step holds the last STATE_LEN committed inputs
    (history). The step processes R rows starting from the buffer's tail
    (the conv window for row t spans [hist_tail, x_0..x_t]). After the
    step, the drafted buffer is [hist, x_0..x_{R-1}] truncated to the last
    STATE_LEN entries; accepting k rows (num_accepted = k, anchor counted)
    commits x_0..x_{k-1}, so the next buffer is that drafted buffer
    shifted left by k, again truncated to the last STATE_LEN entries.
    conv_state: [dim, state_len]; x_rows: [rows, dim]."""
    d = conv_state.shape[0]
    hist = conv_state  # [d, STATE_LEN] committed history
    seq = torch.cat([hist, x_rows.t().float()], dim=1)  # [d, S+R]
    w = ref_weight.float()
    outs = []
    for t in range(x_rows.shape[0]):
        window = seq[:, STATE_LEN + t - (WIDTH - 1): STATE_LEN + t + 1]
        o = (w * window).sum(dim=1)
        if REF_BIAS is not None:
            o = o + REF_BIAS.float()
        if REF_ACT:
            o = torch.nn.functional.silu(o)
        outs.append(o)
    outs = torch.stack(outs, dim=1)  # [d, R]
    drafted = seq  # full [hist, drafts]
    committed = drafted[:, accept:]  # shift left by k (accept = num_accepted)
    L = committed.shape[1]
    if L >= STATE_LEN:
        new_state = committed[:, L - STATE_LEN:]
    else:
        new_state = torch.cat(
            [committed,
             torch.zeros(d, STATE_LEN - L, device=committed.device)], dim=1)
    return outs, new_state


# Setup kernel-side tensors
weight = (torch.randn(DIM, WIDTH, device=DEV) * 0.2).to(DT)
bias = (torch.randn(DIM, device=DEV) * 0.1).to(DT)
ref_weight = weight.float()
REF_BIAS = bias
REF_ACT = True

conv_state = torch.zeros(1, DIM, STATE_LEN, dtype=DT, device=DEV)
# seed some history
conv_state[0, :, :WIDTH - 1] = (torch.randn(DIM, WIDTH - 1, device=DEV) * 0.5).to(DT)
ref_state = conv_state[0].float().t().contiguous()  # [state_len? no] -> want [d, state_len]
ref_state = conv_state[0].float()  # [DIM, STATE_LEN]

indices = torch.zeros(1, dtype=torch.int32, device=DEV)
cu = torch.tensor([0, ROWS], dtype=torch.int32, device=DEV)

ok = True
for step, accept in enumerate([13, 3, 0, 7, 11, 1, 13, 5]):
    x = (torch.randn(ROWS, DIM, device=DEV) * 0.5).to(DT)
    num_acc = torch.tensor([accept + 1], dtype=torch.int32, device=DEV)
    out = causal_conv1d_update(
        x, conv_state, weight, bias, activation="silu",
        conv_state_indices=indices,
        num_accepted_tokens=num_acc,
        query_start_loc=cu, max_query_len=ROWS,
    )
    # reference
    r_out, r_state = reference_step(ref_state, x.float(), accept)
    # compare kernel conv state (layout [1, dim, state_len]) vs ref [d, s]
    k_state = conv_state[0].float()
    d_state = (k_state - r_state).abs().max().item()
    r_out_t = r_out.t()  # [R, d]
    o_diff = (out.float() - r_out_t).abs().max().item()
    line = f"step{step} accept={accept:2d}: state_err={d_state:.4f} out_err={o_diff:.4f}"
    good = d_state < 0.02 and o_diff < 0.02
    ok &= good
    print(line + ("" if good else "  <-- DIVERGENCE"))

print("OVERALL:", "PASS" if ok else "FAIL (conv state diverges under rewind)")
sys.exit(0 if ok else 1)
