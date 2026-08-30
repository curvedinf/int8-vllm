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
    """Dense reference: given the rolling buffer [dim, state_len] holding
    [history..., drafts...] and a new 14-row block, accept `accept` tokens
    from the block's start, produce new state + outputs.
    conv_state: [dim, state_len]; x_rows: [rows, dim]."""
    d = conv_state.shape[0]
    # sequence the conv sees: state tail (history) + new rows
    hist = conv_state[:, : WIDTH - 1]  # only first width-1 are history
    seq = torch.cat([hist, x_rows.t().float()], dim=1)  # [d, W-1+R]
    w = ref_weight.float()
    outs = []
    for t in range(x_rows.shape[0]):
        window = seq[:, t:t + WIDTH]
        o = (w * window).sum(dim=1)
        if REF_BIAS is not None:
            o = o + REF_BIAS.float()
        if REF_ACT:
            o = torch.nn.functional.silu(o)
        outs.append(o)
    outs = torch.stack(outs, dim=1)  # [d, R]
    kept = accept + 1  # accepted drafts + anchor
    total = WIDTH - 1 + x_rows.shape[0]
    new_state = seq[:, total - STATE_LEN:total] if total >= STATE_LEN else \
        torch.cat([seq, torch.zeros(d, STATE_LEN - total, device=seq.device)], dim=1)
    # but only `kept` rows actually happened: state must reflect the
    # accepted prefix, not the drafted tail
    seq_kept = seq[:, : WIDTH - 1 + kept]
    t2 = seq_kept.shape[1]
    new_state = seq_kept[:, max(0, t2 - STATE_LEN):t2]
    if t2 < STATE_LEN:
        new_state = torch.cat([seq_kept, torch.zeros(d, STATE_LEN - t2, device=seq.device)], dim=1)
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
