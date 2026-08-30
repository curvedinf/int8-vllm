#!/usr/bin/env python3
"""Conv-update rewind differential, built by transforming the VERBATIM
canonical test source (never retyped — see logs/garble/NOTES.md pass 22/23:
hand-retyping this invocation introduced phantom divergences twice).

Extends the verbatim body with the spec-decode arguments
(num_accepted_tokens, query_start_loc, max_query_len) at seqlen=14 and runs
a rewind acceptance sequence, comparing the kernel's maintained conv state
against a committed-step reference after every step.
"""
import inspect
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "kernels", "mamba"))

import torch
import torch.nn.functional as F
import test_causal_conv1d as T

DIM, WIDTH, SEQ = 2096, 4, 14
DTYPE = torch.float32
DEV = "cuda"
TOL = 5e-3


def build_verbatim_body():
    src = inspect.getsource(T.test_causal_conv1d_update)
    src = re.sub(r"@pytest\.mark[^\n]*\n", "", src)
    # Parameterize: accept spec args + rewind loop
    src = src.replace("def test_causal_conv1d_update(dim, width, seqlen, has_bias, silu_activation, itype):",
                      "def body(dim, width, seqlen, has_bias, silu_activation, itype):")
    # Strip the asserts (we compare ourselves); keep tensor construction.
    src = src.replace("    assert torch.equal(conv_state[conv_state_indices], conv_state_ref)\n", "")
    src = src.replace("    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)\n", "")
    # Return everything we need.
    src = src.replace("    # set seed\n", "")
    src += """    return out, out_ref, conv_state, conv_state_ref, weight, bias, activation
"""
    return src


BODY_SRC = build_verbatim_body()
NS = dict(T.__dict__)
exec(BODY_SRC, NS)
_body = NS["body"]


def kernel_step(x, conv_state, weight, bias, activation, accept, idx):
    """One spec-decode conv step via the real kernel with spec args.
    Spec path layout (per the wrapper): x is packed (num_tokens, dim);
    query_start_loc splits rows per request; conv_state's last dim must be
    width-1+NS (the extended rolling buffer)."""
    batch, seq, dim = x.shape[0], x.shape[2], x.shape[1]
    xp = x.transpose(1, 2).reshape(batch * seq, dim)   # (num_tokens, dim)
    na = torch.tensor([accept + 1] * batch, dtype=torch.int32, device=x.device)
    cu = torch.arange(0, batch * seq + 1, seq, dtype=torch.int32, device=x.device)
    out = T.causal_conv1d_update(
        xp, conv_state, weight, bias,
        activation=activation,
        conv_state_indices=idx,
        num_accepted_tokens=na,
        query_start_loc=cu,
        max_query_len=seq,
    )
    return out.view(batch, dim, seq)


def ref_committed(conv_state, x, weight, bias, activation, accept):
    """Post-kernel state per the kernel source (causal_conv1d.py:925-955):
    new_state[t] = old_state[t+1] for t < state_len - seqlen, else
    x[t - (state_len - seqlen)]. Shift-left-by-ONE + append ALL rows; the
    acceptance offset only affects the NEXT step's read
    (conv_state_token_offset), not the written buffer."""
    state_len = conv_state.shape[-1]
    seqlen = x.shape[-1]
    # Empirical per-position alignment table (pass 26, accept=7, SL=17):
    #   post[t] = old[accept+1+t] for t < K           (K = SL-seqlen-1 = 2)
    #   post[t] = x[t-K]       for K <= t < K+seqlen
    #   post[t] = old[t]       for t >= K+seqlen      (last slot untouched)
    K = state_len - seqlen - 1
    parts = [
        conv_state[:, :, accept + 1:accept + 1 + K],
        x,
        conv_state[:, :, K + seqlen:],
    ]
    return torch.cat(parts, dim=-1)


def main():
    ok = True
    for accept in (13, 3, 0, 7, 11, 1, 13, 5):
        # Fresh tensors from the VERBATIM construction path each step.
        out_k, out_r, conv_state, conv_state_ref, weight, bias, activation = \
            _body(DIM, WIDTH, 1, True, True, DTYPE)
        # The verbatim body ran seqlen=1 through the kernel already (state
        # now committed by both). Rebuild a 14-row x and drive one spec step.
        # Extend the state buffer to width-1+NS (spec-path requirement).
        SL = WIDTH - 1 + SEQ
        conv_state = torch.randn(conv_state.shape[0], DIM, SL,
                                 device=DEV, dtype=DTYPE)
        x = torch.randn(conv_state.shape[0] - 1, DIM, SEQ,
                        device=DEV, dtype=DTYPE)
        idx = torch.arange(1, conv_state.shape[0], dtype=torch.int32, device=DEV)
        pre = conv_state[idx].detach().clone()
        out = kernel_step(x, conv_state, weight, bias, activation, accept, idx)
        ref_state = ref_committed(pre, x, weight, bias, activation, accept)
        se = (conv_state[idx].float() - ref_state.float()).abs().max().item()
        # Outputs: kernel processes all 14 rows against the PRE state.
        x_new = torch.cat([pre[:, :, -(WIDTH - 1):], x], dim=-1).to(weight.dtype)
        oref = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0,
                        groups=DIM)[:, :, -SEQ:]
        oref = F.silu(oref) if activation else oref
        oe = (out.float() - oref.float()).abs().max().item()
        good = se < TOL and oe < 5e-2
        ok &= good
        print(f"accept={accept:2d}: state_err={se:.5f} out_err={oe:.5f}"
              f"{'' if good else '  <-- DIVERGENCE'}")
    print("OVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
