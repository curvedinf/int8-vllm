#!/usr/bin/env python3
"""Compare a speculative GDN state chain against one-token recurrence.

Every verify window reads the state left by the previous round's accepted
position. Rejected lookahead rows are recomputed in the next window. This
exercises changing acceptance lengths, which single-window kernel tests miss.
"""

import os

import torch

from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)


def main() -> None:
    torch.manual_seed(20260924)
    device = "cuda"
    dtype = torch.bfloat16
    n_tokens = int(os.environ.get("GDN_CHAIN_TOKENS", "512"))
    window, heads, kdim, vdim = 7, 4, 128, 256
    # Keep the optimistic verify window full even on the final round. A
    # production request can stop after committing fewer than seven tokens;
    # the last lookahead positions are still computed before the stop.
    source_len = n_tokens + window
    q = torch.randn(source_len, heads, kdim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(source_len, heads, vdim, device=device, dtype=dtype)
    a = torch.randn(source_len, heads, device=device, dtype=dtype) * 0.3
    b = torch.randn(source_len, heads, device=device, dtype=dtype)
    mixed_qkv = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=1)
    a_log = torch.randn(heads, device=device, dtype=torch.float32) - 1
    dt_bias = torch.randn(heads, device=device, dtype=torch.float32) * 0.1
    state_spec = torch.zeros(
        window + 1, heads, vdim, kdim, device=device, dtype=torch.float32
    )
    state_ref = torch.zeros((2, heads, vdim, kdim), device=device)
    initial = torch.randn_like(state_ref[1]) * 0.03
    state_spec[1].copy_(initial)
    state_ref[1].copy_(initial)
    ref_index = torch.tensor([1], device=device, dtype=torch.int32)
    out_ref = torch.empty(1, 1, heads, vdim, device=device, dtype=dtype)
    max_out_diff = max_state_diff = 0.0
    first_large = None
    rounds = 0
    pos = 0
    previous_accept = 1
    pattern = (1, 2, 5, 7, 3, 6, 4, 1, 7, 2)
    while pos < n_tokens:
        length = window
        indices = torch.arange(1, length + 1, device=device,
                               dtype=torch.int32).unsqueeze(0)
        cu = torch.tensor([0, length], device=device, dtype=torch.int32)
        accepted = torch.tensor([previous_accept], device=device,
                                dtype=torch.int32)
        out_spec, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=a_log, a=a[pos:pos + length], b=b[pos:pos + length],
            dt_bias=dt_bias, q=q[None, pos:pos + length],
            k=k[None, pos:pos + length], v=v[None, pos:pos + length],
            initial_state=state_spec, inplace_final_state=True,
            cu_seqlens=cu, ssm_state_indices=indices,
            num_accepted_tokens=accepted, use_qk_l2norm_in_kernel=True,
        )
        take = min(pattern[rounds % len(pattern)], n_tokens - pos)
        for i in range(take):
            cur = pos + i
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=mixed_qkv[cur:cur + 1],
                a=a[cur:cur + 1], b=b[cur:cur + 1],
                A_log=a_log, dt_bias=dt_bias, scale=kdim ** -0.5,
                initial_state=state_ref, out=out_ref,
                ssm_state_indices=ref_index,
                use_qk_l2norm_in_kernel=True,
            )
            diff = (out_spec[0, i].float() - out_ref[0, 0].float()).abs()
            max_out_diff = max(max_out_diff, diff.max().item())
        state_diff = (
            state_spec[take] - state_ref[1]
        ).abs().max().item()
        max_state_diff = max(max_state_diff, state_diff)
        if first_large is None and (state_diff > 0.002 or max_out_diff > 0.02):
            first_large = (rounds, pos, previous_accept, take,
                           max_out_diff, state_diff)
        previous_accept = take
        pos += take
        rounds += 1
    print(f"rounds={rounds} tokens={pos} max_out_diff={max_out_diff:.7f} "
          f"max_state_diff={max_state_diff:.7f} first_large={first_large}")
    assert max_out_diff < 0.02 and max_state_diff < 0.01


if __name__ == "__main__":
    main()
