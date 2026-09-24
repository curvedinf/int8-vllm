#!/usr/bin/env python3
"""Compare speculative convolution rewind with one-token recurrence."""

import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)


def main() -> None:
    torch.manual_seed(20260924)
    device = "cuda"
    n_tokens, window, dim, width = 512, 7, 256, 4
    source_len = n_tokens + window
    x = torch.randn(source_len, dim, device=device, dtype=torch.bfloat16)
    weight = torch.randn(dim, width, device=device, dtype=torch.bfloat16) * 0.1
    bias = torch.randn(dim, device=device, dtype=torch.bfloat16) * 0.1
    # The speculative row stores width-1 history plus six optimistic slots.
    state_spec = torch.zeros(2, dim, width - 1 + window - 1,
                             device=device, dtype=torch.bfloat16)
    state_ref = torch.zeros_like(state_spec)
    initial = torch.randn(dim, width - 1, device=device,
                          dtype=torch.bfloat16) * 0.1
    state_spec[1, :, :width - 1] = initial
    state_ref[1, :, :width - 1] = initial
    index = torch.tensor([1], device=device, dtype=torch.int32)
    cu = torch.tensor([0, window], device=device, dtype=torch.int32)
    pattern = (1, 2, 5, 7, 3, 6, 4, 1, 7, 2)
    max_out_diff = max_state_diff = 0.0
    first_large = None
    previous_accept = 1
    pos = rounds = 0
    while pos < n_tokens:
        accepted = torch.tensor([previous_accept], device=device,
                                dtype=torch.int32)
        out_spec = causal_conv1d_update(
            x[pos:pos + window].clone(), state_spec, weight, bias,
            activation="silu", conv_state_indices=index,
            num_accepted_tokens=accepted, query_start_loc=cu,
            max_query_len=window,
        )
        take = min(pattern[rounds % len(pattern)], n_tokens - pos)
        for i in range(take):
            out_ref = causal_conv1d_update(
                x[pos + i:pos + i + 1].clone(), state_ref, weight, bias,
                activation="silu", conv_state_indices=index,
            )
            diff = (out_spec[i].float() - out_ref[0].float()).abs()
            max_out_diff = max(max_out_diff, diff.max().item())
        state_diff = (
            state_spec[1, :, take - 1:take + width - 2].float()
            - state_ref[1, :, :width - 1].float()
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
    assert max_out_diff < 0.02 and max_state_diff < 0.002


if __name__ == "__main__":
    main()
