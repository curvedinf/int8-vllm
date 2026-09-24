#!/usr/bin/env python3
"""Validate + bench the GDN exact-prefill final-store-only path (GOALOPT).

Runs fused_sigmoid_gating_delta_rule_update at the chunked-prefill exact
geometry (1 sequence, T tokens, constant ssm_state_indices mapping) with
and without STORE_FINAL_STATE_ONLY; requires bitwise-equal outputs and
final pool line, and reports per-call time.
"""
import os
import sys
import time

import torch

sys.path.insert(0, "/home/curved/vllm-gfx908")

from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (  # noqa: E501
    fused_sigmoid_gating_delta_rule_update,
)

dev = "cuda"
torch.manual_seed(0)

# Production geometry per rank (Qwen3.8-27B GDN at TP4): 4 key heads,
# 12 value heads, K/V head dims 128.
H, HV, K, V = 4, 12, 128, 128
POOL = 64
T = int(os.environ.get("GDN_T", "2048"))

A_log = torch.randn(HV, device=dev, dtype=torch.float32) * 0.1 - 1.0
a = torch.randn(T, HV, device=dev, dtype=torch.bfloat16) * 0.3
b = torch.randn(T, HV, device=dev, dtype=torch.bfloat16)
dt_bias = torch.randn(HV, device=dev, dtype=torch.float32) * 0.1
q = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, T, HV, V, device=dev, dtype=torch.bfloat16)
cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
SLOT = 7
si = torch.full((1, T), SLOT, device=dev, dtype=torch.int32)


def run(flag: bool, pool0: torch.Tensor | None = None):
    pool = (
        pool0.clone()
        if pool0 is not None
        else torch.randn(POOL, HV, V, K, device=dev, dtype=torch.float32) * 0.05
    )
    pool_pre = pool.clone()
    o, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=a, b=b, dt_bias=dt_bias,
        q=q, k=k, v=v,
        initial_state=pool, inplace_final_state=True,
        cu_seqlens=cu, ssm_state_indices=si,
        num_accepted_tokens=None, use_qk_l2norm_in_kernel=True,
        store_final_state_only=flag,
    )
    torch.cuda.synchronize()
    return o, pool, pool_pre


if __name__ == "__main__":
    # Bit-exactness: per-token stores vs final-only must match on o and pool
    # when both arms start from the same initial pool content.
    pool0 = torch.randn(POOL, HV, V, K, device=dev, dtype=torch.float32) * 0.05
    o_ref, pool_ref_after, _ = run(False, pool0)
    o_new, pool_new_after, pool_pre_new = run(True, pool0)
    same_o = torch.equal(o_ref, o_new)
    same_pool = torch.equal(pool_ref_after, pool_new_after)
    untouched = torch.equal(
        pool_new_after.clone().view(-1)[: SLOT * HV * V * K],
        pool_pre.clone().view(-1)[: SLOT * HV * V * K],
    ) if False else None  # pool rows other than SLOT: compare full tensors
    other_rows_same = torch.equal(
        torch.cat([pool_new_after[:SLOT], pool_new_after[SLOT + 1:]]),
        torch.cat([pool_pre_new[:SLOT], pool_pre_new[SLOT + 1:]]),
    )
    print(f"o bit-exact: {same_o}")
    print(f"pool line bit-exact: {same_pool}")
    print(f"other pool rows untouched: {other_rows_same}")
    assert same_o and same_pool and other_rows_same

    def bench(flag, iters=20, warmup=5):
        pool = torch.randn(POOL, HV, V, K, device=dev, dtype=torch.float32)
        for _ in range(warmup):
            fused_sigmoid_gating_delta_rule_update(
                A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v,
                initial_state=pool, inplace_final_state=True,
                cu_seqlens=cu, ssm_state_indices=si,
                num_accepted_tokens=None, use_qk_l2norm_in_kernel=True,
                store_final_state_only=flag,
            )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fused_sigmoid_gating_delta_rule_update(
                A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v,
                initial_state=pool, inplace_final_state=True,
                cu_seqlens=cu, ssm_state_indices=si,
                num_accepted_tokens=None, use_qk_l2norm_in_kernel=True,
                store_final_state_only=flag,
            )
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    t_ref = bench(False)
    t_new = bench(True)
    print(f"per-call ms T={T}: per-token-stores={t_ref:.3f} "
          f"final-only={t_new:.3f} speedup={t_ref / t_new:.2f}x")
