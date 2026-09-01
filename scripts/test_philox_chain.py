#!/usr/bin/env python3
"""Adjacent-position philox correlation on ROCm Triton (garble hunt, pass 76).

The wall's acceptance u's along a spec chain come from tl.rand(seed, pos) at
consecutive positions sharing the seed. If adjacent-counter philox outputs
correlate on this backend, chain acceptances burst => long walls.
"""
import sys

import torch

sys.path.insert(0, ".")
from vllm.triton_utils import tl, triton  # noqa


@triton.jit
def _adj_kernel(u0_ptr, u1_ptr, seeds_ptr, pos_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    seed = tl.load(seeds_ptr + offs)
    pos = tl.load(pos_ptr + offs)
    u0 = tl.rand(seed, pos)
    u1 = tl.rand(seed, pos + 1)
    tl.store(u0_ptr + offs, u0, mask=m)
    tl.store(u1_ptr + offs, u1, mask=m)


@triton.jit
def _chain_kernel(out_ptr, seeds_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    seed = tl.load(seeds_ptr + offs)
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for j in range(13):
        u = tl.rand(seed, 1000 + j)  # chain positions P0..P0+12
        acc += (u < 0.9).to(tl.int32)
    tl.store(out_ptr + offs, acc.to(tl.float32), mask=m)


def main():
    torch.cuda.init()
    N = 1 << 22
    seeds = torch.randint(1, 2**31 - 1, (N,), device="cuda", dtype=torch.int32)
    poss = torch.randint(0, 100000, (N,), device="cuda", dtype=torch.int32)
    u0 = torch.empty(N, device="cuda", dtype=torch.float32)
    u1 = torch.empty(N, device="cuda", dtype=torch.float32)
    _adj_kernel[(triton.cdiv(N, 1024),)](u0, u1, seeds, poss, N, BLOCK=1024)
    u0c, u1c = u0.cpu(), u1.cpu()
    corr = torch.corrcoef(torch.stack([u0c, u1c]))[0, 1].item()
    print(f"corr(u(pos), u(pos+1)) = {corr:+.5f} over {N} pairs")
    joint = ((u0c < 0.5) & (u1c < 0.5)).float().mean().item()
    print(f"P(u0<.5 & u1<.5) = {joint:.4f} (0.25 expected if independent)")

    acc = torch.empty(N, device="cuda", dtype=torch.float32)
    _chain_kernel[(triton.cdiv(N, 1024),)](acc, seeds, N, BLOCK=1024)
    a = acc.cpu()
    full = (a == 13).float().mean().item()
    indep = 0.9 ** 13
    print(f"P(all 13 chain u's < 0.9) = {full:.5f}  (0.9^13 = {indep:.5f} if independent)")
    # distribution of the count vs binomial(13, 0.9)
    import numpy as np
    from scipy import stats as _s  # may not exist; fall back
    try:
        exp = [len(a) * _s.binom(13, 0.9).pmf(k) for k in range(14)]
    except Exception:
        exp = None
    hist = torch.bincount(a.long(), minlength=14).tolist()
    print("chain-accept-count histogram (0..13):", hist)
    if exp:
        print("binomial(13, 0.9) expected:      ", [round(x) for x in exp])


if __name__ == "__main__":
    main()
