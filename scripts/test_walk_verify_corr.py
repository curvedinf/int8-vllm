#!/usr/bin/env python3
"""Joint drafter+verifier differential: does the (seed,pos)-correlated
noise between _selector_walk_kernel and _rejection_kernel bias the
committed-token distribution away from the target's?

Protocol (NS=1 step, V small): for many seeds,
  1. drafter picks d = argmax(scores + noise(seed,pos,token))  [walk]
  2. verifier: accepted iff u(seed,pos) < p(d)/q(d)  [rejection kernel]
  3. on reject, resample from the residual via the verify's gumbel
     (seed,pos,token) lattice.
Commit-token histogram must match p. Arms:
  CURRENT: u from tl.rand(seed,pos) — same philox state as the walk's
           gumbel_seed = tl.randint(seed,pos).
  DECORR:  u from tl.rand(seed ^ 0x9E3779B9, pos) — independent stream.
"""
import sys

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "../aiter")
from vllm.triton_utils import tl, triton  # noqa


@triton.jit
def _walk_pick_kernel(
    scores_ptr, cand_ptr, out_ptr, seed_t, pos_t, V, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    seed = tl.load(seed_t + pid)
    pos = tl.load(pos_t + pid)
    gumbel_seed = tl.randint(seed, pos)
    best = -1e30
    best_x = 0
    for v0 in range(0, V, BLOCK):
        offs = v0 + tl.arange(0, BLOCK)
        m = offs < V
        sc = tl.load(scores_ptr + offs, mask=m, other=-1e30)
        u = tl.rand(gumbel_seed, offs)
        u = tl.maximum(u, 1e-10)
        g = -tl.log(-tl.log(u))
        val = tl.where(m, sc + g, -1e30)
        v_max = tl.max(val, axis=0)
        v_arg = tl.argmax(val, axis=0)
        if v_max > best:
            best = v_max
            best_x = v0 + v_arg
    tl.store(out_ptr + pid, best_x)


@triton.jit
def _u_kernel(out_ptr, seed_t, pos_t, N, DECORR: tl.constexpr):
    pid = tl.program_id(0)
    seed = tl.load(seed_t + pid)
    pos = tl.load(pos_t + pid)
    if DECORR:
        u = tl.rand(seed ^ 1509949171, pos)
    else:
        u = tl.rand(seed, pos)
    tl.store(out_ptr + pid, u)


def main():
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

    torch.manual_seed(5)
    DEV = "cuda"
    V = 256
    N = 20000  # trials
    # target distribution p over V (peaked on 7, spread elsewhere)
    logits = torch.full((V,), -6.0, device=DEV)
    logits[7] = 4.0
    logits[50] = 2.0
    logits[130] = 1.0
    p = torch.softmax(logits, -1)
    # drafter scores q: mildly different (shifted peak)
    dlogits = torch.full((V,), -6.5, device=DEV, dtype=torch.float32)
    dlogits[7] = 3.0
    dlogits[50] = 2.2
    dlogits[130] = 1.4
    q = torch.softmax(dlogits, -1)

    seeds = torch.randint(1, 2**31 - 1, (N,), device=DEV, dtype=torch.int32)
    poss = torch.randint(0, 100000, (N,), device=DEV, dtype=torch.int32)
    d_idx = torch.empty(N, device=DEV, dtype=torch.int64)
    _walk_pick_kernel[(N,)](dlogits, dlogits, d_idx, seeds, poss, V, BLOCK=256)

    for decorr in ("current", "decorr_u", "full"):
        u = torch.empty(N, device=DEV, dtype=torch.float32)
        if decorr == "full":
            u = torch.rand(N, device=DEV)  # fully independent acceptance
        else:
            _u_kernel[(N,)](u, seeds, poss, N, DECORR=(decorr == "decorr_u"))
        # acceptance per trial: u < p(d)/q(d)
        ratio = (p / q)[d_idx]
        accepted = u < ratio
        # resample on reject from the residual (verify's (seed,pos,x)
        # gumbel lattice) — use rejection_sample per rejected trial
        committed = d_idx.clone()
        rej = (~accepted).nonzero().flatten()
        if rej.numel():
            # build a batched call: each rejected trial = its own request
            L = rej.numel()
            tl_batch = logits.unsqueeze(0).expand(2 * L, V).contiguous()
            dl = dlogits.view(1, 1, V).expand(L, 1, V).contiguous()
            dsm = torch.stack([d_idx[rej], torch.zeros(L, device=DEV, dtype=torch.int64)], 1).flatten()
            cu = torch.arange(0, 2 * L + 1, 2, device=DEV, dtype=torch.int32)
            pos_b = poss[rej].repeat_interleave(2).to(torch.int64)
            idx = torch.arange(L, device=DEV, dtype=torch.int32)
            e_idx = idx.repeat_interleave(2)
            e_pos = torch.tensor([0, 1], device=DEV, dtype=torch.int32).repeat(L)
            temp = torch.ones(L, device=DEV, dtype=torch.float32)
            if decorr == "full":
                seed_b = (seeds[rej] ^ 1895158093).to(torch.int64)
            else:
                seed_b = seeds[rej].to(torch.int64)
            s, n = rejection_sample(
                tl_batch, dl, dsm, cu, pos_b, idx, e_idx, e_pos, temp, seed_b,
                1, None, use_fp64=True,
            )
            rows = s.view(L, 2)
            ns = n.tolist()
            committed[rej] = torch.tensor(
                [rows[i, ns[i] - 1].item() for i in range(L)], device=DEV
            )
        # committed histogram vs p
        hist = torch.bincount(committed, minlength=V).float() / N
        err = (hist - p).abs()
        top = torch.topk(err, 3)
        name = {"current": "CURRENT", "decorr_u": "DECORR-U", "full": "FULLDECORR"}[decorr]
        exp_acc = torch.minimum(p / q, torch.ones_like(p)).sum().item()
        print(f"{name}: max|hist-p|={err.max().item():.4f} "
              f"top-errs at {top.indices.tolist()} = {[round(x,4) for x in top.values.tolist()]} "
              f"| hist[7]={hist[7].item():.3f} p[7]={p[7].item():.3f} "
              f"| acc_rate={accepted.float().mean().item():.3f} "
              f"(analytic={exp_acc:.3f})")


if __name__ == "__main__":
    main()
