#!/usr/bin/env python3
"""Decisive trajectory-conditional fidelity for rejection_sample at NS=1.

Measures the empirical P(commit R | current run length k) over chained rounds
and compares against the analytic dist(k)(R). If the engine is exact, the
empirical conditional == the analytic at every k (within sampling error).
A deviation = the engine's joint trajectory distribution deviates.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "../aiter")
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample  # noqa

DEV = "cuda"
V = 1024
R = 777
OTHERS = list(range(100, 118))


def dist(k, draft=False):
    pr = min(0.90, 0.35 * 1.25 ** min(k, 40))
    if draft:
        pr = min(0.95, pr * 1.25 + 0.03)
    rest = (1 - pr) * torch.tensor([1.0 / (i + 1) for i in range(18)], device=DEV)
    rest = rest / rest.sum() * (1 - pr)
    v = torch.full((V,), -30.0, device=DEV)
    v[R] = torch.log(torch.tensor(pr, device=DEV))
    v[OTHERS] = torch.log(rest)
    return v


def main():
    torch.cuda.init()
    NS = 1
    cond_tot, cond_hit = {}, {}
    commit_counts = []
    for traj in range(80):
        k = 0
        seeds_t = torch.tensor([1000 + traj], device=DEV, dtype=torch.int64)
        temp = torch.ones(1, device=DEV)
        for rnd in range(1500):
            rows = [dist(k), dist(k + 1)]
            tl_batch = torch.stack(rows).float()
            dl = torch.full((1, NS, V), -float("inf"), device=DEV)
            q = dist(k, draft=True)
            lq = q - torch.logsumexp(q, -1)
            d = torch.multinomial(lq.exp(), 1).item()
            sup = torch.topk(lq, 16).indices
            dl[0, 0, sup] = lq[sup]
            dsm = torch.tensor([0, d], device=DEV, dtype=torch.int64)
            cu = torch.tensor([0, NS + 1], device=DEV, dtype=torch.int32)
            pos = torch.arange(1000, 1000 + NS + 1, device=DEV)
            idx = torch.zeros(1, device=DEV, dtype=torch.int32)
            e_idx = torch.zeros(NS + 1, device=DEV, dtype=torch.int32)
            e_pos = torch.arange(NS + 1, device=DEV, dtype=torch.int32)
            s, n = rejection_sample(
                tl_batch, dl, dsm, cu, pos, idx, e_idx, e_pos, temp, seeds_t,
                NS, None, use_fp64=True, use_block_verification=False)
            nn = int(n.item())
            committed = s[0, :nn].cpu().tolist()
            commit_counts.append(nn)
            for t in committed:
                cond_tot[k] = cond_tot.get(k, 0) + 1
                if t == R:
                    cond_hit[k] = cond_hit.get(k, 0) + 1
                k = k + 1 if t == R else 0
    print("mean committed/round:", float(np.mean(commit_counts)))
    print("k | n | empirical P(R|k) | analytic dist(k)(R) | ratio")
    for kk in sorted(cond_tot):
        if cond_tot[kk] >= 100:
            emp = cond_hit[kk] / cond_tot[kk]
            ana = min(0.90, 0.35 * 1.25 ** min(kk, 40))
            print(f"{kk:4d} | {cond_tot[kk]:6d} | {emp:.4f} | {ana:.4f} | {emp/ana:.3f}")


if __name__ == "__main__":
    main()
