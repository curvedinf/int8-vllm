#!/usr/bin/env python3
"""Engine-faithful joint drafter+verifier differential (pass 55).

Single-source acceptance: the walk picks d via the (seed,pos,x) gumbel
lattice (exactly like _selector_walk_kernel), then rejection_sample does
BOTH acceptance and resample with the same (seed,pos) stream (exactly like
the engine). The committed-token histogram over many trials must match the
target distribution p. Variants:
  dense-q  : draft logits dense over V
  sparse-q : draft logits sparse (top-k support only, -inf elsewhere) —
             the engine's cache layout
"""
import sys

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "../aiter")
from vllm.triton_utils import tl, triton  # noqa
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (  # noqa
    rejection_sample,
)


@triton.jit
def _walk_pick_kernel(
    scores_ptr, out_ptr, seed_t, pos_t, V, BLOCK: tl.constexpr
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


def main():
    torch.manual_seed(5)
    DEV = "cuda"
    V = 256
    N = 20000
    NS = 1  # single draft step: rows = [draft-check, bonus]
    logits = torch.full((V,), -6.0, device=DEV)
    logits[7] = 4.0
    logits[50] = 2.0
    logits[130] = 1.0
    p = torch.softmax(logits, -1)
    dlogits = torch.full((V,), -6.5, device=DEV, dtype=torch.float32)
    dlogits[7] = 3.0
    dlogits[50] = 2.2
    dlogits[130] = 1.4

    seeds = torch.randint(1, 2**31 - 1, (N,), device=DEV, dtype=torch.int32)
    poss = torch.randint(0, 100000, (N,), device=DEV, dtype=torch.int32)
    d_idx = torch.empty(N, device=DEV, dtype=torch.int64)
    _walk_pick_kernel[(N,)](dlogits, d_idx, seeds, poss, V, BLOCK=256)

    for variant in ("dense", "sparse"):
        if variant == "sparse":
            topk = torch.topk(dlogits, 16)
            base = torch.full((NS, V), -float("inf"), device=DEV)
            base.scatter_(-1, topk.indices.unsqueeze(0), topk.values.unsqueeze(0))
            dl = base.unsqueeze(0).expand(N, NS, V).contiguous()
        else:
            dl = dlogits.view(1, 1, V).expand(N, NS, V).contiguous()
        tl_batch = logits.unsqueeze(0).expand(N * (NS + 1), V).contiguous()
        # kernel convention: dsm = [anchor, d1, ...] — the verified draft
        # for logit row 0 is dsm[1] (the walk's pick). Anchor is unused at
        # NS=1 for the marginal test.
        dsm = torch.stack(
            [torch.zeros(N, device=DEV, dtype=torch.int64), d_idx], 1
        ).flatten()
        cu = torch.arange(0, N * (NS + 1) + 1, NS + 1, device=DEV,
                          dtype=torch.int32)
        pos_b = poss.repeat_interleave(NS + 1).to(torch.int64)
        idx = torch.arange(N, device=DEV, dtype=torch.int32)
        e_idx = idx.repeat_interleave(NS + 1)
        e_pos = torch.arange(NS + 1, device=DEV,
                             dtype=torch.int32).repeat(N)
        temp = torch.ones(N, device=DEV, dtype=torch.float32)
        s, n = rejection_sample(
            tl_batch, dl, dsm, cu, pos_b, idx, e_idx, e_pos, temp,
            seeds.to(torch.int64), NS, None, use_fp64=True,
        )
        rows = s.view(N, NS + 1)
        ns = n.tolist()
        # accepted: row0 = the committed draft; rejected: row0 = resample.
        # Either way the first committed token is rows[i, 0] (Leviathan).
        committed = rows[:, 0].clone()
        hist = torch.bincount(committed, minlength=V).float() / N
        err = (hist - p).abs()
        top = torch.topk(err, 3)
        # expected committed dist = p (spec theorem); acceptance analytic
        q = torch.softmax(dlogits, -1)
        exp_acc = torch.minimum(p, q).sum().item()
        print(f"{variant}: max|hist-p|={err.max().item():.4f} "
              f"top3={[round(x,4) for x in top.values.tolist()]}@{top.indices.tolist()} "
              f"hist[7]={hist[7].item():.3f} p[7]={p[7].item():.3f} "
              f"mean_n={sum(ns)/N:.3f} analytic_acc={exp_acc:.3f}")


if __name__ == "__main__":
    main()
