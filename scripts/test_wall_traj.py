#!/usr/bin/env python3
"""Trajectory-level differential for the wall garble (pass 72).

Chains the ENGINE's real rejection_sample over ~500 rounds/trajectory on a
synthetic repetition attractor, vs an exact single-token reference sampler
with identical per-position distributions. If the engine enters/ sustains
walls more often than the reference, the spec machinery (not the model) is
the class-B driver; if matched, walls are model behavior under exact
sampling.

Attractor dynamics (matched both arms): with a current run of k repeats of
token R in context, p(R) = min(0.995, 0.35 * 1.5**k) for the next token;
the remaining mass is Zipf over 18 other tokens. The drafter mimics p but
is sharper on R: q(R) = min(0.998, p(R)*1.7 + 0.05), top-16 support.
"""
import os
import numpy as np
import torch

from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

DEV = "cuda"
V = 1024
import os as _os
NS = int(_os.environ.get("NS", "13"))
R = 777  # the repeat token
OTHERS = list(range(100, 118))  # 18 non-repeat tokens


def dist(k, draft=False):
    """p (or q) over the 19-token support for run-length k. Returns logits."""
    pr = min(0.90, 0.35 * 1.25 ** min(k, 40))
    if draft:
        pr = min(0.95, pr * 1.25 + 0.03)
    rest = (1 - pr) * torch.tensor(
        [1.0 / (i + 1) for i in range(18)], device=DEV)
    rest = rest / rest.sum() * (1 - pr)
    v = torch.full((V,), -30.0, device=DEV)  # log prob floor ~ 1e-13
    v[R] = torch.log(torch.tensor(pr, device=DEV))
    v[OTHERS] = torch.log(rest)
    return v


def run_reference(M, rng):
    """Exact single-token sampler over M tokens."""
    k = 0
    maxrun, walls = 0, 0
    cur = 0
    for _ in range(M):
        lp = dist(k)
        lp = lp - torch.logsumexp(lp, -1)
        t = torch.multinomial(lp.exp(), 1, generator=None).item()
        if t == R:
            cur += 1
        else:
            if cur >= 20:
                walls += 1
            cur = 0
        maxrun = max(maxrun, cur)
        k = cur
    return maxrun, walls


def run_engine(M, seed):
    """The real rejection_sample driven round-by-round. Single request."""
    torch.manual_seed(seed)
    k = 0
    maxrun, walls, cur = 0, 0, 0
    wallrounds = []
    cond_tot, cond_hit = {}, {}
    seeds_t = torch.tensor([seed], device=DEV, dtype=torch.int64)
    temp = torch.ones(1, device=DEV)
    total = 0
    rnd = 0
    while total < M:
        rnd += 1
        # 14 verify rows: row j conditioned on run k + j (chain rise).
        # Chain-faithful: sample the draft prefix FIRST (q conditioned on the
        # actual drafted prefix run), then set each verify row's target
        # distribution conditioned on [committed prefix + d_1..d_j].
        draft_logits = torch.full((1, NS, V), -float("inf"), device=DEV)
        drafts = []
        run = k  # trailing run of R in [committed prefix + draft prefix]
        for s in range(NS):
            q = dist(run, draft=True)
            lq = q - torch.logsumexp(q, -1)
            d = torch.multinomial(lq.exp(), 1).item()
            drafts.append(d)
            sup = torch.topk(lq, 16).indices
            draft_logits[0, s, sup] = lq[sup]
            run = run + 1 if d == R else 0
        rows = []
        run = k
        for j in range(NS + 1):
            # row j tests d_{j+1}: conditioned on [prefix + d_1..d_j]; row 0 = anchor
            # (committed prefix only, run=k).
            rows.append(dist(run))
            if j < NS:
                run = run + 1 if drafts[j] == R else 0
        target_logits = torch.stack(rows).float()
        dsm = torch.tensor([0] + drafts, device=DEV, dtype=torch.int64)
        cu = torch.tensor([0, NS + 1], device=DEV, dtype=torch.int32)
        pos = torch.arange(1000, 1000 + NS + 1, device=DEV)
        idx = torch.zeros(1, device=DEV, dtype=torch.int32)
        e_idx = torch.zeros(NS + 1, device=DEV, dtype=torch.int32)
        e_pos = torch.arange(NS + 1, device=DEV, dtype=torch.int32)
        s, n = rejection_sample(
            target_logits, draft_logits, dsm, cu, pos, idx, e_idx, e_pos,
            temp, seeds_t, NS, None, use_fp64=True,
            use_block_verification=False,
            salt_u=bool(os.environ.get("SALT_U")), salt_resample=bool(os.environ.get("RSALT")),
        )
        nn = int(n.item())
        total += nn
        committed = s[0, :nn].cpu().tolist()
        # wall-round diagnostics: all-R draft prefix and committed all-R?
        allR_drafts = all(d == R for d in drafts)
        if k >= 20 and allR_drafts:
            # in a wall: did the run CONTINUE (all committed are R)?
            cont = all(t == R for t in committed)
            wallrounds.append((nn, cont))
        # conditional fidelity: P(commit R | run k at round start)
        for t in committed:
            cond_tot[k] = cond_tot.get(k, 0) + 1
            if t == R:
                cond_hit[k] = cond_hit.get(k, 0) + 1
        for t in committed:
            if t == R:
                cur += 1
            else:
                if cur >= 20:
                    walls += 1
                cur = 0
            maxrun = max(maxrun, cur)
            k = cur
    return maxrun, walls


def main():
    torch.cuda.init()
    M = 3000
    N = 60
    eng_runs, eng_walls = [], []
    for i in range(N):
        m, w = run_engine(M, 1000 + i)
        eng_runs.append(m)
        eng_walls.append(w)
    ref_runs, ref_walls = [], []
    for i in range(N):
        m, w = run_reference(M, None)
        ref_runs.append(m)
        ref_walls.append(w)
    er, rr = np.array(eng_runs), np.array(ref_runs)
    ew, rw = np.array(eng_walls), np.array(ref_walls)
    print(f"engine: maxrun mean={er.mean():.1f} p90={np.percentile(er,90):.0f} "
          f"max={er.max()} | walls>=20/run: {(ew>0).mean():.3f}")
    print(f"refer.: maxrun mean={rr.mean():.1f} p90={np.percentile(rr,90):.0f} "
          f"max={rr.max()} | walls>=20/run: {(rw>0).mean():.3f}")
    print(f"long-wall (>=100) rate: engine {(er>=100).mean():.3f} "
          f"vs reference {(rr>=100).mean():.3f}")
    # conditional fidelity dump from the LAST engine trajectory
    try:
        ct, ch = _cond_dump
        print("P(commit R | run k)  empirical-engine vs analytic-dist(k):")
        for k in sorted(ct):
            if ct[k] >= 30:
                emp = ch[k]/ct[k]
                ana = min(0.90, 0.35*1.25**min(k,40))
                print(f"  k={k:3d}: n={ct[k]:4d} emp={emp:.4f} analytic={ana:.4f} ratio={emp/ana:.3f}")
    except Exception as ex:
        print("cond dump err", ex)


if __name__ == "__main__":
    main()
