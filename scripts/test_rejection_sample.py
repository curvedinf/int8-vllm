#!/usr/bin/env python3
"""Offline audit of the spec-verify rejection_sample kernels.

Checks, with temp=1.0 and NS=13:
 1. determinism: identical inputs+seeds -> identical output.
 2. different seeds -> different draws (no frozen RNG).
 3. uniform-logits bias: emitted tokens across many trials must be
    ~uniform; repeated-token emission far above chance = correlated RNG
    (the 'ductductduct' signature).
 4. greedy (temp~0): emitted chain == argmax-vs-draft reference.
 5. analytic acceptance: mean accepted length vs sum(p_t*p_d) collision.
"""
import sys

import torch

sys.path.insert(0, ".")
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

torch.manual_seed(0)
DEV = "cuda"
NS = 13
V = 8192
R = 2  # requests
L = R * (NS + 1)


def make_inputs(temp=1.0, seed_a=1, seed_b=2, uniform=False):
    if uniform:
        tl = torch.zeros(L, V, device=DEV, dtype=torch.float32)
        dl = torch.zeros(R, NS, V, device=DEV, dtype=torch.float32)
    else:
        tl = torch.randn(L, V, device=DEV, dtype=torch.float32)
        dl = torch.randn(R, NS, V, device=DEV, dtype=torch.float32)
    # draft tokens: draw from the draft distribution
    dsm = torch.softmax(dl / temp, dim=-1)
    draft_tok = (
        dsm.reshape(R * NS, V)
        .multinomial(1, replacement=True)
        .view(R, NS)
        .to(torch.int64)
        .flatten()
    )
    cu = torch.arange(0, L + 1, NS + 1, device=DEV, dtype=torch.int32)
    pos = (
        torch.arange(NS + 1, device=DEV, dtype=torch.int64)
        .unsqueeze(0)
        .repeat(R, 1)
        .flatten()
    )
    idx = torch.arange(R, device=DEV, dtype=torch.int32)
    e_idx = idx.repeat_interleave(NS + 1)
    e_pos = (
        torch.arange(NS + 1, device=DEV, dtype=torch.int32)
        .unsqueeze(0)
        .repeat(R, 1)
        .flatten()
    )
    temp_t = torch.full((R,), temp, device=DEV, dtype=torch.float32)
    seed = torch.tensor([seed_a, seed_b], device=DEV, dtype=torch.int64)
    return tl, dl, draft_tok, cu, pos, idx, e_idx, e_pos, temp_t, seed


def run(**kw):
    tl, dl, dsm, cu, pos, idx, e_idx, e_pos, tt, sd = make_inputs(**kw)
    s, n = rejection_sample(
        tl, dl, dsm, cu, pos, idx, e_idx, e_pos, tt, sd, NS,
        None, use_fp64=True,
    )
    return s.cpu(), n.cpu()


def emitted_rows(sampled):
    out = []
    for row in sampled.tolist():
        r = []
        for t in row:
            if 0 <= t < 200000:
                r.append(t)
            else:
                break
        out.append(r)
    return out


# 1+2: determinism / seed sensitivity
s1, n1 = run(temp=1.0)
s2, n2 = run(temp=1.0)
print("determinism:", "PASS" if torch.equal(s1, s2) and torch.equal(n1, n2)
      else "FAIL")
s3, _ = run(temp=1.0, seed_a=99, seed_b=98)
print("seed-sensitivity:", "PASS" if not torch.equal(s1, s3) else "FAIL")

# 3: uniform bias — emitted first-token distribution over trials
from collections import Counter

cnt = Counter()
accs = []
for k in range(60):
    sk, nk = run(temp=1.0, uniform=True, seed_a=1000 + k, seed_b=2000 + k)
    rows = emitted_rows(sk)
    for r in rows:
        if r:
            cnt[r[0]] += 1
    accs.extend(nk.tolist()[:R])
top = cnt.most_common(5)
n_tot = sum(cnt.values())
chi = sum(c * c for _, c in cnt.items()) / n_tot
print(f"uniform-first-token: draws={n_tot} distinct={len(cnt)} "
      f"top5={top} E[chi2/n]={chi:.3f} (uniform~{1 + 1 / max(len(cnt),1):.5f})")

# 4: greedy
sg, ng = run(temp=0.001)
tl, dl, dsm, cu, pos, idx, e_idx, e_pos, tt, sd = make_inputs(temp=0.001)
# reference: argmax chain vs drafts
rows = emitted_rows(sg)
ok = True
targ_am = tl.view(R, NS + 1, V).argmax(-1).cpu()
for r in range(R):
    exp = []
    for i in range(NS):
        d = dsm.view(R, NS).cpu()[r, i].item()
        t = targ_am[r, i].item()
        if t == d:
            exp.append(t)
        else:
            exp.append(t)
            break
    else:
        exp.append(targ_am[r, NS].item())
    got = rows[r][: len(exp)]
    if got != exp:
        ok = False
        print(f"  greedy req{r}: got={got} exp={exp}")
print("greedy-vs-argmax:", "PASS" if ok else "FAIL")

# 5: analytic acceptance on uniform logits: P(collision)=1/V per draft
mean_acc = sum(accs) / len(accs)
print(f"mean accepted (uniform, V={V}): {mean_acc:.3f} "
      f"(geometric expectation ~{1 + 1 / (V - 1):.4f} +1..{2:.2f} cap NS+1={NS+1})")
