#!/usr/bin/env python3
"""Chained in-vitro process test for the spec-decode rejection sampler.

Single-round marginals were exact (greedy one-hot drafts, top-k truncation).
 This
test CHAINS the real rejection_sample kernel round-by-round on a synthetic
LM with a known law and compares the emitted stream's statistics against
direct sampling — the configuration never tested (ACCEPT1 evidence: a
correction-only chain still garbles in vivo).

Target law: bigram LM over V=256 (softmax of a random matrix row), sampled
temperature 1.0 truncated to top-20 (production law). Draft law: noisy
copy, GREEDY (production draft_sample_method).

Chains measured:
  A. correction-only (ACCEPT1-like: commit only emitted[0] each round)
  B. natural (commit all emitted tokens each round, feed the last as next
     anchor)
Each compared against direct target sampling of equal length:
  - token unigram KL, bigram-conditioned KL (the true sufficient statistic
    for a bigram law), top-20 support violation rate.

Usage: HIP_VISIBLE_DEVICES=0 python scripts/test_rejection_chain.py
"""
import sys

import torch

sys.path.insert(0, "/home/curved/vllm-gfx908")
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (  # noqa
    rejection_sample,
)

DEV = torch.device("cuda:0")
V = 256
NS = 6
TOP_K = 20
N = 12000
torch.manual_seed(0)

# Target and draft laws: rows are softmax(T[i]) for previous token i.
Tm = torch.randn(V, V, device=DEV) * 3.0
Dm = Tm + torch.randn(V, V, device=DEV) * 1.5
target_p = torch.softmax(Tm, dim=-1)             # [V, V]
draft_p = torch.softmax(Dm, dim=-1)

# Production law: target tempered 1.0 and truncated to top-20 per row.
topk = target_p.topk(TOP_K, dim=-1).indices
trunc_mask = torch.zeros_like(target_p, dtype=torch.bool)
trunc_mask.scatter_(1, topk, True)
law_p = target_p * trunc_mask
law_p = law_p / law_p.sum(-1, keepdim=True)
law_cdf = law_p.cumsum(-1)

target_logits = Tm.masked_fill(~trunc_mask, float("-inf"))   # [V, V]
draft_greedy = draft_p.argmax(-1)                             # [V]


def law_sample(prev: int, g: torch.Generator) -> int:
    u = torch.rand(1, generator=g).item()
    return int((law_cdf[prev] < u).sum().item())


def run_rejection(prev: int, seed: int):
    """One verify round from anchor=prev: returns emitted tokens."""
    # Verify rows: [anchor, d_1..d_NS]; the anchor row's logits predict
    # position prev+1 given prev. Rows share the anchor's context in this
    # synthetic setup only through prev; later rows use draft-continued
    # contexts (approximating the engine: row i conditioned on the draft
    # chain).
    ctx = prev
    rows_logits = []
    for _ in range(NS + 1):
        rows_logits.append(target_logits[ctx])
        ctx = int(draft_greedy[ctx].item())
    tl = torch.stack(rows_logits).contiguous()                   # [T,V]
    tl = tl.unsqueeze(0).reshape(-1, V).contiguous()             # [T*1, V]
    dl = Dm.masked_fill(~trunc_mask, float("-inf"))
    dl = torch.stack([dl[ctx0] for ctx0 in _draft_chain(prev)]).unsqueeze(0) \
        if False else None  # greedy drafts: pass None (greedy path)
    dchain = _draft_chain(prev)
    dsamp = torch.full((NS + 1,), -2, dtype=torch.long, device=DEV)
    dsamp[1:] = torch.tensor(dchain, device=DEV)
    cu = torch.tensor([0, NS + 1], device=DEV, dtype=torch.int32)
    pos = torch.arange(NS + 1, device=DEV, dtype=torch.int64)
    idx = torch.zeros(1, device=DEV, dtype=torch.int64)
    eidx = torch.zeros(NS + 1, device=DEV, dtype=torch.int64)
    epos = torch.arange(NS + 1, device=DEV, dtype=torch.int64)
    temp = torch.ones(1, device=DEV, dtype=torch.float32)
    s = torch.tensor([seed], device=DEV, dtype=torch.int64)
    sampled, _ = rejection_sample(
        tl, None, dsamp, cu, pos, idx, eidx, epos, temp, s,
        NS, None, use_fp64=False, use_block_verification=False,
    )
    return sampled[0].tolist()


def _draft_chain(prev: int):
    chain = []
    c = prev
    for _ in range(NS):
        c = int(draft_greedy[c].item())
        chain.append(c)
    return chain


def bigram_kl(stream, counts_ref):
    """Empirical bigram-conditional KL against the law (support-restricted)."""
    cnt = torch.zeros(V, V)
    for a, b in zip(stream[:-1], stream[1:]):
        cnt[a, b] += 1
    law_cpu = law_p.cpu()
    tot = cnt.sum(-1).clamp(min=1)
    emp = cnt / tot.unsqueeze(-1)
    kl = 0.0
    n = 0
    for a in set(stream[:-1]):
        law = law_cpu[a]
        sup = law > 0
        if sup.sum() < 2:
            continue
        e = emp[a][sup].clamp(min=1e-9)
        l = law[sup]
        kl += float((e * (e.log() - l.log())).sum())
        n += 1
    return kl / max(n, 1)


def support_violations(stream):
    law_cpu = law_p.cpu()
    bad = 0
    for a, b in zip(stream[:-1], stream[1:]):
        if law_cpu[a, b] <= 0:
            bad += 1
    return bad / (len(stream) - 1)


g = torch.Generator().manual_seed(7)
# Direct reference stream
direct = []
prev = int(torch.randint(0, V, (1,), generator=g))
for _ in range(N):
    nxt = law_sample(prev, g)
    direct.append(nxt)
    prev = nxt

# Chain A: correction-only
corr = []
prev = int(torch.randint(0, V, (1,), generator=g))
for i in range(N):
    emitted = run_rejection(prev, 1000 + i)
    tok = emitted[0]
    if tok is None or tok < 0:
        tok = law_sample(prev, g)
    corr.append(int(tok))
    prev = int(tok)

# Chain B: natural (commit all)
nat = []
prev = int(torch.randint(0, V, (1,), generator=g))
i = 0
while len(nat) < N:
    emitted = run_rejection(prev, 500000 + i)
    i += 1
    toks = [t for t in emitted if t is not None and t >= 0]
    if not toks:
        toks = [law_sample(prev, g)]
    nat.extend(int(t) for t in toks[:7])
    prev = int(toks[-1])
nat = nat[:N]

print(f"N={N} tokens per stream")
for name, s in (("direct", direct), ("correction-only", corr),
                ("natural", nat)):
    print(f"{name:>16}: bigram-KL {bigram_kl(s, None):.5f}  "
          f"support-viol {support_violations(s):.5f}")
print("\nverdict: correction-only/natural KL >> direct's = CHAINED BIAS "
      "convicted; all equal = kernel chain exact")
