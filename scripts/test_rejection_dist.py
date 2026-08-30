#!/usr/bin/env python3
"""Distribution test for the spec-decode rejection sampler.

Runs rejection_sample on a tiny synthetic vocab with a fixed target/draft
distribution many times and compares the emitted token frequencies to the
exact expected distribution (target tempered + truncated to top-k, with the
draft's contribution removed by the standard residual math). A frequency
mismatch convicts the sampler mechanically.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

DEV = torch.device("cuda:0")
V = 256
NS = 4          # speculative tokens
TOP_K = 20
TEMP = 1.0
N = 20000

torch.manual_seed(0)
# Random stable target logits; a draft distribution close to target but noisier.
base = torch.randn(V, device=DEV, dtype=torch.float32) * 3.0
draft_noise = torch.randn(V, device=DEV, dtype=torch.float32) * 2.0

# Exact expected distribution: softmax(base) truncated to top-k.
p = torch.softmax(base, dim=-1)
topk_ids = p.topk(TOP_K).indices
mask = torch.zeros_like(p)
mask[topk_ids] = 1.0
p_trunc = (p * mask) / (p * mask).sum()
expected = p_trunc  # rejection sampling must emit exactly the target dist

# Build per-request tensors: 1 request, NS+1 logits rows (anchor + NS drafts).
# Production truncates the target logits to top-k before rejection_sample
# (apply_sampling_params); replicate that here.
trunc_mask = mask.bool()
target_logits = base.masked_fill(~trunc_mask, float("-inf")).unsqueeze(0).repeat(NS + 1, 1).contiguous()
draft_logits = (base + draft_noise).unsqueeze(0).unsqueeze(0).repeat(1, NS, 1).contiguous()

# Draft tokens sampled from the (untruncated) draft distribution.
g = torch.Generator(device="cpu").manual_seed(1)
draft_tokens = torch.multinomial(
    torch.softmax((base + draft_noise).cpu(), -1), NS, replacement=True
).to(DEV)
draft_sampled = torch.cat(
    [torch.full((1,), -2, dtype=draft_tokens.dtype, device=DEV), draft_tokens]
)
cu = torch.tensor([0, NS + 1], device=DEV, dtype=torch.int32)
pos = torch.arange(NS + 1, device=DEV, dtype=torch.int64)
idx_mapping = torch.zeros(1, device=DEV, dtype=torch.int64)
exp_idx = torch.zeros(NS + 1, device=DEV, dtype=torch.int64)
exp_pos = torch.arange(NS + 1, device=DEV, dtype=torch.int64)
temperature = torch.full((1,), TEMP, device=DEV, dtype=torch.float32)
seeds = torch.full((1,), 12345, device=DEV, dtype=torch.int64)

counts = torch.zeros(V, device=DEV)
steps = 0
for it in range(N):
    seeds[0] = 1000 + it
    sampled, _ = rejection_sample(
        target_logits, draft_logits, draft_sampled, cu, pos,
        idx_mapping, exp_idx, exp_pos, temperature, seeds,
        NS, None, use_fp64=False, use_block_verification=False,
    )
    # emitted tokens = the accepted prefix + correction; count the FIRST
    # emitted token's distribution (position 0 of the stream) which is
    # anchored at row 0.
    tok0 = int(sampled[0, 0].item())
    counts[tok0] += 1
    steps += 1

freq = (counts / counts.sum()).cpu()
exp = expected.cpu()
# Compare on the support
sup = exp > 1e-6
err = (freq[sup] - exp[sup]).abs()
tail_mass_outside = float(freq[~sup].sum())
print(f"steps={steps}")
print(f"max abs freq err on support: {err.max():.4f} (expected ~<0.01)")
print(f"mean abs freq err: {err.mean():.4f}")
print(f"emitted mass OUTSIDE top-{TOP_K} support: {tail_mass_outside:.4f} (expected ~0)")
top5_out = (freq * (~sup).float()).topk(5)
print("worst outside-support tokens (id, freq):",
      [(int(i), round(float(f), 4)) for f, i in zip(top5_out.values, top5_out.indices)])
verdict = "CONVICTED" if (err.max() > 0.02 or tail_mass_outside > 0.01) else "consistent"
print("VERDICT:", verdict)
