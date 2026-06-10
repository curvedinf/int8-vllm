# Genesis P82 — lossy acceptance-threshold OR-clause (gfx908)

Opt-in speculative-decoding lever for gfx908 (MI100). **Default OFF — no behavior
change unless explicitly enabled.**

## What it does
Adds an SGLang-style acceptance-threshold OR-clause to the V1 rejection sampler.
In addition to standard rejection sampling, a draft (MTP) token is **also accepted
when the target model assigns it probability ≥ threshold**. This raises acceptance
length — and therefore MTP throughput — at the cost of output exactness (it emits
draft tokens strict sampling would reject). It is **lossy**.

Provenance: the "P82" patch from Sandermage's *Genesis* vLLM patch set, documented
in <https://alexander-ollman.github.io/qwen3.6-on-rtx3090/qwen3.6-on-rtx3090.html>.
Related downstream work: `curvedinf` PR #3 on this repo.

## Usage
```bash
# enable with a probability threshold in (0, 1); unset/<=0/>=1 disables it
-e VLLM_GFX908_MTP_ACCEPT_THRESHOLD=0.1
# with MTP, e.g.:
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```
Lower threshold = more aggressive acceptance = faster but more lossy. The threshold
operates on the **post-top_k/top_p** distribution, so sampling params matter.

## Recommended config: n=3 @ threshold 0.1
Validated on 4×MI100 (gfx908) TP4, full 12-tier BenchAndReport (sonnet) + GSM8K (300 Q,
thinking, temp=0.6, seeded). Both runs at the SHIPPED/default budget — **n=3 does not
crash** (the long-context crash is n=5-specific, see below).

| model | metric | strict | P82@0.1 | Δ |
|---|---|---:|---:|---:|
| 27B dense | Decode c1 | 86.7 | 103.5 | +19% |
| 27B dense | Single User c1 | 71.3 | 84.1 | +18% |
| 27B dense | GSM8K | 91.7% | 92.0% | neutral |
| 35B-A3B MoE | Decode c1 | 155.6 | 193.1 | +24% |
| 35B-A3B MoE | Single User c1 | 127.4 | 160.8 | +26% |
| 35B-A3B MoE | GSM8K | 88.7% | 90.0% | neutral |

**Accuracy is neutral** despite large token-level divergence from strict — P82 shifts
generation toward the draft's high-probability predictions (more modal), which doesn't
hurt correctness on GSM8K. Stable (0 failures) on both models at default budget.

## Caveats / scope
- **Lossy**: changes output trajectory substantially (≈75-80% token change at @0.1 vs
  same-seed strict). Accuracy-neutral on GSM8K (math), but only one task verified;
  treat as opt-in. IFEval not yet run.
- **n=5 long-context crash (out of scope here):** at the most aggressive depth (n=5),
  high acceptance at long context overflows a `max_num_batched_tokens`-sized buffer
  (HIP illegal access, cumulative). Fixed by raising the budget (budget = max_model_len
  holds at 32K), but the proper fix is a context-aware budget bump — deferred, since
  this lever ships at **n=3**, which is stable at default budget and wins at high
  concurrency anyway.

Full evaluation: `mi100-llm-testing` → `Model_Reports/p82_evaluation/`.
