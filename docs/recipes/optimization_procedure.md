# Performance evaluation and optimization procedure (mandatory)

This document is the authoritative procedure for any performance work in this
fork. It supersedes every earlier performance guideline, benchmark recipe and
throughput claim. If another document, script header, scratch note or ledger
row contradicts it, this file wins and the other text is historical.

Read "Why this exists" before using any number recorded before 2026-11-10.

## Why this exists

The 2026-11-10 measurement audit found the serving recipe running at
**29 tok/s single-stream** while the loop's recorded state was
"650-705 tok/s". Most of the gap is workload, not drift: the benchmark
measured a different machine state. Part of it may also be condition drift —
the identical bench command today reports 223.7 tok/s cold and 552.2 tok/s
warm, below the six runs previously recorded — so the bench's own numbers are
not a stable reference either. The measured facts:

| Measurement (2026-11-10, live server) | Result |
|---|---|
| Bench as previously run, cold (seed 62408) | 223.7 tok/s, TTFT 6.47 s, wall 13.7 s |
| Same command immediately re-run (prefix cached) | 552.2 tok/s, TTFT 0.62 s, wall 5.56 s |
| Single stream, real text, greedy, 2.3k ctx | 86.9 tok/s, 3.08 tokens/step |
| Single stream, real text, greedy, 33.5k ctx | 33.7 tok/s, 1.88 tokens/step |
| Single stream, chat+tools, temperature 1.0 (served recipe) | **28.8-29.6 tok/s**, acceptance 1.04-1.20 |
| C12, real text, temperature 1.0, cold | 101.7 tok/s whole run, TTFT mean 13.5 s |

Five defects produced the gap:

1. The headline was a 12-stream aggregate, never single-stream.
2. The bench sampled greedily (`temperature: 0`) while the recipe serves
   temperature 1.0 / top-p 0.95 / top-k 20. Acceptance falls from 5.3-5.9 to
   1.04-1.20, and throughput is acceptance / step time.
3. Prompts were uniform random token IDs, which make the target nearly
   deterministic and inflate acceptance further.
4. `output_tok_s` divided by wall time, so a warm prefix cache (the same seed
   re-run) silently removed prefill from the denominator.
5. "Tokens" were SSE chunks. `tpot_ms` is the chunk interval, not the token
   interval; with acceptance 5 a 58 ms "TPOT" is a ~11 ms token.

Every rule below exists to make those five mistakes impossible.

## 1. The throughput identity

For speculative decoding on this stack:

```
tokens/s per stream = acceptance / step_ms
aggregate tokens/s   = concurrent streams x tokens/s per stream   (full overlap only)
acceptance           = 1 + sum(per-position acceptance rates)
step_ms              = measured at fixed M = C x (NS + 1)
```

A throughput number is meaningless unless it states workload, sampling
parameters, concurrency, acceptance, step time and cache state. Report the
triple (tokens/s, acceptance, ms/step) together; never one alone.

## 2. Workload matrix

All accept/reject decisions use W1-W4. W5 is a pathology detector and is never
an optimization target.

| ID | Workload | Definition | Purpose |
|---|---|---|---|
| W1 | agent-steady | C12; one shared ~16-32k system prompt with tool schemas, prefix-warmed; 256 output; real text; recipe sampling; tools present | Primary: decode throughput, acceptance, total request time |
| W2 | agent-cold | C12; 12 distinct ~2k real-text prompts; no prefix cache | Prefill and queueing: TTFT distribution, chunk shapes |
| W3 | long-turn | W1 with 2048-4096 output | Acceptance decay, long-output total time |
| W4 | mixed | 6 requests, then 6 more after 2 s (decode and prefill overlap) | Interference, injected-prefill TTFT |
| W5 | witness | C1; one chat+tools request, ~3k prompt, 256 output, recipe sampling | Detects pathologies in ~10 s; never tuned against |

Rules:

- Sampling is the recipe's: `temperature 1.0, top_p 0.95, top_k 20`
  (`--override-generation-config`). A greedy arm may be run as a **diagnostic**
  and must be labelled `greedy diagnostic`; it is never the accepted arm.
- Prompts are real text from a fixed, hash-recorded corpus. Random token IDs
  are prohibited for any measurement that can accept a change.
- Tools are present in W1/W3/W5, because the served traffic carries them and
  the tool preamble changes the distribution.
- Cache state is part of the workload definition and is pinned: W1 is warmed
  explicitly once before measuring; W2 uses distinct prompts and is never
  warmed. Cached and cold runs are never compared with each other.
- Output length is 256 for W1/W2; W3 exists because acceptance decays within a
  turn.

## 3. Metric definitions

- **Tokens** come from `usage.completion_tokens` or the Prometheus counters
  (`vllm:generation_tokens_total`, `vllm:prompt_tokens_total`). SSE chunks are
  never tokens.
- **Per-stream decode tok/s** = `(completion_tokens - 1) / (t_last - t_first)`.
- **Aggregate decode tok/s** is computed over the fully overlapped window
  (the interval where `vllm:num_requests_running` equals C for the whole
  window). Whole-run rates may be reported separately and must be labelled
  `whole run`.
- **Acceptance** from counter deltas: `(accepted + steps) / steps`, plus the
  `..._accepted_tokens_per_pos_total` vector. Per-position rates are always
  recorded.
- **Step time**: derive `tokens/step` and `ms/step` from the same counter
  window, or read `VLLM_STEPWALL`. Both are reported with every throughput row.
- **TTFT** is reported cold and warm separately, from
  `vllm:time_to_first_token_seconds` deltas or client-side first-token timing.
- **Prefill** is prompt tokens divided by prefill-phase wall time, and the
  prefill chunk shapes must be reported alongside it. `[aiter] shape is M:...`
  messages that say `not found tuned config` are part of the result: irregular
  chunked-prefill shapes currently fall back to a default GEMM config.
- **Tail latency**: p95 inter-token latency from
  `vllm:inter_token_latency_seconds`.
- **Primary objective**: per-request total time (TTFT + all output tokens),
  p50/p95, at W1. Aggregate capacity is the secondary number.

## 4. Three-phase optimization cycle

The cycle separates raw inference speed from acceptance. This matters because
acceptance is a multiplier, and because numeric changes made during inference
tuning can require a new PTQR pass.

| Phase | Scope | Isolation | Primary metric | Exit gate |
|---|---|---|---|---|
| 0 | Freeze the numeric contract; classify every candidate change | — | — | Decision list recorded in the ledger |
| 1 | Kernels, graphs, scheduling, prefill, engine handoff (most of the cycle) | Draft runs; every draft is rejected; verify batch keeps production shape | ms/step at M = C x (NS+1), p95 ITL, cold TTFT, prefill tok/s and shapes | Gain exceeds run spread; fixed-prompt output bit-identical |
| 2 | PTQR on target and draft | Offline training | Sampled acceptance (temp 1.0), per-position, KLD, quality battery | Acceptance >= baseline at served sampling; quality pass |
| 3 | NS, draft KV/quant, acceptance interactions, end-to-end | Real traffic model | Total request time at W1/W3, aggregate, acceptance | Beats the phase-0 baseline on total request time |

### Phase 0: numeric contract

Classify every candidate change before tuning:

- **Numerics-preserving** (allowed in phase 1): scheduling, graph capture,
  host-side handoff, memory layout, block sizes, prefix-cache policy, and
  fusions verified bit-identical on a fixed prompt.
- **Numerics-changing** (decided before PTQR, or they force a re-PTQR):
  activation-quantization mode (round vs stochastic), KV dtype, all-reduce
  algorithm, split-K or any different accumulation order, block-scale kernel
  substitution, quantized epilogue seams.

A numerics-changing change that lands after phase 2 invalidates the PTQR
gates. PTQR here is short rungs (15-60 steps), so re-running it is affordable
but must be scheduled, not discovered.

### Phase 1: inference tuning with forced rejection

- The draft executes and NS draft positions are scheduled; the **sampler
  rejects every draft**, so each step emits exactly one token per stream. The
  batch stays at `M = C x (NS + 1)` (96 at C12/NS=7), which is what production
  uses.
- Do not implement this by disabling speculative decoding or by the
  scheduler's `skip_drafts`: both shrink the step to `M = C` and every kernel,
  graph shape and CTA count changes with it.
- Implement it sampler-side. Verify the isolate with two counters:
  `spec_decode_num_accepted_tokens_total` delta == 0, and the scheduled token
  count == `C x (NS + 1)`.
- Fixed NS. Phase 1 does not decide NS, draft pruning, draft size or draft KV
  quantization: with drafts always rejected, draft compute looks like pure
  waste. The draft is still tuned for speed here (its time is in the step),
  but any trade of draft cost against draft accuracy belongs to phase 3.
- Step time is content-independent in this phase. Use that: A/B noise drops
  sharply, and three repeats are usually sufficient.
- Phase 1 cannot produce aggregate capacity or TTFT-under-real-overlap
  numbers: at acceptance 1 every request takes ~8x more steps, so the batch
  never drains the way real traffic does. Projections are labelled as such.

### Phase 2: PTQR

PTQR gates are subject to the same rules as everything else:

- Acceptance is measured at served sampling (temperature 1.0), per position,
  at the serving NS, at two context bands (short 2-8k, long 32k+). Greedy
  acceptance is a diagnostic arm only.
- Draft quality is gated on **draft-vs-target** agreement: top-1 agreement
  with the target's argmax and TV/KL between draft and target next-token
  distributions at several temperatures. Top-1 agreement with corpus ground
  truth is a wiring check, not a quality gate — the 2026-11 draft gate scored
  0.042 for a working draft and could only prove arm equality.
- Every quantization decision on the draft (weights, KV, activations) is
  accepted on the sampled arm. "int8 == bf16 at greedy" is not evidence for
  the served regime.
- `scripts/kld_probe_v2.py` remains the target quality gate (KLD median,
  greedy agreement). It is a quality gate: it never justifies a throughput or
  acceptance claim, and `greedy_agree_40char` truncates at 40 characters, so
  structural drift later in an output is invisible to it.
- The quality battery (long-output reads) is unchanged and still required.

### Phase 3: full pipeline with acceptance

- NS sweep and draft KV/quant decisions, judged on total request time at W1
  (and W3 for acceptance decay), not on step time alone.
- Carry-over check: re-run the two largest phase-1 wins with real acceptance
  to confirm they survive.
- End-to-end A/B against the phase-0 baseline on the primary objective.

## 5. Run protocol

- One warm-up run, then at least three measured repeats. Report median and
  spread; never a single run.
- Pin cache state per workload (see section 2). A cached-vs-cold comparison
  is not an A/B.
- Kernel-level changes: paired/interleaved arms on one boot. Config changes:
  full restart, and re-run the control arm at the end of the session to
  detect drift.
- Accept a change only if it beats the baseline median by more than the
  pooled spread, or wins at least 4 of 5 paired repetitions.
- Record a boot fingerprint with every result: vLLM and aiter SHAs, the
  non-default-args log line, graph-capture success, and GPU clocks.
- Fix the corpus hash and the sampling parameters across arms.

## 6. Acceptance gates

- A change is rejected if W1 median acceptance drops by more than 0.10 below
  the baseline, regardless of its step-time gain.
- Every acceptance result carries its per-position vector.
- NS and draft decisions are made at served sampling only.
- Attribution when acceptance regresses: four arms at temperature 1.0 and
  greedy, same target and prompts, per-position acceptance at ~4k and ~32k:
  int8 draft + int8 KV, int8 draft + bf16 KV, bf16 draft + bf16 KV,
  bf16 draft + int8 KV. This separates weights/PTQR from KV dtype from the
  sampling regime. The bf16 draft for the Flint pair is at
  `<models>/flint-dflash2-bf16`.

## 7. Prohibited measurements (the retired procedure)

- Greedy as the accepted arm.
- Uniform random token IDs as prompts.
- Counting SSE chunks as tokens, or reporting `tpot_ms` as a token time.
- Aggregate-only reporting with no per-stream number.
- Throughput denominators that include prefill while claiming decode speed.
- Comparing a cache-warm run against a cache-cold run.
- Accepting a change from a single run.
- Using C1 as an optimization target (it is the W5 witness).
- Median-only metrics as the sole basis for a throughput or acceptance claim.

## 8. Ledger row requirements

Every performance row in `docs/recipes/surface_experiments_ledger.jsonl`
carries: workload id (W1-W5), concurrency, corpus hash, sampling parameters,
cache state, repeats, median and spread, acceptance with per-position vector,
tokens/step, ms/step, M, TTFT cold and warm, prefill tok/s with chunk shapes,
boot fingerprint, and the verdict.

Rows written before 2026-11-10 do not carry these fields; they are historical
and must not be cited as production throughput.

## 9. Tooling status

`scripts/bench_flint_workloads.py` currently implements the retired synthetic
workload (greedy, random IDs, chunk counting, wall-clock rate). Treat its
output as a diagnostic only. A `--served` mode implementing W1-W5 is required
before the next optimization cycle; until it exists, use direct probes that
follow section 3 (usage-based token counts, Prometheus deltas).

Reference state at the time of writing (2026-11-10, after the audit):
single-stream served path 28.8-29.6 tok/s at acceptance 1.04-1.20; C12 real
text 101.7 tok/s whole run; greedy real text 86.9 / 54.8 / 33.7 tok/s at
2.3k / 9.4k / 33.5k context. These are the numbers any new claim must beat,
measured under the rules above.
