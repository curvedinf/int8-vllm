# INT8 Serving Audit Results

## Scope and evidence

This audit covers the canonical gfx908 TP4/C8 serving stack for:

- target: `curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128`
- draft: `curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128`

It was refreshed on 2026-08-25 (second full rewrite) after the accuracy
program (commits `43bf98431`, `172f6e936`, `17254ec77`). Evidence reviewed:
vLLM `17254ec77..7e6678088` and aiter `e28051d2c`; the live launcher
environment dump of a booted server (`/proc/<pid>/environ`, all pins
verified); the live startup log (kernel/JIT selection lines); a fresh
boot + coherence probe; and a full offline replay of the CURRENT stack's
recorded I/O (`fp32state_round` artifacts) against the BF16 reference
weights. Benchmark artifacts cited below live under
`logs/c8_optimization/` with manifests; replay JSONs under
`~/models/kld/quant_audit/replay/`.

Recorded benchmark labels are not accepted as ground truth: every number
below was cross-checked against its run manifest where one exists, and
numbers produced under the corruption regimes described in "Historical
caveat" are explicitly voided.

The target has 64 layers: 48 GatedDeltaNet linear-attention layers and 16
full attention layers. The DFlash2 draft has five noncausal
sliding-attention layers.

## Executive verdict

The accuracy program landed and the stack is both more accurate and
faster than the pre-fix state:

- Per-token activation quantization now rounds to nearest (fused Triton
  kernel, `VLLM_GFX908_ACT_QUANT=round`, default). In replay on recorded
  serving I/O this halved the dominant error leg on every module family
  (worst family 14.78% -> 7.86% rel-L2 at GEMM output).
- The GDN recurrent state runs FP32. The fp16 round-trip — convicted as
  the generator of runaway early-layer states (||h|| to 63,245, within 4%
  of the fp16 ceiling) — is gone; recorded states on the current stack
  max out at 131.7 with zero blow-ups.
- The old recipe inflated first-token stop probability 10x over BF16
  (3.74% vs 0.36% mean) — the "empty response" driver. The current stack
  is at parity (0.38%).
- 52-prompt KLD gate vs the BF16 reference: median 0.0153 (was 0.0966),
  greedy 40-char agreement 38/52 (was 24/52).
- Perf at TP4/C8 with identical NS=15: mean TPOT 12.52 ms (was 14.44 ms,
  -13%); TTFT ~219 ms; single-stream mean 91.1 tok/s. The speedup is
  structural: one fused quant kernel replaces a 4-pass eager quant chain
  on every W8A8 GEMM, and effective accepted length rose with accuracy.

What is still not the production dataflow (each verified against the live
boot this session):

- All-reduce is vLLM `CUSTOM` (dispatch `['CUSTOM','PYNCCL']`,
  `VLLM_ROCM_USE_AITER_CUSTOM_AR=0`); AITER CAR and the fused epilogue
  remain implemented but inactive (CAR trails 8.8% unfused; the epilogue
  is blocked by gfx908 Inductor corruption).
- The fused AR+RMSNorm+INT8 quant-out epilogue does not execute
  (`fuse_allreduce_rms=false`, compile mode forced NONE on gfx908).
- DFlash2 noncausal draft attention still routes through the vLLM Triton
  unified kernel (`rocm_aiter_unified_attn.py:413` — the aiter kernel is
  causal-only); it honors the int8-PTH cache format.
- The target `lm_head` is INT8 W8A8 (`VLLM_GFX908_INT8_LM_HEAD=1`, gate
  65.6 -> 65.6% acceptance, neutral) — this item IS now active, closing
  the previous audit's gap.
- The CK path still requantizes GS128 group scales to one scale per
  output channel at load (the 0.6-0.9% weight leg; the GS128 originals
  are freed for non-QKV layers, so no duplicate storage remains).
- `a8w8_tuned_gemm.csv` still has ZERO gfx908 rows; every production
  GEMM runs default-config CK (warmup does now call CK, fixed).

## Replay error budget (current stack, 2026-08-25)

Method (unchanged): module I/O recorded once under the exact production
recipe by `vllm/quant_audit_recorder.py` (async ring, probe-armed), then
replayed offline on CPU against BF16 weights with the exact numeric chain
(GPTQ uint8b128 unpack -> fp16 round-trip -> CK per-channel requant ->
recorded act quant). 481/512 recorded instances trusted (31 all-zero-x
exclusions); trusted-instance sanity `xq*s ~= x` mean 6.46% = the
expected per-token quant residual.

Rel-L2 at GEMM output vs BF16 golden (`gemm_budget_round.json`;
pre-fix values from the bootA replay in parentheses):

| family | gptq leg | ck-requant leg | act-quant leg | total |
|---|---|---|---|---|
| linear_attn.in_proj_qkvz | 0.26% | 0.75% | 7.86% (14.78%) | 7.91% (14.80%) |
| linear_attn.out_proj | 0.63% | 0.83% | 3.57% (6.85%) | 3.73% (6.93%) |
| mlp.down_proj | 0.70% | 0.92% | 5.63% (10.80%) | 5.77% (10.86%) |
| mlp.gate_up_proj | 0.37% | 0.78% | 5.26% (9.09%) | 5.33% (9.13%) |
| self_attn.o_proj | 0.62% | 0.87% | 2.54% (4.67%) | 2.78% (4.80%) |
| self_attn.qkv_proj | 0.18% | 0.61% | 6.47% (12.84%) | 6.51% (12.85%) |

Non-GEMM surfaces:

| surface | divergence | verdict |
|---|---|---|
| KV int8-PTH storage | k 0.88% / v 0.84% per token-head, p95 ~1.5% | normal int8 SNR — keep |
| Custom AR | 9.6e-5 mean; bitwise = fp32-accumulate-then-round-once | no accuracy action |
| GDN state (fp32) | max ||h|| 131.7, 0 states >1000 | blow-up eliminated |
| DFlash2 linears | weight domain 1.2%; layer-0 down_proj 0.80% rel-L2, no outlier structure | exonerated — the 0.4168 quant-log metric is calibration-relative, not weight error |

Weight quantization remains sub-1% everywhere and is not a quality
constraint. The act-quant leg is now the only material error source;
within it, per-token absmax scaling (forced by the CK rowwise epilogue)
on kurtosis-239-10k activations is the structural limit. Clip-percentile
(99.0-99.9) and SmoothQuant (alpha 0.5-0.85) variants were swept in
replay and all measured WORSE than plain trunc — rejected. Caveat: the
sweep's folding implementation mean-normalized the per-channel vector,
which manufactures outliers; correctly-implemented folding remains a
live candidate, and blockwise GS128 scale kernels (aiter
`GFX908_BUILD_PLAN.md` Step 5) are the real fix.

## Accuracy gates

52-prompt KLD vs `R0_bf16_ref` (bf16 weights, auto KV, fp32 mamba, no
spec; probe `scripts/kld_probe_v2.py`, gate harness
`scripts/kld_gate_boot.sh`):

| recipe | mean | median | p95 | greedy agree |
|---|---|---|---|---|
| pre-fix (fp16 state + trunc) | 1.832 | 0.0966 | 11.36 | 24/52 |
| fp32 state only | 1.860 | 0.0660 | 11.55 | 22/52 |
| current (fp32 + round) | 1.732 | 0.0153 | 11.92 | 38/52 |

Two findings from the gate series worth recording:

1. The fp32-state gate eliminated the state blow-ups outright yet left
   the KLD tail unchanged — proof that two separate mechanisms were at
   work and that acceptance-style short-horizon gates cannot see either.
2. Residual tail attribution: top-5 prompts carry ~24% of KLD mass, and
   text inspection shows same-answer-different-style divergence (correct
   content, alternate phrasing vs the greedy reference). This is
   distribution shift inherent in comparing two greedy continuations,
   not corruption. Further tail reduction requires shrinking the
   act-quant leg itself.

Soaks on the current recipe: 300 raw completions @ temp 0.7 C8 (293/300
non-empty; 7 first-token stops = the expected ~2% sampling tail now that
stop-prob is at BF16 parity; 0 corrupt), 300 greedy (0/300 empty),
200 chat-template (0/200; length-cut cases surface thinking under
`message.reasoning` — clients ignoring that field with small max_tokens
see "empty" by API semantics, not corruption).

## Perf record

Current honest numbers (bench_quick manifest `fp32state_round_c8`, fresh
boot, greedy C8, 8x32-in/1000-out):

| metric | current | pre-fix |
|---|---|---|
| TG rate (8 / TPOT) | 348 tok/s sustained | ~554 nominal (voided regime) |
| Mean TPOT | 12.52 ms | 14.44 ms |
| Mean TTFT | 219 ms | ~440 ms |
| Single-stream | 91.1 tok/s mean | — |

At fixed NS=15, TPOT scales inversely with effective accepted length, so
the -13% TPOT with better KLD implies acceptance improved alongside
quality (direct spec counters are off in this launch; treat the
acceptance implication as TPOT-derived, not counter-read).

Component benches retained from the 2026-08-24 isolation series (still
valid: measured after the corruption bisect, before the accuracy program
— unchanged components):

- GEMM: CK W8A8 63.49 tok/s C8 median vs valid W8A16 48.17 (+31.8%);
  SS 14.56 vs 10.64 (+36.8%); TPOT 105.0 vs 165.7 ms.
- All-reduce: vLLM CUSTOM 63.49 / AITER CAR 58.34 / PYNCCL 53.04 tok/s
  C8 medians; greedy outputs byte-identical across backends.
- Native MTP2 one-run (kept as context, not a baseline): 96.41%
  acceptance, 28.18 ms TPOT at C8 — ~283.9 tok/s TPOT-derived; the MTP
  sidecar itself is BF16 (not W8A8) and this was a single run.
- Cold 8x8192 prefill: 1,889-1,925 tok/s input throughput; warm-cache
  prefill numbers (2,174 tok/s) are void (prefix-cache hits).
- Int8 embedding gather: 14.1x local op speedup, 586 MiB less peak
  temporary; end-to-end prefill neutral (bracketing legs).

## Component status table

| Component | Status | Evidence |
|---|---|---|
| Target GEMMs | CK W8A8 everywhere, round act quant | live JIT lines + replay |
| lm_head | int8 W8A8 ACTIVE | env pin + neutral acceptance gate |
| KV target + draft | int8-PTH both | spec JSON + config line |
| GDN state | fp32 REQUIRED (int8 corrupts w/ int8 KV; fp16 blow-ups) | bisect + replay + gate |
| DFlash2 surfaces | conv+selector W8A8; codebooks bf16; ctx-KV proj bf16 | dtype ladder gates |
| Attention | AITER-UA causal; noncausal draft -> vLLM Triton (int8-PTH aware) | `rocm_aiter_unified_attn.py:413` |
| All-reduce | vLLM CUSTOM (fastest measured); CAR implemented, 8.8% behind unfused | iso bench 3-repeat |
| Fused epilogue | implemented, INACTIVE (Inductor corrupts on gfx908) | compile bisect |
| GS128 duplicates | freed for non-QKV layers | `aiter_w8a16.py:275-297` |
| CK warmup | calls CK (blockscale warmup fixed) | `aiter_w8a16.py:298-335` |
| CK tune table | ZERO gfx908 rows — all shapes default-config | `a8w8_tuned_gemm.csv` |

## Changes that still should not be made

- **No bare int8 GDN state.** int8-KV + int8-mamba TOGETHER corrupt
  generation (2x2 bisect, greedy-deterministic). fp32 is the
  checkpoint-declared dtype and the quality default; a scaled int8 state
  (block scales + SR) is a bandwidth project, not a quality fix.
- **Keep the GDN conv cache, depthwise conv, A_log/dt/gates, RoPE, P@V,
  softmax accumulation in floating point.** Error feeds the recurrence;
  the arithmetic/bandwidth saved is noise.
- **Keep KV scales FP32** (metadata is tiny vs the int8 payload).
- **Keep selector codebooks bf16** (candidate-row gathers, ranking-sensitive).
- **Do not describe CAR as int8 collective transport** — partials are
  fp16; the int8 label applies to the post-norm quant-out only.
- **Do not cite any pre-bisect acceptance/throughput figure** (the
  71-77% acceptance / 770 tok-s / 554 tok-s regimes) — corruption-inflated.

## Documentation inconsistencies to remove

Resolved this pass: recipe/README/AGENTS no longer restate the component
table (docs/recipes/README.md is the single source, with a "Current
production status" section); the stale "mamba int8 gate-passed" claim and
the 2026-08-22 KLD table were deleted; README/AGENTS point at the recipe
doc. Remaining: none known; new claims should be added to the recipe doc
first and referenced elsewhere.

## Updated priority order

1. **Blockwise GS128 scale kernels** (W-side first: eliminates the
   0.6-0.9% CK requant leg; then A-side: attacks the remaining 3-8%
   act-quant leg — the only material error source left). Build economics
   and variant table: aiter `GFX908_BUILD_PLAN.md` Steps 2/5.
2. **gfx908 CK tune-table fill.** Zero rows today; every GEMM runs
   default-config. Pure perf, no quality risk.
3. **Fused AR+RMSNorm+int8-quant epilogue** — enablement blocked on
   gfx908 Inductor corruption; eager-seam alternative documented in the
   recipe doc.
4. **NS sweep re-run on the honest stack.** The NS=17 collapse analysis
   predates the accuracy fixes; re-verify the cliff.
5. **Scaled int8 GDN state kernel** (bandwidth win; quality already at
   fp32 parity).
6. **AITER CAR tuning** — forced-naive kernel geometry on gfx908;
   re-bench after tune before any backend switch.
