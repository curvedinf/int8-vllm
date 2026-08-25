# INT8 Serving Audit Results

Re-audit of the canonical gfx908 TP4/C8 serving stack, 2026-08-25 (v2).
This document supersedes the previous audit entirely; its record-once /
replay-offline methodology is unchanged, and all numbers below are fresh
measurements taken against the CURRENT recipe (fp32 GDN state +
round-to-nearest act quant), not carried over from the old doc.

The canonical component/flag/perf state lives in `docs/recipes/README.md`
("Current production status"). This doc covers: method, the current error
budget, accuracy/perf gates, what was convicted and fixed, what remains.

- target: `curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128`
- draft: `curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128`
- vLLM `17254ec77`, aiter `e28051d2c` (both `mi100-optimized-sync`)
- arch: 64 layers = 48 GDN linear-attention + 16 full attention; TP4 rank0
  evidence; recorder artifacts under `~/models/kld/quant_audit/` (not in git)

## Method (unchanged from v1)

1. **Record once under real serving**: `vllm/quant_audit_recorder.py`
   (async pinned-ring, armed via the probe's first request) captures GEMM
   x/xq/xs, KV pre/post-quant, GDN states, AR partials/outs per rank.
   `VLLM_QUANT_AUDIT=<dir>` arms it; `scripts/kld_gate_boot.sh <tag>` wraps
   boot -> 52-prompt probe -> compare -> stop.
2. **BF16 reference**: `R0_bf16_ref` — bf16 weights, auto KV, fp32 mamba,
   no spec, same 52 prompts, same probe (`scripts/kld_probe_v2.py`).
3. **Replay offline** (CPU): `scripts/quant_replay_gemms.py` rebuilds the
   exact numeric chain (GPTQ uint8b128 unpack -> fp16 round-trip -> CK
   per-channel requant -> recorded act quant) and decomposes error legs
   against BF16 weights. `scripts/quant_replay_state.py` (KV/GDN/AR),
   `scripts/quant_replay_drafter.py` (draft chain),
   `scripts/quant_sweep_actquant.py` (quantizer variant sweep).
4. **Gate end-to-end**: KLD over matched greedy positions + top-20 dumps;
   greedy 40-char agreement; soaks at C8.

Recorder integrity (this audit): 481/512 GEMM instances trusted; 31
excluded (all-zero x captures); trusted-instance sanity `xq*s ≈ x` mean
6.46% — exactly the expected per-token quant residual.

## Current error budget (round-stack replay, 2026-08-25)

Rel-L2 at GEMM output vs BF16 golden, recorded rank0 inputs from the
`fp32state_round` boot (the live production chain):

| family | gptq leg | ck-requant leg | act-quant leg (round) | total |
|---|---|---|---|---|
| linear_attn.in_proj_qkvz | 0.26% | 0.75% | **7.86%** | 7.91% |
| linear_attn.out_proj | 0.63% | 0.83% | 3.57% | 3.73% |
| mlp.down_proj | 0.70% | 0.92% | 5.63% | 5.77% |
| mlp.gate_up_proj | 0.37% | 0.78% | 5.26% | 5.33% |
| self_attn.o_proj | 0.62% | 0.87% | 2.54% | 2.78% |
| self_attn.qkv_proj | 0.18% | 0.61% | 6.47% | 6.51% |

Comparison to the pre-fix stack (same method, bootA recordings): the
act-quant leg halved on every family (e.g. in_proj_qkvz 14.78% -> 7.86%,
qkv 12.84% -> 6.47%). Weight legs are unchanged and remain sub-1%.

Non-GEMM surfaces (v1 measurements, re-confirmed against the current
artifacts where noted):

| surface | divergence | verdict |
|---|---|---|
| KV int8-PTH storage | k 0.88% / v 0.84% per token-head (p95 ~1.5%) | normal int8 SNR — keep |
| Custom AR | 9.6e-5 mean; bitwise = fp32-accumulate-then-round-once | no accuracy issue |
| GDN state (fp32, current) | max ||h|| 131.7 across all recorded snapshots; zero states >1000 | blow-up eliminated (was 63,245 pre-fix) |
| DFlash2 linears | weight domain 1.2%; layer-0 down_proj exonerated (0.80%, no outlier structure) | no requant needed |

## Accuracy gates (52-prompt KLD vs R0_bf16_ref)

| recipe | mean | median | p95 | greedy agree |
|---|---|---|---|---|
| pre-fix (fp16 state + trunc) | 1.832 | 0.0966 | 11.36 | 24/52 |
| fp32 state only | 1.860 | 0.0660 | 11.55 | 22/52 |
| **current (fp32 + round)** | 1.732 | **0.0153** | 11.92 | **38/52** |

Position-0 stop probability (the empty-response driver): pre-fix 3.74%
mean / 41% max — 10x BF16 inflation; current 0.38% (BF16 is 0.36%). At
par.

Residual tail attribution: top-5 prompts carry ~24% of KLD mass; text
inspection shows same-answer-different-style divergences (correct content,
alternate phrasing vs the greedy reference continuation). This is
distribution shift inherent to comparing two greedy continuations, not
corruption. Further tail reduction requires shrinking the act-quant leg
itself — blockwise GS128 kernels (see aiter `GFX908_BUILD_PLAN.md` Step 5).

## Perf gates (TP4/C8, bench_quick, fp32state_round_c8)

| metric | current | pre-fix |
|---|---|---|
| TG rate (8 / TPOT) | 348 tok/s sustained | ~554 nominal (corrupted-acceptance regime) |
| Mean TPOT | 12.52 ms | 14.44 ms |
| TTFT | 219 ms | ~440 ms |
| Single-stream | 91.1 tok/s mean | — |

-13% TPOT at identical NS=15 with better accuracy: at fixed spec depth TPOT
scales inversely with effective accepted length, so acceptance improved
alongside quality; the fused one-kernel RN quantizer also replaces the
4-pass eager aiter quant chain on every GEMM.

Soaks on the current recipe: 300 raw completions @ temp 0.7 C8 (293/300
non-empty; 7 first-token stops = the expected ~2% sampling tail, 0
corrupt), 300 greedy (0/300 empty), 200 chat-template (0/200 empty;
length-cut cases surface thinking under `message.reasoning` — clients that
ignore that field with tiny max_tokens see "empty" by API semantics).
Coherence spot-check: correct math + reasoning on the live boot.

## What was convicted and fixed (2026-08-25 accuracy program)

1. **Per-token act quant truncation** (aiter `pertoken_quant`, absmax/127,
   trunc-toward-zero on kurtosis-239-10k activations). Fix: fused Triton
   round-to-nearest kernel `act_quant_rn.py`, selected by
   `VLLM_GFX908_ACT_QUANT=round` (default in launcher + envs.py). Variant
   sweep rejected clipping (99.0-99.9 pct) and SmoothQuant folding
   (alpha 0.5-0.85) — all worse than trunc in replay. Note: the folding
   implementation in the sweep was flawed (mean-normalization manufactured
   outliers); correctly-implemented folding remains a candidate.
2. **GDN fp16 state round-trip**: decode re-stores h in fp16 every token;
   once |h|*2^-11 > beta*|v| the delta-rule cancellation fails and grows
   multiplicatively (early layers carry exp(A) up to 60). Fix:
   `--mamba-ssm-cache-dtype float32` (the checkpoint's declared dtype;
   state is cache-resident, 37.7 MB/seq/rank, no per-token bandwidth cost).
   Int8 state remains banned: int8-KV + int8-mamba TOGETHER corrupt
   generation (2x2 bisect 2026-08-25); a properly scaled int8 state kernel
   is future work, not a quality requirement.

Exonerated (no action): GPTQ weights, CK per-channel requant, custom AR,
KV int8-PTH, DFlash2 layer-0 down_proj.

## Historical caveat (why older perf numbers are void)

Every acceptance/throughput figure recorded before 2026-08-25 afternoon
(e.g. "770 tok/s / 71-77% acceptance") was measured under the int8-KV +
int8-mamba corruption or the fp16-state regime — repetitive degraded
output is trivially predictable and inflates acceptance. The recipe doc
marks these superseded; do not cite them as baselines.

## Priority order (next work)

1. Blockwise GS128 scale kernels (W-side first: kills the 0.6-0.9% CK
   requant leg; then A-side for the remaining act-quant tail) — spec and
   build economics in aiter `GFX908_BUILD_PLAN.md` Steps 2/5.
2. gfx908 CK tune-table fill (`a8w8_tuned_gemm.csv` has zero gfx908 rows —
   every GEMM runs default-config today).
3. Fused AR+RMSNorm+int8-quant epilogue enablement (kernel done, blocked
   on gfx908 Inductor corruption — see recipe doc).
4. NS sweep re-run on the honest stack (NS=17 cliff analysis predates the
   accuracy fixes).
5. Scaled int8 GDN state kernel (bandwidth win; quality already at parity
   with fp32).
