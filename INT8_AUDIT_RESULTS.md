# INT8 Serving Audit Results — Coverage Edition

## Purpose

This document enumerates every place the production serving path (Qwen3.8-27B
+ DFlash2, TP4/C8, gfx908) still touches non-int8 data — weights, activations,
caches, collectives, kernels — and classifies each as an optimization target,
a dead-storage cleanup, or an audited exception. It is a gap inventory for the
int8-native doctrine, not an accuracy report: quality gates (KLD, soaks) are
run per-change and recorded in the run manifests / recipe doc; only coverage
questions live here.

Ground rules for this edition:

- **AITER Custom All-Reduce is NOT a plan item.** The all-reduce optimization
  target is vLLM's own `CUSTOM` implementation tuned specifically for
  4x MI100 XGMI. AITER CAR exists, is coherent, and stays parked.
- Accuracy verdicts from the 2026-08-25 program (act-quant round, fp32 GDN
  state, KLD/stop-prob parity) are settled context, not repeated here.
- Every size below was measured from the actual checkpoints (safetensors
  scan, 2026-08-25) or the live server, not estimated from configs.

## Active float surfaces in the serving path (the real targets)

Ranked by expected decode impact. TP4-rank figures where relevant.

| # | Surface | Where | Dtype / size | Class | Feasibility |
|---|---|---|---|---|---|
| 1 | **Draft `fc` GEMM** | `qwen3_dflash.py:509` — `ReplicatedLinear`, registered under `quant_config` but the checkpoint ships BF16 `fc.weight` (no GPTQ params), so it executes as a **dense bf16 GEMM** every draft step | 262 MB bf16, replicated per rank (NOT sharded — `ReplicatedLinear`) | compute + 4x replicated bandwidth | HIGH: quantize to GPTQ GS128 (same recipe as the rest of the draft) → CK W8A8. Halves the largest single bf16 read in the decode path and removes 4x replication of the bf16 copy |
| 2 | **All-reduce payload (fp16 partials)** | vLLM `CUSTOM` CAR, every row-parallel layer output | fp16 [T,5120] x4 ranks per AR | bandwidth | **vLLM CAR tuned for 4x MI100 XGMI** — launch geometry, IPF/IPC staging, multi-block overlap for C8 decode sizes. Int8 collective transport (scale-carrying) is a research item only; fp16 payload stays until a scaled scheme is proven |
| 3 | **Inter-op activation stream** | residual stream + norm outputs flow fp16 between ops; int8 exists only at GEMM inputs (each linear re-quantizes) | fp16 activations, quantized redundantly per consumer | bandwidth + launches | the fused AR+RMSNorm+int8-quant-out epilogue (kernel exists, `ec90fc933`) removes the fp16 round-trip AND the per-consumer quant; blocked by gfx908 Inductor corruption → build the eager seam in the decoder layer |
| 4 | **Draft `base_kernel` conv weights** | `qwen3_dflash2.py:67` — `_grouped_conv` runs bf16; CORRECTED 2026-08-25 (E4): the taps are 10x [2,2,5120] bf16 = **0.4 MB** (the 131.5 MB in this row was the `kernel_projection` GEMM weights, which ARE already W8A8) | negligible | E4 built scaled-int8 dequant-on-add (env `VLLM_GFX908_DF2_CONV_I8`) and REJECTED on ROI — surface too small to matter |
| 5 | **Target `in_proj_a`/`in_proj_b`** | 48 GDN layers; standalone bf16 GEMMs [48,5120] feeding the decay/gate math | 47 MB bf16 total | launches + small bandwidth | pack/append into `in_proj_qkvz` (shares the quantized input read, one extra GEMM launch removed); do NOT quantize standalone (feeds exp(A) gates — audited sensitivity) |
| 6 | **Attention P@V + softmax** | UA attention internals | fp16/fp32 math | compute | KEEP FLOAT (doctrine): probability accumulation and P@V are accuracy-critical; the KV read is already int8-PTH |
| 7 | **Noncausal draft attention kernel** | DFlash2 draft attention runs the vLLM Triton unified kernel (`rocm_aiter_unified_attn.py:413`, aiter kernel is causal-only) — KV read IS int8-PTH | fp16 P against int8 KV | launches | an aiter-native noncausal variant would drop a Triton launch per draft layer; needs the aiter kernel extended to noncausal masks |
| 8 | **Draft ctx-KV projection** | bf16 dense GEMM by measured dtype ladder (71.2 > 69.2 > 66.0% acceptance) | bf16 | compute | AUDITED EXCEPTION — re-gate only after the act-quant round landed (the ladder predates it; int8 may now pass) |

## Dead storage (zero runtime risk, checkpoint hygiene)

Measured bf16 tensors in the deployed checkpoints that never execute:

| Item | Size | Note |
|---|---|---|
| Target **visual tower** (`model.visual.*`, 117 tensors) | 1,656 MB | `--language-model-only` never touches it; strip from the published checkpoint (and from loader memory if loaded at all) |
| Target **native MTP sidecar** (`mtp.*` incl. `mtp.fc` 105 MB) | ~250 MB | DFlash2 is the drafter; the sidecar is dormant — strip, or keep only if MTP is re-benched |
| Target `lm_head.weight` stored BF16 | 2,543 MB disk | runtime IS int8 W8A8 (`VLLM_GFX908_INT8_LM_HEAD=1`) — the checkpoint just ships the bf16 master; publish an int8-stored variant to halve the download |
| Target `embed_tokens` stored BF16 | 2,552 MB disk | runtime IS int8 (gather-then-cast, `vocab_parallel_embedding.py:144`); same: ship int8 storage |

Net effect of the two "ship int8 storage" items: the published checkpoint
halves from ~30 GB toward ~18 GB with zero runtime change.

## Runtime state / caches (non-int8 by audit)

| Surface | Dtype | Status |
|---|---|---|
| GDN recurrent state | fp32 | audited exception — the accuracy program settled fp32 (int8 corrupts with int8 KV; fp16 blew up states). A scaled int8 state remains a BANDWIDTH project (151 MB/GPU fp16-era figure), gated on a future kernel with per-block scales + SR |
| GDN conv cache | fp16, ~20 MB/GPU at C8 | keep (tiny; no scale path exists) |
| GDN depthwise conv1d, A_log, dt_bias, norms | fp16/bf16, <5 MB total | keep float (recurrence-feeding) |
| KV scales | fp32 | keep (metadata dwarfed by int8 payload) |
| DFlash2 selector codebooks | bf16 254 MB | audited exception (candidate-row gathers, ranking-sensitive); revisit only with per-row scales + rank-preserving gate |
| Draft `candidate_selector.hidden_projection` | — | already W8A8 (`VLLM_GFX908_DF2_W8A8=1`) with the conv `kernel_projection` — covered |

## Coverage summary

By tensor bytes in the ACTIVE decode path (per full model, TP4 aggregate):

| class | bytes | share |
|---|---|---|
| int8 W8A8 weights (GPTQ → CK) | ~14.4 GB | dominant |
| bf16 draft `fc` | 262 MB (x4 replicated) | largest active float GEMM |
| bf16 draft `base_kernel` convs | 131 MB | conv math stays bf16 |
| bf16 target `in_proj_a/b` | 47 MB | packing candidate |
| bf16 draft ctx-KV proj + codebooks | audited exceptions | ladder/codebook gates |
| fp16 AR payloads + inter-op stream | transient | epilogue/vLLM-CAR work |

Every large dense GEMM in the TARGET is int8. The remaining float GEMM mass is
concentrated in the DRAFT (fc, ctx-KV) — the drafter is the next int8 front.

## Priority order (coverage work)

1. **Draft `fc` → GPTQ GS128 W8A8.** Single largest active float GEMM, 4x
   replicated, quantization recipe already proven on this drafter's other
   linears. Gate: acceptance + KLD (standard).
2. **vLLM CUSTOM all-reduce tuned for 4x MI100.** XGMI launch geometry,
   decode-size specialization (C8, NS=15 → T in {1..16} rows x 5120), overlap
   with the epilogue seam. NOT AITER CAR.
3. **Eager AR+RMSNorm+int8-quant epilogue seam** (removes fp16 stream +
   redundant per-consumer quant; kernel already bit-exact vs the unfused
   chain).
4. **Checkpoint hygiene pass**: strip visual tower + MTP sidecar, ship
   int8-stored lm_head/embed variants of the published checkpoints.
5. **Draft `base_kernel` conv weights → scaled int8** (acceptance-gated).
6. **`in_proj_a/b` packing into `in_proj_qkvz`** (no standalone quant).
7. **Re-gate draft ctx-KV projection int8** on the post-round stack.
8. **Noncausal aiter-native draft attention** (launch-count win; KV already
   int8).

## Experiment outcomes (2026-08-26, logs/surface_experiments/ledger.jsonl)

All 8 surfaces were experiment-gated (speed+quality A/B, paired controls):

| Surface | Verdict | Evidence |
|---|---|---|
| 1. Draft fc GEMM | **REJECT** — not a bottleneck | cast-quant W8A8 (0.49% GEMM err, cos 0.99999): TPOT +3.4% paired; fc read amortized across draft step |
| 2. vLLM CAR geometry | **REJECT e2e** | microbench 256/8 cuts AR time 18% at T=16-32, end-to-end neutral (+1.4-1.9%): AR ~10% of decode. Knobs `VLLM_GFX908_AR_THREADS/_AR_BLOCKS` kept in kernel |
| 3. Inter-op fp16 stream | **NEUTRAL** (prefill-only by design — decode is graph-captured) | seam implemented + booted clean (`VLLM_GFX908_EAGER_EPILOGUE=1`); known gap: PREQUANT stash bypassed under ACT_QUANT=round |
| 4. Draft conv taps | **REJECT on ROI** | surface is 0.4 MB (audit misattribution corrected; the 131 MB was kernel_projection, already W8A8). Mechanism built + validated behind `VLLM_GFX908_DF2_CONV_I8` |
| 5. in_proj_a/b pack | **REJECT on ROI + doctrine** | already runtime-fused as one `in_proj_ba` GEMM; remaining gain ~1% (below measurement floor) and packing quantizes exp(A)-feeding gates |
| 6. Attention P@V/softmax | **KEEP FLOAT confirmed**; config sweep done | tile 32 (current) optimal: 64→+1.3%, 16→+1.8% TPOT, all fast-regime paired boots. Knobs `VLLM_GFX908_ATTN_TILE/_WARPS/_STAGES` kept |
| 7. Noncausal draft attention | same kernel as 6 — verdict carries | earlier legs invalidated by regime skew; scale-fold parked (negligible vs floor) |
| 8. Draft ctx-KV projection | **REJECT — bf16 exception stands** | paired 13.86 vs 12.56 control (+10.4% TPOT = acceptance drop) even with RN act quant; ladder unchanged |

**Measurement doctrine (new, ledgered):** this box has a 24% per-rank sclk
skew under load (GPU2/3 at 731 MHz vs GPU0/1 at 962 MHz; power-limited
MI100 DVFS ramp). TP4 decode is straggler-bound → TPOT is bimodal
(12.5-13.9 fast / 16.6-20 slow regime). Sub-15% single-boot deltas are
unmeasurable; legs need paired fast-regime boots (TTFT ~200 ms marks the
fast regime; ~420+ the slow one). Clock pinning needs sudo — deferred.

**Perf number definitions (do not mix):** steady-state TG rate =
concurrency ÷ mean TPOT (e.g. 8 ÷ 12.50 ms = 640 tok/s) is the project's
primary decode metric. `vllm bench serve`'s "Output token throughput"
(~349 tok/s on the same boot) is wall-clock: total tokens ÷ total wall
time including the staggered finish as requests hit EOS — concurrency
decays after the first finisher, so it reads ~0.55× steady-state on
8-equal-length benches by construction. Ledger TPOT deltas are comparable
across both; tok/s numbers are only comparable within the same definition.
See docs/recipes/README.md "Current production status".

Net: the audited exceptions ARE the optimum at the current measurement
floor. The genuinely-open items are structural: Inductor fix (E3 fusion
in decode), blockwise GS128 kernels (act-quant leg), and the CK gfx908
tune table — none measurable to their potential on this box until the
clock skew is pinned.

## Explicit non-goals

- AITER CAR enablement or tuning (parked; vLLM CAR is the target).
- Quantizing softmax, P@V, RoPE, GDN recurrent math, norms, KV scales,
  selector codebooks, or the GDN conv cache (doctrine; see exceptions above).
- Any accuracy-motivated dtype change without a fresh gate — the current
  stack is at BF16-parity on stop-prob and state health; do not reopen
  settled ground (fp32 GDN state) without new kernel work justifying it.
