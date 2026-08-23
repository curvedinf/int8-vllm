# UA-fix evaluation — Qwen3.6-27B-GPTQ-8bit (gfx908 / 4×MI100)

**Date:** 2026-06-11
**Image A** = `btbtyler09/vllm-rocm-gfx908:v0.21.0rc1.dev-aitersync` (439-commit AITER sync, UA restructure / corruption fix)
**Image B** = `btbtyler09/vllm-rocm-gfx908:v0.21.0rc1.dev` (pre-sync prod, anchor only)
**Common config:** dtype=half, TP4, max-model-len 32768, max-num-batched-tokens 8192, gpu-util 0.95, `--compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'`, `VLLM_MI100_TORCH_COMPILE=1`.
**MTP arms:** P82 `VLLM_GFX908_MTP_ACCEPT_THRESHOLD=0.1` + `{"method":"mtp","num_speculative_tokens":3}`. Bench dataset = sonnet (real text → realistic MTP acceptance ~85–88%).
**UA arm:** `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`, no `--attention-backend` (main decode = ROCM_AITER_UNIFIED_ATTN, rocm.py:615). **TRITON arm:** `--attention-backend TRITON_ATTN` (main decode = TRITON_ATTN, rocm.py:606).
Noise floor (from anchor): **~±2%**.

---

## TL;DR — best settings on dense 27B-8bit

| Workload | Best backend | Why |
|---|---|---|
| **Dataset generation (4k in / 4–8k out, c=32–64)** | **UA + MTP-P82** | **+15% throughput, −12% TPOT** |
| Interactive / single-user (c=1–2) | **UA + MTP-P82** | +6–8% tok/s |
| Long context (16K) | **UA + MTP-P82** | +29% tok/s |
| Short-context high-throughput (in≤512, out=256, c≥16) | TRITON + MTP-P82 | UA −10% there |
| Mid/high concurrency, short output (out=256) | tie | ±2% |

- **MTP n=3 + P82 is the dominant lever regardless of backend** — ~doubles c=1 (50→89 tok/s).
- **UA's advantage is MTP-specific on dense 27B**: without MTP, UA ≈ TRITON everywhere; with MTP it wins at low-c, long-ctx, and long-output batch. Mechanism: each MTP decode step is a 4-token verify mini-batch, and UA's unified prefill+decode kernel handles batched attention better than Triton's decode path — the edge compounds over long outputs.
- **AITER sync = no regression** (anchor within ±2.2% on all 12 tiers).
- **UA corruption bug = fixed** on the model in use: ~1,200-req soak under production config, coherence pristine.

---

## 1. Headline: A1 (UA) vs A2 (TRITON), both MTP-P82, full 12-tier (tok/s)

| scenario | in/out | c | UA | TRITON | UA Δ% |
|---|---|--:|--:|--:|--:|
| Single User Latency | 2048/512 | 1 | 88.8 | 82.3 | **+7.9** |
| Decode Stress | 128/2048 | 1 | 108.4 | 102.5 | **+5.8** |
| Long Context | 16384/1024 | 4 | 78.7 | 61.1 | **+28.8** |
| Short Context | 512/256 | 16 | 322.6 | 359.3 | **−10.2** |
| Mixed Traffic | 2048/512 | 8 | 211.0 | 205.6 | +2.6 |
| Conc c=2 | 1024/256 | 2 | 129.3 | 123.5 | +4.7 |
| Conc c=4 | 1024/256 | 4 | 180.7 | 175.7 | +2.8 |
| Conc c=8 | 1024/256 | 8 | 211.7 | 207.2 | +2.2 |
| Conc c=16 | 1024/256 | 16 | 307.8 | 308.2 | −0.1 |
| Conc c=32 | 1024/256 | 32 | 347.3 | 352.3 | −1.4 |
| Conc c=64 | 1024/256 | 64 | 391.9 | 396.8 | −1.2 |
| Conc c=128 | 1024/256 | 128 | 399.6 | 395.1 | +1.1 |

Coherence pre+post PASS (6/6) both arms.

## 2. No-MTP A/B: A4 (UA) vs A5 (TRITON), full 12-tier (tok/s) — isolates the MTP interaction

UA ≈ TRITON across all 12 tiers (UA −1 to −4%, within noise): Single User 50.9/51.9, Short Ctx 225.5/234.1, Long-16K 62.1/61.9, c=128 380.9/380.6. **Without MTP, the two backends are equivalent on dense 27B** → UA's MTP-arm wins are genuinely produced by the MTP verify path, not a baseline kernel difference. (Opposite of MoE-35B, where UA helped only *without* MTP.)

## 3. Workload A/B (W) — dataset generation, in=4096 / out=6144, sonnet, MTP-P82

| conc | UA tok/s | TRITON tok/s | UA Δ | UA TPOT | TRITON TPOT |
|---|--:|--:|--:|--:|--:|
| c=32 | **745.7** | 646.0 | **+15.4%** | 38.2 ms | 43.1 ms |
| c=64 | **742.7** | 648.0 | **+14.6%** | 57.9 ms | 66.7 ms |

The decisive result for dataset generation: **UA is +15% throughput and ~12% faster per token.** The standard suite's short-context −10% does NOT apply to long-output workloads — it flips hard in UA's favor because 6k output = thousands of MTP verify-batches and UA's edge compounds. (Workload saturates aggregate throughput by c=32; c=64 only adds queueing, TTFT 19s→118s.)

## 4. Anchor (A3) — sync regression check: A3 (prod ImageB) vs A2 (sync ImageA), both TRITON+MTP

All 12 tiers within ±2.2% (noise). **The 439-commit AITER sync caused no regression** at full-suite scale.

## 5. Corruption soak (S1) — UA + MTP-P82 27B-8bit: PASS

~1,200 requests total (192 workload-AB at 4k/6k + 1,000 mixed soak across c=8/16/32/48/64, varied 256–1536 in / 256–640 out), 100% success. Coherence pre + post pristine (6/6, no degeneration) — far past the historical ~200-req corruption threshold. **The production config (UA + MTP-P82) is corruption-free on dense 27B.** (122B soak skipped — not in use.)

---

## Recommendation

For **dense GPTQ-8 models on gfx908 running MTP-P82** — especially **dataset generation and interactive/long-context serving** — **switch the default attention backend to UA** (`VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`, drop `--attention-backend`). It is a clear win (+15% on the dataset-gen workload, +6–8% interactive, +29% long-ctx) with no correctness cost. Keep TRITON only for short-context high-throughput (in≤512, out=256) serving, where it is ~10% better.

This is model- and MTP-dependent: the MoE-35B showed the opposite pattern, so this recommendation is scoped to **dense + MTP**. A flag flip (not a hard default) keeps both available.

Raw data: `/home/tyler/aiter-sync-builds/ua_eval/{A1..A5,W_UA,W_TRITON,S1_soak}.*`, per-arm reports `ua_eval_A{1..5}_*.md`.
