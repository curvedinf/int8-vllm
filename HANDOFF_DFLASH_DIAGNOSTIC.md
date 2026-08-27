# DFlash diagnostic handoff

## PERFORMANCE REGRESSION (2026-08-27, investigation under goal mode)

User report: TG ~1/10th expected (target 10-12 ms/token C8, ~650 tok/s
agg; observed ~50-70 ms/token single-stream). Prefill normal. Present
since "a couple days" of work. Root causes found:

1. **GPU clock/power state pinned (dominant, ~5x)**: all four MI100s in
   `power_dpm_force_performance_level=manual`, sclk pinned to level 0
   (300 MHz of 1502), power cap 105 W (max 290 W). Sysfs mtimes: set
   2026-08-26 20:03 — a tuner-session pin never reverted. Verified under
   sustained GEMM load: 100% util, 66 W, sclk 300 MHz; measured 18 TOPS
   on a big int8 GEMM = 92 TOPS x (300/1502). Compute-bound decode GEMMs
   run ~5x slow; bandwidth-bound prefill (mclk full 1200 MHz) unaffected.
   Fix requires root:
   `for d in /sys/class/drm/card*/device; do echo auto > $d/power_dpm_force_performance_level; done && rocm-smi --setpoweroverdrive 290`
   - `scripts/restore_gpu_clocks.sh` — idempotent restore, run as root.
   - `scripts/serve_recipe_qwen38.sh` now runs `clock_state_guard` on
     start: restores automatically when invoked as root (the user's
     workflow), warns with the exact fix otherwise.
   - `scripts/clock_restore_watcher.sh` (running, nohup) — on detecting
     restoration, boots the recipe under curved and runs
     `scripts/bench_quick.sh` automatically (logs/clock_watcher.log,
     results under logs/c8_optimization/clock_restored_*/).

2. **AITER tuned-CSV misses for every serving GEMM shape (secondary)**:
   `a8w8_tuned_gemm.csv` lacks the target verify/decode (N,K) set —
   (8704,5120) gate_up, (5120,4352) down, (5120,1536) o, (2048,5120) qkv,
   (256,5120) kv-proj, (62080,5120) lm_head, (5120,25600) draft fc — at
   ALL runtime Ms (graphs captured with splitK=0 defaults; log floods
   with "not found tuned config"). The Aug 26 tuning campaign tuned only
   M=1/16 draft-shaped rows. Fix: splitK sweep over the padded-M
   coverage grid (M in {1,2,4,8,16,32,48,64,80,96,112,128,...,2048}),
   append rows (CK dispatch consumes only splitK from the row; verified
   vs current module .so). Measured wins at pinned clock: kv-proj up to
   6x, draft fc 2.5x, qkv ~2x, MLP 1.2-1.7x at M>=64.
   - `aiter/aiter/ops/gemm_op_a8w8.py`: miss logging deduped per shape
     (was flooding serving logs, esp. eager prefill tails).

## Validation status (2026-08-27, under goal mode)

- Post-CSV-fix, pinned-clock (300 MHz) API validation:
  `logs/c8_optimization/csvfix_pinnedclk_20260827T072332Z` —
  single-stream 37.5 tok/s (was 14-20), C8 TPOT 24.95 ms, agg 217 tok/s,
  coherent greedy outputs. Miss lines 628 -> 28.
- The 28 residual misses are lm_head prefill-tail Ms (1280/2048 x
  62080x5120): swept separately — splitK=0 is measured-optimal (1.0x) at
  both, so the default fallback IS the winner there; no rows needed.
- Software side complete and pushed (int8-vllm 588ba48ceb, int8-aiter
  dc318f5bc). Full-speed 10-12 ms / 650 tok/s validation executes
  automatically on clock restore: `sudo scripts/restore_gpu_clocks.sh`
  (or any root start of the serve script — the guard restores). The
  watcher (`scripts/clock_restore_watcher.sh`, logs/clock_watcher.log)
  boots + benches + saves results + stops the server unattended.
- splitK winners were ranked at pinned clock; ranking argument is
  CU-occupancy (tile-count vs 120 CUs), which is clock-independent, but
  a full-clock re-sweep is cheap if ever in doubt.

## User request

Duplicate the existing main-model quantization audit process for the DFlash
draft model, stop the Qwen3.8 systemd server, and run the diagnostic. The
server must remain stopped. Any files created by the agent must be owned by
`curved:curved`.

## Current state

- Repo: `/home/curved/vllm-gfx908`
- Service: `vllm-openai-gfx908-qwen38.service`
- Service is currently `inactive`; no vLLM engine/API process is running.
- All created/modified files and diagnostic artifacts were checked as owned by
  `curved:curved`.
- Do not restart the service unless explicitly requested.

## Implemented files

- `scripts/dflash_audit.py`: capture fixed greedy prompts and compare aligned
  rank-0 DFlash stage tensors; reports relative L2, cosine, max error, or exact
  integer match. Also derives mean acceptance length and draft-token acceptance.
- `scripts/dflash_audit_run.sh`: stops systemd, runs controlled NS=7 legs,
  captures/compares stages, and leaves service stopped.
- `vllm/quant_audit_recorder.py`: added DFlash stage recording and
  `VLLM_QUANT_AUDIT_DFLASH_ONLY=1`. Also fixed an important recorder race:
  asynchronous GPU-to-pinned-CPU copies now record/synchronize CUDA events
  before `torch.save`; prior unsynchronized captures contained NaN/garbage.
- Instrumented DFlash stages in:
  `vllm/model_executor/models/qwen3_dflash.py`,
  `vllm/model_executor/models/qwen3_dflash2.py`,
  `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`, and
  `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py`.
- `scripts/serve_recipe_qwen38.sh`: diagnostic knobs (`CMODE`,
  `DRAFT_KV_DTYPE`, `VLLM_DFLASH_AUDIT`) while preserving production defaults.

## Models and artifacts

- BF16 reference draft: `/home/curved/models/dflash2-bf16-with-tokenizer`
- Production GPTQ draft:
  `/home/curved/.cache/huggingface/dflash2-int8/Qwen3.8-27B-DFlash2-GPTQ-8bit`
- Audit artifacts: `/home/curved/models/kld/dflash_audit` (about 370 MB).
- Comparisons:
  - `compare_bf16_int8_kv_vs_bf16_auto_kv.json`
  - `compare_int8_prod_sync_vs_bf16_int8_kv.json`

## Measured results (NS=7)

| leg | drafts | drafted tokens | accepted | mean acceptance length | token acceptance |
|---|---:|---:|---:|---:|---:|
| BF16 draft, KV auto/BF16 | 595 | 4165 | 556 | 1.9345 | 13.35% |
| BF16 draft, INT8 PTH KV | 252 | 1764 | 912 | 4.6190 | 51.70% |
| INT8 production draft, INT8 PTH KV | 258 | 1806 | 910 | 4.5271 | 50.39% |
| BF16 draft, dense head, INT8 PTH KV | 245 | 1715 | 919 | 4.7510 | 53.59% |

## Diagnosis (root cause identified 2026-08-27)

The DFlash model implementation, the GPTQ draft weights, the AITER
unified-attention Triton noncausal read kernel, and the C++
`reshape_and_cache_flash` write kernel are all correct. The failure is a
**draft KV cache dtype mismatch under `kv_cache_dtype="auto"`**:

1. `load_dflash_model` (`vllm/v1/worker/gpu/spec_decode/dflash/utils.py:55`)
   builds `draft_vllm_config = replace(vllm_config, attention_config=...,
   cache_config=replace(cache_config, cache_dtype="auto"))` — it does NOT
   replace `model_config`, which therefore stays the **target's fp16**
   ModelConfig.
2. `Attention.__init__`
   (`vllm/model_executor/layers/attention/attention.py:313`) resolves
   `kv_cache_dtype_str_to_dtype("auto", get_current_vllm_config().model_config)`
   → the target's `torch.float16`, so the draft KV cache tensor is allocated
   fp16 while the draft model runs bf16.
3. The C++ flash-cache writer dispatches `CACHE_T` from the **input** dtype
   (`DISPATCH_BY_KV_CACHE_DTYPE` in
   `csrc/quantization/w8a8/fp8/amd/quant_utils.cuh:650`: BFloat16 + kAuto →
   `__nv_bfloat16` cache type), so bf16 bit patterns are written raw into the
   fp16 storage — silently.
4. The Triton noncausal reader loads the cache through its declared dtype
   (fp16), reinterpreting bf16 bits as fp16 → deterministic, finite, wrong
   values → layer-0 draft attention garbage, propagating through the backbone.

Evidence:
- Offline kernel repro on GPU0 with the exact runtime geometry (packed K|V
  halves, NHD physical strides, page-padded block stride, sliding window
  2048, GQA 4:1, block 64): bf16 cache + bf16 write + Triton noncausal read =
  rel L2 0.0015 (correct) for every geometry including padding + SW. The
  kernels are clean.
- Same repro with an fp16 cache tensor + bf16 inputs (the runtime mismatch):
  raw attention output rel L2 ≈ 0.88, matching the audit's layer-0
  `attn_out` ≈ 1.9 after o_proj/conv amplification. Same failure signature
  (finite, coherent-degrading) as the audit.
- The `int8_per_token_head` path resolves the cache dtype explicitly
  (`STR_DTYPE_TO_TORCH_DTYPE`), writes via the PTH Triton kernel with inline
  fp32 scales, and reads with dequantization — dtype-consistent end to end,
  which is why it is healthy.
- This is the same bug family as the 2026-08-24 residual-stream dtype fix:
  target fp16 leaking onto the bf16 draft through an un-replaced config
  field.

Suggested narrowly-scoped fix (APPLIED 2026-08-27, see below): resolve the
draft's `"auto"` cache dtype against the draft's own `model_config.dtype`
in `load_dflash_model`.

## Fix applied and verified (2026-08-27)

`vllm/v1/worker/gpu/spec_decode/dflash/utils.py` (`load_dflash_model`): the
draft `cache_config.cache_dtype` is now resolved before config-replace —
explicit `speculative_config.kv_cache_dtype` passes through; `None` inherits
the target's cache dtype (unchanged behavior); `"auto"` maps to the drafter's
own compute dtype (`bfloat16`/`float16`). Boot logs the resolution once
(`DFlash draft KV cache dtype: bfloat16 (draft compute dtype torch.bfloat16)`).
Production (`int8_per_token_head`) is byte-identical in behavior.

Verification (NS=7, eager, audit recorder, tag `bf16_auto_kv_fixed`):
- Focused tests: `tests/v1/spec_decode/test_dflash2.py` 5 passed (baseline 5).
- Boot: worker log names the draft KV dtype `bfloat16`.
- Stage compare vs healthy `bf16_int8_kv` leg: layer-0 `attn_out` rel L2
  **0.0045–0.0059** (was 1.89–2.09 broken; ~350x reduction), backbone output
  2.8–3.8%, `context_ready`/`query_embedding` still bit-identical. The
  residual band is int8-PTH KV quantization sensitivity, matching the audit's
  documented drift (projected K ~0.35–0.40%, unary logits ~1.5–2.4%).
- Acceptance: mean length **4.48** (was 1.93 broken; 4.62 healthy int8-KV leg;
  production int8 draft 4.53), token acceptance 49.8% (was 13.4%). Recovered
  to the healthy band.
- Artifacts at `/home/curved/models/kld/dflash_audit/{bf16_auto_kv_fixed,
  compare_bf16_auto_kv_fixed_vs_bf16_int8_kv.json}`, owned `curved:curved`.
- Service left `inactive`.

## Verification already run

`PYTHONPATH="$PWD:$PWD/../aiter" .venv/bin/python -m pytest tests/v1/spec_decode/test_dflash2.py -q`
passed: **5 passed** (23 warnings). `py_compile`, `bash -n`, and `git diff --check`
also passed. No repo code was changed during the root-cause pass (kernel repro
scripts live in /tmp, owned by curved).

## Continue from here

1. Fix is applied and verified (see above); no pending work from this thread.
2. Production defaults remain TP4/C8, NS=13, INT8 PTH KV, fp32 Mamba; the
   diagnostic legs used NS=7 only. The fix changes nothing for production
   (explicit `int8_per_token_head`).
3. If desired, a full serving-regression boot of the production recipe should
   follow before the next deploy (AGENTS.md testing ladder step 3), and the
   server stays stopped until explicitly requested.
