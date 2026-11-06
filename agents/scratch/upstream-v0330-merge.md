# Upstream v0.33.0 merge (branch merge/upstream-v0.33.0, started 2026-11-06)

Goal: bring fork main (base a556f3fc, 984 local commits) to upstream v0.33.0
(tag 4ba787ec) which contains Flint support (PR #62938, `vllm/models/flint/`).
Doctrine: merge, not history rewrite; deliberate per-hunk resolution; E2E gates
before main promotion.

## Merge decisions log

- Fork-invented scripts stay under `scripts/` (upstream renamed the dir to
  `tools/`; 235 files moved back, systemd/docs/AGENTS hardcode scripts/ paths).
- Fork's `csrc/gdn/fused_gdn_decode_kernel_rocm.cu` follows upstream's
  `libtorch_stable/gdn/` -> `csrc/gdn/` move (sibling kernel moved).
- Helion autotune scripts stay deleted (fork's fresh-slate policy).
- MambaDType keeps fork "int8" literal; MambaCacheMode takes upstream drop of
  "all" (no fork usage). --mamba-cache-mode CLI takes upstream deprecation
  (warn-only, no passthrough into CacheConfig; serve scripts don't use it).
- speculative.py: upstream get_hf_text_config helper + deepseek_v41 dspark
  guard adopted; fork's precise num_mtp_layers warning kept.
- torch_utils: fork OOM-grace pinning kept; new `_pinned_staging` helper
  combines upstream's contiguous pinned staging with grace retries.
- linear.py: fork's gfx908 pre-bound `_vllm_unquantized_gemm` takes priority
  over upstream's `_gemm_impl`; rocm post-load does gfx11 cache-aliasing pad
  THEN gfx908 bind.
- modelopt.py: fork's ModelOptNvFp4LinearMethod + W4A16 (packed_g128 gfx908)
  classes kept alongside upstream's unified ModelOptLinearMethod; both the
  fork IQ2_XXS/Q8_0 packed-G128 dispatch and upstream B12X codec dispatch kept
  (fork branch first). GroupQuantScaleParameter re-imported.
- gptq_shuffle: fork 3-arg (q_weight, q_perm, bit) Python def kept; upstream
  csrc has 2-arg (q_weight, bit) — upstream exllama.py/rdna3_w4a16.py
  auto-merged to 2-arg calls, fork's compressed_tensors_moe_wna16_rdna3.py
  still 3-arg. NEEDS one csrc op with overloads or Python shim. OPEN ITEM.
- q_gemm.cu conflict: kept fork gfx908 v_dot2c_f32_f16 helpers; file now at
  upstream path csrc/quantization/gptq/q_gemm.cu (rename from
  libtorch_stable/quantization/gptq/), fork's repacked-GEMM kernels survived
  the merge in-body.

## Build reconciliation (RESOLVED 2026-11-06)

- Upstream consolidated `csrc/libtorch_stable/*` back to `csrc/*`; the merged
  CMake builds `_C_stable_libtorch` from the consolidated paths and already
  includes `csrc/quantization/gptq/q_gemm.cu` (fork repacked kernels in-body).
- Kept `csrc/libtorch_stable/ops.h` + `torch_bindings.cpp` as a FORK-DELTA
  registration file (STABLE_TORCH_LIBRARY_FRAGMENT): registers only
  masked_situ_and_mul, gptq_w8a16_repacked_gemm, and the 3-arg gptq_shuffle
  overload; upstream's csrc/torch_bindings.cpp registers the shared surface.
  Dropped permute_cols/allspark_w8a16_gemm/rearrange_kn_weight_as_n32k16_order
  (kernels pruned by consolidation, no Python callers).
- Restored the fork's perm-aware `shuffle_exllama_weight` (5-arg) +
  make_sequential_* kernels + 3-arg gptq_shuffle impl into the merged
  q_gemm.cu; upstream's 4-arg variant replaced by NULL-perm call.
- GDN decode ROCm kernel path updated to csrc/gdn/fused_gdn_decode_kernel_rocm.cu
  (follows upstream rename); FUSED_GDN_DECODE + FUSED_KDA_CHUNK compile-def
  blocks both kept.
- csrc/custom_collective_common.cuh: kept fork barrier_at_start_release
  (vLLM CUSTOM all-reduce machinery).

## Other notable resolutions

- KVQuantMode fork INT8_BLOCK_G4..G128 renumbered 20-25 (upstream took 10-14
  for NVFP4_DS_MLA/ULTRAQUANT/OCTAVE); is_int8_block/int8_block_group updated.
- kv_cache_utils: fork's 764-line block dropped — superseded by upstream's
  vllm/v1/core/kv_cache_planning.py (fork lineage, evolved; padding now covers
  any non-MLA attention spec instead of the indexes_kv_by_block_stride opt-in).
- single_type manager: upstream replay_boundaries param subsumes fork
  prefix_replay_offset (same aligned math; coordinator still sets it, now
  unused there); fork prefix_checkpoint_tokens kept as fork_boundaries in the
  mamba partial-tail checkpoint junction check.
- dflash2 speculator: fork init/instrumentation + upstream
  enable_adaptive_verification hook (getattr-guarded); self.top_k aliased.
- dflash/utils: fork draft-KV dtype auto-resolution vs drafter dtype kept
  (2026-08-24 overflow fix) + upstream get_draft_load_config both passed.
- lilicorr model/docs/test: fork side taken (Bolt-production int8 residual
  paths; upstream's copy is the fork's work stripped of them). Speculator
  merged separately by agent.
- rocm_aiter_unified_attn: get_kv_cache_shape API gone upstream; sizing flows
  through customize_spec/state_content_bytes (merged: upstream checks + fork
  int8_block/PTH content computation + hs%G validation).
- exllama/rdna3 gptq_shuffle auto-merged to upstream 2-arg calls; fork
  compressed_tensors_moe_wna16_rdna3.py still 3-arg — both served by the
  overload pair now registered.

## Status

- 19 conflict files remain, owned by running subagents (engine group:
  model_runner/gpu_worker/engine core/multiproc/parallel_state/custom_all_reduce;
  backends+misc: gdn_attn, qwen_gdn_linear_attn, triton_unified_attention,
  chunk_o, compilation passes, serving, profiler, topk_topp; offload:
  offloading scheduler + shared_offload_region; spec-decode group incl.
  lilicorr speculator + samplers + sample/gumbel/prompt_logprob + buffer/
  cudagraph utils + mamba_hybrid).
- After conflicts: syntax sweep → import smoke on pinned venv → rebuild
  extensions → validation ladder → serve boot.

## Serve-boot campaign (2026-11-06 evening, in progress)

Merge E2E gate: micro PASS + battery 4/4 done; serve boot iteratively
deeper. Fixed in sequence: orjson dep; dup gptq_shuffle op registration
(delta file = gptq_w8a16_repacked_gemm only); c.has_g_idx getattr;
phantom 4-tuple _get_weight_params unpacks x4; AIS read-plan 4-tuple;
w_gidx_name default; fork-latent NameError dflash_target_rope_is_neox_style
(defined, attn0 convention); fork-only is_encoder_only guarded getattr.

OPEN MYSTERY: VLLM_GFX908_RESIDUAL_INT8 arrives SET in workers (proven by
control flow: quantized=True raise + draft entering decoder_int8
projection) with no setter in tree/script/profile/systemd/sitecustomize;
isolated subprocess eval says False; in-script env dump got truncated by
the launcher's log redirect. Contained: seam self-disables for hidden
2048 (degrade-not-raise) AND the qwen38 script now pins the env OFF
explicitly. Bolt path untouched (1024/4096 + its own script sets =1).
Boot 14 (pinned) in flight at target-weight AIS load.

PTQR Flint pipeline (all committed): FlintRef+trainer (65 surfaces,
router excluded), corpus 4.22M tok, TP4 equivalence PASS (experts shard
BY EXPERT - double-sharding was the bug), shard-streaming ckpt loader,
PTQRLinear-aware student sharding, torchrun launch path. Rung-0 launch
is next after the boot gate clears (GPUs are busy serving the boot).

## Boot-fix ledger (append-only, through boot 17)

Pattern observed: ~half the blockers are LATENT fork attributes/paths that
gfx908 production never executed and the merged upstream flows now reach:
is_encoder_only (guard), num_layers/num_state_types on MambaSpecDecodeGPUContext
(derive from state arrays), enable_fused_gdn_decode (never assigned anywhere;
guard falls back to the spec_decode attr, False on ROCm). Plus the phantom
residual seam (contained at 3 layers). Boot 15 = first full worker init;
16 = engine assembly reached; 17 in flight.

## MERGE E2E GATE CLOSED (2026-11-06 22:29)

Boot 21: server HEALTH-OK; greedy completion request returned COHERENT
output ("The capital of France is Paris. The capital of Germany is
Berlin..."). Full path exercised: TP4 workers, AIS repack load (target +
GPTQ draft), DFlash2 speculator graph capture, int8_block_g128 KV both
models (startup lines confirmed), mamba fp32, scheduler + offload
connector, first-request fixes. Total campaign: 21 boots, 10 merge
regressions fixed (7 latent fork attrs/paths + phantom-env containment x3
+ 2 API drifts + 1 dup op registration + orjson dep).
Remaining known-open: phantom env source (contained); boot-fix commits
all on merge/upstream-v0.33.0.
