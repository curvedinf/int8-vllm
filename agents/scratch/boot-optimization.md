# Bolt boot-time optimization

## Baseline (clean main, AIS cache hit, 105 W)

| phase | duration |
|---|---:|
| API server + engine + worker spawn | 96 s |
| model init + 213 MB manifest parse | 17 s (11 s JSON) |
| target AIS cache load (28,885 tensors) | 190 s |
| draft cache load | 4 s |
| warmup + dense/expert caches | 48 s |
| CUDA graph capture + startup | 39 s |
| **total to health** | **~6 min 39 s** |

The loader-only time was 127.5 s; a 213 MB `rank0.meta.json` parsed in
11.2 s / 2.2 GB RSS per rank. Storage is a 5400-RPM HDD RAID1 fronted by a
950 GB NVMe bcache (96% available). Warm AIS reads run ~1.3 GiB/s; the
bottleneck was per-tensor `torch.empty` (~5 ms), a pathologically slow
strided `copy_`, and 1,024-iteration Python `run_groups` for down experts,
not the disk.

## Implemented (this checkout)

- `vllm/model_executor/model_loader/ais_loader.py`
  - one device sync per 32 MiB staging window instead of per tensor;
  - one reusable destination arena for caches >= 1,024 tensors (draft keeps
    per-tensor allocations because its LiLiCorr head retains tensors);
  - Triton byte-scatter kernel replaces strided uint8 `copy_`;
  - tqdm `mininterval=1.0`.
- `vllm/fs_gpu/repack.py`
  - fast manifest: `rankN.meta.fast.json` (5.7 MB) + `rankN.runs.npy`
    (225 MB, mmap), with the legacy JSON as fallback and one-time
    `_publish_fast` migration;
  - vectorized numpy `run_groups`.
- Existing cache migrated for both target and draft, all four ranks.
- `VLLM_GFX908_SHARED_OVERLAP` defaults off: the side stream trips the
  native custom AR during graph capture (illegal access at 6/8).

## Measured loader-only (rank 0)

- manifest validate: 0.19 s (was 11.2 s)
- 28,885 tensors / 8.03 GiB cache read: **8.21 s** (was 127.5 s)
- byte parity: embedding, LM head, all 40 expert up/down slices (1,024 runs
  each), draft cache 179 tensors: PASS.

## Remaining boot phases (not yet addressed)

- worker spawn/import/device init: ~96 s;
- converted dense INT8 + expert scale construction: ~48 s;
- CUDA graph capture (8 sizes x target+draft): ~39 s;
- eager vision/audio weight + processor load (cache tail).

## Fused expert cache (implemented)

- `vllm/fs_gpu/fused_experts.py`: per-layer `w13`/`w2` blocks built from the
  repack compact runs by reconstructing each expert's full checkpoint tensor
  and narrowing to the same TP slice the model's `weight_loader` uses. The
  naive compact-concatenation is not valid: one expert in some layers has
  merged/overlapping runs (e.g. layer 3 expert 99, 136,804 vs 135,168 bytes).
- `scripts/build_bolt_fused_experts.py` builds `rankN.experts.bin`
  (4.39 GiB/rank) + `rankN.experts.json`; ranks 0-2 complete, rank 3 building.
- `AisModelLoader` skips `.mixer.experts.` tensors when the fused cache is
  valid (`skip_experts` in `ais_repacked_weights_iterator`) and calls
  `FusedExpertCache.load_into(model)` after the normal weight load.
- Validation: fused layer-1 w13/w2 byte parity True; skip iterator yields
  1,237 non-expert tensors (expected 28,885 - 27,648).

## Worker startup measurements

- Four parallel `import vllm.v1.worker.gpu_worker`: 11.5-11.7 s each, 12 s wall.
- Four parallel HIP context init: 0.14 s each, 3 s wall.
- Therefore the observed 12 s worker stagger is not import or device context;
  it is in the parent spawn/config or distributed-init path. A persistent
  worker pool remains the only way to remove the ~96 s phase.

## Other changes

- Launcher: `CUDAGRAPH_SIZES` defaults to `16,32,48,64` (was 8 sizes).
- Launcher `start`: removes stale `/dev/shm/vllm_offload_*.mmap` and `psm_*`
  when no supervisor is alive (the last failure was a leaked 24 GiB mmap).

## Runtime-path bisect (in progress)

The fused/loader/buffered base booted healthy eager in 393 s (390 s cold).
The fused cache loaded 40/40 layers; non-expert 65.6 s, fused 70-94 s cold.
The native custom-AR crash is therefore in a runtime path. Driver
`agents/tmp/bisect_runtime.sh` runs five eager boots against the base,
writing `agents/tmp/bisect_results.txt`:

1. safe-warm (dense off, gate off, page bf16, SSU off)
2. SSU on only
3. dense split-K/tiers on only
4. gate-i8 on only
5. page FP16 on only

After the culprit is identified, the healthy paths are combined and the
graph-capture + 52-prompt quality + startup measurement follow. The SSU
config is temporarily in `agents/tmp/ssu_config.json.bak` for the off legs.

## Bisect result (2026-10-20)

Five eager boots against the fused/loader base:

- safe-warm (all runtime paths off): 417 s, healthy
- SSU config on: 416 s, healthy
- dense split-K/tiers on: 391 s, **crash** (illegal access in
  `packed_dense_cache.int8_gemm` during warmup; custom AR)
- gate-i8 on: 397 s, healthy
- page FP16 on: 430 s, healthy

Verdict: only the dense decode dispatch crashes. SSU, gate-i8 and page FP16
are safe; the SSU config is restored. `VLLM_GFX908_DENSE_DECODE_DISPATCH=0`
is required until the split-K/custom-AR interaction is fixed.

## I/O path measurements (2026-10-20)

- bcache: 950 GiB device, 95% free, `sequential_cutoff=4 MiB`. Reads >=4 MiB
  bypass and stream from the HDD RAID1 at 50-75 MiB/s; the working set is
  ~30 GiB and only a fraction stays cached. `stats_total` refreshes slowly;
  `stats_five_minute` is the live view.
- Buffered pread of a page-cached 1 GiB region: 5.4-6.8 GiB/s.
- AIS/hipFile of a page-cached region: 1.13 GiB/s at 2 MiB, 1.33 GiB/s at
  32 MiB. Cold (page cache dropped via `fadvise(DONTNEED)`): 60-75 MiB/s.
- Conclusion: hipFile is fine; the boot was HDD-bound. bcache cannot be
  relied on without root.

## Implemented for startup

- `scripts/prefetch_bolt_cache.py`: 8-thread buffered prefetch of the repack
  ranges (manifest order) then the fused-expert files; writes
  `/dev/shm/vllm_bolt_prefetch.done` when finished.
- `ais_loader.py`: `_wait_for_prefetch()` blocks loaders until the sentinel
  exists (env `VLLM_BOLT_WAIT_PREFETCH=1`, 360 s timeout). With 2 MiB reads
  and the wait, the loader was 5.05 s vs 63-72 s cold.
- `multiproc_executor.py`: with spawn, the four `Process.start()` handshakes
  now run in a thread pool; all four workers reached distributed init in the
  same second (was 12 s apart), saving ~36 s.
- Prefetch boot measured 310 s (vs 417-430 s); the 90 s budget only covered
  14.9 GiB, so the fused load was still 76-114 s. Full budget is now 300 s.

## Quality bug found and fixed (2026-10-20)

The combined boot with the fused cache failed the 52-prompt probe (13
repetition loops; all prior corpora were 0/52). Root cause:
`_expert_destination` used `first = start // 256` for w2, where
`start = (rank * local_i) % 256`, so `first` was always 0. Ranks 1-3 got
block 0 of the wrong slice (zeros). Fixed to `first = begin // 256`, matching
`PackedG128MoEMethod.weight_loader`. Regression test:
`tests/model_executor/test_fused_experts.py` (3 passed).

Isolation boot (fused disabled, page FP16 on, no prefetch): healthy 344 s and
the 52-prompt probe passed 0/52 failures, so page FP16 is clean and the fused
cache was the sole quality regression.

`build_fused_cache` now reads one span per layer/key instead of one pread per
expert (27,648 tiny reads made a rank take ~4 min). All four ranks are being
rebuilt and byte-verified against the safetensors loader (rank0 layer 3
experts 0/99 matched before the span change).

## Final combined boot measurements (2026-10-21)

- F1 (fixed fused, full prefetch): **340 s**, healthy.
- F2 (steady state): **316 s**, healthy. Phase log:
  - target construction 0.90 s
  - loader waited 141.5 s for the prefetch sentinel
  - repack load 5-7 s (3.58 GiB), fused experts 4.1 s, draft 6.9 s
  - post-load weight processing 0.43 s
  - prefetch: 30.17 GiB in 203 s = 152 MiB/s (HDD-bound)
- Quality: **PASS 52/52** with fused fixed + page FP16 + gate-i8 + SSU.
- Bisect: dense split-K is the only crasher; launcher default is now 0.
- Parallel spawn: all four workers at the same second (was 12 s apart).

The boot is now: engine+spawn (~60 s, overlapped), HDD prefetch (203 s,
dominant), load (~10 s), post-load/warmup/capture/KV (~110 s). The prefetch
is bounded by the HDD array at ~152 MiB/s; chunk size 2/8/32 MiB made no
difference (150/142/161).

The 24 GiB KV CPU offload evicts the weight page cache between boots, so the
prefetch is cold every time. `KV_OFFLOAD_BYTES` was added to the launcher to
A/B the tier size; 0 GiB boots are being measured to see whether the 30 GiB
working set survives in RAM (61 GiB total). If it does, routine restarts skip
the HDD read entirely; the tradeoff is KV prefix-tier capacity.

## Warm page cache across boots (2026-10-21)

`cpu_bytes_to_use=0` is rejected by the offloading connector. With the
minimum accepted tier (1 GiB):

- prefetch: 30.17 GiB in **18.2 s = 1.7 GiB/s** (all page-cached)
- model loading: 17-20 s; weight loading 2.6-5.1 s
- **boot: 156 s healthy** (from 417-430 s), same fixed fused cache,
  page FP16, gate-i8, SSU, dense decode off.

The 24 GiB CPU KV offload was what evicted the 30 GiB weight working set
between boots. A smaller tier keeps it in the 61 GiB of RAM (weights 30 GiB
+ process ~18 GiB). Tradeoff: less KV prefix-tier capacity. The launcher
default remains 24 GiB; `KV_OFFLOAD_BYTES` was added for A/B boots. Next:
confirm persistence over a real serving lifetime and try 8 GiB.

## KV tier A/B (2026-10-21)

| Tier | Boot | Prefetch | Notes |
|---:|---:|---:|---|
| 24 GiB | 316 s | 203 s (150 MiB/s) | weights evicted every boot |
| 8 GiB | 162/164 s | 17/50 s (613 MiB/s) | partially warm |
| 1 GiB | 156/157 s | 18-24 s (1.3-1.8 GiB/s) | fully warm |

`cpu_bytes_to_use=0` is rejected. The launcher default stays 24 GiB;
`KV_OFFLOAD_BYTES` selects the tier for A/B boots. 8 GiB keeps most of the
benefit (only 6-8 s slower than 1 GiB) with 8x the KV tier capacity.

Remaining ~160 s budget: engine+spawn ~50 s, load 17-20 s, post-load
(warmup/capture/KV/multimodal) ~85 s. The prefetch is no longer the critical
path when the weights stay resident. To go further: trim graph capture to the
two used sizes, reduce the warmup dummy steps, lazy multimodal init, and a
persistent worker pool for the ~50 s startup.

## Runtime optimization bundle (2026-10-21)

Implemented and quality-gated (52/52 PASS) against the then-effective config:

- Dense split-K re-enabled. Root cause of the "custom-AR" illegal access was
  an unmasked scale load in the A8W8 blockscale split-K kernel: for K=1344
  the rounded SPLITK_BLOCK_SIZE makes the last K block index exceed the 11
  scale groups. Fixed by clamping the K loop to cdiv(K, BLOCK_SIZE_K) in
  `aiter/.../gemm_a8w8_blockscale.py` (both kernels). Component tests 6/6.
- MoE latent+shared ARs merged into one collective (`all_reduce_pair`), and
  the G128 AR kernel gained a 64-CTA signal table + `VLLM_GFX908_G128_AR_BLOCKS`
  (built to `vllm/_residual_g128_cap.so`, default remains the 16-CTA .so).
- MoE branch combine deferred to the next norm (`ResidualPair` + a two-payload
  `_add_norm` variant); toggle `VLLM_GFX908_MOE_PAIR=0` for rollback.
- Gate-i8 is used for decode as well as prefill (`_gate_logits_i8` has no
  prefill guard); the fused top-k epilogue was not needed.
- Adaptive spec verification (`VLLM_DFLASH_ADAPTIVE_NS=margin`): the scheduler
  trims each request's next draft block to last accepted + margin, default off.

Serving bench (random 4k/256, C8, 16 prompts; acceptance ~1.05):
pre-bundle 34.75 tok/s / TPOT 143.4 ms; bundle 47.60 tok/s / TPOT 125.9 ms.

## Env propagation fix (critical)

Fork-local flags exported by the launcher did not reach EngineCore/workers:
the worker initial env contained only a registered subset (AITER overrides,
expert cache GiB, ...), so `DENSE_DECODE_DISPATCH`, `G128_PAGE_MMA`,
`ADAPTIVE_NS`, `RESIDUAL_G128_LIBRARY`, the AR cap and `BOLT_WAIT_PREFETCH`
were silently ignored. The launcher now writes all `VLLM_*` vars (minus the
API key) to `/dev/shm/vllm_bolt_runtime_env.json`; `run_engine_core` and
`WorkerProc.worker_main` apply it before reading flags
(`vllm/utils/bolt_runtime_env.py`). A boot log line reports the applied count.

All earlier A/B results measured code defaults, not the intended flags; the
true bundle config needs a fresh quality gate after this fix.

## True bundle validation and adaptive result (2026-10-21)

After the env propagation fix, the true config (dense split-K, page FP16,
gate-i8, pair fusion, 64-CTA AR) applies all 31 launcher vars in the engine
and every worker. Quality: **PASS 52/52**.

Adaptive DFlash verification (VLLM_DFLASH_ADAPTIVE_NS) was implemented in the
scheduler (trim each request's next draft block to last accepted + margin) and
made to force synchronous scheduling (async scheduling bypasses the trim).
It works (draft tokens 28k -> 4.2k, trim log active) but **regresses**:
29.2 tok/s / TPOT 171 ms vs 43-47 tok/s / 120-126 ms. Cause: the draft still
proposes 7 tokens per step while fewer are verified, so the fixed draft cost
is paid per emitted token; forcing sync scheduling adds overhead. The lever
stays default-off. A future version must reduce the draft proposal length
too, not just verification.

## Final runtime A/B (2026-10-21, random 4k/256, C8, acceptance ~1.0)

| Config | tok/s | TPOT ms | TTFT ms |
|---|---:|---:|---:|
| pre-bundle (old .so, 16 CTA) | 34.75 | 143.4 | 18.8 |
| bundle, 16 CTA | 35.64 | 144.3 | 17.9 |
| bundle, **64 CTA** | **42.12** | **126.3** | **13.9** |
| bundle + adaptive NS (sync) | 29.22 | 171.2 | 21.3 |

The 64-CTA G128 AR is the dominant measured win (+18% throughput over the
same binary at 16 CTA); dense split-K + pair fusion + page FP16 + gate-i8 are
roughly neutral on this low-acceptance random corpus. Quality: true bundle
52/52 PASS. Launcher now defaults to the 64-CTA .so
(`vllm/_residual_g128.so`, old 16-CTA kept as `_residual_g128_16cta.so`) and
`VLLM_GFX908_G128_AR_BLOCKS=64`. Adaptive NS stays default-off (negative).

Caveat: random prompts give draft acceptance ~1.0; production agent prompts
measured ~3.8, where dense/pair/AR-merge contributions can differ.

## Shared overlap and admission (2026-10-21)

- Shared-vs-routed overlap is now capture-safe: healthy boot at 159 s with
  the 8-size graph, 52/52 quality, and **45.36 tok/s / TPOT 120.7 ms /
  TTFT 12.3 s** vs 42.12 / 126.3 / 13.9 serial. Promoted to the launcher
  default (`VLLM_GFX908_SHARED_OVERLAP=1`).
- Prefill-informed expert admission (`VLLM_BOLT_PREFILL_ADMIT=1`) measured
  **42.49 tok/s / TPOT 124.6 / TTFT 13.9**, worse than the default on this
  corpus; left off.
- 128-row expert tiles are blocked: `packed_experts` accepts only 16/32/64
  and the earlier experiment was sub-noise; not implemented.
- Final launcher defaults: dense dispatch 1, shared overlap 1, 64-CTA G128
  AR, page FP16, gate-i8, pair fusion on, adaptive NS off, admission off.
  The 64-CTA kernel is `vllm/_residual_g128.so` (16-CTA copy kept as
  `_residual_g128_16cta.so`).
