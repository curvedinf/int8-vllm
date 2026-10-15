# Bolt continuous optimization

Keep optimizing until the user stops. Target: approximately 900 aggregate
C8 decode tok/s. Preserve current fidelity baseline; only read outputs for
basic coherence now. Full fidelity checks deferred until the user stops.
Report prefill and decode numbers after every successful iteration and in
progress updates; keep prose minimal. Production activation remains deferred.

User clarified to CONTINUE optimizing after shutdown. Diagnostic server is running with VLLM_BOLT_DENSE_CACHE_GIB=2, GPU_MEMORY_UTILIZATION=0.70.
CPU KV is currently 8 GiB. User requested 24 GiB on the NEXT restart;
launcher updated to 24 GiB. Do not restart solely for this change.
Memory watchdog is agents/tmp/bolt-opt/watch-memory.py (30-minute sample loop;
stops server below 8 GiB host available or above 31 GiB GPU used).
Dense-cache integrated micros matched original unsplit packed outputs on 15
shapes; optimized native comparisons show ~3.5x Mamba input / ~4x LM head
speedup at M64. Small N<=512 decode retains native dispatch.
Accepted AITER cache commit: 3f7898b3a. Quick gate: 1537.20 prefill /
321.98 decode; repeat 1438.56 / 321.32. 268 entries, 2 GiB/GPU cached.
Memory ~24.4 GiB/GPU and ~30 GiB host available after recaptures.
Cache-disabled same-boot comparison developed a rejection tail: 1199/98.24. New worker RPC
get_bolt_kernel_memory reports cache entries/bytes and allocator usage.
Next experiment: micro-cached-tiles.py bounded tile/split sweep; no concurrent
serving benchmark during micros. No broad fidelity check.

Diagnostic configuration before shutdown: localhost8021, TP4/C8/NS7, official packed weights,
INT8 GS128 compute/KV, fp32 Mamba. Tuning/dev mode enabled. Key stays in
ignored agents/tmp/bolt-eval/regression.key. Preserve unrelated user changes.

Prior uncached AITER incumbent: fe49fc27a, 1176.45 prefill / 250.93 decode.
Previous CK bridge measured 1315.52 prefill / 252.16 decode, coherent.
Its source is saved ignored at agents/tmp/bolt-opt/packed_aiter_bridge.py and
aiter-ck-wrapper.py. Current cache experiment replaces it with persistent
GS128 weights and AITER blockscale using enable_fp_fusion=False.
Rejected AITER GS128 bridge: 1248.79 / 114.35 (draft-rejection tail).
Rejected extending CK to NVFP4 prefill: 1445.89 / 99.17 (rejection tail).
Rejected LDS codebook: 1198.56 / 185.62. Original shared-UP native split
and ordered reduction also caused rejection tails. All artifacts ignored
under agents/tmp/bolt-opt; micro-ck.py and micro-ck-nv4.py tested real weights.
AITER blockscale launch currently uses FMA; try enable_fp_fusion=False
when testing resident GS128 against packed accumulation.

Original 32-request benchmark differs from current 8-request C8 quick gate;
do not mix their rates. Ledger contains the latest AITER bridge results.
Protected unrelated root changes: AGENTS.md, logs/README.md deletion,
scripts/serve_recipe_qwen38.sh, docs/recipes/qwen38_flash_next_support.md.

Latest short profile (opt7-chat under profile/gpu): routed native experts
2.23s, remaining Triton GEMMs 1.62s, native NVFP4 1.18s of 9.07s GPU time. Profiling has been enabled in live workers via
ignored enable-profile.py; host lifecycle wrappers remain off. No active
GPU profiler. Cached scale metadata passed codec/routed smoke and E2E.
Next structural target: routed IQ2 native path and remaining GEMM overhead.

Reload helper requires idle workers, creates fresh HIP pools, and recaptures
both graph managers (~3s cached). Failed collective RPCs desynchronize worker
response queues: restart instead of retrying. Never overlap GPU micros,
profiling, and unprofiled benchmarks. Warm new prefill JIT shapes before timing.
