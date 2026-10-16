# Bolt continuous optimization

Keep optimizing until the user stops. Target: approximately 900 aggregate
C8 decode tok/s. Preserve current fidelity baseline; only read outputs for
basic coherence now. Full fidelity checks deferred until the user stops.
Report prefill and decode numbers after every successful iteration and in
progress updates; keep prose minimal. Production activation remains deferred.

User clarified to CONTINUE optimizing after shutdown. Diagnostic server is running with VLLM_BOLT_DENSE_CACHE_GIB=2, GPU_MEMORY_UTILIZATION=0.70.
CPU KV is currently 8 GiB. User requested 24 GiB on the NEXT restart;
launcher updated to 24 GiB. Do not restart solely for this change.
Next launch uses 262144 context ceiling (user requested); live ceiling32768.
Shared GPU cache reports589475 tokens; CPU offload caches prefixes, not extra
simultaneously active GPU KV. TP4/C8 retained; user discussed C6 but only
explicitly requested the ceiling change.
Watchdog: agents/tmp/bolt-opt/watch-memory-2.py, two-hour loop, log
 dense-cache-memory-2.jsonl; stops below8GiB host or above31GiB/GPU.
Current accepted AITER f320662ff:1511.49/340.02 with dense small tiles,
then1541.14 prefill/362.51 decode with partial routed K576 native handling.
All4 rank offsets exact at M8/M64; short chat coherent. Dense cache remains
268entries/2GiB per GPU; about24.5GiB total/GPU and30GiB host available.
New profile: routedIQ2 33.53%, denseGS12817.17%, AR14.88%, BF16 GEMM9.26%,
Mamba4.47%; packed fallback eliminated14.25% ->0.15%. Reprofile major wins.
Latest prototypes ignored under agents/tmp/bolt-opt:
- routed-cache: fully resident expert INT8 saves22-30% C8 kernel time, exact.
- dynamic_cache_probe: GPU histogram and frequency-ranked64-slot cache,
  refresh every256 calls; persistent120-block refresh kernel avoids launch
  overhead. Uniform replay C8 gain ~4% UP/~2% DOWN; static routes overstate
  long-term cache hits. No production cache yet.64 slots/all layers=2.805GiB.
- routed-signed-grid:256KiB signed lookup slows kernels; reject.
Real C8 chat route-frequency probe completed; normal kernels restored.
Top32 experts capture48.2%, top64 62.8%, top128 80.9% of selections.
80 projections, all ranks agree. routing-frequencies.json contains results.
Next: bounded32-slot dynamic cache E2E trial (~1.4GiB/rank).

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

Original 32-request benchmark differs from current 8-request C8 quick gate;
do not mix their rates. Ledger contains the latest AITER bridge results.
Protected unrelated root changes: AGENTS.md, logs/README.md deletion,
scripts/serve_recipe_qwen38.sh, docs/recipes/qwen38_flash_next_support.md.


Reload helper requires idle workers, creates fresh HIP pools, and recaptures
both graph managers (~3s cached). Failed collective RPCs desynchronize worker
response queues: restart instead of retrying. Never overlap GPU micros,
profiling, and unprofiled benchmarks. Warm new prefill JIT shapes before timing.
