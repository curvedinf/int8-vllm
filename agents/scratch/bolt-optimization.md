# Bolt continuous optimization

Keep optimizing until the user stops. Target: approximately 900 aggregate
C8 decode tok/s. Preserve current fidelity baseline; only read outputs for
basic coherence now. Full fidelity checks deferred until the user stops.
Report prefill and decode numbers after every successful iteration and in
progress updates; keep prose minimal. Production activation remains deferred.

Live diagnostic: localhost8021, TP4/C8/NS7, official packed weights,
INT8 GS128 compute/KV, fp32 Mamba. Tuning/dev mode enabled. Key stays in
ignored agents/tmp/bolt-eval/regression.key. Preserve unrelated user changes.

Accepted AITER incumbent: a1dfa926d. Quick gate: 1179 prefill / 217.82 decode
tok/s. Ledger contains A/B details. Original 32-request benchmark differs
from current prior 8-request quick gate; do not mix their decode rates.
Artifacts, harnesses, and rollback kernels: agents/tmp/bolt-opt/.

Current experiment: native-routed. Native HIP now handles routed IQ2 expert
matrices on small-M requests using the existing sorted route plan. Loaded
successfully on all four ranks; run-quick.py native-routed is running.
Native dense Q8/IQ2 and NVFP4 are accepted. Native module is reloaded by the
wrapper, clearing the library cache automatically. Native M8192/SK8 hit
process-local OOM; keep native paths limited to small M.

Latest short profile (opt3-* under profile/gpu): packed GEMMs still 82% of
chat GPU time, 70% prefill. Profiling has been enabled in live workers via
ignored enable-profile.py; host lifecycle wrappers remain off. No active
GPU profiler. Cached scale metadata passed codec/routed smoke and E2E.
Next structural target: routed IQ2 native path and remaining GEMM overhead.

Reload helper requires idle workers, creates fresh HIP pools, and recaptures
both graph managers (~3s cached). Failed collective RPCs desynchronize worker
response queues: restart instead of retrying. Never overlap GPU micros,
profiling, and unprofiled benchmarks. Warm new prefill JIT shapes before timing.
