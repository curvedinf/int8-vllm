# Bolt continuous optimization

Keep optimizing until the user stops. Target: approximately 900 aggregate
C8 decode tok/s. Preserve current fidelity baseline; only read outputs for
basic coherence now. Full fidelity checks deferred until the user stops.
Report prefill and decode numbers after every successful iteration and in
progress updates; keep prose minimal. Production activation remains deferred.

Live diagnostic: localhost8021, TP4/C8/NS7, official packed weights,
INT8 GS128 compute/KV, fp32 Mamba. Tuning/dev mode enabled. Key stays in
ignored agents/tmp/bolt-eval/regression.key. Preserve unrelated user changes.

Accepted AITER incumbent: 91a4eda03. Quick gate: 1169 prefill / 237.10 decode
tok/s. Live server restored to this incumbent. Ledger contains A/B details.
Original 32-request benchmark differs from current prior 8-request quick gate;
do not mix their decode rates. Artifacts and harnesses: agents/tmp/bolt-opt/.

Rejected native shared up split-K: 1136 prefill / 104.25 decode; one request
entered near-total draft rejection, possibly from accumulation rounding.
Reverted. Larger native M tiles were neutral; reciprocal metadata caching
slowed expert micros and was discarded. Official packed weights remain resident.

Current uncommitted AITER candidate: hoist routed down IQ2 scale loads,
aligned routed indexing, unsigned decode indices. Micro results are mixed;
micro-routed-graphs.py is measuring against old-routed.so in one process
with GPU graphs to remove Python overhead. Not loaded into server yet.
Native module reload is wired; native M8192/SK8 caused process-local OOM,
so keep native paths limited to small M.

Latest short profile (opt7-chat under profile/gpu): routed native experts
2.23s, remaining Triton GEMMs 1.62s, native NVFP4 1.18s of 9.07s GPU time. Profiling has been enabled in live workers via
ignored enable-profile.py; host lifecycle wrappers remain off. No active
GPU profiler. Cached scale metadata passed codec/routed smoke and E2E.
Next structural target: routed IQ2 native path and remaining GEMM overhead.

Reload helper requires idle workers, creates fresh HIP pools, and recaptures
both graph managers (~3s cached). Failed collective RPCs desynchronize worker
response queues: restart instead of retrying. Never overlap GPU micros,
profiling, and unprofiled benchmarks. Warm new prefill JIT shapes before timing.
