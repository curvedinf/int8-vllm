# Bolt continuous optimization

Keep optimizing until the user stops. Target: approximately 900 aggregate
C8 decode tok/s. Preserve current fidelity baseline; only read outputs for
basic coherence now. Full fidelity checks deferred until the user stops.
Report prefill and decode numbers after every successful iteration and in
progress updates; keep prose minimal. Production activation remains deferred.

Live diagnostic: localhost8021, TP4/C8/NS7, official packed weights,
INT8 GS128 compute/KV, fp32 Mamba. Tuning/dev mode enabled. Key stays in
ignored agents/tmp/bolt-eval/regression.key. Preserve unrelated user changes.

Accepted AITER incumbent: 8584635aa. Quick gate: 1014 prefill / 151.31 decode
tok/s. Ledger contains A/B details. Original 32-request benchmark differs
from current prior 8-request quick gate; do not mix their decode rates.
Artifacts, harnesses, and rollback kernels: agents/tmp/bolt-opt/.

Current experiment: split-static (split-K8 NVFP4 at small M). Earlier combined
split-dense regressed decode; it also made every non-split K loop dynamic.
The new candidate preserves static K bounds for SPLIT_K=1 and incumbent tiles.
Idle reload in progress; then warm prefill, read coherence, run quick gate.
Next: short GPU profile via enable-profile.py if needed; packed shared-memory
layouts/kpack and IQ2 wide loads gave little benefit or regressed in micros.

Reload helper requires idle workers, creates fresh HIP pools, and recaptures
both graph managers (~3s cached). Failed collective RPCs desynchronize worker
response queues: restart instead of retrying. Never overlap GPU micros,
profiling, and unprofiled benchmarks. Warm new prefill JIT shapes before timing.
