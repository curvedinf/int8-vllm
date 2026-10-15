# Bolt continuous optimization

User request: optimize continuously until stopped; current implementation is
the fidelity baseline. Use the established performance methodology, read
outputs for basic coherence, and defer thorough fidelity testing until the
user stops the run. Report current prefill and decode tok/s after every
successful iteration. Production systemd activation remains deferred.

Initial incumbent: vLLM 64101bc3ad, AITER a6aacf7d52. Running diagnostic Bolt
on localhost8021, TP4/C8/NS7, official packed resident weights, INT8 GS128
compute/KV, fp32 Mamba. Preserve unrelated Qwen launcher/AGENTS changes.

Metrics: standard vLLM random32-input/1000-output, 32 requests, C8, greedy,
seed0, two warmups; report output throughput as decode benchmark tok/s.
Prefill: two cold unique-prefix8192-input/1-output requests, input tokens
divided by TTFT, fixed seed/content construction. Keep these distinct from
the earlier tutorial and repetitive-copy profiling rates. Repeat a candidate
when gain is near noise; seek >=5% without a material other-metric regression.
Use natural tutorial outputs for basic coherence. No extensive fidelity gate
during the optimization loop by explicit user instruction.

Batch1 hypotheses, in order of information gain/cost:
1. Replace generic NVFP4/E4M3 ldexp decoding with exact bit construction.
2. Amortize packed conversion across larger M tiles for dense prefill.
3. Replace expensive elementwise correctly-rounded divisions with a shared
   reciprocal for GS128 quantization; preserve round-to-nearest output.
4. Improve decode parallelism via split-K or specialized small-M packed kernels.
5. Reduce repeated packed loads/scales and tune routed-expert tile occupancy.

Benchmark/variant artifacts and original kernel/wrapper rollback copies are
under ignored agents/tmp/bolt-opt/. Baseline: decode 44.95 tok/s (32,000 outputs / 711.85 s); cold prefill
TTFT 32.397/33.403 s, approximately 249 input tok/s. Candidate: exact NVFP4
bit decode, per-group reciprocal quantization, dense NVFP4 BM32/BM256.
Accepted candidate: 555.92 prefill tok/s and 60.33 decode tok/s; coherent
128-token tutorial. User target: approximately 900 aggregate decode tok/s.
Quick-gate incumbent: 601.94 prefill / 136.34 decode tok/s (8 requests).
Split-K8 plus all-codec dense tiling loaded through the fixed idle RPC;
all four workers returned matching hashes and 8 target + 8 draft graphs.
Candidate quick benchmark is running after prefill JIT warmup/coherence.
Next: split-K8 for small-M NVFP4 (micro win); IQ2 wide loads rejected (slower).
Idle reload needs a fresh HIP graph pool before clearing/capturing graphs;
a failed collective RPC leaves response queues misaligned, requiring restart.
Do not run GPU microbenchmarks concurrently with E2E measurements.
