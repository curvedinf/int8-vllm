# Profile-driven mixed-load optimization — 2026-10-11

Follow-up to [the lifecycle profile](agent_mixed_profile_results_20261011.md).
The numerical/model contract remains PTQR target/draft, TP4/C6/NS6, W8A8,
INT8-G128 KV, BF16 gate weights, FP32 recurrent state and custom all-reduce.
All tests use an isolated listener with Nginx stopped and runtime-masked.

## Candidates and isolated checks

1. **Compile-safe tiny b/a projection.** Move the token-count decision inside
   an opaque custom op. Previously a large tracing shape specialized away the
   enabled tiny projection. Large prefills still use the same F.linear; small
   batches use the existing validated FP32-accumulating kernel. Dynamic
   compilation first at M=1664, then M=42/7/64/65/1/0, graph replay, and FP32
   reference tests pass. The op is registered at model import, before cached
   AOT graphs load; its enable switch is included in the compilation cache key
   so rollback does not reuse an optimized graph. A subprocess import test and
   compile-factor regression cover these boot/cache cases. Actual checkpoint checks cover all
   48 layers and four TP shards: 525,312 outputs, 225 BF16 differences from
   rocBLAS, maximum absolute difference 0.015625; FP32-reference tolerance
   rtol=0.004 / atol=0.00002 passes. This is not a bitwise-equality claim.
2. **G128 decode MFMA tile.** `VLLM_G128_I8DOT_MFMA=16` selects a different
   instruction tile without changing Q/K quantization, BF16 P/V operands or
   FP32 accumulation. At 40,334 context, C1/64 splits changes 410.54 to
   308.82 microseconds; C6/32 splits changes 1751.16 to 1356.11 microseconds.
   The compiler reports eight spills before, zero after. Raw FP32 partials
   have mean-relative differences 0.00000009 / 0.00000017 and maximum absolute
   differences 0.00000858 / 0.00001526. This is not a dtype rollback.
3. **Segmented G128 prefill.** `VLLM_G128_PREFILL_SEGMENTS=8` divides the long
   key scan between CTAs and merges stable-softmax partials. It reuses the
   existing shared scratch capacity (256 decode rows x 64 splits can hold
   2048 prefill rows x eight splits), with a capacity guard and serial fallback.
   No extra per-layer memory or cache dtype change. At 40,334 context,
   M=1664 changes 23.50 to 18.35 ms (1.28x); M=512 changes 10.73 to 6.33 ms
   (1.69x). Mean-relative differences versus the existing serial prefill are
   approximately 0.00018; repeated outputs and guard regions pass. Ragged
   sequences include short contexts with empty causal segments. Short contexts
   retain serial dispatch; isolated short-context forced-split tests regressed.

These are isolated kernel results, **not end-to-end serving speedups**.
Wider 8/16-warp decode launches and smaller 32-row query tiles were also
screened and rejected: they were slower. Increasing kpack was slower too.

Reproducible screens (idle GPU, repository ROCm environment):

```bash
HIP_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/kernels/test_gfx908_tiny_ba.py -q
HIP_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/kernels/benchmark_gfx908_i8dot_launch.py
HIP_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/kernels/benchmark_gfx908_prefill_segments.py
```

## Serving gate

The three-kernel candidate completed both agent runs, 24/24 requests each with
exact token accounting. Its compiled graph contains 48 `gfx908_tiny_ba` calls.
A short serving trace confirms 768 actual tiny-kernel launches, mean 8.16 us
(48 calls take 0.392 ms in that short-context sample).

| Same frozen agent workload | Baseline | Three-kernel candidate |
|---|---:|---:|
| Cold wall time | 532.98 s | 457.97 s |
| Cold whole-run output tok/s | 35.44 | 41.42 |
| Warm wall time | 235.14 s | 211.51 s |
| Warm whole-run output tok/s | 80.05 | 89.07 |

These are single sampled, response-dependent runs per arm, not fixed-output
microbenchmarks. Outputs, acceptance and follow-up bodies can differ. For
example, candidate cold c5 calls a tool in its first response, changing later
prompt lengths. Cold c0-t1 reaches the queue before two later initial arrivals,
so its TTFT reduction (264.75 to 168.59 s) also includes an admission-order
change. Per-stream results and exact bodies remain in the run artifacts. These
measurements show approximately 17% cold / 11% warm whole-run throughput gains,
**not 3x**. The individual mixed-step pauses are still substantial.

### Bounded cached-follow-up admission (separate candidate)

`VLLM_GFX908_PREFIX_FAIR_ADMISSION=1` permits one locally cached, short replay
tail to bypass each long FCFS queue head. The request must have at least an 80%
local hit and at most three scheduler blocks left. Lookups are read-only:
no block ownership changes, cache events, tier I/O or new numerical paths.
Normal lookup/allocation checks still run at admission. Priority scheduling,
running requests and the shared prefill budget are unchanged. A per-head guard,
cleared on admission/finish, prevents repeated warm arrivals from indefinitely
bypassing cold work. The scan examines at most 16 queued candidates. Ten new
CPU tests and the existing scheduler suite pass (162 tests combined).

The original combined candidate is not adopted; see the completed quality
checks and ablations below. Artifacts:
`logs/agent_mixed/candidate_cold/`, `candidate_warm/`,
their `*_analysis.json` files and `candidate_host/`.

The admission candidate's cold run completed 24/24 in 458.39 s (39.75 output
tok/s), versus 457.97 s for the kernel-only candidate. Natural output length
differs, including an early EOS in c6-t2, so the whole-run rates are not a
fixed-output comparison. Cached c1-t1/c2-t1 TTFT changes 223.62/184.49 to
124.35/136.00 s; c0-t1 changes 168.59 to 175.16 s. This is bounded admission
relief, not preemption of an already running prefill or elimination of queueing.
Its warm run completed 24/24 in 211.81 s, 88.91 output tok/s, essentially the
same as the kernel-only candidate's 211.51 s / 89.07 tok/s. All four candidate
agent runs have zero exact-token-accounting discrepancies.

Final regression checks before the control boot: 303 CPU/client tests, seven
tiny-projection GPU tests, INT8 KV micro PASS, and battery 4/4 PASS. The control
uses `TINY_BA=0`, `PREFILL_SEGMENTS=1`, `I8DOT_MFMA=0`, `PREFIX_FAIR_ADMISSION=0`
(full environment names above), with all numerical/model settings unchanged.
Its exported graph has zero tiny-projection calls and 48 general projection
GEMMs; see `logs/agent_mixed/control_computation_graph.py`. This is not a
rollback to the historical numerically broken configuration.

### Greedy acceptance check (combined candidate rejected)

The frozen gate (`logs/agent_mixed/greedy_gate.json`, SHA-256
`7b8244a8b40d8668fb606c270c665066a8bb311a6e1157937b29bad75b83c84b`)
uses six simultaneous copies of one 40k shared-prefix prompt, 512 greedy
output tokens each, thinking disabled. This is one prompt repeated at C6,
not six independent quality examples. Generated continuations differ.

| Gate | Control accepted / drafted | Three-kernel candidate |
|---|---:|---:|
| Cold | 2139 / 5670 (37.72%) | 2074 / 6030 (34.39%) |
| Warm | 2142 / 5676 (37.74%) | 2013 / 6414 (31.38%) |

This does not pass a +/-1 percentage-point acceptance criterion. Although no
draft dtype changed and this single prompt cannot establish a general quality
regression, the combined candidate was held back; the completed ablation and
exact-token clean-prefill scoring follow below. Do not present its throughput
numbers as a production-deployed improvement.

Clean-prefill scoring on the control passes **0 / 5170 outside top 20**
(maximum rank 13): candidate cold c0-t0 (1536 tokens), c2/c4/c6-t2 (768 each),
admission-candidate c0-t1 (50) and c4-t2 (768), and candidate greedy warm c0-t0
(512, maximum rank 3). Exact prompt/output IDs and scores are retained under
`logs/agent_mixed/quality/`. This is evidence against gross mixed/cache-path
numerical corruption, not bitwise equivalence or a broad model-quality eval.

The attention-only ablation (tiny projection off, MFMA=16, prefill segments=8,
admission bypass off) scores **2094/5886 = 35.58% cold** and
**2080/5982 = 34.77% warm** on the same greedy gate. Thus the projection is not
the sole source of the acceptance difference. This arm also fails the gate;
its full mixed-workload run was automatically withheld. Artifacts:
`logs/agent_mixed/attention_greedy_{cold,warm}/`.

The serial-prefill arm restores the original prefill reduction and keeps the
compile-safe tiny projection, MFMA=16 decode tile and bounded cached admission.
It scores **2206/5250 = 42.02% cold** and **2159/5544 = 38.94% warm**, versus
37.72%/37.74% on the control. There is no acceptance loss on this gate. The
admission bypass does not change this shared-prefix greedy case: once cached,
the queue head itself has a short replay tail and is not bypassed. The full
agent runs for this arm completed 24/24 each: **471.14 s / 38.39 output tok/s
cold**, **220.91 s / 85.24 tok/s warm**. Versus the profiled baseline, elapsed
time falls about 12% cold / 6% warm; this is not 3x. Natural output lengths and
follow-up bodies differ (serial c4-t2 reaches 51k context), so these remain
response-dependent workload comparisons. Segmented prefill remains off.
Artifacts: `logs/agent_mixed/serial_{cold,warm}/` and
`serial_greedy_{cold,warm}/`.

An additional serial-path oracle scores **0 / 2560 outside top 20**, maximum
rank 12. It covers serial cold c2-t2 (768), c4-t2 (first 512), c6-t2 (768), and
serial greedy warm c0-t0 (512). The reference requests use full prompt
recomputation and large query chunks on the unchanged serial attention and
general projection paths; the optimized decode dispatches are not used for
these prompt logprobs. The c4 slice was shortened to keep the final chunk above
256 tokens. Scores are in `logs/agent_mixed/quality_serial/`.

### Cross-queue admission correction

The serial cold trace exposed a gap in the first admission candidate: c0-t1
entered the regular queue at 77.27 s, alongside three older cold requests in
the deferred queue, and was not admitted until 249.15 s. At admission all
39,936 reused tokens were local; no external cache transfer explains the wait.
The helper had searched only the selected queue, so deferred cold work could
still block a regular cached turn. The corrected helper scans both queues with
one shared 16-candidate bound, after a blocked head becomes runnable, and
retains the one-bypass-per-head guard. Two additional regression cases cover
cross-queue promotion and the shared scan bound; all 164 scheduler tests pass
with the feature enabled. Its completed live results are below.
The serial arm's server had already imported the one-queue implementation:
editing the checkout during that run did not hot-patch its scheduler. Its warm
client snapshot therefore describes a newer checkout than the live scheduler;
use the boot/variant record and retained scheduler spans to distinguish them.
The cross-queue arm is a fresh boot of the corrected source.

The combined final CPU/client suite passes **305 tests** with the policy enabled
(`logs/agent_mixed/final_cpu_tests.xml`). A remaining startup limitation is
visible on this boot: cached AOT loading reports `Kernel index 0 not found in
id_to_kernel` and safely falls back to recompilation (77 s reported). The
missing custom-op registration was fixed, but this separate cache-load failure
has not been resolved. Do not claim every restart now reuses the compiled graph.

The first cross-queue warm run completed 23/24 requests. Its last request lost
the HTTP connection in 0.92 ms before headers and has no corresponding frontend
or scheduler span; the engine remained healthy. This is consistent with the
five-second agent wait racing pooled-connection keep-alive expiry. The client
now uses fresh TCP connections rather than silently replaying inference. The
failed run remains in `crossqueue_warm/` and is not a successful throughput
result; `crossqueue_warm_retry/` is the explicit rerun with the connection policy
recorded. Cold timing used the earlier pooled client and completed 24/24.

## Adopted configuration and final results

The launcher enables compile-safe tiny projection, `VLLM_G128_I8DOT_MFMA=16`
and cross-queue `VLLM_GFX908_PREFIX_FAIR_ADMISSION=1`.
`VLLM_G128_PREFILL_SEGMENTS=1` explicitly retains serial prefill. No model,
quantization, KV dtype, TP, concurrency, speculative count or power-cap change.

| Frozen 40k agent workload | Baseline | Adopted package |
|---|---:|---:|
| Cold wall time | 532.98 s | 468.48 s |
| Cold whole-run output tok/s | 35.44 | 40.19 |
| Warm wall time | 235.14 s | 204.21 s |
| Warm whole-run output tok/s | 80.05 | 92.24 |
| Cold mean cached tool-turn TTFT (t1) | 145.78 s | 69.56 s |
| Cold first chat final completion from run start | 463.39 s | 195.77 s |

Both final runs complete 24/24 with zero token-accounting discrepancies.
These are approximately **13% cold / 15% warm aggregate throughput gains,
not 3x**, and not per-user decode rates. The sampled continuations differ.
The warm result is the explicit fresh-connection rerun described above.
Cache parity was verified: baseline and final warm initial requests each reuse
38,272 tokens; all 16 follow-ups reuse 39,936, all local, with zero externally
loaded cached tokens. All eight cold initial requests have zero cached tokens.
The workload manifest SHA-256 is
`0ab76f18e27687ee0190bfded739778926b457db35329eb682792fb1806bb666`.
Artifacts: `logs/agent_mixed/crossqueue_cold/`, `crossqueue_warm_retry/` and
their `*_analysis.json` files. Bounded fairness has a tradeoff: the last cold
arrival's TTFT is 268.4 s versus 248.4 s with the one-queue prototype (295.2 s
baseline). It does not eliminate cold queueing or interrupt running prefills.

Final exact-token clean-prefill scoring passes **0/2354 outside top 20**,
maximum rank 13, across cold c0-t1, c0-t2, c1-t2 and c2-t2. The reference
recomputes the saved prompts using unchanged serial-prefill numerical paths;
query chunks stay above 256 so neither optimized decode dispatch runs in the
reference. See `quality_crossqueue/`. This supplements the accepted greedy
gate above; it is not a broad model-quality evaluation. Final checks include
305 CPU/client tests, seven tiny-projection GPU tests, INT8 KV micro PASS,
battery 4/4 PASS, and 12 client tests after the connection-policy fix.

### Final serving kernel trace

The same six-copy greedy gate was additionally profiled, with all four workers'
host spans and five rank-0 GPU chunks: **723,732 kernels**. It is a different
workload from the full baseline agent trace, not a phase-timing A/B.
`final_greedy_kernel_analysis.json` and `final_greedy_csv/` retain the analysis.
There are 13,056 actual tiny-projection launches averaging **10.31 us**:
48 calls cost approximately **0.495 ms**, versus approximately 16.5–17 ms
for the 48 general projections in baseline agent decode steps. Large prefills
retain the general projection. Rank-0 GPU activity is 98.48% inside captured
chunks; the remaining decode costs are GEMM 7.84 s, attention 6.01 s and
collectives 2.37 s. Mixed steps still reach **1.25 s**. Long-prefill attention,
GEMMs and collectives remain substantial opportunities; the rejected segmented
prefill experiment must not be enabled without passing its acceptance gate.

Disposable baseline/candidate/rank-0 test prefix-cache namespaces were removed
after their servers stopped, reclaiming about 74 GiB. Their workload artifacts,
metrics and traces remain under `logs/agent_mixed/`; production cache data was
not removed. The cache entries can be regenerated by replaying the manifests.
