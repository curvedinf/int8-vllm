# 40k agent/chat lifecycle profile — 2026-10-11

This is a diagnostic baseline, **not a performance fix or speedup claim**.
The test reproduces slow long-context streaming and burst pauses while other
requests enter prefill. It does not reproduce every reported 5–10 tok/s case.
The numerically corrected production recipe is unchanged.

## Workload and controls

- vLLM base `d363a0351`, AITER `83e07b940`; TP4/C6/NS6, PTQR target/draft,
  AITER W8A8, INT8-G128 target/draft KV, FP32 Mamba state, round activation
  quantization, custom all-reduce. The current decode-priority scheduler stays on.
- Eight staggered 40,334-token chats, three turns each: long answer, actual
  `read_file` tool call, mocked tool result and analysis. Output budgets
  1536/256/768, natural EOS, production sampling defaults. No host tool executes.
- Nginx stopped for all measurements; loopback API only. An automated restart
  during a later boot was caught by the guard before test requests were sent;
  the service was then temporarily runtime-masked. No background load
  generator. Separate disposable disk-cache namespaces leave production cache
  intact. All server/client work runs as `curved`.
- Frozen manifest digest:
  `0ab76f18e27687ee0190bfded739778926b457db35329eb682792fb1806bb666`.
  Baselines use response-dependent agent turns. GPU profiles replay their exact
  saved bodies and send offsets; generated outputs and cache-tail hits can differ.

See [the protocol](agent_mixed_profile.md) for reproduction and metric definitions.

## Baseline: what the client actually sees

Both runs completed 24/24 requests with zero difference between streamed token-ID
counts and reported completion usage. These are transport completions: some
budget-limited replies spend the entire output budget in reasoning, so this is
not a 24/24 useful-answer quality score.

| Measurement | Cold agent run | Warm agent run |
|---|---:|---:|
| Entire workload elapsed | 532.98 s | 235.14 s |
| Output tok/s, including queue and prefill | 35.44 | 80.05 |
| Initial prompt prefix hits | 0 | 38,272 |
| Representative slow analysis-stream decode | 16.92 tok/s | 17.21 tok/s |

The cold initial streams decode at roughly 22–28 tok/s. They repeatedly pause
for 1.1–1.3 seconds between streamed bursts; one reaches 2.79 seconds. Warm cache
does not eliminate approximately 1.3-second pauses during replay prefills.
These are speculative **stream-event gaps**, not individual-token ITL.

For example, cold `c2-t2` runs at 16.87 tok/s while another request is waiting for
its first token and 29.01 tok/s afterward. This is a within-run phase comparison,
not a matched optimization A/B. The contention definition includes queue wait;
GPU mixed-batch attribution is analyzed separately.

## Cache hits versus queueing

Cold `c0-t1` reuses **39,936 of 41,912 prompt tokens**, but client TTFT is
264.75 seconds. Server metrics separate **261.49 seconds in queue** from
**3.12 seconds after admission**. `c1-t1` similarly spends 271.18 seconds queued
despite a 39,936-token hit. This is not a 265-second cache lookup or copy.

The test therefore disproves a blanket “prefix caching is off” explanation.
It does not prove that disk promotion, recurrent-state replay, or admission
policy is optimal. Cached tails still need computation, and the current policy
can make a warm agent follow-up wait behind earlier long prompts.

The 39,936-token hit still leaves roughly 2k tokens to replay on a tool-call
turn, and about 4k on the subsequent analysis turn. A high hit count is not
equivalent to zero prefill work for this hybrid recurrent/attention model.

Frontend render/tokenization spans average **80 ms cold / 88 ms warm** across
the 24 requests (maxima 106/111 ms). Scheduler CPU execution totals only
**2.62 s cold / 1.75 s warm**. These measurements point to admission policy and
GPU execution, not hundreds of seconds spent tokenizing or running Python
scheduler code. Host spans are nested/asynchronous and cannot be added to GPU
time as if they were disjoint elapsed-time components.

Both baseline runs retain the existing **105 W per-GPU power cap**. Device busy
gauges average about 97–98%; observed clocks commonly fall between 731 and
1087 MHz. No clock/power changes or hardware A/B were performed.

## Compiled gate-projection dispatch

The deployed environment has `VLLM_GFX908_TINY_BA=1`, but the saved compiled
backbone graph contains `rocm_unquantized_gemm_gfx908` for BF16 `[24, 5120]`
GDN b/a weights, not the tiny-BA kernel. In the first all-rank trace segment,
rank 0 executes **1,728** instances of the corresponding rocBLAS
`Cijk_...MT128x128x64...` kernel across **36 decode steps**: exactly 48 per step.
They take 0.565 seconds in total, approximately **15.7 ms per step**, against a
71.5 ms median decode GPU window. No `_tiny_ba_kernel` appears in that segment.

This identifies a concrete dispatch opportunity while retaining BF16 gate
weights and FP32 recurrent state. The source's `num_tokens <= 64` Python branch
is absent from the captured graph; tracing-time shape specialization is a
plausible explanation, not yet a separately validated root cause. A fix should
make the intended shape dispatch survive compilation and pass the existing
numerical/acceptance gates. Do **not** re-enable the numerically rejected
LLMM1/wvSplitK paths or change state precision to chase this time.

The full compiled graph is preserved locally as
`logs/agent_mixed/deployed_computation_graph.py`. The exact kernel symbols and
per-phase counts remain in `cold_preview_analysis.json` and its raw CSVs.

The completed rank-0 cold capture confirms this is not just a startup artifact:
169,584 of these kernels across 3,533 decode steps (48 per step) take **58.47 s**,
or **16.55 ms per decode step**. The warm capture gives 67,056 calls across
1,397 decode steps, **23.73 s / 16.98 ms per step**. This is an opportunity,
not a claim that removing this cost alone would deliver a 3x speedup.

## Complete request-lifecycle kernel results

Both final replays completed 24/24 requests with exact streamed/usage token
accounting. Cold recording contains **9,061,701 kernels in 60 chunks**; warm
contains **3,477,714 kernels in 24 chunks**. All recorded kernels have a HIP
launch correlation. Coverage is continuous request execution on **rank 0**,
plus CPU spans on all four ranks; separate short traces cover all four GPUs.
It is not full-workload GPU coverage of every rank.

| Rank-0 GPU step window | Cold median / p95 | Warm median / p95 |
|---|---:|---:|
| Decode only | 74 / 116 ms (3,533 steps) | 115 / 146 ms (1,397 steps) |
| Mixed decode + prefill | 962 / 1,231 ms (245 steps) | 709 / 1,266 ms (85 steps) |
| Prefill only | 889 / 1,074 ms (26 steps) | 452 / 1,142 ms (3 steps) |

These are first-to-last kernel windows associated with a scheduler step,
including target, sampling and draft work, not per-token latencies. Cold/warm
have different batch occupancy and history, so the table is not a matched
kernel speed comparison. A representative warm mixed step schedules five
7-token speculative decode sequences beside a 1,664-token replay prefill and
takes 1.266 s: attention 514 ms, GEMMs 386 ms, collectives 194 ms, GDN recurrence
77 ms, with smaller remaining operations. Decode priority spaces these stalls
out; it does not make the individual prefill step short.

| Summed kernel time on rank 0 | Cold decode | Cold mixed | Warm decode | Warm mixed |
|---|---:|---:|---:|---:|
| GEMMs | 137.14 s | 80.74 s | 65.73 s | 21.24 s |
| Attention | 60.31 s | 64.66 s | 57.50 s | 26.23 s |
| Collectives | 14.54 s | 36.79 s | 9.38 s | 8.92 s |
| GDN recurrence | 9.08 s | 15.27 s | 4.16 s | 3.67 s |

The leading decode kernels are the BF16 small b/a projection above and
`_g128_i8dot_kernel`: the latter consumes **57.67 s cold / 56.20 s warm**.
Mixed attention is led by `g128_prefill_core` (**53.25 / 20.45 s**); the
two-stage custom all-reduce consumes **34.71 / 8.24 s**. Exact CK tile symbols,
counts, maxima, phase breakdowns and the 20 longest steps are in the analysis
JSON, avoiding ambiguous labels such as “all GEMM time is target INT8.”

Launch-origin attribution assigns **456.16 / 202.11 s** to target execution,
**48.40 / 18.96 s** to draft proposal, and **10.22 / 2.91 s** to sampling.
Categories are summed kernel durations and may overlap; they are not additive
wall-clock budgets. Rank-0 union kernel busy time is **515.70 / 224.33 s**,
about **98.9% / 99.0% inside captured chunk windows**. This is a busy GPU
execution path, not a mostly idle GPU waiting for Python scheduling.

Recorded HIP `hipEventSynchronize` totals **501.75 / 221.37 s**. Those are
host waits for GPU completion, not an additional independent compute cost.
Recorded DtoD copies total **2.20 / 0.81 s**, DtoH **0.22 / 0.06 s**. These
counts do not include all transfers implemented as kernels, peer accesses or
storage-side operations; they must not be interpreted as total KV traffic.

Filesystem spans show real tier activity: cold has 224 store batches totaling
20.96 s and three load batches totaling 0.17 s; warm has 27 store batches
totaling 1.06 s and 13 load batches totaling 2.82 s (maximum 0.53 s).
Filesystem lookup totals only 29 ms cold / 5 ms warm. These background I/O
spans can overlap compute. They do not account for the multi-minute admission
delays. Both runs finished with no outstanding cache promotion/cascade work.

### Prioritized optimization opportunities

1. **Reduce the duration of each mixed prefill step**, not merely how often it
   runs. Profiled stalls are dominated by the existing G128 prefill attention,
   CK GEMMs and collectives. Any smaller work quantum must respect the hybrid
   model's recurrent-state/cache alignment; measure prefill progress and queue
   latency alongside decode, rather than hiding starvation in aggregate tok/s.
2. **Make small b/a dispatch survive compilation.** The configured fast path is
   absent, with about 17 ms/step spent on the general BF16 projection. Keep its
   numerical contract and verify the actual compiled graph after changing it.
3. **Optimize G128 long-context decode attention.** It is the largest single
   warm decode kernel family. Preserve INT8 KV and current numerical gates.
4. **Reduce recurrent replay tails and improve follow-up admission fairness.**
   Prefix hits leave 2–4k tokens of work; a cached follow-up can sit behind cold
   requests for minutes. Cache-hit percentage alone misses both problems.

These are measured targets for subsequent A/B work, not implemented fixes or
promises that any one change reaches 3x.

## Trace collection and limitations

The GPU-only Kineto preflight passed on all four workers, including custom
all-reduce graph replay and RCCL broadcast. A full-model smoke produced kernel,
runtime-call and copy traces on every rank. Native rocprofv3 passed single-GPU
testing but failed bounded TP4 preflights with excessive host-memory growth;
the patched serving runtime was not replaced to accommodate the profiler.

CPU JSONL spans cover frontend rendering, scheduling/admission, cache lookup and
I/O, worker execution, input/attention preparation, sampling and draft proposal.
GPU traces retain individual graph-replay kernels and HIP launch correlations.
Clock calibration connects their epoch-relative timestamps to host/client
monotonic time. CPU stack sampling complements these spans. The final process-tree
capture used 5 Hz, idle/nonblocking sampling without native frames: 286,651
samples, 4,334 sampling errors and two lag warnings. It is qualitative evidence,
not a CPU utilization measurement.

The first monolithic cold capture completed the requests but failed export at
the isolated unit's memory limit. PyTorch's compressed exporter staged a large
uncompressed file in RAM-backed `/tmp`. Increasing bounded swap and serializing
export did not save that capture. A temporary swap file was removed afterward;
the partial export and client/CPU records were retained, not counted as a
successful full GPU capture. The replacement uses 64-step chunks and disk
staging. A second all-rank attempt retained event memory across chunks and
exhausted its limit. Explicit collection releases PyTorch's cyclic profile
objects (verified against the installed build), but residual native-recorder
growth remains. Continuous capture therefore selects **rank 0**, with
all-worker CPU spans and separate all-rank mixed-phase samples. Its unit has a
50 GiB RAM cap and 8 GiB swap allowance. A CPU barrier after export prevents
cross-rank export skew in the final run. Profiling pauses mean end-to-end
timings are not clean overhead estimates or substitutes for the baselines.
Model loading/compilation is outside these request-lifecycle captures.

The final cold/warm profiled wall times are 687.48 / 288.87 s, with
27.48 / 65.16 whole-run output tok/s. Rank-0 rotation spans total
174.92 / 66.94 s (59 / 23 rotations). Export pauses, changed batching and
replay prefix-tail differences prevent treating these as clean profiler-overhead
A/Bs. The successful unit used no swap. No model kernels or numerical settings
were changed by this work; lifecycle wrappers are absent when disabled.

## Local artifacts

Under ignored `logs/agent_mixed/`:

- `manifest_v3.json`, `cold_v3/`, `warm_v3/`: frozen corpus, complete request
  bodies/results, exact token bursts, metrics and system snapshots.
- `cold_v3_analysis.json`, `warm_v3_analysis.json`: per-stream timings and
  client-side contention intervals.
- `baseline_host_v2/`: baseline host spans (filter by each run's monotonic window).
- `profile_cold/`, `profile_host/`: initial monolithic attempt (incomplete GPU
  export; excluded from kernel conclusions).
- `chunk_cold/`, `chunk_host/`: partial all-rank capture; not a completed run.
- `rank0_cold/`, `rank0_warm/`, `rank0_host/`: continuous selected-rank replay
  evidence and all-worker host spans, with per-chunk clock calibration.
- `rank0_cold_kernel_analysis.json`, `rank0_warm_kernel_analysis.json`:
  full kernel/host summaries; matching `rank0_*_csv/` retain every event.
- `rank0_cpu.speedscope.json`: final bounded process-tree CPU sampling.

Calibration failures are retained separately and excluded from baseline results.
Raw artifacts include documentation and generated text; review before publishing.
