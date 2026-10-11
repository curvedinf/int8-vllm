# Model-free CAR quantization study — 4× MI100

The initial study below is model-free. A separate **MLP-only full-model
follow-up** is recorded at the end of this page; its results must not be
confused with the synthetic format comparison.

**Verdict: keep production CAR unchanged.** None of the tested BF16-input
compressed implementations beat the native automatic CAR path. The most
promising *format* for another implementation is linear INT8 with FP32 group
scales. Adaptive-log quantization preserves small magnitudes but costs more
and is particularly vulnerable to cancellation.

No target or draft checkpoint was loaded for this study. Nginx was stopped
and runtime-masked; the vLLM service was stopped and all four GPUs confirmed
idle. Only standalone tensor kernels ran. The unchanged production service
was restarted afterward. There is no model-quality or generation-speed claim.

## What was measured

The standalone HIP library reuses native CAR's **uncached IPC allocations**
and signal layout, with SYSTEM-scope release/acquire synchronization. It
does not install a new serving mode or modify the production library.

- BF16 and FP16 inputs, TP4, hidden width 5120.
- Primary shapes: 7, 42, 128, 1664 and 2048 rows. An initial pass also
  covered 1 and 512 rows.
- Native CAR: current automatic selection and forced one-stage; both include
  the production staging copy. BF16→FP16→CAR→BF16 is timed with both casts.
  That boundary-cast control uses forced one-stage; an automatic/two-stage
  boundary-cast control was not run.
- Compressed one-stage: pack each rank's input, read peers' codes/scales,
  decode and sum in FP32 in rank order, then cast once to the output dtype.
- Compressed two-stage: pack, compressed reduce-scatter, FP32 accumulation,
  output-dtype intermediate, then **uncompressed output-dtype all-gather**.
  No second quantization of the sum. A conservative final barrier is retained.
- HIP graph bursts of 16 collectives, 10 repetitions/trial, five trials.
  Tables use the median of per-trial maximum rank times. No GPU profiling
  during these latency runs. Native graph pointer slots are registered before
  replay, just as in production.

The restored serving process exports `VLLM_TP_AR_FP32=car1` but leaves
`VLLM_CUSTOM_ALLREDUCE_ALGO` unset; its effective native CAR dispatch is
automatic. This study did not change that behavior. Native two-stage rotates
FP32 accumulation order by slice owner, unlike the fixed-order prototypes;
it is a runtime control, not a claim of numerical equivalence to one-stage.

Synthetic checks use an independent CPU encoder, an exact sum of GPU-decoded
rank payloads, zero groups, outliers, wide log ranges, nearly cancelling
partials, `1e10`, `1e30`, `1e-30`, and BF16 endpoints. The supplemental sweep
includes **all 65,280 finite BF16 bit patterns**, grouped in magnitude order.
Another test places the largest and smallest BF16 magnitudes in the *same*
group. These are different tests: the ordered sweep alone hides within-group
range problems.

Across the primary and supplemental runs, 3,240 changing-input graph replays
per rank passed, including rank-dependent arrival skew. GPU-decoded peer sums
matched the implementation's finite outputs exactly. Expected nonfinite
results from unsuitable scale encodings are recorded, not treated as safe.

## Transport formats

| Input packing | Scale/parameter dtype | Packed bytes/value | Reduction/output |
|---|---|---:|---|
| Native | None | 2.000 | FP32 sum → BF16/FP16 |
| Linear INT8 GS32 | FP32 | 1.125 | FP32 sum → input dtype |
| Linear INT8 GS16 | FP32 | 1.250 | FP32 sum → input dtype |
| Linear INT8 GS32 | FP16 | 1.0625 | FP32 sum → input dtype |
| Linear INT8 GS16 | FP16 or BF16 | 1.125 | FP32 sum → input dtype |
| Linear INT8 GS16, square/cube scale | FP16 root, expanded in FP32 | 1.125 | FP32 sum → input dtype |
| Adaptive-log INT8 GS16 | Two FP16 parameters | 1.250 | FP32 `exp2` + sum → input dtype |
| Adaptive-log INT8 GS16 | Two FP32 parameters | 1.500 | FP32 `exp2` + sum → input dtype |

Different ranks have different scales: summing INT8 codes directly is invalid.
The byte counts describe packed **input**, not the complete two-stage protocol.
For example, GS32/FP32 reduces one-stage remote payload by 43.75%, but only
21.875% across the two-stage reduce-scatter + uncompressed all-gather phases.

The adaptive-log representation is
`sign(q) * 2^(offset + (abs(q)-1)*step)`, with `q=0` reserved for zero.
It implements the adjustable logarithmic spacing proposed as `S1^(W*S2)`;
the additional offset anchors the group's magnitude interval. Signed INT8
provides 127 nonzero magnitude levels per sign, not 256 positive levels.
Calibration and encoding use the **rounded transmitted parameters**.

## End-to-end collective latency

BF16 input/output, µs; packing, synchronization and all-gather included.
Compressed rows below use two-stage unless marked otherwise.

| Transport | 7 rows | 42 rows | 128 rows | 1664 rows | 2048 rows |
|---|---:|---:|---:|---:|---:|
| Native automatic | 13.06 | 34.10 | 50.14 | **546.67** | **666.37** |
| Native forced one-stage | **12.57** | **34.04** | 88.92 | 1090.20 | 1313.53 |
| FP16 wire, BF16 boundary casts, one-stage | 25.62 | 60.46 | 121.95 | 1184.35 | 1466.42 |
| Linear INT8 GS32 / FP32 scale | 23.61 | 39.62 | 72.99 | 726.60 | 899.01 |
| Linear INT8 GS16 / FP32 scale | 22.87 | 37.84 | 71.54 | 712.16 | 878.32 |
| Linear INT8 GS16 / FP16 scale | 24.19 | 39.01 | 72.15 | 711.86 | 896.65 |
| Linear INT8 GS16 / BF16 scale | 25.86 | 39.99 | 74.40 | 772.02 | 961.72 |
| Square-scale INT8 GS16 / FP16 root | 26.25 | 39.89 | 75.24 | 767.96 | 968.50 |
| Cube-scale INT8 GS16 / FP16 root | 26.06 | 40.32 | 75.22 | 779.43 | 978.78 |
| Adaptive-log GS16 / FP16 parameters | 27.81 | 42.04 | 81.59 | 912.41 | 1145.31 |
| Adaptive-log GS16 / FP32 parameters | 27.17 | 42.45 | 81.88 | 904.51 | 1134.12 |
| Bounded adaptive-log GS16 / FP16 parameters¹ | 28.02 | 42.36 | 82.71 | 917.47 | 1132.43 |

¹ Supplemental run; its native automatic control was 12.86 / 33.70 / 50.11 /
550.35 / 674.64 µs. Do not treat differences of a few µs across runs as wins.

At 1664 rows, one-stage GS32/FP32 was 832.52 µs: faster than native
**one-stage**, but slower than native automatic **two-stage**. Algorithm choice
must not be confused with a quantization win. FP16-source tensors have a
narrow exception: 42-row GS16/FP32 two-stage was 37.57 µs versus native
automatic 50.16 µs. That is not a BF16-production transport result.

## Numerical behavior

Relative L2 error of the reduced BF16 output versus fixed-rank FP32 summation
of the original BF16 partials, followed by BF16 rounding. Percentages are
synthetic diagnostic errors, **not** model quality or acceptance scores.

| Format | Normal | Sparse ×300 outliers | 24-bit log range | Near cancellation |
|---|---:|---:|---:|---:|
| Linear GS32 / FP32 | 0.581% | 1.053% | 0.635% | 116.934% |
| Linear GS16 / FP32 | **0.522%** | 0.730% | **0.519%** | **103.597%** |
| Linear GS16 / FP16 | 0.523% | 0.742% | 0.521% | 103.691% |
| Linear GS16 / BF16 | 0.528% | 0.780% | 0.537% | 103.561% |
| Square scale / FP16 | 0.524% | 0.750% | 0.522% | 103.880% |
| Cube scale / FP16 | 0.524% | 0.748% | 0.522% | 103.968% |
| Adaptive-log / FP16, including bounded | 0.877% | 0.298% | 2.006% | 171.824% |
| Adaptive-log / FP32 | 0.875% | **0.153%** | 1.985% | 173.001% |

Log quantization really does retain small values. On rank 0's 40,960-value
log-range tensor, GS16 linear/FP32 zeroed 24,921 nonzero inputs; log quantization
zeroed none. However, protecting relative accuracy of tiny inputs costs
accuracy on large inputs, which dominate the L2 metric. On the outlier test,
log quantization both preserves small values and improves the reduced L2 error.

Cancellation is the main quality risk for CAR: errors on large independent
rank partials can exceed the small true result. More dynamic range, smaller
groups and FP32 accumulation do not remove error already introduced before
the sum. Actual callsite histograms and quality/acceptance gates are still
required before any deployment.

### Full BF16 range

- Linear FP32 scales produced finite outputs for the complete BF16 sweep,
  with no nonzero values dropped in its magnitude-ordered groups. Relative L2
  errors were 0.312% for GS32 and 0.294% for GS16. Mixed extreme groups still
  lose tiny values: this is quantization resolution, not scale overflow.
- BF16 scales failed on 130 sweep outputs: `amax/127` can underflow even
  though the original value itself is representable in BF16.
- Ordinary FP16, squared FP16 and cubed FP16 scales failed on 54,784, 44,288
  and 33,792 sweep outputs respectively. Squaring/cubing extends range but
  does **not** cover BF16. These prototypes explicitly return NaN for invalid
  scales; they do not silently select an uncompressed fallback.
- Raw FP16 log parameters passed the ordered sweep but overflowed 1,280
  outputs in the mixed-endpoint test. Rounded `step` can move the top code
  above the finite BF16/FP32 endpoint.
- Bounding reconstructed magnitudes to finite BF16 fixed that endpoint
  failure; the bounded variant retained both endpoints, including the smallest
  subnormal. FP32 log parameters also passed both tests. The sweep's L2 error
  was 0.111% for all three log variants; this favorable ordered distribution
  is not representative of ordinary activations.
- FP16 wire cast is not BF16-range safe. Its CPU numerical oracle overflowed
  28,672 sweep outputs and zeroed 26,112 nonzero values. Actual GPU timing was
  measured on normal-range inputs, including both boundary casts.

## Kernel trace

A separate bounded GPU-only Kineto run retained **1,054 kernels on each of
four ranks**, plus runtime/copy events. Native rocprof was not used: previous
TP4 preflights on this patched runtime had host-memory growth problems.
Profiler warnings about duplicate flow starts and HtoD correlation mean
launch/copy associations are not claimed to be complete; kernel timings are
present on all ranks. Graph staging-copy events are incompletely represented,
so end-to-end controls above are authoritative.

1664 rows, maximum of the four per-rank median kernel durations, µs:

| Kernel | Pack | Communication/decode/reduce/gather |
|---|---:|---:|
| Native BF16 two-stage | Staging copy, not quantization | **508.48** |
| Linear GS32 / FP32 | 155.84 | 517.60 |
| Linear GS16 / FP32 | **127.12** | 535.44 |
| Linear GS16 / BF16 | 156.00 | 568.72 |
| Cube scale / FP16 | 163.60 | 564.00 |
| Bounded adaptive-log / FP16 | 157.04 | 693.36 |
| Adaptive-log / FP32 | 142.16 | 691.52 |

Even before packing, these compressed kernels are not materially faster than
the native two-stage kernel. Log decoding adds substantial work. Profiled
kernel medians, isolated phase event timings and complete unprofiled graph
bursts are different measurements and should not be added or substituted
for one another. This is not a hardware-counter roofline classification.

## Recommendation

Do not enable a blanket compressed CAR mode. If pursuing it further, start
with **GS32 INT8 + FP32 scales**, keep GS16/FP32 as the accuracy comparison,
and investigate producer-epilogue packing and a better two-stage transport.
GS32/FP32 has the same packed byte count as GS16/FP16 without the latter's
range restriction or power-scale reconstruction. The current kernel results
do not establish a speedup even with packing fused away.

Use callsite-specific cancellation/range gates and an uncompressed fallback.
An FP16 transport is reasonable only where measured range permits it; it
does not reduce bytes relative to BF16. MI100's FP16 **matrix** throughput
advantage does not accelerate this FP32-add/bandwidth-based collective.
Adaptive-log is worth retaining as an outlier/relative-error experiment,
not the default transport.

## Reproduction and retained evidence

Sources: `benchmarks/kernels/benchmark_gfx908_car_quant.py` and
`benchmarks/kernels/car_quant_study.cpp`. Build only the standalone library;
never replace a serving `.so` with it. Set the repository's required ROCm env
and sibling-aiter `PYTHONPATH`; block Nginx and stop the service first.

```bash
/opt/rocm/bin/hipcc -O3 -std=c++17 --offload-arch=gfx908 -shared -fPIC \
  benchmarks/kernels/car_quant_study.cpp -o /tmp/car_quant_study.so
PYTHONPATH="$PWD:$PWD/../aiter" .venv/bin/python -m torch.distributed.run \
  --nproc_per_node=4 benchmarks/kernels/benchmark_gfx908_car_quant.py \
  --library /tmp/car_quant_study.so --rows 7,42,128,1664,2048 \
  --output logs/agent_mixed/car_quant_reproduction.json
```

For a bounded all-rank trace, add `--skip-accuracy --stress 0 --rows 1664
--dtypes bf16 --iterations 2 --trials 2 --burst 4 --algorithms 2stage
--trace-dir logs/agent_mixed/car_quant_trace`; choose a few `--formats`.
Restore the original service and Nginx afterward. No HTTP load generator
was created or run.

Retained under `logs/agent_mixed/` (gitignored):

- `car_quant_final_20261011.json`: primary unprofiled BF16/FP16 A/Bs.
- `car_quant_edges_20261011.json`: all formats, BF16 bit-pattern/endpoint checks.
- `car_quant_bounded_20261011.json`: bounded-log unprofiled controls and checks.
- `car_quant_profile_20261011.json`, `car_quant_trace_20261011/rank*.json.gz`:
  separate kernel capture. JSONs record source/library hashes where available.

The production INT8 KV micro-test passed and the unloaded-GPU battery was
**4/4 PASS** after testing. Production model, precision, topology, speculation,
collective selection and scheduler configuration were not changed.

---

## MLP-only full-model follow-up — 2026-10-11

**Outcome: keep this implementation experimental and off by default.** The
selected token-oracle checks passed, but it did not produce a useful
mixed-load decode improvement. It is not a 3× speedup or a lossless transport.
Production and Nginx were restored after a successful completion smoke check;
the GS32 flag and profiling/developer flags are absent from production workers.

### Implementation and controls

`VLLM_MLP_CAR_INT8_GS32=1` selects the experimental transport only for
`RowParallelLinear` prefixes ending in `.mlp.down_proj`: target and DFlash2
draft. It defaults off and participates in the compile-cache hash. The
model/activation dtype, checkpoints, attention, KV format, scheduler, TP4,
C6 and NS6 are unchanged. The baseline revision is `e5092f0f2`; sibling
AITER is `83e07b940`. Both arms use the same added code with the flag off/on.

| Callsite | Partial input | Wire input | Scale | Accumulator | Output / immediate consumer |
|---|---|---|---|---|---|
| Target MLP down projection | BF16 | Signed INT8, GS32 | FP32 | FP32 | BF16 → residual/normalization |
| Target GDN output projection | BF16 | Native BF16 | None | Native FP32 | Unchanged BF16 residual stream |
| Target full-attention output projection | BF16 | Native BF16 | None | Native FP32 | Unchanged BF16 residual stream |
| Target token embedding | BF16 | Native BF16 | None | Native FP32 | Unchanged BF16 model input |
| Draft MLP down projection | BF16 | Signed INT8, GS32 | FP32 | FP32 | BF16 → residual/normalization |
| Draft attention output projection | BF16 | Native BF16 | None | Native FP32 | Unchanged BF16 residual stream |
| Draft token embedding | BF16 | Native BF16 | None | Native FP32 | Unchanged BF16 draft input |

The transport also supports FP16 inputs/outputs, tested synthetically, but
the serving experiment does not change these BF16 model streams. Each rank
packs its own partial with symmetric round-to-nearest quantization and an
independent `amax/127` scale per 32 values. Decoded values are accumulated in
fixed rank order in FP32 and rounded once to the output dtype. INT8 codes
with different scales are **not** directly added.
Native two-stage CAR uses a rotating summation order, so the A/B includes
both quantization and summation-order changes; it does not isolate their
individual numerical contributions.

Private uncached IPC buffers and barrier epochs isolate this transport from
interleaved native CAR. The native automatic 512 KiB cutoff is retained:
one-stage below it, two-stage at/above it. Two-stage all-gather remains BF16;
there is no second quantization of the reduced sum. Packed input is 1.125
bytes/value, saving 43.75% of one-stage remote payload but only 21.875% of
the complete two-stage remote payload. Separate packing adds a launch.

### Test protocol

Nginx is stopped and runtime-masked throughout isolated testing. Every
server/client runs as `curved`; no persistent traffic generator is installed.
The harness uses private disk-cache roots, bounded service memory/lifetime,
and restores the original production service and Nginx in its finalizer.
Stopped test KV directories are removed after their logs are retained;
these contain regenerable cache entries, not model weights or requests.

The frozen workload is **eight chats, two turns each, 16 requests**. The
stored manifest is not the older three-turn/24-request workload. Initial
contexts are approximately 40k tokens; arrivals are at 0, 7, 19, 29, 43,
61, 83 and 109 seconds. Output caps are 1536 then 768 tokens, with natural
EOS, sampled decoding, fixed seeds and 2/5/8-second follow-up waits. HTTP
outstanding-request cap is eight; engine concurrency remains six. Normalized
manifest SHA256: `7f5934b1334bd26e6f864a55b6e77dbd0e0435d375d1798e0bfade3e486370a1`.

The suite includes:

- Production INT8 KV micro-test and the four-part gfx908 battery.
- CPU scheduler, prefix-admission, disk-tier, environment/hash and client
  regression tests, plus the new MLP selector/fake-op tests.
- Four-rank exact quantized-sum checks, BF16/FP16 range/cancellation cases,
  mixed native/quantized graph replay and unprofiled collective timings.
- A 52-prompt, 10,610-token baseline reference. Candidate teacher forcing
  uses **identical reference token contexts**, not independently generated
  continuations. Reported KL is a coarsened lower bound: common returned
  top-20 tokens plus an OTHER bucket, not full-vocabulary KL.
- Cold/warm C6 greedy gates: one shared 40k prompt, 512 output tokens per
  stream, 3072 fixed output tokens per leg, NS6 acceptance counts.
- Cold/warm mixed runs, followed by separate cold/warm kernel captures on
  fresh baseline/candidate boots. Profiling replays the exact saved baseline
  request bodies, including follow-up contexts, rather than rebuilding them
  from different candidate responses.
- Clean-baseline prefill-oracle scoring of six candidate long-context
  traces and six corresponding baseline controls: all 1536 tokens of the
  first two long responses (or until their earlier stop), and up to 512
  tokens of each other trace. Late tails from token 1024 are reported
  separately to expose failures hidden by prefix-only checks.

Request-lifecycle captures cover rank 0; model loading/compilation is
excluded. The separate bounded micro capture
covers all four ranks. Timings use GPU events/Kineto plus calibrated host
lifecycle spans, not hardware performance counters. Unprofiled client
timings remain authoritative for performance. These legs have GPU capture
off but the same lightweight host lifecycle tracing on in both arms.
Whole-run output rate includes
prefill, arrivals and queueing; it is not per-stream decode throughput.
Profile legs are **open-loop replays of saved send offsets**, not new
closed-loop conversations. Profiler delays can make a follow-up coexist
with its still-running predecessor and change cache availability or batch
composition. Their phase/call counts describe that diagnostic replay;
the unprofiled closed-loop legs are the realistic performance comparison.
All four captures completed 16/16 requests with zero accounting errors.
Their submitted bodies match the saved native request hashes, and their
cache-hit/created-token details match between arms for every request.
Recorded runtime source hashes also match. Maximum start/end clock-offset
drift was 0.760 µs; maximum calibration uncertainty was 2.210 µs.

### Serving-transport micro results

The INT8 KV micro-test passed; the battery passed **4/4**; **308 CPU tests
passed**. All 15 numeric cases per rank matched the independent CPU-decoded
quantized-sum oracle, and all **270 changing-input graph replays per rank**
passed with native CAR interleaved. This is implementation correctness, not
a claim that quantization preserves the uncompressed model's outputs.

BF16 relative L2 errors versus uncompressed FP32 summation followed by BF16
rounding: normal 0.581%, outliers 1.053%, log-range 0.635%, near-cancellation
116.934%. The full finite BF16 sweep had no nonfinite results; mixed extreme
groups still lost small values. FP32 scales preserve range, not arbitrary
within-group resolution.

Unprofiled BF16 collective latency in microseconds, including packing and
communication; median of five maximum-rank trials, 10 repetitions of a
16-operation graph burst:

| Rows | Native automatic | INT8 GS32 |
|---:|---:|---:|
| 1 | 9.41 | 14.93 |
| 7 | 12.96 | 17.10 |
| 42 | 33.91 | 39.76 |
| 48 | 37.32 | 41.01 |
| 52 | 30.16 | 50.64 |
| 56 | 30.96 | 49.19 |
| 128 | 50.55 | 73.10 |
| 1664 | 552.19 | 726.36 |
| 2048 | 672.03 | 890.73 |

In the separate 1664-row all-rank capture, native staging-copy medians were
41.44–42.88 µs and native two-stage medians 513.84–518.08 µs. INT8 packing
took 165.76–170.48 µs, with compressed reduction/gather at 520.40–522.24 µs.
These are per-kernel medians from profiled runs, not additive substitutes
for the complete unprofiled timings above. Packing is expensive, and this
implementation's compressed reduction does not yet turn byte savings into
a faster reduction.

Compiler resource diagnostics report 11 VGPRs and zero scratch for packing;
one-/two-stage reductions use 38/41 VGPRs and 72 bytes/lane private scratch,
with no compiler-reported register spills. Private scratch is not itself
proof of register spilling or a bandwidth-bound roofline classification.

### Aligned short-context quality gate

| Metric | GS32 candidate against native baseline |
|---|---:|
| Prompts / identical-context reference tokens | 52 / 10,610 |
| Token-level top-1 agreement | 96.852% |
| Mean coarsened KL lower bound | 0.0136051 |
| Mean absolute reference-token logprob change | 0.0430498 |
| Reference tokens outside candidate top 20 | 0 |
| Worst reference-token rank | 5 |
| Nonfinite returned logprobs | 0 |

This is not a 96.852% task-accuracy score and not lossless communication:
334 reference positions change top-1 prediction. The baseline itself is
the already quantized serving model, not a full-precision teacher. Long
contexts and speculative acceptance require their separate gates below.

### Unprofiled end-to-end results

All eight measured legs completed: six requests per greedy leg, 16 per
mixed leg, with **zero token-accounting errors** throughout. These are
single paired runs, not repeated/interleaved confidence intervals.

| Workload | Native wall, s | GS32 wall, s | Native output tokens | GS32 output tokens | Native whole-run tok/s | GS32 whole-run tok/s |
|---|---:|---:|---:|---:|---:|---:|
| C6 greedy, cold | 58.261 | 55.223 | 3072 | 3072 | 52.73 | 55.63 |
| C6 greedy, warm | 34.232 | 32.498 | 3072 | 3072 | 89.74 | 94.53 |
| Mixed, cold | 462.930 | 480.727 | 16,675 | 18,432 | 36.02 | 38.34 |
| Mixed, warm | 206.600 | 204.429 | 18,432 | 18,432 | 89.22 | 90.16 |

The cold mixed baseline stopped `c1-t0` and `c1-t1` early with tool calls
(444 and 103 output tokens). The candidate used the full budgets there.
Consequently the cold wall times and rates do not measure identical work.
All initial cold contexts missed local prefix cache in both arms; one
baseline follow-up matched 38,272 instead of 39,936 tokens. Warm mixed
cache counts matched for **every request**: 38,272 initial, 39,936 follow-up.

| Mixed-run client metric | Native cold | GS32 cold | Native warm | GS32 warm |
|---|---:|---:|---:|---:|
| Median per-stream decode, tok/s | 24.17 | 24.71 | 24.32 | 24.37 |
| Per-stream tok/s while another request awaits first token | 23.89 | 23.67 | 19.28 | 18.05 |
| Mean first-turn TTFT, s | 155.54 | 167.30 | 5.59 | 5.08 |
| Mean follow-up TTFT, s | 56.21 | 38.50 | 11.08 | 7.41 |

The contention metric is weighted over per-stream token-seconds. It
includes another request's queue wait, not just actual overlapping prefill.
Speculative burst gaps are stream-event gaps, not single-token ITL.
The mixed results do **not** establish a useful decode speedup: warm median
stream rate is unchanged, warm whole-run rate is only 1.06% higher, and the
contention metric is lower. TTFT movements in one sampled pair are not a
separate cache implementation improvement; cache policy was not changed.

| NS6 acceptance, accepted / drafted | Native | GS32 | Change, percentage points |
|---|---:|---:|---:|
| Greedy cold | 2085 / 5946 = 35.07% | 2124 / 5730 = 37.07% | +2.00 |
| Greedy warm | 2101 / 5856 = 35.88% | 2130 / 5682 = 37.49% | +1.61 |
| Mixed cold | 11270 / 32514 = 34.66% | 12412 / 36156 = 34.33% | −0.33 |
| Mixed warm | 12438 / 36048 = 34.50% | 12539 / 35502 = 35.32% | +0.82 |

Acceptance is aggregated from raw counts, not averaged request percentages.
Mixed acceptance stays within the ±1-point parity band. The greedy changes
are increases, but lie outside strict ±1-point parity; they accompany
different generated continuations. Greedy wall improvement must not be
attributed to a faster CAR kernel: acceptance and scheduled work also changed.

### Long-context clean-prefill oracle

Six retained traces per arm were teacher-forced through a **native-CAR
baseline** server. This tests the exact emitted token IDs under their exact
contexts, including the late portions of the two 1536-token candidate
responses; it does not compare independently generated positions as KL.

| Check | Native-generated controls | GS32-generated traces |
|---|---:|---:|
| Scored tokens | 4028 | 5120 |
| Tokens outside clean-prefill top 20 | 0 | 0 |
| Worst rank | 20 | 16 |
| Nonfinite logprobs | 0 | 0 |
| Tokens checked from position 1024 onward | 512 | 1024 |
| Late-tail tokens outside top 20 | 0 | 0 |

The differing counts reflect the native `c1-t0` tool-call stop at 444
tokens, not omitted failing tokens. These checks found no numerical
catastrophe in the selected long outputs. They do not establish unchanged
general task accuracy, exact output equivalence, or multi-day stability.
Mean logprobs of different generated texts must not be treated as an
accuracy improvement.

### Native reference profile

The native cold/warm captures contain **8,446,995 / 3,405,436 GPU kernels**
on rank 0, in **56 / 23 chunks**. All retained kernels were launch-correlated
by the converter/analyzer. Kernel-interval coverage within chunks was
98.79% / 98.85%; do not call that compute utilization, because a running
collective can be waiting for peers. Export pauses lie between chunks.

In the cold replay, median GPU step windows were **59.71 ms decode-only**
and **985.70 ms mixed**, with mixed p95 **1237.86 ms**. These phase groups
contain different batch sizes; they are not fixed-shape microbenchmarks.

| Kernel family in native cold mixed batches | Summed GPU seconds | Share |
|---|---:|---:|
| GEMMs, across all projections | 75.943 | 37.40% |
| Full-attention KV kernels | 57.425 | 28.28% |
| All collectives | 37.045 | 18.24% |
| GDN recurrence | 14.561 | 7.17% |
| Normalization | 5.749 | 2.83% |
| Other kernels | 5.019 | 2.47% |
| Copy/store kernels | 3.765 | 1.85% |
| Activation quantization | 2.525 | 1.24% |
| GDN convolution | 0.850 | 0.42% |
| Sampling | 0.195 | 0.10% |

These are implementation families, not disjoint model-layer labels: GEMMs
include MLP, attention projections and other linear work. GDN is an
attention mechanism; its recurrence/convolution kernels are listed separately
from the full-attention KV kernels. MLP down-projection CAR is only a subset
of the collective row. The latter includes synchronization wait time, not
just arithmetic or transferred bytes.

Native BF16 reduction kernels alone totaled **39.014 s** over 29,866
two-stage calls and **23.273 s** over 466,709 one-stage calls in the cold
capture. Corresponding warm totals were **7.110 s / 6,170 calls** and
**14.338 s / 198,162 calls**. Their means include a mixture of sizes and
peer-arrival delays. Generic staging copies are not uniquely attributable
to CAR in the full trace; the fixed-shape micro table includes native
staging and remains the clean complete-collective comparison.

### GS32 candidate profile

The candidate cold capture contains **8,339,795 rank-0 kernels** in 55
chunks. Kernel-interval coverage inside chunks is 97.14%. Its median
decode-only step was **59.46 ms** and median mixed step **996.53 ms**
(mixed p95 **1248.12 ms**), versus 59.71/985.70/1237.86 ms in the native
capture. These are diagnostic phase distributions, not matched-shape
speed ratios: batch composition, continuation tokens and speculative work
can differ even when replayed request bodies match.

Cold candidate kernel components, across all phases:

| Kernel | Calls | Summed GPU seconds | Mean, µs |
|---|---:|---:|---:|
| MLP GS32 packing | 241,438 | 4.690 | 19.42 |
| MLP GS32 one-stage reduction | 226,609 | 7.619 | 33.62 |
| MLP GS32 two-stage reduction/gather | 14,829 | 16.965 | 1144.06 |
| Remaining native BF16 one-stage | 233,390 | 14.350 | 61.48 |
| Remaining native BF16 two-stage | 15,061 | 23.299 | 1546.98 |

In mixed steps alone, packing costs **2.886 s / 14,766 calls**, or
195.42 µs on average. GS32 two-stage costs **14.905 s / 12,864 calls**,
or 1158.65 µs. The much larger remaining-native means demonstrate why
individual kernel means cannot be treated as matched collective latency:
callsite/shape mix and peer arrival differ, and reductions include waits.
The fixed-shape all-rank microbenchmark above is the controlled comparison.
Native staging copies are in the copy category whereas GS32 packing is
classified as a collective; blindly subtracting category totals would
also mix accounting boundaries.

The warm candidate capture contains **3,381,318 kernels** in 23 chunks,
all launch-correlated, with 98.90% within-chunk kernel-interval coverage.
Median decode-only/mixed step windows were **108.49 / 1205.37 ms**, versus
native **95.08 / 1188.87 ms**; mixed p95 was **1297.58 versus 1278.47 ms**.
The same cautions about phase composition and profiler perturbation apply.
Warm candidate component totals:

| Kernel | Calls | Summed GPU seconds | Mean, µs |
|---|---:|---:|---:|
| MLP GS32 packing | 99,991 | 1.258 | 12.58 |
| MLP GS32 one-stage reduction | 96,919 | 4.740 | 48.90 |
| MLP GS32 two-stage reduction/gather | 3,072 | 3.071 | 999.62 |
| Remaining native BF16 one-stage | 99,776 | 10.154 | 101.76 |
| Remaining native BF16 two-stage | 3,120 | 4.446 | 1425.15 |

The measured implementation has not converted reduced input traffic into
end-to-end speed. A further transport experiment should remove the separate
packing pass (for example, a down-projection producer epilogue), and optimize
the two-stage reduction/gather itself. Merely changing the wire dtype with
this pack/reduce path does not address the dominant mixed-step stalls.
The reproducible A/B record is ledger entry
`MLP_DOWN_PROJ_CAR_INT8_GS32_20261011` in
`docs/recipes/surface_experiments_ledger.jsonl`.

### Unprofiled host lifecycle

Warm closed-loop run, median host-span milliseconds. Worker entries use
rank 0; frontend/scheduler/cache entries use their owning process. These
are asynchronous, sometimes nested spans, **not additive wall-time parts**.

| Host span | Native | GS32 |
|---|---:|---:|
| Chat rendering/tokenization | 80.332 | 78.219 |
| Scheduler call | 0.903 | 0.885 |
| Prefix lookup | 0.069 | 0.088 |
| Prepare inputs | 0.375 | 0.431 |
| Prepare attention metadata | 0.147 | 0.171 |
| Target model call/enqueue | 6.313 | 6.906 |
| Draft propose call/enqueue | 1.293 | 1.565 |
| Sampling call/enqueue | 1.133 | 1.416 |
| Filesystem load task (7 / 8 samples) | 176.708 | 176.589 |
| Filesystem store task (23 / 24 samples) | 160.868 | 117.144 |

Lookup call time is not the entire asynchronous cache-promotion delay, and
filesystem tasks can overlap other work. This transport experiment changes
no cache policy. The recorded request-level cache-hit counts, TTFT and
GPU phase timings must be considered together; small host-span differences
in one pair do not establish a causal optimization.

### Reproducing the full-model follow-up

Sources: `csrc/custom_all_reduce_int8_gs32.hip`,
`benchmarks/kernels/benchmark_mlp_car_gs32.py`, the opt-in communicator and
MLP dispatch, and `scripts/{run_mlp_car_experiment,mlp_car_campaign,
mlp_car_suite,eval_mlp_car,summarize_mlp_car}.py`.

Build from the repository root using the required ROCm environment:

```bash
export ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib
export PYTORCH_ROCM_ARCH=gfx908 GPU_ARCHS=gfx908 VLLM_TARGET_DEVICE=rocm
export PYTHONPATH="$PWD:$PWD/../aiter"
mkdir -p build
/opt/rocm/bin/hipcc -O3 -ffp-contract=off -std=c++17 \
  --offload-arch=gfx908 -shared -fPIC csrc/custom_all_reduce_int8_gs32.hip \
  -o build/mlp_car_int8_gs32.so
```

Tested library SHA256:
`5467c190c9bc689d9f2af0e2801d14a66a462389d92585633e0808cc3236576c`.
The flag requires a process restart and never builds a library inside graph
capture. It does not change the default serving recipe.

The following root-only campaign stops external ingress for its entire
duration. Choose a **new** output directory; existing evidence is never
overwritten. The two manifest variables below select the exact retained
workload rather than regenerating a potentially different corpus:

```bash
export MLP_CAR_CAMPAIGN_DIR="$PWD/logs/agent_mixed/mlp_car_repeat"
export MLP_CAR_MANIFEST="$PWD/logs/agent_mixed/mlp_car_gs32_20261011/manifest.json"
export MLP_CAR_GREEDY_GATE="$PWD/logs/agent_mixed/mlp_car_gs32_20261011/greedy_gate.json"
systemctl stop nginx.service
systemctl mask --runtime nginx.service
systemctl stop vllm.service
.venv/bin/python scripts/mlp_car_campaign.py
```

The controller boots baseline/candidate, runs accuracy and closed-loop
timings, then boots both again for cold/warm captures and native-oracle
checks. It restores the original `vllm` service, verifies a completion, and
only then unmasks/starts Nginx. `--baseline-already-running` is available
for an explicitly prebooted isolated baseline. The original measured run
used that preboot workflow before it was moved under the restoration guard.

If the retained manifests are unavailable, prepare new ones with
`scripts/bench_agent_mixed.py --prepare` and `--greedy-gate-from`, then point
the variables at them. That is a new workload, not a reproduction of the
two-turn numbers here. The controller freezes both manifests into the
campaign directory. Its test clients and transient servers run as `curved`.

The separate GPU micro/battery checks were run before the serving campaign,
with all four GPUs idle and Nginx stopped. They are not invoked by the
controller. After building the library, use the required environment above
and run as the service owner:

```bash
runuser -u curved -- env PYTHONPATH="$PWD:$PWD/../aiter" \
  ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib \
  PYTORCH_ROCM_ARCH=gfx908 GPU_ARCHS=gfx908 VLLM_TARGET_DEVICE=rocm \
  .venv/bin/python scripts/test_int8_kv_micro.py
runuser -u curved -- env PYTHONPATH="$PWD:$PWD/../aiter" \
  ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib \
  PYTORCH_ROCM_ARCH=gfx908 GPU_ARCHS=gfx908 VLLM_TARGET_DEVICE=rocm \
  HIP_VISIBLE_DEVICES=0 .venv/bin/python scripts/battery_gfx908.py
runuser -u curved -- env PYTHONPATH="$PWD:$PWD/../aiter" \
  ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib \
  PYTORCH_ROCM_ARCH=gfx908 GPU_ARCHS=gfx908 VLLM_TARGET_DEVICE=rocm \
  VLLM_MLP_CAR_INT8_LIBRARY="$PWD/build/mlp_car_int8_gs32.so" \
  .venv/bin/python -m torch.distributed.run --nproc_per_node=4 \
  benchmarks/kernels/benchmark_mlp_car_gs32.py \
  --output "$MLP_CAR_CAMPAIGN_DIR/micro.json" \
  --trace-dir "$MLP_CAR_CAMPAIGN_DIR/micro_trace"
```

Create that artifact directory with `curved` ownership first. The 308-test
CPU run covered `tests/test_envs.py`, `tests/distributed/test_mlp_car_gs32.py`,
`tests/v1/core/{test_scheduler,test_mamba_align_chunk_split,
test_prefix_fair_admission}.py`,
`tests/v1/attention/test_rocm_g128_mixed_partitions.py`,
`tests/v1/kv_offload/tiering/test_fs_tier.py`, and
`scripts/test_bench_agent_mixed.py`.

Summarize completed runs without issuing any requests:

```bash
.venv/bin/python scripts/summarize_mlp_car.py "$MLP_CAR_CAMPAIGN_DIR" \
  --output "$MLP_CAR_CAMPAIGN_DIR/results.json"
```

The experiment's retained artifacts are under
`logs/agent_mixed/mlp_car_gs32_20261011/` (gitignored): `micro.json`,
`micro_trace/`, `reference52.json`, `candidate52.json`, the four unprofiled
mixed/greedy legs per arm, `oracle_*.json`, `*_capture.json`, per-boot
`host/gpu/*.json.gz` plus clock calibration, `*_kernel_analysis.json`,
`unprofiled_host_timings.json`, `profile_validation.json`, and `results.json`.
Use `scripts/kineto_to_agent_csv.py` followed by `scripts/analyze_agent_profile.py`
to regenerate kernel summaries; pass the matching profile run to `--run`
and its boot's `host` directory to `--host`.

## Decode INT8 intermediates — GS128 producer and transport

Experimental, **off by default**: `VLLM_DECODE_INT8_INTERMEDIATES=1`.
This is a separate experiment from the GS32 boundary packer above. It applies
to target and draft `mlp.down_proj`, for 1–64 input rows, including the
decode-sized speculative verification batches. Other projections, larger
prefill batches, KV precision, and recurrent-state precision are unchanged.

The small-M path uses the checkpoint's original signed INT8 GS128 weights
with their scales promoted to FP32, rather than CK's per-channel requantized
copy. Activations are quantized per row and 128 K elements. The GEMM uses
INT8 MFMA with INT32 dot products for each K group, then FP32 scaled
accumulation. Its epilogue writes signed INT8 output groups of 128 channels
and one FP32 scale per group, without writing BF16 GEMM-output values.

The custom all-reduce transports those **INT8 GS128 codes plus FP32 scales**
from each rank. It does not transport BF16 or requantize the reduced sum.
Peer dequantization, rank summation, residual addition, and RMSNorm use FP32
inside one HIP kernel; the existing BF16 rounding of the reduced activation
is preserved in registers. RMSNorm output and the residual stream are BF16.
The draft's two-tap convolution is folded into this consumer, preserving its
BF16 arithmetic boundaries. Five target auxiliary snapshots needed by DFlash
are explicit dense consumers and require five additional reductions.

There are still two staging copies per collective (codes and scales).
The shape-stable compiled ABI also allocates an unused BF16 placeholder for
small-M calls; it is neither written with GEMM values nor consumed there.
Runtime dispatch lives inside opaque custom ops because vLLM drops Dynamo
shape guards. A Python size branch would risk freezing the warmup choice.
Large-M calls dispatch to the existing CK GEMM, BF16 native all-reduce, and
the corresponding native/HIP norm arithmetic.

The experiment retains the original down-projection weights alongside CK's
prefill copy. The first full-model boot at the production 19 GB KV allocation
ran out of memory during graph capture. The paired experiment therefore uses
**17 GB KV in both arms**, with the original weights and private transport
buffers retained in both (`VLLM_DECODE_INT8_RETAIN_GS128=1`). This capacity
adjustment is confined to the test harness; it is not a production recipe
change. The 40k-context workload is unchanged. These results must not be
presented as a capacity-neutral change.

Build with the repository's ROCm environment:

```bash
/opt/rocm/bin/hipcc -O3 -ffp-contract=off -std=c++17 \
  --offload-arch=gfx908 -shared -fPIC csrc/decode_int8_intermediates.hip \
  -o build/decode_int8_intermediates.so
export MLP_CAR_EXPERIMENT=decode_int8
export MLP_CAR_CAMPAIGN_DIR="$PWD/logs/agent_mixed/decode_int8_intermediates_20261011"
export MLP_CAR_MANIFEST="$PWD/logs/agent_mixed/mlp_car_gs32_20261011/manifest.json"
export MLP_CAR_GREEDY_GATE="$PWD/logs/agent_mixed/mlp_car_gs32_20261011/greedy_gate.json"
export VLLM_DECODE_INT8_LIBRARY="$PWD/build/decode_int8_intermediates.so"
.venv/bin/python scripts/mlp_car_campaign.py \
  --isolate --with-micro --with-battery --candidate-preflight
```

Use a fresh campaign directory when reproducing. The guard stops/masks Nginx,
runs GPU checks before booting, and restores production before reopening
ingress. `MLP_CAR_KVMEM` can explicitly override the paired test's 17 GB arena.
The new `scripts/eval_decode_int8.py` complements the existing teacher-forced
prefill control: it forces one reference token, then compares the next
decode step's **raw, pre-mask** top-20 logprobs at identical histories. Its
common-top20-plus-OTHER KL is a coarsened lower bound, not full-vocabulary KL.
A raw-logprob canary checks that mask-contaminated scores are not compared.
The force uses `allowed_token_ids`, not `logit_bias` (the latter is rejected
by the speculative-decoding API).

Validation: 314 CPU tests, 4/4 GPU battery, KV micro, and 112 four-rank
numeric cases passed. The micro also exercised 84 changing-input graph
replays per rank. The candidate booted with full and piecewise graphs, and
six concurrent smoke requests returned coherent answers and finite scores.

### Unprofiled full-model results

**Do not deploy this implementation.** It does not improve the target
workload, and it fails the strict fixed-token draft-acceptance parity gate.
The exact frozen manifests are the same as the GS32 experiment above, but
these are fresh paired runs, both with the 17 GB KV arena and retained
original weights. Numbers below are not a comparison against the old run.

| Measurement | Native baseline | Decode INT8 GS128 |
|---|---:|---:|
| Fixed C6, cold: 3,072 tokens, wall time | 55.68 s | 67.57 s |
| Fixed C6, warm: 3,072 tokens, wall time | 32.11 s | 38.48 s |
| Fixed C6, warm: median per-stream decode | 28.84 tok/s | 22.83 tok/s |
| Fixed C6, warm: per-stream contention rate | 21.75 tok/s | 17.44 tok/s |
| Fixed C6, warm: accepted / drafted | 41.23% | 33.25% |
| Mixed cold: 18,432 tokens, wall time | 488.72 s | 488.10 s |
| Mixed cold: median per-stream decode | 26.35 tok/s | 25.27 tok/s |
| Mixed cold: per-stream contention rate | 24.05 tok/s | 24.92 tok/s |
| Mixed warm: median per-stream decode | 25.63 tok/s | 23.71 tok/s |
| Mixed warm: per-stream contention rate | 19.56 tok/s | 19.71 tok/s |
| Mixed warm: output tokens | 16,647 | 18,432 |
| Mixed warm: wall time | 201.19 s | 210.09 s |

Contention means another client is awaiting its first token, not proof of
simultaneous execution on the GPU. Warm mixed whole-run throughput is not a
fixed-work comparison: the baseline ended some responses early. Every leg
completed all planned requests with zero token-accounting errors. There is
only one paired run, not a statistical significance claim. The baseline cold
leg includes a 15.22-second stream gap and a mid-run cleanup of old derived
CSV exports; do not treat its tiny aggregate difference as an improvement.
Pre-existing XGMI WAFL/FCInitTimeoutErr reports were present in kernel logs
before this experiment and continued in the native baseline; no hardware
causality is established by these measurements.

The fixed-token warm run needed 323 candidate `runner.sample`/`draft.propose`
calls versus 285 baseline calls. Median CPU `worker.execute_model` spans
were nearly unchanged (6.47 vs 6.46 ms); these asynchronous CPU launch spans
are **not** GPU step latency. The extra rounds and the 7.97-point acceptance
loss are important confounders when attributing the end-to-end slowdown to
the kernels alone.

### Accuracy results

| Aligned comparison | Cases | Mean coarsened KL | Top-1 agreement |
|---|---:|---:|---:|
| Teacher-forced control, 52 prompts | 10,803 tokens | 0.01219 | 97.07% |
| Aligned next-token decode (2-token requests) | 415 positions | 0.01495 | 95.90% |

The short masked requests assess next-token fidelity, not normal draft
acceptance; the latter is measured separately by the unforced C6 runs.
The teacher-forced reference token remained in the candidate top 20 at every
position (worst rank 7). Maximum coarsened decode KL was 0.27753. These are
fidelity comparisons against the native baseline, not task accuracy or
full-vocabulary KL. The fixed-token draft-acceptance gate fails: cold acceptance
fell from 42.55% to 32.39%, and warm from 41.23% to 33.25%.

The native-baseline long-response oracle scored 5,120 retained continuation
tokens per arm across six requests, including two 1,536-token continuations
and their tails beyond token 1,024. All scores were finite, with zero tokens
outside the native top 20; worst rank was 11 for baseline continuations and
14 for candidate continuations. This checks plausibility of these sampled
continuations, not task correctness or equivalence of their free-running
outputs. Artifacts: `oracle_{baseline,candidate}_*.json` in the campaign.

This is not an isolated communication-format A/B: it changes activation
grouping, uses the original checkpoint's grouped weights instead of CK's
per-channel requantization, quantizes projection outputs, and changes both
target and draft paths together. The fidelity and acceptance changes cannot
be attributed specifically to the INT8 wire format from this experiment.

### Microbenchmarks

Complete quantization + GEMM + transport + residual/Gemma RMSNorm chain,
unprofiled median GPU latency using the slowest TP rank:

| M | K | Native chain | INT8 GS128 chain |
|---:|---:|---:|---:|
| 1 | 256 | 51.30 us | 53.99 us |
| 7 | 256 | 56.32 us | 56.23 us |
| 1 | 4352 | 551.30 us | 140.78 us |
| 7 | 4352 | 126.79 us | 141.98 us |
| 42 | 4352 | 185.00 us | 210.79 us |
| 48 | 4352 | 605.52 us | 216.96 us |
| 64 | 4352 | 596.01 us | 270.60 us |

These strongly shape-dependent micro wins do not translate to the fixed-C6
model run. All four rank traces were retained. At M=42, profiled medians
across ranks were 5.76–5.92 us for activation quantization, 74.40–75.36 us
for GEMM plus quant-out, and 49.92–51.68 us for fused CAR/residual/RMSNorm.
Interleaved native CAR was 31.04–32.16 us. These constituent profiled timings
must not be added or substituted for the unprofiled complete-chain timings.
The native collective figure excludes its separate RMSNorm, whereas the
candidate kernel includes it. The compiled GEMM uses
`v_mfma_i32_16x16x16i8`, verified in generated assembly.

Tested HIP source SHA256:
`da364d04654dab06d2dd2fbcc40def2b8391e5dff429b48b1df8a1c87c12ee93`.
Library SHA256:
`48108efa0b249d976a2d0edae9a50afdb6cf33c0d0567ff61ecfb773ed724121`.

### Full-model lifecycle profiles

All four rank-0 lifecycle captures completed: cold and warm for both arms,
16 requests per capture, zero token-accounting errors. Request-body hashes
and reported cache-token counts match across arms for all 32 paired requests.
This is exact-body, open-loop replay of the native baseline's retained
requests; it is not a replay of identical generated tokens or scheduler steps.
Cold captures each produced 18,432 tokens. Warm captures produced 16,704
baseline versus 17,799 candidate tokens, so their aggregate times are not
equal-work speed comparisons.

There are 79 baseline and 78 candidate GPU trace chunks. Maximum measured
clock-offset drift is 720 ns. Chunk flushing introduces multi-second pauses;
the analysis includes phase totals excluding each chunk's first worker step.
Kernel durations still include synchronization waits. Full-model profiles
cover rank 0; the separate micro captures cover all four ranks. These are
timeline profiles, not hardware-counter roofline measurements.

Artifacts are under
`logs/agent_mixed/decode_int8_intermediates_20261011/`: `results.json`,
`{baseline,candidate}_{cold,warm}_kernel_analysis.json`, and raw
`{baseline,candidate}_profile_boot/host/gpu/*.json.gz` with clock sidecars.

Warm capture, decode-phase kernel-duration sums, excluding chunk-boundary
worker steps (seconds; **not** per-request or end-to-end latency):

| Kernel family | Native baseline | Decode INT8 GS128 |
|---|---:|---:|
| Projection/other GEMMs | 45.16 | 38.06 |
| New GS128 GEMM + quantized output | — | 8.59 |
| Native all-reduce | 14.05 | 11.62 |
| New fused GS128 CAR + residual/RMSNorm | — | 9.24 |
| Separate normalization | 5.93 | 3.57 |
| Paged/full-attention kernels | 37.85 | 40.42 |
| GDN recurrence (linear attention) | 4.17 | 4.42 |

These are operation-family labels: attention kernels can internally perform
matrix multiplication, and GDN is linear attention. The GEMM rows are not a
count of every matrix instruction in the model. The new fused collective
includes reduction, optional draft convolution, residual addition, RMSNorm,
and synchronization waits; it must not be labeled as all-reduce alone.

The candidate's collective-plus-normalization sum is 24.43 s versus 19.98 s
for baseline. Combined GEMM rows are 46.65 s versus 45.16 s. This identifies
the new consumer as a substantial cost, but does not isolate its causal
slowdown: warm outputs and batch trajectories differ. Across the complete
warm capture, the new GEMM/quant-out kernel averages 91.18 us and fused CAR
averages 91.93 us, versus 6.18 us for the new activation quantizer. These
averages mix shapes and include boundary effects; the separate micro is the
controlled shape comparison.

The design still stages codes and scales separately, and performs five extra
collectives for auxiliary snapshots. Compressing the peer payload to 51.56%
of BF16 bytes does not remove those costs. Together with the measured
fixed-token acceptance loss and extra draft rounds, this explains why the
implementation cannot be treated as a communication-only speedup. The
experiment does not establish that an optimized INT8-GS128 transport would
be slower than native CAR.

The equal-output cold capture reinforces that distinction. Its decode-phase
GEMM sums decrease from 79.67 s to 76.47 s (including the new quant-out GEMM),
while collective-plus-normalization sums increase from 35.52 s to 41.04 s.
The candidate has 3,241 attributed decode steps versus 3,332 baseline steps,
so this is not explained by a larger decode-step count in that trace. Batch
shapes still differ; this is evidence to target the fused consumer, not an
isolated attribution of every extra millisecond to communication.

| Profile step GPU window, median | Native baseline | Decode INT8 GS128 |
|---|---:|---:|
| Cold: decode | 59.42 ms | 60.12 ms |
| Cold: mixed | 982.00 ms | 977.13 ms |
| Warm: decode | 94.65 ms | 109.80 ms |
| Warm: mixed | 1,198.00 ms | 1,196.29 ms |

These windows are descriptive, not matched-shape kernel benchmarks. Every
kernel in the four full-model traces was launch-correlated: 8,523,686 and
3,344,777 baseline kernels, versus 8,232,074 and 3,350,027 candidate kernels
(cold and warm respectively). The native production service and Nginx were
restored after a successful completion smoke test. The experimental flags
are unset in production; the production KV allocation was not changed.
