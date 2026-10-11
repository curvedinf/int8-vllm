# Model-free CAR quantization study — 4× MI100

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
