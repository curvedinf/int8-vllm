# Bolt performance profile — 2026-10-15

The current packed matrix implementation is the main bottleneck. It accounts
for 87.5% of GPU kernel time in the natural-language capture and 90.5% in the
8k prefill capture. The implementation passes its GS128 arithmetic checks, but
it is not a tuned counterpart of Qwen's CK W8A8 matrix path. Retaining official
checkpoint bytes also does not establish fidelity equivalence to the official
execution: the additional GS128 quantization already has measurable error.

This investigation changes no serving kernels, precision settings, model
weights, or production service selection.

## Timing measurements

All rates below are client-observed output tokens divided by whole-run elapsed
time, including prefill, admission, and completion tails. They are **unprofiled**.

| Workload | Concurrency | Output | Elapsed | Output tok/s |
| --- | ---: | ---: | ---: | ---: |
| Technical tutorials, run 1 | 8 | 2,048 | 39.45 s | 51.92 |
| Same tutorial requests, run 2 | 8 | 2,048 | 43.90 s | 46.66 |
| One tutorial | 1 | 256 | 16.90 s | 15.15 |
| Repetitive 32-token completion, run 1 | 8 | 2,048 | 15.94 s | 128.47 |
| Same repetitive requests, run 2 | 8 | 2,048 | 16.86 s | 121.46 |
| Repetitive completion, single stream | 1 | 256 | 7.91 / 7.98 s | 32.35 / 32.06 |

Tutorial requests have 44–48 input tokens and ask for long technical answers;
all eight generated 256 tokens. Stream-derived generation rates were roughly
6.5–7.9 tok/s per stream at C8 and 15.4 tok/s at C1. These are bounded samples,
not statistical performance estimates or broad quality evaluations.

The repetitive prompt produces highly predictable continuations. Server logs
reached acceptance length 8.0, whereas representative tutorial intervals were
around 3.6–3.7, including the anchor token. The repetitive test is a favorable
speculation case and must not be presented as typical chat throughput.

Cold 8,192-input/1-output requests with distinct initial prefixes took
24.90 and 25.56 seconds, with first visible token at 24.79 and 25.45 seconds.
That is approximately 321–329 input tokens/s including first-token work.
The earlier 42 tok/s smoke was a different short-answer workload; the new
natural-language results confirm that low useful throughput persists with
longer outputs.

The default mixed-load policy also contributes to admission delay: in the
second tutorial run the first request received text at 0.36 s, while the other
seven waited approximately 4.75 s. The source admits mixed prefill every 16
rounds. This is a secondary latency issue; it cannot account for the dominant
matrix-kernel time below.

## GPU captures

GPU-only Kineto captured TP rank 0; host lifecycle spans covered all workers.
The TP4 graph/collective profiler preflight passed on all four ranks. Captures
used the unchanged recipe with graph execution and 32-step trace rotation.
Latency measurements above were collected separately with GPU profiling off.

| Fraction of summed rank-0 kernel time | Repetitive C8 capture | Tutorial C8 capture | 8k prefill capture |
| --- | ---: | ---: | ---: |
| Packed `_gemm`, including conversion and MFMA | 86.98% | 87.51% | 90.47% |
| Collectives, including custom AR and NCCL kernels | 3.44% | 3.35% | 4.71% |
| Unified attention core | 0.26% | 0.28% | 2.67% |
| Separate input activation `_quant` | 0.45% | 0.41% | 0.50% |

The tutorial trace contains 190,147 kernels, including 20,724 packed GEMMs
taking 22.28 of 25.46 summed GPU seconds. Median decode-step GPU window was
510 ms; the repetitive capture's median was 436 ms. Phase attribution uses
correlated launches and scheduler metadata; restricting the tutorial capture
to decode still assigns 87.46% to packed GEMMs.

Rank-0 kernel busy time occupies 99.56% of the tutorial capture windows after
excluding gaps between trace chunks. Thus host launch gaps are not the main
cost in this trace. Busy time includes collective waiting and is not a measure
of arithmetic utilization. One concurrent GPU snapshot showed all four GPUs
busy, at approximately 1.03–1.15 GHz and 65–67 C junction temperature; it is not
a sustained power or throttling study.

Prefill contains 358 packed GEMM launches taking 25.13 seconds. Launch geometry
and model dimensions provide a more specific attribution:

- Forty launches with grid `[512,37,1]` take **10.10 seconds**, or **36.36% of
  all kernel time**. They match the forty NVFP4 Mamba input projections:
  local output width 4,640, input width 4,096, and 8,192 input rows.
- Routed-expert launch grids take **3.20 seconds**, or **11.50% of all kernel
  time**. Dense and other non-routed packed GEMMs take **21.93 seconds**, or
  **78.96%**. Shared-expert projections are included in the latter category.
- The `[512,32,1]` group alone takes 8.44 seconds across 129 launches. Several
  distinct projections share this geometry, so it is not assigned to one
  codec or module here.

This rules out treating the problem as only an MoE-routing problem. Dense
packed projections, especially NVFP4 Mamba projections, need attention first.

Percentages are sums on one rank, not sums across four GPUs. Profiler export
and synchronization perturb latency; the prefill capture also includes GPU
work drained after the client receives its token. Its 27.79-second GPU window
is not the unprofiled TTFT. The graph events share the `_gemm` symbol, limiting
per-codec and per-module attribution during decode. No hardware instruction,
bandwidth, or occupancy counters were collected.

## Mechanism and recommended order of work

The relevant code is AITER's
`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_packed_g128.py` and its wrapper
`aiter/ops/triton/gemm/basic/gemm_packed_g128.py`, called through
[modelopt_gfx908.py](../../vllm/model_executor/layers/quantization/modelopt_gfx908.py).

The generic kernel uses a fixed 16-row by 128-column tile, four warps, one
pipeline stage, and a serial loop over 128-element K groups. Each iteration
decodes packed weights into FP32 values, computes group maxima, divides and
rounds them to INT8, executes INT8 MFMA, then scales the partial result into
FP32. The same conversion is repeated for each M tile. At 8k prefill, that is
512 M tiles per output tile. This is source-confirmed repeated work; the trace
does not separately time decoding, division, reduction, and MFMA inside it.

Compiled packed-kernel cache variants use 158–252 VGPRs without scratch spills.
That is consistent with register pressure from decoding and quantization, but
does not by itself measure live occupancy or establish the limiting hardware
resource. There is no shape autotuning or separate prefill/decode dispatch in
the packed wrapper comparable to the tuned Qwen CK path.

Recommended sequence, preserving packed resident weights:

1. **Specialize the dense packed kernels, starting with NVFP4 Mamba
   projections.** Separate prefill and small-M dispatch; improve packed-byte
   loading and conversion reuse, tune M/N tiles and pipelining, and evaluate
   split-K only where reduction overhead pays off. Keep group boundaries,
   rounding, and accumulation behavior under the component oracle. Label
   kernels by codec/shape to get finer attribution in the next trace.
2. **Tune IQ2 routed and shared experts.** Measure actual rows per expert,
   padding, and weight-conversion reuse. Choose routing tile sizes for measured
   occupancy, rather than assuming the same 16-row tile suits every layer.
   Do not expand the resident checkpoint into an INT8 weight cache.
3. **Reassess draft cost and acceptance after matrix fixes.** The tutorial
   workload commits far fewer tokens per verification than the repetitive
   workload. Capture target/draft timings separately and test the supported
   selection behavior; do not assume lowering NS preserves the draft's trained
   eight-token block contract or improves end-to-end speed.
4. **Then tune mixed admission latency.** Use the same tutorial and long-context
   manifests with fresh cache control. Do not prioritize attention/AR changes
   for this short-context bottleneck: together their measured share is small.

With the tutorial's 87.5% packed-kernel fraction, making that kernel family
3x faster would give an idealized GPU-time speedup of about **2.4x**; 10x would
give about **4.7x**, if all other costs stayed fixed. Eliminating every other
kernel could improve this capture only about **1.14x**. These are Amdahl
estimates, not promised serving gains; acceptance, scheduling and bottlenecks
can change after optimization.

## Fidelity and the Qwen comparison

Official bytes remain intact, but the compute path additionally quantizes
decoded weights and activations to GS128. Existing component evidence records
relative L2 output errors of 0.862% and 0.888% for sampled real expert up/down
projections, 0.960% for a sampled NVFP4 Mamba input projection, and 1.735% for a
sampled Q8 LM-head projection, against FP32 multiplication of decoded official
weights and the probe's BF16 inputs. The quantized implementation matched its
explicit INT8 oracle. These are component measurements with random inputs,
not model-level quality-loss percentages.

Therefore **zero additional fidelity loss has not been demonstrated**. Tuning
can preserve the current quantized arithmetic, but establishing parity with
official execution requires an end-to-end reference gate: logits/KLD, greedy
agreement, representative task quality, and draft acceptance on held-out short
and long contexts. Component correctness and coherent generation are necessary
but insufficient. Universal numerical identity should not be assumed for this
additional quantization.

The Qwen recipe records historical short C6 results around 319–352 tok/s and
32k single-request TTFT around 15–16 seconds. Those used different prompts,
output lengths, concurrency, and configurations; no new matched Qwen run was
performed here, so no exact Bolt/Qwen speed ratio is claimed. Bolt's 9B active
parameter count does not imply proportional runtime: it still executes 88
target blocks, shared/dense projections, routing, packed conversion, and draft
verification. The profile supports substantial implementation headroom; it
does not establish that Bolt will run several times faster than Qwen.

## Reproduction and evidence

Code identity: vLLM `f827e37a50`, AITER `a6aacf7d52`. Recipe: official target and
LiLiCorr revisions recorded in [the model recipe](nemotron_bolt.md), TP4/C8,
NS7, INT8-G128 KV on both models, FP32 Mamba state, BF16 residuals, custom AR,
full decode graphs, default mixed-prefill policy, CPU tier only. No numerical
or performance settings were changed during this investigation.

From the checkout, with `MODEL_DIR` and `API_KEY_FILE` pointing at the local
checkpoint and private diagnostic credential:

```bash
.venv/bin/python scripts/bench_bolt_profile.py \
  --tokenizer "$MODEL_DIR/tokenizer.json" --api-key-file "$API_KEY_FILE" \
  --chat --c 8 --tokens 256 --out logs/bolt-chat-c8.json
```

Repeat for the second trial. Omit `--chat` for the repetitive 32-input probe;
use `--c 1` for the single-stream probe. For cold prefill, omit `--chat`, set
`--prompt-tokens 8192 --tokens 1`, and change `--salt` at the beginning of each
request. Inspect outputs: `ignore_eos` fixes work and is not a quality test.

Profiling uses the existing [lifecycle capture procedure](agent_mixed_profile.md)
on an isolated localhost diagnostic boot. Use `VLLM_LIFECYCLE_GPU_CHUNK_STEPS=32`
and capture rank 0. Warm up first, then profile separate `--c 8 --tokens 128`
runs with and without `--chat`, and a distinct-prefix 8k/1 request. Convert with
`scripts/kineto_to_agent_csv.py` and analyze with `scripts/analyze_agent_profile.py`.
Restore the normal diagnostic server with development endpoints disabled after
capture; production service activation remains deferred by the user.

Restoration was verified: Bolt is running on localhost port 8021, authenticated
generation and health checks pass, unauthenticated model access returns 401,
and the development RPC endpoint returns 404.

Raw local evidence is retained under ignored `agents/tmp/bolt-profile/`:
`base-*.json`, `chat-c*.json`, `profile-*.json`, `host/gpu/*.json.gz`, host JSONL,
`*-analysis.json`, `*-summary.json`, `report-data.json`, GPU snapshots, and
`preflight.log`. The workload JSON retains request bodies, responses and stream
timestamps without credentials. The initial prefill trial saved its complete
result before an auxiliary rate-print division-by-zero; that print was fixed,
and no failed request was substituted. The temporary boot-readiness timeout
was extended while the original server continued loading; it was not a second
model launch.
