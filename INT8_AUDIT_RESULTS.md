# INT8 Serving Audit Results

## Scope and evidence

This audit covers the canonical gfx908 TP4/C8 serving stack for:

- target: `curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128`
- draft: `curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128`

It was refreshed on 2026-08-24 after reviewing vLLM commits
`16e2b51e0..e2ec34d25`, the corresponding AITER changes through
`ec90fc933`, the current launcher, the active server process, and the live
startup log, plus the raw benchmark directories under `logs/c8_optimization/`.
It then reran isolated four-GPU benchmarks with auditable manifests under the
`iso_*_20260824` result directories.

The recorded benchmark labels are not accepted as ground truth. This audit
cross-checks each leg against its startup log and kernel selection where those
artifacts still exist. Where the exact server configuration cannot be
reconstructed, the result is treated as an observation rather than evidence
that one implementation wins.

The target has 64 layers: 48 GatedDeltaNet linear-attention layers and 16 full
attention layers. The DFlash2 draft has five noncausal sliding-attention layers.

## Executive verdict

The overnight work made real progress:

- AITER CK W8A8 is active for the GPTQ linear layers and delivers 31.8% higher
  median C8 output throughput than the valid gfx908-tuned W8A16 baseline.
- AITER Custom All Reduce (CAR) was repaired for eager and graph replay. It is
  coherent under repeated TP4/C8 service, but current unfused medians trail
  vLLM CUSTOM by 8.8%.
- Actual PYNCCL/RCCL is slower than both custom implementations; the old
  reported RCCL win was a mislabeled vLLM CUSTOM run.
- A fused CAR + residual + RMSNorm + per-token INT8 quant-out kernel and vLLM
  fusion patterns now exist.
- The CK GEMM and activation quantizer are registered as custom ops so the
  compiled graph can represent them.
- A one-run native-MTP2 follow-up reached 252.22 whole-request output tok/s at
  C8 with 96.41% draft-token acceptance, versus about 64 tok/s for the matching
  DFlash2 stack. Its TPOT-derived post-first-token generation rate was 35.49
  tok/s per stream, or 283.89 tok/s when normalized to eight active streams.
  These are different metrics and must not be interchanged.

However, the advertised full INT8 recipe is still not the production dataflow:

- the live launcher is using vLLM `CUSTOM` all-reduce, not RCCL/PYNCCL and not
  AITER CAR;
- the fused INT8 epilogue is inactive because torch.compile is disabled and
  `fuse_allreduce_rms` defaults false;
- draft KV is FP16;
- DFlash2 noncausal attention still falls back to the vLLM Triton kernel;
- GDN recurrent-state INT8 remains an unscaled integer store;
- the target `lm_head` remains BF16;
- the CK path changes the checkpoint's GS128 weights into per-output-channel
  INT8 weights at load time.

The current documentation repeatedly describes intended features as active even
when the launcher and live log prove otherwise.

## Phase-1 replay error budget (2026-08-25, supersedes prior conviction order)

Method: module I/O was recorded once under the exact production recipe
(`bootA`: GPTQ-gs128 W8A8 + CK requant + int8 PTH KV + fp16 GDN state + DFlash2
NS=15, TP4, CUDA graphs, recorder armed by the probe), and once under a BF16
reference boot (`bootB`: bf16 weights, auto KV, fp32 mamba state, no spec).
Artifacts live under `~/models/kld/quant_audit/` (not in git; ~2 GB/boot).
Replay is offline CPU against the BF16 weights: `scripts/quant_replay_gemms.py`
(GEMM leg decomposition), `scripts/quant_replay_state.py` (KV + GDN state +
AR), `scripts/quant_replay_drafter.py` (draft chain). JSON results under
`~/models/kld/quant_audit/replay/`.

End-to-end KLD gate (52 prompts x 256 tokens greedy, top-20 logprobs,
`scripts/kld_probe_v2.py`): recipe-vs-BF16 KLD mean 1.83 / median 0.097 /
p95 11.36, greedy 40-char agreement 24/52. The median is fine; the tail is
catastrophic. That tail is the quality bug being fixed.

Error budget at GEMM output (rel-L2 vs BF16 golden, recorded rank0 inputs):

| family | gptq leg | ck-requant leg | act-quant leg | total |
|---|---|---|---|---|
| linear_attn.in_proj_qkvz | 0.29% | 0.72% | 14.78% (p95 27.0%) | 14.80% |
| linear_attn.out_proj | 0.62% | 0.82% | 6.85% | 6.93% |
| mlp.down_proj | 0.69% | 0.92% | 10.80% | 10.86% |
| mlp.gate_up_proj | 0.39% | 0.76% | 9.09% | 9.13% |
| self_attn.o_proj | 0.61% | 0.89% | 4.67% | 4.80% |
| self_attn.qkv_proj | 0.21% | 0.60% | 12.84% | 12.85% |

Verdicts, in new conviction order:

1. **Per-token int8 activation quantization is the dominant error source**
   (10-15% mean, up to 27% p95 per GEMM), 10-30x every weight leg combined.
   The aiter `pertoken_quant` path is absmax/127 with trunc-toward-zero on
   heavy-tailed activations (hidden-state kurtosis 239-10,421). Fixes to
   evaluate in replay first: rounding instead of truncation; per-token
   percentile/clipped absmax; per-channel outlier migration folded into the
   weights (SmoothQuant-style) so the serving GEMM stays plain CK W8A8.
2. **GDN state blow-ups in early layers**: 17 bootA-only snapshots with
   ||h|| up to 63,245 (bf16 ref: ~1.2) at layers 2/5/8/10 - within 4% of the
   fp16 overflow ceiling (65,504). This is the catastrophic-tail generator.
   The int8-native fix is the Nemotron-style scaled int8 state (block-scaled
   + stochastic rounding, granularity swept offline); fp32 state is the
   fallback, not the answer.
3. **Weight quantization is exonerated**: GPTQ leg 0.2-0.7% and CK
   per-channel requant leg 0.6-0.9% are sub-1% everywhere, including the
   previously suspected DFlash2 layer-0 down_proj (0.80% weight rel-L2;
   the 0.4168 in the quant log is a calibration-relative metric, not weight
   error). The planned "DFlash2 down_proj requant" is cancelled.
4. **Custom AR is exonerated**: recorded outputs are bitwise identical to
   fp32-accumulate-then-round-once (mean rel err 9.6e-5, 1.5x better than
   naive fp16 sequential sum). No AR change is warranted for accuracy.
5. **KV int8 PTH storage is exonerated**: k 0.88% / v 0.84% mean rel-L2 per
   token-head (p95 ~1.5%) - normal int8 SNR. Keep as is.

Recorder integrity notes: 142/512 GEMM records were gated out (99 all-zero x,
43 xq/xs from a different forward); the replay scripts gate on xs-consistency
and report exclusions. The recorder fp16-casts float tensors on disk; replays
recompute per-token scales from x. Proxy inputs for the drafter were assembled
from recorded target activations where K aligns (bootA captured no draft
hidden states; only draft KV).

### AITER CK W8A8 GEMMs: active and substantially faster than valid W8A16

**Status: active. Performance conclusion: confirmed for the current default
CK implementation, despite missing gfx908 tuning entries.**

`AiterW8A16LinearKernel` now dequantizes the GPTQ GS128 weights and requantizes
each output row to one INT8 scale. Activations are quantized with one dynamic
scale per token, and `gemm_a8w8_CK` performs the INT8 GEMM. The compatibility
class name is therefore misleading, but the active compute really is W8A8.

The 2026-08-24 isolation rerun fixed vLLM `CUSTOM` all-reduce, compilation mode
NONE/FULL_DECODE_ONLY, fusion off, deterministic greedy requests, two C8
warmups, and identical target/draft/cache settings. Startup logs and manifests
prove the selected kernels. Three repeats produced:

| GEMM route | SS median (range) | C8 output median (range) | TPOT median | TTFT median |
|---|---:|---:|---:|---:|
| CK W8A8 | 14.56 (14.07-15.89) tok/s | 63.49 (60.98-68.65) tok/s | 105.00 ms | 437.49 ms |
| valid W8A16 | 10.64 (10.60-12.12) tok/s | 48.17 (47.31-50.60) tok/s | 165.68 ms | 547.63 ms |

CK therefore improves median C8 output throughput by 31.8%, single-stream by
36.8%, and TPOT by 36.6% versus the valid baseline. A late CK control after all
other legs still reached 60.17 tok/s, ruling out the original lead being only a
favorable early run.

The valid baseline explicitly selected `TritonW8A16LinearKernel`; at small M it
routes to the earlier gfx908-tuned `gptq_w8a16_repacked_gemm`. The apparent
AITER blockscale alternative was also isolated with CK disabled, but it emitted
the same degenerate `duct Register Register...` text for all three prompts. Its
67.13 tok/s C8 result is invalid and must never be cited as a performance
baseline.

The CK result establishes the value of the shipped default, not its optimized
ceiling. AITER's `a8w8_tuned_gemm.csv` still has no gfx908 entries, live startup
logs still report default configs for production shapes, and `_warmup()` still
warms blockscale rather than CK. A production-shape CK tuning pass remains
warranted.

This path does **not** preserve GS128 scaling during GEMM. GS128 describes the
source checkpoint. At load time the code performs:

```text
GPTQ INT8 + per-128-group scales
  -> dense FP16 reconstruction
  -> INT8 + one scale per output channel
```

That adds another weight quantization step and discards group-local dynamic
range. The August 22 KLD files predate the August 24 CK integration, so the
published `+ act quant` KLD row is not evidence for this active CK-requantized
path. A fresh real-artifact KLD comparison is still required.

The implementation also retains both the original unpacked GS128 INT8 weights
and the new `_ck_q` INT8 copy on GPU. This approximately duplicates quantized
weight storage for affected layers and reduces memory available to KV cache.
Where the GS128 fallback/dequant view is not required, a load-only conversion or
selective release should be considered.

Finally, `_warmup()` still invokes `gemm_a8w8_blockscale`, not
`gemm_a8w8_CK`. The live startup log consequently reports missing CK tuned
configs for many production `(M,N,K)` shapes during graph capture and first
inference. The active CK shapes should be tuned and warmed directly.

### AITER CAR repair: coherent, but current unfused path trails vLLM CUSTOM

**Status: implemented and gated off by `CAR=0`. Correctness repair confirmed;
current performance characterized, optimized ceiling not established.**

The AITER changes address both previously observed corruption modes:

- eager calls stage through an uncached input pool;
- captured calls stage through the pre-registered pool so graph replay does not
  reuse an incoherent peer-L2 view.

The old "RCCL" leg was vLLM `CUSTOM`, not RCCL. The isolation rerun explicitly
verified all three dispatch modes with CK held on:

| All-reduce backend | Verified dispatch | SS median (range) | C8 output median (range) | TPOT median | TTFT median |
|---|---|---:|---:|---:|---:|
| vLLM CUSTOM | `['CUSTOM', 'PYNCCL']` | 14.56 (14.07-15.89) tok/s | 63.49 (60.98-68.65) tok/s | 105.00 ms | 437.49 ms |
| AITER CAR | `['AITER_CUSTOM', 'PYNCCL']` | 13.42 (13.10-14.71) tok/s | 58.34 (57.16-60.58) tok/s | 114.27 ms | 447.87 ms |
| PYNCCL/RCCL | `['PYNCCL']` | 12.18 (11.93-13.23) tok/s | 53.04 (52.26-57.57) tok/s | 133.89 ms | 470.36 ms |

Greedy AITER-CAR and vLLM-CUSTOM samples had byte-for-byte identical beginnings
for the three correctness prompts. Repeated TP4/C8 service completed without a
CAR replay failure, supporting the repair. Current medians put vLLM CUSTOM 8.8%
ahead of AITER CAR and 19.7% ahead of actual PYNCCL. A late CUSTOM control was
60.17 tok/s versus AITER CAR's cold 60.58 tok/s, so cold-start CUSTOM versus CAR
can be within noise; their thermally stabilized third repeats favor CUSTOM by
6.7%. The defensible conclusion is that CUSTOM leads sustained warmed service,
not that every individual pass beats CAR.

This still is not a comparison of optimized CAR ceilings. On gfx908
`csrc/include/custom_all_reduce.cuh` forces `use_new=false` and uses generic
`threads=512`, `block_limit=16`; no production gfx908 tuning sweep was found.
Compilation was disabled and `fuse_allreduce_rms=false`, so the intended fused
CAR + residual + RMSNorm + INT8 quant-out design was not exercised. The fused
path remains unbenchmarked rather than a performance loser.

### Fused CAR + RMSNorm + per-token INT8 quant-out: implemented but inactive

**Status: kernel and compiler pattern implemented; not active in production.**

AITER `ec90fc933` provides `fused_ar_rms_int8_per_token_quant()`, and vLLM
`89e808d3e` adds the custom op and both RMSNorm pattern variants. This matches
the active CK consumer:

- floating-point partial input;
- floating-point residual output;
- per-token INT8 normalized activation;
- FP32 `[M,1]` activation scale.

The residual remains floating point, so fusion should add no quantization loss
beyond the standalone per-token activation quantizer it replaces.

The AITER micro-suite reportedly matched the unfused payload bit-for-bit at 24
two-rank cases, with only reduction-order ULP differences in scales. That is
useful kernel evidence, but it is not a TP4 end-to-end validation of the vLLM
pattern.

The live server shows:

- `VLLM_ROCM_USE_AITER_CUSTOM_AR=0`;
- `compilation_config.mode=NONE` after the gfx908 platform override;
- `fuse_allreduce_rms=false`;
- an explicit warning that Inductor-only optimizations are ignored.

Therefore none of the fused epilogue patterns execute in the production run.
The attempted compiled path produced incoherent generations even with CAR and
AR fusion disabled, so enabling this fusion currently depends on either fixing
the gfx908 Inductor path or adding a carefully placed eager fusion seam.

This fused kernel still transports floating-point partials. Its INT8 label
describes the post-RMSNorm output, not the all-reduce payload.

### Compile-path custom ops: necessary scaffolding, not a production speedup yet

**Status: implemented; dormant while compile is disabled.**

The CK GEMM wrapper avoids tracing AITER's JIT-config lookup, and the
`input_guard` change avoids tracing a skip-marked accelerator query. They remove
two compile-time blockers. They do not resolve the observed compiled-output
corruption and do not affect the current mode-NONE production path.

No vLLM test was added for the new pattern match, custom-op fake contracts, or
TP4 fused execution. The AITER kernel test alone does not cover those layers.

### Native MTP2 follow-up: 252.22 whole-request output tok/s in one C8 run

**Status: native sidecar confirmed and one controlled run completed. Result is
promising enough to prioritize, but it is not yet a repeated production
winner.**

The GS128 target checkpoint contains its native Qwen MTP sidecar in the final
safetensors shard. The sidecar has `mtp.fc`, one full-attention MTP decoder
layer, its MLP and attention projections, and the associated norms. The config
declares `mtp_num_hidden_layers=1`.

The follow-up replaced DFlash2 with:

```json
{"method":"mtp","num_speculative_tokens":2}
```

while retaining the best measured target stack: TP4/C8, CK W8A8 target GEMMs,
vLLM CUSTOM all-reduce, AITER unified attention, target
`int8_per_token_head` KV, INT8 Mamba state, `max_num_batched_tokens=2048`, and
gfx908's forced mode-NONE/FULL_DECODE_ONLY graph policy. vLLM resolved the
sidecar as `Qwen3_5MTP`, shared the target embedding and `lm_head`, and warned
that the one learned MTP layer is reused modulo the layer count for the second
proposal. That reuse did not prevent high acceptance in this run.

The directly comparable decode workload was eight concurrent requests with 32
random input tokens and 1,000 greedy output tokens each, after two warmups:

| Metric | Native MTP2 result |
|---|---:|
| Successful requests | 8/8 |
| Benchmark duration | 31.72 s |
| Whole-request output throughput | 252.22 tok/s |
| Total token throughput | 260.30 tok/s |
| Mean TTFT | 546.94 ms |
| Mean TPOT | 28.18 ms |
| Draft acceptance rate | 96.41% |
| Mean acceptance length | 2.93 |
| Position-0 acceptance | 97.44% |
| Position-1 acceptance | 95.38% |
| Accepted / drafted tokens | 5,264 / 5,460 |

The benchmark's 252.22 tok/s is generated tokens divided by total wall time. It
includes TTFT/prefill and therefore is not a pure token-generation rate. The
mean TPOT of 28.18 ms is the relevant post-first-token measurement: 35.49
tok/s per stream, or 283.89 tok/s normalized to eight simultaneously active
streams. The short 32-token prompts make the whole-request figure
decode-dominant, but they do not change the metric's definition.

The 252.22 whole-request figure is approximately 3.97x the current CK + vLLM
CUSTOM + DFlash2 median of 63.49 tok/s and approximately 1.97x the historical
128.0 tok/s no-spec result. The first comparison holds the current target stack
closely aligned; the historical no-spec comparison crosses checkpoint and
launcher revisions and is context rather than a clean A/B. The benchmark's
separate 104 tok/s `peak` field is a sampling-window statistic and is not a
replacement for either whole-request throughput or TPOT-derived generation.

The acceptance result also explains the large difference from DFlash2. The
later DFlash2 diagnostic accepted zero draft tokens and paid pure proposal and
verification overhead. Native MTP2 accepted nearly both proposed positions on
almost every step and produced close to three useful tokens per target
verification cycle.

This run does not establish an optimized ceiling:

- it is one measurement, not a three-repeat median with a late control;
- graph capture and first inference reported missing gfx908 A8W8 tuning entries
  for MTP/verification batch shapes including `M=3`, `M=6`, and `M=9`;
- the MTP sidecar itself is not INT8. Its 15 tensors are BF16, including about
  850 MB of linear weights globally; and
- the target `lm_head` shared with MTP remains BF16.

The MTP sidecar therefore makes this the fastest observed serving
configuration, not an all-W8A8 implementation. Quantizing the sidecar and
shared head would need an acceptance and KLD gate before being treated as an
optimization.

A prefill-dominant C8 follow-up used eight cold 8,192-token prompts and one
output token per request. The trusted run used a new random seed and zero
warmups so benchmark warmups could not populate the prefix cache:

| Metric | Cold C8 prefill result |
|---|---:|
| Successful requests | 8/8 |
| Total input / output tokens | 65,536 / 8 |
| Benchmark duration | 34.04 s |
| Approximate input throughput | 1,925.01 tok/s |
| Total token throughput | 1,925.24 tok/s |
| Mean / median TTFT | 19.85 / 19.92 s |
| P99 TTFT | 33.80 s |

An earlier pass reported 2,174.74 total tok/s and 17.41-second mean TTFT, but
its two warmup prompts were reused by the measured request set and the server
reported prefix-cache hits. That warm-cache result is retained only as a
methodology warning and must not be cited as cold-prefill performance. MTP adds
little meaning to the one-token prefill test; this result primarily measures
the target's chunked-prefill path with a 2,048-token scheduling budget.

Auditable artifacts are under:

- `logs/c8_optimization/mtp2_ck_custom_single_20260824/`
- `logs/c8_optimization/mtp2_ck_custom_prefill_8x8192_20260824/`

### Corrected warmed 8,192-input / 512-output metric split

A later suite warmed the server with the complete C8 8,192-input/512-output
workload, then used fresh random seeds for the measured legs. It resolves the
earlier misuse of output throughput as token generation:

| Leg | Correct interpretation | Result |
|---|---|---:|
| PP: 8 x 8,192 input, 1 output | Input tokens divided by wall time | 1,889.44 tok/s |
| TG: 8 x 8,192 input, 512 output | TPOT-derived post-first-token rate | 16.55 tok/s per stream; 132.36 tok/s C8-normalized |
| Combined: 8 x 8,192 input, 512 output | All input and output tokens divided by wall time | 1,202.74 tok/s |

The PP leg completed in 34.69 seconds with mean TTFT 20.214 seconds. The TG leg
had 60.44 ms mean TPOT, 95.63% acceptance, and 2.91 mean acceptance length. Its
reported 73.51 output tok/s is only whole-request output accounting because it
includes the long prompt phase. A preload attempt did not isolate decode: the
server continued to report 0.0% prefix-cache hits, so that attempted method is
not evidence for a pure TG number.

The fresh combined leg completed in 57.89 seconds, with 22.477-second mean
TTFT, 63.66 ms mean TPOT, 89.57% acceptance, and 2.79 mean acceptance length.
Its 70.75 output tok/s is likewise not token generation. Artifacts are under
`logs/c8_optimization/mtp2_correct_pp_tg_combined_8192_512_20260824/`.

## Prefill INT8 and W8A8 architectural audit

**Method: the architecture findings began as a static source,
checkpoint-metadata, configuration, and log audit. The embedding item was then
implemented and measured in a separate GPU A/B follow-up described below.**

The target has 400 GPTQ/W8A8 matrix instances across its 64 target layers: six
in each of the 48 GatedDeltaNet layers and seven in each of the 16 full-
attention layers. The active GS128 path reconstructs the checkpoint weights at
load, requantizes each output channel to INT8, and for every matrix invocation
per-token-quantizes its FP16 activation before calling AITER CK A8W8. Thus the
large target projections and MLPs really use INT8 GEMMs during prefill, but the
surrounding graph remains a series of FP16 producers, separate INT8
quantizers, INT8 GEMMs, and FP16 consumers. `in_proj_a` and `in_proj_b` in the
48 GDN layers, their depthwise convolutions, norms, recurrent math, and the
target head are floating point.

The following changes are ordered by expected prefill effect. The ranking is
architectural judgment from the dataflow and tensor sizes, not benchmark proof:

1. **Gather INT8 embedding rows before dequantization. Implemented; local
   operation effect: high; end-to-end prefill effect: below measurement noise;
   quality cost: none.** The old `embedding()` evaluated `w_q.to(fp16)` before
   `F.embedding`, eagerly converting the entire vocab-sharded table instead of
   only requested rows. For this 248,320 x 5,120 table it allocated about 646
   MiB per TP4 rank in the measured operation. The replacement gathers INT8
   rows first, then casts and applies their per-row scales. It preserves the
   exact old output while avoiding the whole-table conversion.

   An actual-shape one-GPU microbenchmark used a 62,080 x 5,120 TP4 shard and
   2,048 token IDs. Median lookup time fell from 3.6935 ms to 0.2616 ms
   (14.1x), throughput rose from 0.554 million to 7.830 million rows/s, and
   peak extra allocation fell from 646.2 MiB to 60.0 MiB. A late old-path
   control reproduced 3.6923 ms. Outputs were bit-identical.

   Full TP4/C8 serving impact was neutral within the system's order drift.
   After one 8 x 8,192/512 warmup, three 8 x 8,192/1 passes produced:

   | Implementation | Input-throughput samples | Median |
   |---|---|---:|
   | Old whole-table cast | 1,897.35, 1,866.68, 1,837.07 tok/s | 1,866.68 tok/s |
   | Gather first | 1,861.49, 1,834.34, 1,813.04 tok/s | 1,834.34 tok/s |
   | Late old-path control | 1,818.03, 1,791.05, 1,770.09 tok/s | 1,791.05 tok/s |

   Both within-server and across-server performance declined monotonically.
   The gather-first median is 0.30% above the midpoint of the bracketing old
   medians, so the defensible conclusion is no measurable end-to-end throughput
   change. The fix is still worthwhile for its deterministic allocation and
   bandwidth reduction. Artifacts and source hashes are under
   `logs/c8_optimization/int8_embedding_gather_ab_20260824/`.
   Post-change validation also passed the focused embedding unit test, the
   INT8-PTH attention micro suite (`OVERALL: PASS`), and the gfx908 battery
   (`4 PASS, 0 FAIL`).

2. **Fuse floating producers directly into the per-token INT8 representation
   consumed by CK, and reuse it across compatible projections. Expected
   effect: high; quality cost: none beyond the active W8A8 format.** The CK
   wrapper currently flattens/casts and launches a separate activation
   quantizer for every linear. High-value eager fusions are RMSNorm-to-INT8,
   RMSNormGated-to-INT8, SiLU-and-mul-to-INT8 for `down_proj`, and the existing
   row-parallel all-reduce + residual + RMSNorm + INT8 quant-out design. The
   compile-only patterns are inactive in mode NONE, so gfx908 needs eager
   seams. A shared quantized hidden-state input could also feed projections
   that consume the exact same normalized tensor.

3. **Tune and warm the actual CK prefill kernels. Expected effect: high to
   medium; quality cost: none.** The serving log reports missing gfx908 tuning
   for the dominant M values around 2,016-2,048 and tail values such as 277,
   565, and 853, across the model's major N/K families. The shipped AITER
   tuning table has no gfx908 rows. Worse, `_warmup()` invokes the blockscale
   kernel even though production uses per-token-scale CK. Add measured CK
   configurations for these shapes and call the CK custom ops during warmup.
   This is the most immediate optimization, although it is kernel tuning rather
   than a model-architecture change.

4. **Add a gfx908-native fused GDN prefill path. Expected effect: medium to
   high; quality cost: none if recurrence stays floating point.** Forty-eight
   of 64 layers currently run separate QKVZ and BA projections, causal conv,
   `fused_post_conv_prep`, a Triton/FLA chunked gated-delta recurrence, state
   conversion/store, RMSNormGated, and output projection. Fuse causal conv,
   post-conv preparation, chunk recurrence, and final-state write to avoid
   materializing Q/K/V/g/beta and reduce launches. Keep the recurrent state,
   decay/gates, and accumulation floating; bare INT8 recurrence is not a safe
   arithmetic optimization.

5. **Fuse full-attention Q/K norm + RoPE with INT8-per-token-head KV
   quantization/cache writes, and let prefill attend directly to current-chunk
   FP16 K/V. Expected effect: medium; quality cost: none.** The existing fused
   RoPE/cache APIs explicitly reject INT8-per-token-head mode, so a separate
   Triton cache-update kernel writes INT8 K/V plus FP32 scales. Unified
   attention then reads the cache. A hybrid kernel can use the current chunk's
   already-live FP16 K/V while simultaneously quantizing it for future chunks
   and decode, dequantizing only older paged-prefix blocks. That avoids
   quantize-then-immediately-dequantize traffic in the 16 full-attention layers.

6. **Pack or fuse GDN BA with QKVZ, preferably sharing activation
   quantization. Expected effect: medium to small; quality risk: low but must be
   measured.** The 96 `in_proj_a`/`in_proj_b` matrices are BF16, but their output
   width is small relative to QKVZ. Standalone W8A8 conversion is unlikely to
   matter much and perturbs sensitive gates. Appending/packing them into the
   large projection can remove a BF16 GEMM and input reread for little added N;
   validate KLD and long-context state behavior.

7. **Overlap row-parallel GEMMs with floating-point collectives, and evaluate
   larger prefill chunks. Expected effect: medium but uncertain; quality cost:
   none.** `out_proj` and MLP `down_proj` produce floating partials and use a
   floating custom all-reduce. An eager fused epilogue and tiling/pipelining can
   overlap communication with compute. Separately test 4,096/8,192 token
   scheduler budgets against the current 2,048-token budget; larger M may
   improve GEMM efficiency and reduce launches, but activation memory and
   queueing can reverse the gain. Sequence parallelism may merely exchange an
   all-reduce for reduce-scatter/all-gather, so it needs measurement.

The prefill audit does **not** recommend quantizing softmax, attention or GDN
accumulation, GDN depthwise convolution, decay/gate parameters, FP32 KV scales,
or rank-local collective payloads. It also does not prioritize the BF16 MTP
sidecar or target `lm_head` for ordinary prefill: MTP is a decode path, and
serving normally computes logits only at sampling positions unless prompt
logprobs are requested. Those changes carry quality risk for little prefill
benefit.

## Benchmark-integrity findings

`scripts/bench_quick.sh` now records a per-run manifest with both repository
SHAs, server PID/argv, a safe whitelist of serving environment controls, GPU
state, and selected server configuration lines. It also forces greedy C8
sampling, uses two warmups, preserves response samples for correctness review,
and reports sustained as well as peak output throughput.

The new artifacts resolve the earlier label/configuration ambiguity. They also
expose strong within-server heat/order sensitivity: CK C8 throughput declined
from 68.65 to 60.98 tok/s over three consecutive runs, and every other backend
showed the same direction. Report medians and ranges, compare matching run
positions, and include a late control after the alternatives. Peak throughput
alone is not sufficient because scheduler quantization left multiple materially
different sustained-throughput runs at the same 64 or 72 tok/s peak.

Remaining methodology limitation: the legs were repeated in backend groups
rather than fully interleaved because each server restart reloads the 29.6-GiB
model and recaptures graphs. Matching cold/warm positions plus the late CUSTOM
control makes the ordering credible, but smaller CUSTOM-versus-CAR differences
should still be described as ranges, not immutable constants.

## Highest-priority corrections and remaining work

### 1. Tune the active gfx908 CK and AITER CAR production shapes

**Expected effect: potentially medium; current defaults are now measured but
not their optimized ceilings.**

The launcher sets:

```text
VLLM_ROCM_USE_AITER_CUSTOM_AR=0
AR=1
```

`CAR=0` disables only AITER CAR. Because `AR=1` omits
`--disable-custom-all-reduce`, vLLM's own custom all-reduce remains enabled.
The live server reports the dispatch order:

```text
['CUSTOM', 'PYNCCL']
```

Therefore the current default is vLLM `CUSTOM`, and the rerun confirms that it
is the fastest verified sustained-service choice. `AR=0` or
`--disable-custom-all-reduce` selects the slower actual PYNCCL/RCCL leg.

Keep vLLM CUSTOM as the default while tuning CK's missing gfx908 shapes and
AITER CAR's forced-naive kernel. After tuning, rerun the same captured matrix.
Do not extrapolate the unfused CAR result to the dormant fused epilogue.

### 2. Run a real KLD gate for the active CK-requantized W8A8 path

**Status: DONE 2026-08-25 — see "Phase-1 replay error budget" at the top.**
KLD mean 1.83 / median 0.097 / p95 11.36, greedy 24/52 vs the BF16 reference.
The gate now exists (`scripts/kld_probe_v2.py`) and is re-run after every
accepted accuracy fix.

**Expected effect: correctness evidence, not direct speed.**

The old KLD sweep predates per-channel weight requantization. Qualitative
coherence and a soak heuristic do not measure distribution shift. Capture and
compare the exact current artifact and serving kernels, including the CK
`_ck_q/_ck_s` weights, per-token activation quantization, INT8 embedding, and
the selected Mamba state dtype.

### 3. Activate the fused INT8 epilogue without enabling a corrupt compiler path

**Expected effect: potentially very large. Quality cost: none intended beyond
the already-active per-token W8A8 quantization.**

The implementation now exists, but the production blocker moved from missing
kernel support to graph integration. The two viable paths are:

1. fix and validate torch.compile/Inductor on gfx908, then enable AITER CAR and
   the fusion; or
2. add an eager model seam that fuses the row-parallel reduction, residual add,
   RMSNorm, and quantizer while preserving graph-capture behavior.

The current default vLLM `CUSTOM` observation is faster than the untuned,
unfused AITER CAR observation, but the intended fused AITER path needs to be
activated, tuned, and measured before choosing a production winner.

### 4. Quantize the target `lm_head` to INT8

**Expected effect: large, especially during speculative verification. Quality
cost: modest but directly affects token probabilities.**

The untied target still stores `lm_head.weight` as BF16 `[248320,5120]`, about
2.54 GB globally or roughly 635 MB per TP4 rank. The checkpoint metadata has
`lm_head: false`. This requires a requantized artifact or a deliberate runtime
INT8 head; the serving launcher alone cannot fix it.

Any head quantization must receive its own KLD and acceptance-rate gate because
its error is applied directly to logits.

### 5. Restore INT8 draft KV only after fixing the noncausal serving path

**Expected effect: medium. Current quality status: failed in end-to-end
serving.**

The launcher now explicitly uses FP16 draft KV. The target uses
`int8_per_token_head`, but the DFlash2 speculative JSON uses `float16`.
Earlier end-to-end runs reported corruption with INT8 draft KV, so the current
fallback is justified until the bug is isolated.

Selecting the AITER unified-attention backend does not make DFlash2 attention an
AITER kernel. `RocmAiterUnifiedAttentionImpl.forward()` explicitly falls back
to vLLM Triton whenever `attn_metadata.causal` is false. The live first-request
JIT log for `kernel_unified_attention_2d` is consistent with that fallback.

The right optimization is a correct gfx908 noncausal/sliding-window kernel with
INT8-PTH cache support, followed by an actual DFlash2 acceptance/coherence gate.
A standalone noncausal micro-test is insufficient evidence for the speculative
serving layout and cache-update path.

### 6. Tune and warm the active CK production shapes

**Expected effect: medium to small, with startup and tail-latency benefits.**

The live log repeatedly says production CK shapes are absent from
`a8w8_tuned_gemm.csv` and fall back to defaults. Add tuned entries for the
actual target and draft shapes and change warmup to call the CK kernel rather
than the broken blockscale fallback.

### 7. INT8 embedding gather fixed; pin it explicitly

**Status: implemented and tested. End-to-end effect: neutral within noise.
Quality cost: none.**

The active implementation now gathers INT8 rows before converting them to the
scale dtype, eliminating the full sharded-table FP16 temporary. A focused CPU
test, a bit-identical GPU actual-shape comparison, and three full TP4/C8
serving legs passed. `VLLM_GFX908_INT8_EMBEDDING` still defaults true but is
absent from the live process environment. Set it explicitly in the canonical
launcher so a default change cannot silently restore BF16 embedding storage.

## Changes that still should not be made

### Do not use bare INT8 for the GDN recurrent state

The launcher still passes:

```text
--mamba-ssm-cache-dtype int8
```

The model itself declares `mamba_ssm_dtype=float32`, and the live server warns
that the launcher overrides it with INT8. The current GDN recurrence has no
per-head or per-block state scale. It loads and stores using the state tensor's
declared dtype, so this is an unscaled integer state rather than a conventional
scaled INT8 cache.

The documented "0.6% drift gate" does not validate the production behavior.
`scripts/mamba_state_int8_probe.py`:

- computes a dynamic absmax scale;
- quantizes and immediately dequantizes back to FP32;
- quantizes only after the 512-token chunk;
- runs the following 64 decode steps with FP32 state and no per-step INT8
  store;
- uses synthetic random tensors rather than the real model.

The actual serving path has no such scale and stores to INT8 on recurrent
updates. The probe therefore tests a different, much higher-quality format.
The 0.6% number must not be used as evidence for the live flag.

The theoretical state-bandwidth opportunity is real: the target's recurrent
state is about 151 MB per GPU in FP16 at C8 and is continually accessed during
decode. A scaled INT8 recurrence may be worthwhile, but it needs scale storage,
kernel support, and real-model long-context KLD/acceptance testing. Until then,
the production-quality choice is FP16 or the model-declared FP32.

### Do not set the GDN convolution cache to INT8

Keep `mamba_cache_dtype` at the model dtype. The causal-convolution update still
does `x = x.to(conv_state.dtype)` with no scale tensor. INT8 would directly
round the incoming continuous activation before the convolution.

The benefit is limited: at TP4/C8/NS7 the 48 GDN convolution caches are only
about 20 MB per GPU in FP16.

### Keep the GDN depthwise convolution floating point

`QwenGatedDeltaNetAttention.conv1d` remains outside `quant_config`. This is
appropriate: it is a width-four depthwise convolution with little weight or
arithmetic cost, while its error feeds the recurrent state. The large
`in_proj_qkvz` and `out_proj` projections are the correct W8A8 targets. The
small BF16 `in_proj_a`/`in_proj_b` gates are worth packing into the large input
projection only if that removes a launch/input reread and passes a quality
gate; quantizing them as standalone GEMMs is low-value.

### Keep recurrent parameters and sensitive nonlinear math floating point

Keep `A_log`, `dt_bias`, decay/gating calculations, softmax, attention
probability accumulation, P-times-V, RoPE, and persistent residuals in their
current floating-point representations. Their quality sensitivity exceeds the
small theoretical storage or arithmetic saving.

Fuse residual, norm, and nonlinear operations where profitable, but quantize
only the normalized activation immediately consumed by a W8A8 GEMM.

### Keep KV scales in FP32

`int8_per_token_head` stores INT8 K/V content with FP32 per-token/per-head
scales. The scale metadata is tiny relative to the 256 INT8 K/V values it
describes. Lower-precision scales save essentially nothing while worsening
reconstruction.

### Keep the DFlash2 selector codebooks floating point

The BF16 `[248320,256]` predecessor and successor codebooks are accessed by
candidate-row gathers, not full-vocabulary scans. Quantizing them can alter path
ranking and acceptance for little runtime benefit. The selector's
`5120 -> 256` projection is also not a major decode cost.

### Do not describe CAR as genuine INT8 collective transport

Both the standalone CAR path and the new fused epilogue communicate
floating-point partials. A genuine scaled INT8 collective would need to
communicate rank-local scales, dequantize differently scaled contributions,
accumulate safely, and preserve the floating-point residual. A bare integer sum
would combine incompatible scales and can overflow.

The implemented fused path is accurately described as:

```text
FP16 partial outputs
  -> AITER FP16 Custom All Reduce
  -> fused residual add + RMSNorm
  -> per-token INT8 quant-out + FP32 scale
  -> AITER CK W8A8 GEMM
```

## Documentation inconsistencies to remove

The canonical recipe, root README, AGENTS.md, and launcher comments currently
mix the original target contract with the measured fallback. They should not
claim all of the following are simultaneously active:

- AITER CAR mandatory while the default sets `CAR=0`;
- RCCL production while the launcher leaves vLLM `CUSTOM` enabled;
- fused INT8 quant-out mandatory while compile and the fusion flag are off;
- both target and draft KV INT8 while the speculative JSON says `float16`;
- AITER attention for DFlash2 while noncausal calls fall back to Triton;
- per-group GS128 activation quant-out while the active CK path is per-token;
- "INT8 all-reduce" when the collective payload is floating point;
- 0.6% production Mamba drift based on a scaled FP32 simulation that does not
  emulate the unscaled INT8 store.

## Updated priority order

Revised 2026-08-25 after the Phase-1 replay error budget (see top). The
accuracy program now leads; perf items that were already exonerated or
convicted move down or drop out.

Accuracy program (P2, conviction order from measured budget):

1. Activation-quant repair: rounding + clipped/percentile absmax in the
   per-token int8 quantizer; then per-channel outlier migration folded into
   weights (needs one checkpoint requant only if replay proves the win).
   Validate each variant offline in replay before any serving boot.
2. Scaled int8 GDN state (block-scaled + stochastic rounding, granularity
   swept offline against recorded bootA trajectories) to kill the early-layer
   fp16-state blow-ups that generate the KLD tail.
3. Re-run the 52-prompt KLD gate after each accepted fix; target: KLD mean
   <= 0.2, p95 <= 1.0, greedy 40-char agreement >= 45/52 while keeping the
   C8 TG rate within noise of the 42.9%-acceptance baseline boot.

Cancelled or demoted by evidence:

- DFlash2 layer-0 down_proj requant: exonerated offline (0.80% weight
  rel-L2, no outlier structure).
- AR backend change for accuracy: custom AR is bitwise fp32-accumulate.
- KV int8 storage change: sub-1% per token-head, normal int8 SNR.

Performance program (unchanged unless it touches the above):

4. Add eager producer-to-INT8 seams for mode NONE: norm-to-quant,
   SiLU-and-mul-to-quant, and row-parallel all-reduce + residual + RMSNorm +
   quant-out. Reuse a quantized activation wherever projections consume the
   same normalized input.
5. Tune and directly warm CK for the observed gfx908 prefill M/N/K shapes,
   especially M around 2,048 and the chunk tails. Then test larger scheduler
   token budgets separately.
6. Build a fused gfx908 GDN prefill core and fused full-attention
   QK-norm/RoPE/INT8-PTH cache-update path. Keep recurrent and attention
   accumulation floating point, and use current-chunk FP16 K/V directly.
7. Keep the verified CK W8A8 + vLLM CUSTOM default, tune AITER CAR's
   forced-naive launch geometry, then rerun the captured repeated matrix.
8. Either fix gfx908 Inductor or build an eager seam for the implemented fused
   CAR + RMSNorm + INT8 quant-out path, then measure it at TP4/C8 rather than
   extrapolating from the unfused CAR result.
9. Quantize and validate the target `lm_head`; separately evaluate the native
   BF16 MTP sidecar for W8A8 conversion with acceptance and KLD gates.
10. Fix noncausal DFlash2 INT8-PTH attention/cache handling and restore INT8
   draft KV only after an end-to-end acceptance/coherence gate.
11. Explicitly pin the already-defaulted INT8 embedding path after fixing its
    implementation; pinning the current default alone preserves the full-table
    cast defect.
