# Nemotron Bolt with official packed weights on MI100

This peer recipe runs the official NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB
checkpoint and its LiLiCorr draft on four MI100 GPUs. Static dense projections
convert once to INT8 GS128 and release their original packed GPU tensors and
codec metadata. Routed experts retain their packed originals and use a bounded
INT8 cache for frequently selected experts; misses convert tiles inside the
matrix kernels. The checkpoint files on disk are unchanged.

**Serving gates passed:** eager and graph TP4/C8
serving, LiLiCorr acceptance, 28k context/prefix replay, RAM-prefix restoration,
reasoning, tools, image/video and streaming pass. All eight AIS rank caches
match the official checkpoint bytes. The final post-loader Qwen regression
passed. A subsequent eight-cache-hit restart also passed C8, reasoning, tools,
image, streaming, API authentication and disabled-development-endpoint checks.
The root-managed `vllm-bolt.service` serves this recipe on port 8020.
The Qwen recipe remains independently available.

## Standard optimized combination (2026-10-17)

`scripts/serve_recipe_bolt.sh` selects the accepted combination: the initial
short-prefill cohort, 64-row packed expert prefill, weighted expert reduction,
the dedicated HD128/GQA8 4,256-token page attention kernel, and native MoE
RMSNorm/residual plus GS128 activation quantization. Each specialized path
checks its supported geometry and retains its ordinary fallback. The legacy
`VLLM_GFX908_FUSED_NORM_QUANT` producer stays off. The matching sibling AITER
revision is `3c74adc198e55aeb5bcd41516bddfb3ccadf6725`; the accepted vLLM
runtime revision is `4711913204533dfe702bd7e9ed7d23d07a08503b`.

The final accepted five-run 4,096-input/256-output C8 sample at **105 W per
GPU** measured median **2,300.42 aggregate prefill tok/s** and **168.28
aggregate decode tok/s**. Decode is timed separately, from the latest first
token until the final token, counting only tokens emitted during that window.
The same eight prompts, seed, cache resets and five repetitions define this
measurement; it is not comparable to the older short-prompt results below.
Subsequent experiments failed the retention gates and are excluded. Raw phase
rates, source identities and decisions are in the experiment ledger.

The standard memory settings (since campaign-2 B13, 2026-10-18) are
3 GiB/GPU for static dense INT8 weights
with their original packed GPU tensors released, 11 GiB/GPU for dynamic expert
INT8 copies (10,024 pairs, 240-259 slots/layer) with packed expert originals
retained, 7 GB/GPU for KV (1,641,644 tokens, 6.26x full 262,144-token
requests), and
24 GiB total for CPU prefix offload. Prefill-informed expert admission
(`VLLM_BOLT_PREFILL_ADMIT=1`) re-ranks the expert cache from a 64-token
prefill subsample each forward so resident hits converge immediately. TP4/C8, LiLiCorr NS=7, BF16 residuals,
INT8-G128 target/draft KV, FP32 Mamba state and custom all-reduce remain fixed.
The 105 W results above are fixed-power benchmark measurements. Production
runs with nginx and the thermal power controller enabled; its live caps vary.

### Thermal power control

The controller only changes GPU power caps. It never stops Bolt, including
when a temperature remains high, the minimum cap is reached, or the controller
encounters a sensor/driver fault. The separate thresholds are:

| Sensor | Reduce by 1 W per five minutes | Cut by 20% immediately |
| --- | --- | --- |
| GPU edge/junction | Above 90°C | Above 95°C |
| Memory | Above 89°C | Above 90°C |

A five-minute GPU average above 90°C also requests the 1 W reduction.
Emergency cuts repeat after ten seconds if still over the emergency threshold;
all reductions stop at the configured minimum cap. Thermal reductions apply
regardless of utilization. Increases remain 1 W per five minutes, after warmup,
with cool averages/peaks and every card above 95% GFX utilization.

The boot cap remains 90 W and the configured range remains 60–145 W in
`/etc/default/rocm-downclock`. Restarting the controller preserves the live
cap. Its failsafe lowers caps to the minimum without stopping workloads. Bolt
has no required dependency on the controller, so controller failure or restart
does not stop the server.

The canonical source is `scripts/mi100_thermal_controller.py`, deployed to
`/usr/local/sbin/mi100-thermal-controller.py` for the existing
`mi100-thermal-controller.service`. After updating it:

```bash
sudo install -m 0755 scripts/mi100_thermal_controller.py \
  /usr/local/sbin/mi100-thermal-controller.py
sudo systemctl restart mi100-thermal-controller.service
```

Policy and fault regressions run with
`.venv/bin/python -m pytest tests/test_mi100_thermal_controller.py tests/test_bolt_launcher.py`.
The launcher checks supervisor identity with Bash built-ins, so its systemd
stop/start path does not require ripgrep in the service PATH.

## Routine validation (2026-10-17)

The final combination passes INT8 KV micro, the gfx908 battery **4/4**, and
**339 focused checks** covering packed loading, static and dynamic
caches, norm/quantization parity, page attention, LiLiCorr, FP32 Mamba state
and scheduling, plus ten checker regressions. The standard launcher boots all
four ranks from eight AIS cache hits with development endpoints disabled. Live arithmetic, exact-format
output, reasoning with a final answer, tool calls, image/video and streaming
pass. The C8 answer smoke has acceptance length **4.55**. A 28,067-token
retrieval returns the identical correct phrase cold and on prefix replay.

The short-answer soak passes **500/500 coherence checks** with the original
case indices, seeds and output budgets: 52 completed serially, then the
remaining 448 at C8 without repeating completed cases. The harness explicitly
disables reasoning for its 64–256-token final-answer budgets; reasoning is
tested separately. It checks final content, so reasoning-only responses cannot
mask an empty answer. To reproduce a complete C8 soak against the standard
server, set `API_KEY_FILE` to your existing credential file and run:

```bash
.venv/bin/python scripts/ua_live_soak.py --model nemotron-bolt \
  --api-key-file "$API_KEY_FILE" --concurrency 8 -n 500
```

The standard 52-prompt logprob corpus also passes **52/52** through Bolt's
chat template, with finite top-token logprobs and readable final answers.
The ten strict-format responses match their requested formats. The original
raw-completion capture is retained as a numerical artifact; its continuation
format is unsuitable as a Bolt chat-quality gate. The Bolt adapter is:

```bash
.venv/bin/python scripts/probe_bolt_quality.py \
  --api-key-file "$API_KEY_FILE" --output logs/bolt-quality.json
```

The checker now accepts code-alignment whitespace and Markdown table dividers,
while retaining empty-answer, repeated-visible-character and repeated-prose
checks. The same captured 52 and 500 replies were checked again after this
repair without new generation runs. No matched official-reference KLD
comparison was performed.

These are implementation and coherence gates, not a task-accuracy certificate.
Code spot checks still found invented `Math.clamp` availability claims,
including in greedy answers; the tested Node.js 22 runtime has no such API.
The origin has not been isolated to the model or additional quantization, and
no optimizer regression is established. Official-reference full-model fidelity
and full 256k-context validation remain unverified.

The historical [2026-10-15 performance profile](bolt_profile_20261015.md)
describes the stack before this campaign and the remaining official-reference
fidelity gap.

## Earlier measurements and validation

The user accepted the static dense / 5 GiB expert-cache configuration on
2026-10-16. Five quick probes across two boots at 105 W power caps measured
1,413–1,503 prefill tok/s and 168–422 aggregate decode tok/s. Combined decode
throughput was 303 tok/s (40,000 tokens / 132.04 s), with acceptance lengths
5.10–7.56. The requested historical reference was yesterday's 1,541 / 363;
these variable results do not establish a speed gain. All requests completed,
metrics account for all benchmark traffic, and short chat responses were
coherent. The new optimization campaign uses a separate 4k-input/256-output
C8 workload and five repetitions per experiment. Full fidelity and
256k-context validation remain pending; see the experiment ledger for the
individual outcomes and comparison limits.

The cache-aware prefill experiment (AITER `0d8d783ab`) was withdrawn in
`15a3e4195` after correcting the aggregate decode measurement. Its historical
7.90% end-to-end gain remains a whole-run result; the original per-request
decode metric did not establish aggregate decode non-regression. In five new
trials per implementation, the restored reference measured median aggregate
prefill of 925.79 tok/s and aggregate decode of 96.71 tok/s, versus
1,021.20 / 89.35 for the experiment. Decode counts only tokens arriving after
all eight requests produced their first token, divided by the remaining
generation time. Staggered admission leaves five or six active requests in
that interval, so it measures the decode-only tail rather than steady C8.
The experiment improved pooled whole-run output throughput by 5.70%, but
pooled aggregate decode fell 2.39%; it failed the no-degradation gate. The
five-run samples remain noisy and do not prove a causal decode slowdown.
The requested dense/expert memory configuration remains accepted.

The initial short-prefill cohort policy is accepted from batch 02 experiment 5
(2026-10-17). Exactly five runs of the frozen 4k-input/256-output C8 workload
at 105 W measured median aggregate prefill of **1,418.58 tok/s** and aggregate
decode of **156.63 tok/s**, versus the retained reference's 925.79 / 96.71.
Pooled whole-run throughput improved 21.37%; median and pooled rates improved
for both phases. This reuses the reference's existing five runs; no additional
reference or candidate repetitions were added. Earlier admission keeps all
eight requests active in the candidate's decode-only tail, versus five or six
under the reference. The gain reflects scheduling and batch occupancy rather
than an isolated kernel speedup. Scheduler/alignment/fairness checks pass
(55), INT8 KV micro passes, and the gfx908 battery is 4/4. The workload, source
hashes, raw rates and comparison limits are recorded in the experiment ledger.

Batch 03 experiment 1 additionally shares packed expert reconstruction across
64 prefill rows. Exactly five runs measured median aggregate prefill of
**1,999.54 tok/s** and decode of **161.53 tok/s** at the same 105 W caps,
versus 1,418.58 / 156.63 for the accepted cohort policy. Pooled whole-run
throughput improved 25.71%; median and pooled phase rates passed the
non-regression gate. These five runs become the next experiment's reference.
The native decode kernel is unchanged; the small observed decode difference
does not establish a kernel improvement. AITER `3f8045a90` supplies the matching
route-tile API and is required with this change. Component, real-checkpoint,
held-out shape and graph checks preserve exact INT8 values and FP32 scales.

To reproduce the aggregate calculation from streamed trial JSON containing
shared-clock token events, run
`PYTHONPATH="$PWD" .venv/bin/python scripts/report_bolt_phases.py`
with the raw trial paths. It reports median and pooled phase rates separately
from whole-run output throughput. Input throughput includes admission delays
and interleaved decode before the latest first token; the decode rate covers
only the remaining tail. Per-request clocks are rejected.

## Checkpoints and execution

| Component | Pinned official revision |
| --- | --- |
| [Target](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) | `a326822af111290f2af5ac59ef4b5040347dabc4` |
| [LiLiCorr draft](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr) | `4c6a5613b6d887194a3a5493e3a18a5b3930bb86` |

The target combines 40 Mamba2, 40 latent-MoE and eight attention blocks.
Its experts vary in count and width by layer. The draft has five attention
layers and a separate candidate-correlation head; it is specific to this
checkpoint. The Qwen DFlash2 draft cannot be reused.

| Surface | Execution |
| --- | --- |
| IQ2_XXS experts and linears | Static dense INT8 GS128; experts use cached INT8 or tile conversion from retained packed originals |
| Q8_0 linears and LM head | Static INT8 GS128 with FP32 group scales |
| Embedding lookup | Checkpoint BF16 embedding, sharded by vocabulary |
| NVFP4 linears | Static INT8 GS128; original nibbles, FP8 block scales and fused-projection global scales are released after conversion |
| MoE/shared-expert ReLU2 intermediate | ReLU2 and GS128 INT8 output, consumed directly by the down projection |
| Other packed GEMM inputs | GS128 INT8 activation quantization |
| Residual streams and excluded draft weights | BF16; storage dtype never determines activation dtype |
| Target and draft KV | `int8_block_g128` |
| Mamba recurrent state | FP32 |
| MoE routers | FP32 weights and computation |
| Tensor parallel communication | vLLM custom all-reduce; Ring/Simple for other collectives |

AITER's sibling checkout owns the packed kernels. INT32 group partials are
scaled and accumulated in FP32. The dense cache holds at most 3 GiB of INT8
weights per GPU (`VLLM_BOLT_DENSE_CACHE_GIB`, 0 disables it), using AITER
blockscale GEMMs with FP32 scaling and unfused accumulation.
`VLLM_BOLT_DENSE_CACHE_DROP_PACKED=1` materializes these weights during loading
and removes original packed tensors and codec metadata before graph capture.
Both the layer and the cache relinquish source references; their FP32 GS128
scales remain. Budget overflow leaves that projection's original packed path
intact. Setting the flag to 0 retains the original packed dense tensors.

A separate 5 GiB per-GPU expert-cache budget holds both INT8 projections of
frequently selected experts. `VLLM_BOLT_EXPERT_CACHE_SLOTS=auto` registers all
40 layer geometries before warmup, chooses equal per-layer capacity, then
fills leftover bytes with the smallest complete expert pairs. This checkpoint
fits 4,565 expert pairs: 114 slots per layer, with five additional slots in the
smallest layer. The total payload and ranking metadata fit within 5 GiB. It
learns from decode/verification routes and refreshes every 128 calls, keeping
surviving slots and expanding only newly admitted experts. Misses use the
ordinary IQ2 kernel. `VLLM_BOLT_EXPERT_CACHE_GIB=0` disables this cache;
an explicit numeric `VLLM_BOLT_EXPERT_CACHE_SLOTS` retains fixed capacity.
`VLLM_BOLT_EXPERT_CACHE_REFRESH` controls refresh cadence. Decode/verification
calls with at most 64 input tokens use the cache and own admission and refresh.
Larger prefills use the Triton packed path with 64-row tiles when total routes
average at least 64 per actual expert; sparse batches retain 16-row tiles.
Both expert projections share the same matching route alignment plan.
All original packed expert weights remain
resident, including boundary blocks for
unaligned TP slices. The static dense payload is 2,387,607,552 INT8 values
(2.224 GiB) in 277 fused projections per GPU. FP32 group scales are retained
for all experts, including those without expanded INT8 copies. Some floating
intermediates remain between operations; the recipe does not claim an
entirely INT8 residual stream.

AIS reads weights directly into GPU memory. On a repack-cache miss, the
launcher enables `VLLM_AIS_REPACK_CPU_WRITE=1`: packed cache payloads are copied
back through a temporary host buffer for buffered file writes. This avoids
filesystem-journal stalls from thousands of small hipFile expert writes.
It does not expand or requantize the weights, and cache-hit reads still use
AIS directly. The Qwen launcher's cache-writing path is unchanged.

## Validation and scheduler tradeoff (2026-10-15)

The packed component oracle, boundary slices, routed experts and graph replay
pass. The loader suite passes five checks, Mamba recurrence/chunked prefill
eight, and the post-context-fix LiLiCorr suite 32. INT8 KV micro passes and the
gfx908 battery is 4/4. The CPU cache writer passes short-write, alignment,
publication and source-invalidation checks. All four target rank caches
(28,885 tensors each) and four draft rank caches (179 each) match source bytes.
Loaded target plus draft use 7.14 GiB of model memory per GPU.

In the final short C8 smoke, the cold/warm runs produced 644/657 output tokens
in 16.45/14.78 seconds, with acceptance lengths 4.59/4.67. These variable-length
short runs do not establish a throughput improvement over the earlier graph
baseline (638/649 tokens in 13.13/13.53 seconds).

The matched-input long C8 workload uses a shared 28k-token synthetic archive
and eight questions, after the same retrieval/replay warmup:

| Metric | No mixed-load policy | Adapted policy |
| --- | ---: | ---: |
| First completed response | 221.33 s | 46.33 s |
| Median completed response | 225.25 s | 158.92 s |
| Whole-run elapsed time | 228.63 s | 251.59 s |
| Output tokens | 569 | 538 |
| Acceptance length | 4.445 | 4.402 |

The recipe adopts the policy to protect decoder progress. It staggers new
prefills and increases total elapsed time in this workload. This is one paired
run with freely generated responses of different lengths, not an equal-work
throughput or statistical-significance claim.
For the no-policy comparison, restart the same launcher with
`VLLM_GFX908_MIXED_PREFILL_TOKENS=0 VLLM_GFX908_PREFIX_FAIR_ADMISSION=0`;
restore the defaults for the adapted policy. Use a fresh server for each leg
and the same prefix-probe/benchmark sequence below.

The 28,067-token retrieval returns the same phrase cold and on replay:
100.68/27.02 seconds. After a GPU-only prefix reset, the server restores
339,735,040 bytes from RAM, reuses 17,024 tokens, and returns the same phrase in
46.57 seconds. No disk prefix tier is configured. Reasoning, automatic tool
calls, image/video input and non-thinking streaming also passed functional
checks. These checks do not replace a broad model-quality evaluation.

## Changes needed in the fork

Relevant support from upstream `77154ca6116ae34cbfe314beef1016bde9b63989`
and LiLiCorr PR #57934 was selectively integrated with the existing DFlash
implementation. The CUDA-specific upstream packed backend was replaced by
the gfx908 AITER kernels described above.

The integration adds ModelOpt mixed-format loading, packed embedding/linear
methods, and grouped routed-expert execution. Expert loading distinguishes
individual three-dimensional packed tensors from fused expert arrays. The
Mamba SSD kernel loads initial and previous-chunk state separately to avoid
a Triton 3.6 AMD pointer-canonicalization failure.

LiLiCorr adds normalized vocabulary-parallel candidate scores, target-context
anchors and the trained correlation head. The draft's context projection
uses its configured compute dtype: interpreting the NVFP4 weight storage dtype
as the activation dtype silently converted target features to unsigned bytes
and collapsed acceptance. The regression suite covers this boundary.

## Launch and measure

Use the repository serving venv, patched ROCm runtime and sibling AITER
checkout described in [the stack baseline](README.md). Model directories
default to `$HOME/models/<official repository name>` and can be overridden
with `MODEL_DIR` and `DRAFT_MODEL_DIR`.

```bash
scripts/serve_recipe_bolt.sh start
scripts/serve_recipe_bolt.sh status
scripts/serve_recipe_bolt.sh stop
```

The launcher reads `VLLM_API_KEY`, `LLAMA_API_KEY`, or `API_KEY_FILE` (default
`/etc/llama/llama-api.key`). It serves `nemotron-bolt` on localhost port 8020.
It fixes TP4/C8 and seven speculative tokens, matching the draft's trained
eight-token block including the anchor. Defaults are 262,144 context tokens,
8,512 batched tokens, 7,000,000,000 bytes of GPU KV per GPU (`KVMEM`), separate
3 GiB dense and 11 GiB routed-expert cache ceilings, and a 24 GiB CPU prefix tier. Explicit GPU
KV bytes override the GPU memory-utilization setting, as in the Qwen recipe.
There is no disk prefix tier. `EAGER=1` disables graphs for diagnostics.
The 256k ceiling and 24 GiB CPU tier pass the short C8 serving gate; validation
at the full context length is pending. The 7 GB GPU KV arena reports
1,641,644 effective shared tokens (6.26 full-length requests). CPU offload
pins each rank's own slots with `VLLM_ROCM_PIN_LOCAL_OFFLOAD=1`, avoiding
four registrations of the entire 24 GiB region and AMD's host-memory limit.

The Qwen mixed-load scheduler policy is adapted to Bolt's 4,256-token hybrid
page: normally at most one page of prefill is admitted every 16 rounds while
decoders are active, with fair prefix admission. Bolt additionally fills an
initial cohort of single-page text prompts on intervening rounds while every
running request and the waiting head remain eligible. Accepted output plus
pending output must be at most 64 tokens under C8/NS7. The shared 4,256-token
prefill budget and 8,512-token total budget remain in force. Long prompts,
mature decoders, multimodal/LoRA requests, preempted requests and blocked or
stale requests retain the ordinary admission cadence. Set
`VLLM_BOLT_SHORT_PREFILL_COHORT=0` to disable this addition. Filling the cohort
improves aggregate throughput but can delay early responses while their peers
prefill. Qwen's specialized HD256/GQA6 attention kernels do not apply
to Bolt's HD128/GQA8 layout. Bolt uses AITER unified attention with its
dedicated page-outer prefill kernel for eligible shapes, generic fallback,
INT8-G128 KV and split-K speculative verification. Its larger Mamba state also
requires a different page alignment. The Qwen-specific draft convolution
calibration is disabled for the official LiLiCorr draft.

The official `ultra_v3` reasoning plugin and `qwen3_coder` tool parser are
configured. Requests can disable reasoning with
`chat_template_kwargs={"enable_thinking": false}`. One image or video per
request is enabled; video sampling is capped at 256 frames.

Against an otherwise idle server, retain two runs to separate cold execution
from warm execution:

```bash
.venv/bin/python scripts/bench_bolt_recipe.py \
  --api-key-file "$API_KEY_FILE" --output logs/bolt-c8-cold.json
.venv/bin/python scripts/bench_bolt_recipe.py \
  --api-key-file "$API_KEY_FILE" --output logs/bolt-c8-warm.json
```

This bounded C8 smoke records prompts, responses, tokens, elapsed time and
speculative counter deltas. It is not a general quality benchmark.
`--prefix-file` prepends an identical document to every question for a retained
long-context workload; `--max-tokens` controls each response budget.
The A/B record belongs in [the experiment ledger](surface_experiments_ledger.jsonl).

To reproduce the 28k-token retrieval and repeated-prefix workload:

```bash
.venv/bin/python scripts/probe_bolt_prefix.py \
  --api-key-file "$API_KEY_FILE" --tokenizer "$MODEL_DIR/tokenizer.json" \
  --output-dir logs/bolt-prefix
.venv/bin/python scripts/bench_bolt_recipe.py \
  --api-key-file "$API_KEY_FILE" --prefix-file logs/bolt-prefix/bench-prefix.txt \
  --max-tokens 128 --output logs/bolt-long-c8.json
```

On a diagnostic server started with `VLLM_SERVER_DEV_MODE=1`, adding
`--reset-gpu` to the prefix probe resets the GPU prefix cache while preserving
the CPU tier, then repeats retrieval and records transfer metrics. Keep this
development mode off in production.

For systemd, the separate `scripts/vllm-bolt.service` and
`scripts/vllm-qwen38.service` units use the existing credential envfile
and run as `curved`. See [installation and recipe
selection](services.md). Install/activate only after the intended configuration
passes the serving gates.
