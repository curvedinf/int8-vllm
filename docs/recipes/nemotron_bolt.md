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

The [2026-10-15 performance profile](bolt_profile_20261015.md) identifies the
packed matrix kernels as the dominant bottleneck and documents the remaining
fidelity-validation gap.

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

The first retained optimization on the new workload reuses resident expert
INT8 copies during prefill (AITER `0d8d783ab`). Five 4,096-input/256-output C8
trials measured 51.29 aggregate output tok/s versus 47.54 in five fresh
incumbent controls. Pooled prefill improved 9.15% and per-request decode
6.17%; end-to-end throughput improved 7.90%. Inputs, warmup, cache limits and
105 W caps were held fixed. These are bounded workload results, with full
fidelity and full-context validation still pending.

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
calls with at most 64 input tokens own admission and refresh. Larger prefills
reuse the existing copies without changing ranking or allocation; misses use
native packed conversion. Before caches exist, loading/profiling prefills use
the original Triton packed path. All original packed expert weights remain
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
8,512 batched tokens, 14,000,000,000 bytes of GPU KV per GPU (`KVMEM`), separate
3 GiB dense and 5 GiB routed-expert cache ceilings, and a 24 GiB CPU prefix tier. Explicit GPU
KV bytes override the GPU memory-utilization setting, as in the Qwen recipe.
There is no disk prefix tier. `EAGER=1` disables graphs for diagnostics.
The 256k ceiling and 24 GiB CPU tier pass the short C8 serving gate; validation
at the full context length is pending. The 14 GB GPU KV arena reports
3,283,288 effective shared tokens (12.52 full-length requests). CPU offload
pins each rank's own slots with `VLLM_ROCM_PIN_LOCAL_OFFLOAD=1`, avoiding
four registrations of the entire 24 GiB region and AMD's host-memory limit.

The Qwen mixed-load scheduler policy is adapted to Bolt's 4,256-token hybrid
page: at most one page of prefill is admitted every 16 rounds while decoders
are active, with fair prefix admission. This trades new-request latency for
decoder progress. Qwen's specialized HD256/GQA6 attention kernels do not apply
to Bolt's HD128/GQA8 layout; Bolt uses the generic unified attention path with
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
