# Nemotron Bolt with official packed weights on MI100

This peer recipe runs the official NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB
checkpoint and its LiLiCorr draft on four MI100 GPUs. It keeps the checkpoint's
packed weights in GPU memory. Dense projections reuse a bounded INT8 GS128
weight cache; uncached projections convert tiles inside the matrix kernels.
It does not produce a replacement quantized checkpoint.

**Serving validated; production activation pending:** eager and graph TP4/C8
serving, LiLiCorr acceptance, 28k context/prefix replay, RAM-prefix restoration,
reasoning, tools, image/video and streaming pass. All eight AIS rank caches
match the official checkpoint bytes. The final post-loader Qwen regression
passed. A subsequent eight-cache-hit restart also passed C8, reasoning, tools,
image, streaming, API authentication and disabled-development-endpoint checks.
Production service activation remains pending.
The Qwen recipe remains independently available.

The [2026-10-15 performance profile](bolt_profile_20261015.md) identifies the
packed matrix kernels as the dominant bottleneck and documents the remaining
fidelity-validation gap.

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
| IQ2_XXS experts and linears | Original packed bytes; tile decode, GS128 INT8 weight quantization and INT8 MFMA |
| Q8_0 linears and LM head | Original packed bytes; same GS128 INT8 compute path |
| Q8_0 embedding | Decode only the requested rows |
| NVFP4 linears | Original nibbles, FP8 block scales and individual fused-projection global scales; tile conversion to GS128 INT8 |
| MoE/shared-expert ReLU2 intermediate | ReLU2 and GS128 INT8 output, consumed directly by the down projection |
| Other packed GEMM inputs | GS128 INT8 activation quantization |
| Residual streams and excluded draft weights | BF16; storage dtype never determines activation dtype |
| Target and draft KV | `int8_block_g128` |
| Mamba recurrent state | FP32 |
| Tensor parallel communication | vLLM custom all-reduce; Ring/Simple for other collectives |

AITER's sibling checkout owns the packed kernels. INT32 group partials are
scaled and accumulated in FP32. The dense cache adds at most 2 GiB of INT8
weights per GPU (`VLLM_BOLT_DENSE_CACHE_GIB`, 0 disables it), using AITER
blockscale GEMMs with FP32 scaling and unfused accumulation. Routed experts
remain packed. The launcher defaults to 70% GPU memory utilization; the CPU
KV tier is 24 GiB (effective on the next restart). Unaligned tensor-parallel
slices retain the necessary
original packed boundary blocks. Some floating intermediates remain between
operations; the recipe does not claim an entirely INT8 residual stream.

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
eight-token block including the anchor. Defaults are 32,768 context tokens,
8,512 batched tokens, 90% GPU memory utilization, and an 8 GiB CPU prefix tier.
There is no disk prefix tier. `EAGER=1` disables graphs for diagnostics.
The configured 32k limit is intentional for this validated recipe; raising
`MAX_MODEL_LEN` does not establish validation at the checkpoint's full 256k limit.

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

For systemd, the separate `scripts/vllm-bolt@.service` and
`scripts/vllm-qwen38@.service` templates use the existing credential envfile
and run as the named instance user. See [installation and recipe
selection](services.md). Install/activate only after the intended configuration
passes the serving gates.
