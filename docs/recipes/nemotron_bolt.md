# Nemotron Bolt with official packed weights on MI100

This peer recipe runs the official NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB
checkpoint and its LiLiCorr draft on four MI100 GPUs. It keeps the checkpoint's
packed weights in GPU memory and converts weight tiles to INT8 GS128 inside
the matrix kernels. It does not produce a replacement quantized checkpoint.

**Validation in progress:** eager and graph TP4/C8 serving, LiLiCorr acceptance,
28k context/prefix replay, reasoning, tools, image/video and streaming pass.
Final loader/scheduler checks and production service activation remain pending.
The Qwen recipe remains independently available.

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
| MoE/shared-expert ReLU2 intermediate | Fused activation and GS128 INT8 output, consumed directly by the down projection |
| Other packed GEMM inputs | GS128 INT8 activation quantization |
| Residual streams and excluded draft weights | BF16; storage dtype never determines activation dtype |
| Target and draft KV | `int8_block_g128` |
| Mamba recurrent state | FP32 |
| Tensor parallel communication | vLLM custom all-reduce; Ring/Simple for other collectives |

AITER's sibling checkout owns the packed kernels. INT32 group partials are
scaled and accumulated in FP32. Weight tiles are never persisted as expanded
INT8/BF16 matrices. Unaligned tensor-parallel slices retain the necessary
original packed boundary blocks. Some floating intermediates remain between
operations; the recipe does not claim an entirely INT8 residual stream.

AIS reads weights directly into GPU memory. On a repack-cache miss, the
launcher enables `VLLM_AIS_REPACK_CPU_WRITE=1`: packed cache payloads are copied
back through a temporary host buffer for buffered file writes. This avoids
filesystem-journal stalls from thousands of small hipFile expert writes.
It does not expand or requantize the weights, and cache-hit reads still use
AIS directly. The Qwen launcher's cache-writing path is unchanged.

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

For systemd, `scripts/vllm-bolt@.service` uses the existing credential envfile
and runs as the named instance user. It conflicts with the Qwen service to
prevent simultaneous GPU ownership. Install/activate only after the intended
configuration passes the serving gates.
