<p align="center">
  <img src="int8-vllm-banner.png" alt="int8-vllm banner" width="880">
</p>

# int8-vllm

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Hardware](https://img.shields.io/badge/hardware-4%C3%97%20AMD%20MI100%20(gfx908)-red)

**vLLM with opt-in INT8 acceleration for AMD Instinct MI100.**

This fork retains all mainline [vLLM](https://github.com/vllm-project/vllm)
features through backports and can run **all models supported by stock vLLM**,
subject to the same hardware and backend requirements. The normal vLLM APIs,
serving commands and model workflows remain available.

**All fork-specific INT8 features are opt-in.** Enable them through the
relevant configuration, or use a recipe launcher that selects them for you.
The recipes below are heavily optimized for four 32 GB MI100 GPUs connected
over XGMI, combining INT8 matrix kernels, INT8 KV caching and speculative
decoding to reduce memory traffic and accelerate serving.

## Supported models and optimized recipes

Stock vLLM model support extends beyond this table. These are the model pairs
with dedicated tuning and validation; each recipe uses a target model and its
matching speculative draft.

| Model | Checkpoints | Recipe |
| --- | --- | --- |
| Qwen3.8-27B | [INT8 PTQR target](https://huggingface.co/curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128-PTQR) + [DFlash2 draft](https://huggingface.co/curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128-PTQR) | [Qwen setup](docs/recipes/README.md) |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B-24GB | [Official packed target](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) + [LiLiCorr draft](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr) | [Bolt setup](docs/recipes/nemotron_bolt.md) |

Qwen uses INT8 weights retrained against a BF16 reference. Bolt loads the
official packed checkpoints, converts dense weights to INT8 and caches INT8
expert weights. Both recipes use grouped INT8 KV for the target and draft,
AITER attention kernels and custom XGMI all-reduce.

These recipes are tested on **4 × MI100 (gfx908), with XGMI full mesh**.
Model-specific precision choices, validation results and limitations are
documented in the linked recipes.

## Getting started

Use the setup below to run one of the optimized recipes on four MI100 GPUs.

### 1. Prepare the environment

Use Linux with ROCm at `/opt/rocm`, a repository-local `.venv` and a sibling
[int8-aiter](https://github.com/curvedinf/int8-aiter) checkout:

```bash
cd "$HOME"
git clone --branch main https://github.com/curvedinf/int8-vllm.git
git clone --branch main https://github.com/curvedinf/int8-aiter.git aiter
cd int8-vllm
```

The tested serving environment uses **Python 3.12, PyTorch 2.11.0+rocm7.1
and Triton 3.6.0**, with patched **HIP/HSA 7.2.53211** runtime libraries.
Follow the [runtime setup](docs/recipes/README.md#the-stack) before building:
the original ROCm 7.1 runtime leaks host memory during graph replay.
Keep AITER on `PYTHONPATH` from the sibling checkout; do not pip-install it
into the serving environment.

### 2. Build and validate

After provisioning the serving environment:

```bash
export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm
export PYTORCH_ROCM_ARCH=gfx908 GPU_ARCHS=gfx908 VLLM_TARGET_DEVICE=rocm
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export PATH="$PWD/.venv/bin:/opt/rocm/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/../aiter"
MAX_JOBS=32 .venv/bin/python setup.py build_ext --inplace

# Required by Bolt's default INT8 decoder path.
scripts/build_residual_g128.sh

# Run on idle GPUs before serving.
.venv/bin/python scripts/test_int8_kv_micro.py
HIP_VISIBLE_DEVICES=0 .venv/bin/python scripts/battery_gfx908.py
```

The battery should report **4/4 PASS**. Allow time for AITER's first-use
kernel compilation.

### 3. Download models and start a recipe

Download the target and matching draft linked above. Use the default model
directories in the chosen recipe, or set `MODEL_DIR` and `DRAFT_MODEL_DIR`.
Provide your API key through `VLLM_API_KEY` before starting.

Choose one recipe:

```bash
# Qwen3.8-27B + DFlash2
scripts/serve_recipe_qwen38.sh start
```

```bash
# Nemotron 3.5 Bolt + LiLiCorr
scripts/serve_recipe_bolt.sh start
```

Both launchers default to **127.0.0.1:8020**, so run one at a time.
Use `status`, `stop` or `restart` in place of `start` to manage the server.
The API model names are `qwen3.8-27b-gptq8` and `nemotron-bolt`;
logs are in `logs/serve_recipe_qwen38/` and `logs/serve_recipe_bolt/`.
For persistent deployment, see [systemd setup](docs/recipes/services.md).

## Quality control

The optimized recipes pair faster inference with explicit checks on model
fidelity, response quality and serving reliability. Validation follows the
whole path from checkpoint loading to complete answers, including long
conversations and simultaneous requests.

| Check | What is tested | What it protects |
| --- | --- | --- |
| **Model fidelity** | Run the same **52 prompts** against the INT8 and BF16 recipes. Compare mean, median and p95 KLD from top-token probabilities, agreement on the first 40 output characters, and stop-token probability. | Measures how closely quantized predictions and responses follow the reference model. |
| **Checkpoint integrity** | Byte-compare loaded and repacked weights with the source checkpoint; verify that exported quantized weights reproduce the trained quantization grid. | Loads the intended model correctly across every GPU. |
| **INT8 arithmetic and fusion** | Packed-GEMM checks require relative L2 error below **0.002%**, quantized-epilogue error below **2%**, exact INT8 codes, scale relative tolerance **0.0002%**, and identical eager/graph outputs. Fused operations are compared with their separate reference operations. | Keeps accelerated matrix operations, quantization and fusion numerically faithful to their references. |
| **Attention and KV cache** | Compare prefill, decode and mixed batches with reference attention. Basic INT8 KV tests require maximum absolute error below **0.1**, or **0.05** for noncausal attention. Additional checks cover grouped heads, sliding windows, variable lengths and page boundaries; the kernel/import battery requires **4/4 passes**. | Checks that compressed context is read correctly across the attention paths used during serving. |
| **Recurrent state and multi-GPU execution** | Compare state updates and speculative rollback with sequential references across accepted-token counts, chunk seams, migrations and page boundaries. Exercise collectives with changing payloads and scales, interleaved operations and repeated graph replay. | Preserves conversation state and coordination between GPUs as requests advance. |
| **Sampling and speculative decoding** | Check top-k/top-p ordering and seeded/unseeded sampling. Compare rejection-sampling frequencies with known probabilities over **20,000 trials**. Compare INT8 and BF16 draft predictions on identical target states; measure accepted tokens per round, acceptance fraction, rejection tails and decay over long contexts. | Checks token selection and the draft model's ability to accelerate generation faithfully. |
| **Short answers and formatting** | A **500-request soak requires zero detected empty, repetitive or character-collapse responses**. The **52-prompt chat check requires nonempty answers and finite token log-probabilities**; strict-format prompts check the requested output format. | Produces usable final answers and checks the formats consumed by applications. |
| **Long-context coherence** | Review complete responses from **eight 22k-token inputs with up to 4,096 output tokens each**, using two serial and six concurrent requests. Full-output gates reject structural garble, repetition walls, unintended restarts and progressive breakdown. | Tests whether long answers remain readable through their ending, including under concurrency. |
| **Token-by-token consistency under load** | Replay exact generated token IDs through a clean reference and measure mean absolute log-probability drift and positions exceeding **1 nat**. The mixed-load gate runs **three 32k-context, 2,048-token decoders alongside three fresh 32k prefills**, requiring **zero generated tokens outside the reference's top 20**. | Detects changes in model behavior when generation crosses chunk boundaries or shares the server with new prompts. |
| **Prefix reuse and offload restoration** | Compare retrieval answers and generated token IDs across cold/warm requests, GPU cache reset, RAM restoration, service restart and appended prompts; verify reused-token counters. | Checks that cached conversation history survives reuse and movement between memory tiers. |
| **Application compatibility** | Exercise arithmetic, reasoning with a final answer, automatic tool calls, image/video input, streaming, authentication and disabled development endpoints. Recheck the other optimized recipe after shared changes. | Keeps application features working alongside the accelerated serving paths. |
| **Memory and service stability** | Run sustained concurrent requests while checking all GPU workers, finite activations, memory growth, allocation failures, cache residency, startup/shutdown and unexpected restarts. | Tests the service's ability to keep answering requests throughout continued use. |

Performance validation measures prefill throughput, decode throughput,
time to first token and speculative acceptance on matched workloads.
See the [recipe guides](docs/recipes/README.md) for configuration and validation details.

## Documentation

- [Qwen recipe](docs/recipes/README.md) and [Bolt recipe](docs/recipes/nemotron_bolt.md) — configuration, benchmarks and validation.
- [INT8 coverage audit](INT8_AUDIT_RESULTS.md) — precision choices and floating-point exceptions.
- [Experiment ledger](docs/recipes/surface_experiments_ledger.jsonl) — measured changes and adoption decisions.
- [Incremental builds](docs/contributing/incremental_build.md) — development workflow.
- [License](LICENSE) — Apache 2.0.
