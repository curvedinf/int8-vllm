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

| Model | Checkpoints | Default concurrency / shared GPU context | Recipe |
| --- | --- | --- | --- |
| Qwen3.8-27B | [INT8 PTQR target](https://huggingface.co/curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128-PTQR) + [DFlash2 draft](https://huggingface.co/curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128-PTQR) | 6 simultaneous sequences (C6); ~1.61M shared tokens | [Qwen setup](docs/recipes/README.md) |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B-24GB | [Official packed target](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) + [LiLiCorr draft](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr) | 8 simultaneous sequences (C8); ~1.99M shared tokens | [Bolt setup](docs/recipes/nemotron_bolt.md) |

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

The optimized recipes are built for fast, dependable inference. **Rigorous
quality control covers the entire serving path**, from quantized weights and
GPU kernels to complete responses and sustained concurrent workloads.
Optimizations are evaluated for model fidelity, response quality and service
stability alongside their speed and memory benefits.

- **Model fidelity:** compare 52 prompts with BF16 using KLD, output agreement and stop probability to measure quantization drift.
- **Weight integrity:** byte-check loaded/repacked weights and verify exports against the trained quantization grid to catch loading errors.
- **INT8 arithmetic:** require relative L2 error below **0.002%** for packed GEMMs and **2%** for quantized epilogues to bound numerical error.
- **Quantization and fusion:** check exact INT8 codes, **0.0002%** scale tolerance and fused-operation reference agreement to preserve calculations.
- **Attention and KV:** test masks, grouped heads, windows and page boundaries; basic reference checks limit maximum absolute error to **0.1**, or **0.05** for noncausal attention.
- **Kernel compatibility:** require **4/4** attention, GEMM and import checks so production kernels and serving interfaces work together.
- **Recurrent state:** compare updates and speculative rollback with sequential references across chunk boundaries, page boundaries and migrations to protect conversation state.
- **Multi-GPU execution:** test interleaved collectives and repeated graph replay against reference outputs to catch synchronization and replay errors.
- **Sampling correctness:** use **20,000 trials** against known probabilities, plus top-k/top-p and seeded/unseeded checks, to detect sampling bias.
- **Draft quality:** compare predictions on identical target states and track acceptance, rejection tails and long-context decay to protect speculative decoding efficiency.
- **Short answers:** require **zero detected empty or collapsed responses across 500 requests**, finite log-probabilities on 52 prompts and compliance with requested formats.
- **Long answers:** review eight **22k-input/4,096-output-token** workloads, serially and concurrently, for structural garble, repetition and progressive breakdown.
- **Token consistency:** replay exact generated tokens and measure log-probability drift, including deviations above **1 nat**, to detect inconsistent model state.
- **Mixed-load quality:** run three **32k-context/2,048-output-token** decoders alongside three fresh 32k prefills; require **zero tokens outside the reference's top 20**.
- **Cache continuity:** compare answers and token IDs after prefix reuse, RAM restoration, resets, restarts and appended prompts to verify preserved context.
- **Application behavior:** exercise arithmetic, reasoning, tools, image/video input, streaming, authentication and cross-recipe regressions to protect application compatibility.
- **Serving stability:** stress concurrent requests while checking finite activations, memory growth, allocation failures, startup/shutdown and unexpected restarts.

See the [recipe guides](docs/recipes/README.md) for configuration and validation details.

## Documentation

- [Qwen recipe](docs/recipes/README.md) and [Bolt recipe](docs/recipes/nemotron_bolt.md) — configuration, benchmarks and validation.
- [INT8 coverage audit](INT8_AUDIT_RESULTS.md) — precision choices and floating-point exceptions.
- [Experiment ledger](docs/recipes/surface_experiments_ledger.jsonl) — measured changes and adoption decisions.
- [Incremental builds](docs/contributing/incremental_build.md) — development workflow.
- [License](LICENSE) — Apache 2.0.
