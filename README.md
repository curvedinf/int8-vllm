# int8-vllm

*INT8 inference for AMD Instinct MI100.*

A hardware-focused [vLLM](https://github.com/vllm-project/vllm) fork for
**4 × MI100 32 GB GPUs connected over XGMI**. It pairs INT8 inference kernels
with speculative decoding to serve Qwen3.8-27B through a vLLM API.

[Serving recipe](docs/recipes/README.md) ·
[Models](#models) ·
[Getting started](#getting-started) ·
[Benchmarks](#benchmarks)

## What it does

- **W8A8 inference:** AITER INT8 matrix multiplication for both prompt
  processing and token generation, including the language-model head.
- **Compact KV cache:** grouped INT8 storage for both the target and draft,
  with prefix reuse across GPU memory, host RAM, and disk.
- **Speculative decoding:** a matching INT8 DFlash2 draft proposes tokens
  for the target to verify.
- **MI100-specific tuning:** attention kernels, XGMI collectives, and
  decode-priority scheduling for mixed prefill/decode workloads.

INT8 is used where it passes quality checks—not for every intermediate.
Recurrent state stays FP32, and selected sensitive surfaces remain BF16.

This is a specialized deployment fork, not a drop-in package for arbitrary
GPUs. The validated configuration is **TP4, up to six concurrent requests,
and six speculative tokens** with the model pair below. Other hardware and
models require their own validation.

## Benchmarks

Measured on this project's four-MI100 server. Workload definitions matter:
short-prompt throughput is not a prediction of long-context mixed-load speed.

| Date | Workload | Aggregate output rate | What was measured |
|---|---|---:|---|
| 2026-10-10 | Short prompts, six concurrent streams | 218.1 tok/s | 32 → 1,000 tokens; 32 greedy requests; whole-run rate |
| 2026-10-10 | Long contexts, six concurrent streams | 162.6 tok/s | 32k → 2,048 tokens; sampled; common decode window |
| 2026-10-10 | Mixed load, before this update | 12.6 tok/s | Three decoding streams while three fresh 32k-token prompts are prefilling |
| 2026-10-10 | Mixed load, decode-priority update | **75.4 tok/s** | Matched prompts, models, and hardware; **6.0×** decode rate during the common overlap window |

The mixed-load test uses greedy decoding over documentation-based 32k-token
contexts, 2,048 output tokens per decoder, and 16 output tokens per injected
prefill request. The updated streams delivered **24.3–25.9 tok/s each**
during overlap.

**Tradeoff:** this policy prioritizes active decoders. The three injected
prefills reached their first token in **43 / 82 / 100 seconds**, versus
19 / 35 / 52 seconds before. It is not a claim of higher total-work throughput.
A repeated 32k prefix reached its first token in **1.51 seconds** versus
2.64 seconds before, reusing 29,952 cached tokens.

See the [recipe](docs/recipes/README.md) for validation and tuning details,
and the [A/B ledger](docs/recipes/surface_experiments_ledger.jsonl) for
measurements, controls, and reproduction commands.

## Models

Download both published checkpoints; the draft is not a standalone model.

| Role | Checkpoint | Launcher default location |
|---|---|---|
| Target | [Qwen3.8-27B INT8 PTQR](https://huggingface.co/curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128-PTQR) | `~/models/Qwen3.8-27B-PTQR-R10S60` |
| Draft | [DFlash2 INT8 PTQR](https://huggingface.co/curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128-PTQR) | `~/models/dflash2-ptqr-r1` |

PTQR means the quantized weights and scales were retrained against a
higher-precision reference. Set `MODEL_DIR` and `DRAFT_MODEL_DIR` if your
checkpoints live elsewhere.

## Getting started

### 1. Check the requirements

| Component | Tested deployment |
|---|---|
| GPUs | 4 × AMD Instinct MI100, 32 GB each, XGMI full mesh |
| System | Linux with ROCm at `/opt/rocm` |
| Serving environment | Python 3.12 in repository-local `.venv`; PyTorch `2.11.0+rocm7.1`, Triton `3.6.0` |
| HIP runtime | Patched `7.2.53211` HIP/HSA libraries in the serving environment |
| Kernel library | Sibling [int8-aiter](https://github.com/curvedinf/int8-aiter) checkout, loaded through `PYTHONPATH` |
| Storage | Space for both checkpoints, build caches, and the prefix-cache disk tier |

> [!IMPORTANT]
> Installation currently requires a manually provisioned ROCm build environment;
> there is no one-command installer for this fork. The pinned PyTorch wheel's
> original ROCm 7.1 runtime leaked host memory during graph replay. Preserve
> the tested runtime override described in the [recipe](docs/recipes/README.md),
> or validate a replacement before production use. Unconstrained dependency
> installation can replace the ROCm packages with incompatible builds.

### 2. Check out both repositories

Run these commands as the account that will own the service. The launcher
expects this layout under that account's home directory.

```bash
cd "$HOME"
git clone --branch main https://github.com/curvedinf/int8-vllm.git
git clone --branch main https://github.com/curvedinf/int8-aiter.git aiter
cd int8-vllm
```

Keep AITER as a sibling checkout; **do not pip-install it into the serving
environment**. Its kernels rebuild lazily from that checkout.

### 3. Build and validate

After provisioning `.venv` with the pinned ROCm dependencies and runtime
above, build the local extensions:

```bash
export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm
export PYTORCH_ROCM_ARCH=gfx908 GPU_ARCHS=gfx908 VLLM_TARGET_DEVICE=rocm
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export PATH="$PWD/.venv/bin:/opt/rocm/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/../aiter"
MAX_JOBS=32 .venv/bin/python setup.py build_ext --inplace

# Run with an idle GPU before starting the server.
.venv/bin/python scripts/test_int8_kv_micro.py
HIP_VISIBLE_DEVICES=0 .venv/bin/python scripts/battery_gfx908.py
```

Expect all four battery checks to pass. Allow time for AITER's first-use
kernel compilation. See [incremental builds](docs/contributing/incremental_build.md)
for development workflow details.

### 4. Start the server

Place the model checkpoints in the locations above and provide a private API
key through `VLLM_API_KEY` or the launcher's configured key file.

```bash
scripts/serve_recipe_qwen38.sh start
scripts/serve_recipe_qwen38.sh status
# Stop when needed:
scripts/serve_recipe_qwen38.sh stop
```

The API listens on **127.0.0.1:8020** with model name
`qwen3.8-27b-gptq8`. Logs are under `logs/serve_recipe_qwen38/`.
Review the launcher's CPU affinity (cores 0–47) and host/disk cache budgets
for your server before starting it.
For persistent deployment, adapt [the systemd unit](scripts/vllm.service)
to your account and paths before installing it; the checked-in unit uses
this server's `/home/curved` layout.

## Further reading

- [Serving recipe](docs/recipes/README.md): current configuration, runtime
  requirements, cache tiers, operational controls, and rollback options.
- [INT8 accuracy audit](INT8_AUDIT_RESULTS.md): precision choices and quality gates.
- [Experiment ledger](docs/recipes/surface_experiments_ledger.jsonl):
  reproducible performance experiments and adoption decisions.
- [License](LICENSE): Apache 2.0.
