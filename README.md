# int8-vllm

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Hardware](https://img.shields.io/badge/hardware-4%C3%97%20AMD%20MI100%20(gfx908)-red)

**INT8-native vLLM serving for AMD Instinct MI100 — Qwen3.8-27B at measured quality parity with BF16.**

On MI100, the INT8 matrix rate is 2× every other dtype at half the memory
bandwidth, so INT8 is this GPU's native language. This hardware-focused
[vLLM](https://github.com/vllm-project/vllm) fork rebuilds the serving path
around that fact: INT8 weights, activations, KV cache, and a retrained INT8
speculative draft, all validated by gates against BF16 references. The result
is a 27-billion-parameter hybrid model served interactively on four 32 GB
MI100 GPUs over XGMI.

```bash
scripts/serve_recipe_qwen38.sh start   # after setup, below
```

[Why INT8](#why-int8-on-mi100) ·
[What runs in INT8](#what-runs-in-int8) ·
[Evidence](#evidence) ·
[Models](#models) ·
[Getting started](#getting-started)

## Why INT8 on MI100

MI100 (gfx908 / CDNA1) runs INT8 matrix math at twice the rate of BF16 and
above fp16 bandwidth pressure at half the bytes moved. A BF16 serving stack
therefore leaves most of the chip idle. This fork moves every large surface
to INT8 and keeps the few measured exceptions in higher precision (next
section), so a 27B target plus a speculative draft fit and decode fast on
4 × 32 GB.

This is a deployment fork with one validated configuration: **TP4, up to six
concurrent sequences, six speculative tokens**, on 4 × MI100 with XGMI full
mesh, serving the model pair below. Other GPUs, models, or shapes require
their own validation — treat this as an appliance, not a general package.

## What runs in INT8

| Surface | Precision | Notes |
|---|---|---|
| Model weights | INT8, GPTQ format, group size 128 | Retrained by [PTQR](#models), not one-shot calibration |
| All large GEMMs | AITER W8A8 INT8 | Decode, prefill, language-model head, embedding gather |
| Activations | INT8, fused round-to-nearest quant | Halved the dominant activation-quantization error leg |
| KV cache, target and draft | `int8_block_g128` | Grouped INT8 with fp16 group scales; both models |
| Attention cores | Gluon MFMA kernels, fp16 dots | INT8 K/V and BF16 Q are exact in fp16; fp32 accumulation |

Where measurements said no, higher precision stays: the **residual stream
computes in BF16** (fp16 overflows at long context), the **Mamba/GDN
recurrent state is FP32** (fp16 and INT8 variants both failed quality gates),
and the draft's selector codebooks and context-KV projection remain BF16.
Every precision choice is gated and recorded in the
[accuracy audit](INT8_AUDIT_RESULTS.md).

## Evidence

### Quality gates

| Gate | Result |
|---|---|
| 52-prompt KLD vs BF16 reference (target) | 0.0069 median, vs 0.0110 for the one-shot checkpoint |
| Greedy agreement with BF16 reference | 42/52 prompts, vs 38/52 one-shot |
| Draft fidelity (quantized vs BF16 draft) | top-1 divergence 0.042 — equal to the BF16 draft's own noise |
| Speculative acceptance | At or above the BF16-draft baseline on the serving gate |
| Kernel battery (attention, GEMMs, boot chain) | 4/4 pass; CPU regression suite 335 passed |
| Mixed-load oracle | ≤1 of 6,144 committed decoder tokens outside the clean target's top-20 (0 on greedy legs; 1 rank-21 token on one sampled leg) |

### Throughput

Measured on this project's four-MI100 server; workload definitions matter —
short-prompt throughput does not predict long-context mixed-load speed.

| Date | Workload | Aggregate output rate | What was measured |
|---|---|---:|---|
| 2026-10-10 | Short prompts, six concurrent streams | 218.1 tok/s | 32 → 1,000 tokens; 32 greedy requests; whole-run rate |
| 2026-10-10 | Long contexts, six concurrent streams | 162.6 tok/s | 32k → 2,048 tokens; sampled; common decode window |
| 2026-10-10 | Mixed load, before decode-priority update | 12.6 tok/s | Three decoders while three fresh 32k prompts prefill |
| 2026-10-10 | Mixed load, decode-priority update | **75.4 tok/s** | Matched prompts, models, hardware; **6.0×** during overlap |
| 2026-10-11 | 40k agent-style mixed load, baseline | 35.4 cold / 80.1 warm tok/s | Eight staggered chats × three turns; whole-run rate |
| 2026-10-11 | Same 40k workload, kernel and admission fixes | **40.2 cold / 92.2 warm tok/s** | Cached tool-turn mean TTFT 146 → 70 s |

**Tradeoff:** the decode-priority policy favors active decoders — injected
prefills reached first token in 43/82/100 s versus 19/35/52 s before. It is
not a claim of higher total-work throughput. A repeated 32k prefix hit first
token in 1.51 s versus 2.64 s, reusing 29,952 cached tokens.

Methodology, controls, and reproduction commands live in the
[experiment ledger](docs/recipes/surface_experiments_ledger.jsonl); the
[40k optimization report](docs/recipes/agent_mixed_optimizations_20261011.md)
and [lifecycle profile](docs/recipes/agent_mixed_profile_results_20261011.md)
record the accepted changes and remaining mixed-step stalls.

## Models

Download both published checkpoints; the draft is not a standalone model.

| Role | Checkpoint | Launcher default location |
|---|---|---|
| Target | [Qwen3.8-27B INT8 PTQR](https://huggingface.co/curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128-PTQR) | `~/models/Qwen3.8-27B-PTQR-R10S60` |
| Draft | [DFlash2 INT8 PTQR](https://huggingface.co/curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128-PTQR) | `~/models/dflash2-ptqr-r1` |

**PTQR** (post-training quantization with retraining) settles INT8 weights
*and* group scales against a frozen BF16 reference on serving-shaped traffic,
closing the distribution gap one-shot GPTQ calibration leaves. Set
`MODEL_DIR` and `DRAFT_MODEL_DIR` if your checkpoints live elsewhere.

## Getting started

### 1. Check the requirements

| Component | Tested deployment |
|---|---|
| GPUs | 4 × AMD Instinct MI100, 32 GB each, XGMI full mesh |
| System | Linux with ROCm at `/opt/rocm` |
| Serving environment | Python 3.12 in a repository-local `.venv`; PyTorch `2.11.0+rocm7.1`, Triton `3.6.0` |
| HIP runtime | Patched `7.2.53211` HIP/HSA libraries in the serving environment |
| Kernel library | Sibling [int8-aiter](https://github.com/curvedinf/int8-aiter) checkout, loaded through `PYTHONPATH` |
| Storage | Space for both checkpoints, build caches, and the prefix-cache CPU tier |

> [!IMPORTANT]
> There is no one-command installer. The pinned PyTorch wheel's original
> ROCm 7.1 runtime leaked ~80 KiB of host memory per graph replay (a global
> OOM every ~10 h under load; root-caused 2026-10-05). Preserve the tested
> runtime override described in the [recipe](docs/recipes/README.md), or
> validate a replacement before production use. Unconstrained dependency
> installation can silently replace the ROCm packages with incompatible
> builds.

### 2. Check out both repositories

Run these as the account that will own the service; the launcher expects
this layout under that account's home directory.

```bash
cd "$HOME"
git clone --branch main https://github.com/curvedinf/int8-vllm.git
git clone --branch main https://github.com/curvedinf/int8-aiter.git aiter
cd int8-vllm
```

Keep AITER as a sibling checkout and **do not pip-install it into the serving
environment** — its kernels rebuild lazily from that checkout.

### 3. Build and validate

After provisioning `.venv` with the pinned ROCm dependencies and runtime
override above (20–40 minutes):

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

Expect all four battery checks to pass; allow time for AITER's first-use
kernel compilation. See [incremental builds](docs/contributing/incremental_build.md)
for the development workflow.

### 4. Start the server

Place the checkpoints at the locations above and provide a private API key
through `VLLM_API_KEY` or the launcher's configured key file.

```bash
scripts/serve_recipe_qwen38.sh start
scripts/serve_recipe_qwen38.sh status
scripts/serve_recipe_qwen38.sh stop     # or: restart / supervise
```

The API listens on **127.0.0.1:8020** with model name `qwen3.8-27b-gptq8`;
logs are under `logs/serve_recipe_qwen38/`. Prefix-cache reuse spans GPU
memory and an 8 GiB pinned-RAM tier; an optional NVMe third tier exists
behind a launcher lever and is disabled in the current deployment (consumer
SSD endurance). Review the launcher's CPU affinity (cores 0–47) and cache
budgets for your server before starting. For persistent deployment, adapt
[the systemd unit](scripts/vllm.service) to your account and paths — the
checked-in unit uses this server's `/home/curved` layout.

## Further reading

- [Serving recipe](docs/recipes/README.md) — the canonical baseline:
  configuration, runtime requirements, cache tiers, controls, rollback.
- [INT8 accuracy audit](INT8_AUDIT_RESULTS.md) — precision choices and the
  quality-gate method behind them.
- [Experiment ledger](docs/recipes/surface_experiments_ledger.jsonl) —
  reproducible A/B experiments and adoption decisions.
- [AGENTS.md](AGENTS.md) — operating instructions for coding agents working
  in this repository.
- [License](LICENSE) — Apache 2.0, inherited from vLLM.
