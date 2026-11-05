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

The following setup is for the optimized MI100 recipes.

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

Validation covers checkpoint bytes, kernel numerics, model behavior and live
serving. An optimization is compared with its reference on fixed inputs,
then checked in the actual recipe, including concurrent and long-running
requests. Source revisions, model artifacts, sampling settings, cache state
and benchmark conditions are recorded with the results. Checks also verify
that the intended optimized path actually runs on all four ranks.

### Correctness and numerical gates

| Gate | Checks and recorded results |
| --- | --- |
| **Checkpoint loading and export** | Byte-compare loaded/repacked weights with the source checkpoint and verify PTQR export/dequantization equivalence. Bolt: **0 mismatches** across all eight target/draft rank caches; Qwen repack: **2,405 target + 186 draft entries** byte-identical. |
| **INT8 KV and attention** | Prefill, decode, causal/noncausal, sliding-window, grouped-query, ragged and page-boundary cases against reference attention. KV microtests pass; basic KV max-absolute-error limits are **0.1**, or **0.05** for noncausal/interface checks. The bounded G128 prefill campaign measured maximum mean-relative error **0.00133**. |
| **Kernel/import battery** | Production attention shapes, attention API compatibility, boot-critical imports and GEMM reference checks: **4/4 passed**. |
| **Packed GEMM, quantization and fusion** | Independent checkpoint decoding, actual tensor-parallel slices, partial groups, scales, normalization/residual outputs and expert routing. Packed-G128 checks require **exact INT8 codes**, scale relative tolerance **2e-6**, GEMM relative L2 error **below 2e-5**, quantized-epilogue relative L2 error **below 0.02**, finite outputs and exact eager/graph replay. The accepted norm/quant fusion passed **96 actual-checkpoint cases with zero output/residual/code/scale mismatches**. |
| **Recurrent state and speculative rollback** | Compare convolution/SSM updates with sequential references across accepted-token counts, chunk seams, migrations and page boundaries. Recorded checks include **500 protocol rounds**, convolution output difference **0**, GDN state difference **0** with output error about **1e-4**, and bit-identical convolution state at **11 chunk seams**. |
| **Collectives and graph safety** | Test cross-rank visibility, varying payloads/scales, chunking, interleaved collectives and repeated graph replay. Bolt's INT8 decoder passed **83 GPU checks** plus TP4 collective tests; eager and graph serving boots also passed. |
| **Distribution and greedy-output screen** | The fixed **52-prompt** corpus reports mean/median/p95 KLD, compared/skipped positions, nonempty outputs and first-40-character agreement. The PTQR target with BF16 draft recorded median **0.00694**, **42/52** agreement; the INT8 target+draft pair recorded **0.00798**, **42/52**, versus one-shot **0.0110**, **38/52**. |
| **Stop probability and numerical health** | Monitor first-token stop probability, finite logits/activations and recurrent-state overflow. The historical Qwen serving gate recorded **0.38% stop probability vs 0.36% BF16**; failing FP16/INT8 recurrent-state variants were rejected and FP32 state retained. |
| **Draft fidelity** | Compare quantized and BF16 drafts on the same captured target states, then verify the exported quantization grid. Both scored **0.042 next-token agreement over 120 probes**. This is agreement with held-out text, not a divergence metric or a general accuracy score. |
| **Sampling correctness** | Check top-k/top-p ordering, seeded/unseeded paths, rejection-sampling marginals and conditional chains. The production greedy-proposal rejection test used **20,000 trials**, with maximum frequency error **0.0013**; proposed-token emission was **0.0200 vs 0.0195 expected**. |
| **Decode/prefill consistency** | Replay the exact generated token IDs through a clean prefill reference and measure chosen-token logprob drift. A passing 20k-context seam gate recorded mean absolute difference **0.0339 nats**, with **0/256 positions above 1 nat**. |

The legacy KLD tool compares truncated top-logprob distributions along
independently generated continuations. Its reported values are a screening
proxy, not aligned full-vocabulary KL or proof of BF16 equivalence. Exact-token
replay and live-output gates provide separate checks.

### Serving and regression gates

| Gate | Checks and recorded results |
| --- | --- |
| **Speculative acceptance** | Compare accepted tokens per round, accepted/drafted fraction, rejection tails and acceptance decay under matched traffic. At 42k context, the INT8 Qwen draft recorded mean acceptance **3.825 vs 3.88** for the BF16 draft, above the original **3.67** reference. Bolt's C8 functional smoke recorded acceptance length **4.55**. These are workload-specific values, not interchangeable percentages. |
| **Full-output long-context review** | Read complete outputs for structural garble, repeated walls, restarts, token splices and progressive drift. The harness includes **eight 22k-input/4,096-output legs** (two serial, six concurrent). Later Qwen zero-structural-garble gates passed seeds **301/307/305 plus an unseeded 4k-output leg**; a separate multi-boot stress check completed **24/24** greedy 20k/4k legs. Earlier eight-leg failures remain in the ledger. |
| **Short-answer soak and format checks** | Require nonempty final answers, no repeated-character/prose collapse, finite top-token logprobs and requested formats. Bolt passed **500/500** soak cases, **52/52** chat-corpus cases and **10/10** strict-format responses. Reasoning is disabled for short-budget soak cases and tested separately. |
| **Mixed-load token oracle** | Run three 32k/2,048-token decoders while three fresh 32k prompts prefill; score committed token IDs against the clean target. Greedy traces passed **0/6,144 tokens outside the target's top 20**. The sampled run had **1/6,144**, ranked 21, and did **not** pass the strict zero-miss gate. |
| **Prefix and offload restoration** | Check cold/repeated retrieval, cached-token accounting, GPU reset with RAM restoration, service restart and appended prompts. Bolt returned the same correct phrase for a **28,067-token** prompt, with **28,056 tokens reused** on repeat; the earlier RAM-restoration gate reused **17,024 external tokens** with the same answer. Qwen's local repeat matched **16/16** greedy IDs; its restart check matched **15/16**, so cross-boot bitwise equality is not claimed. |
| **API and cross-recipe regression** | Arithmetic, reasoning plus a final answer, automatic tool calls, image/video input, streaming, authentication and disabled development endpoints passed for Bolt. The shared loader changes also passed the unchanged Qwen launcher's **six concurrent greedy queries**. |
| **Memory and service stability** | Check all ranks, health, allocation failures, leaks, cache residency, startup/teardown and unexpected restarts. The November Bolt offload gate passed **385 regressions**, **8/8 concurrent requests**, HTTP **200**, **zero KV swap bytes** across workers and **zero automatic service restarts**. Earlier campaigns passed **335 Qwen CPU checks** and **339 Bolt focused checks**. |

Performance changes also run the complete workload gates: Qwen short C6
(32 requests, 32 input/1,000 output), 32k C6 steady/long runs (800 and 2,048
output), solo 32k time-to-first-token, mixed prefill/decode, greedy coherence
and temperature-1 copy probes. Bolt campaigns use fixed C8 workloads,
warmup-separated five-run comparisons, exact request/token counters,
held-out shapes/workloads and independent-boot confirmation. Prefill,
decode, whole-run throughput and acceptance are evaluated separately;
a speed gain alone does not clear the quality gates.

Results above come from different recorded campaigns, not one universal
all-green certification. Bolt's INT8 decoder was adopted with an explicit
exception to its one-percentage-point acceptance-loss limit:
**35.65% to 34.22%**. Coherence is not task accuracy; Bolt's matched
official-reference KLD and full 256k-context validation remain pending.

Gate implementations include the [kernel battery](scripts/battery_gfx908.py),
[packed-weight probe](scripts/probe_bolt_packed_int8.py),
[distribution screen](scripts/kld_probe_v2.py),
[draft probe](scripts/ptqr_draft_fidelity.py),
[full-output battery](scripts/qat_battery.py),
[soak](scripts/ua_live_soak.py) and
[Bolt chat probe](scripts/probe_bolt_quality.py).
The [Qwen recipe](docs/recipes/README.md),
[Bolt recipe](docs/recipes/nemotron_bolt.md) and
[experiment ledger](docs/recipes/surface_experiments_ledger.jsonl) retain
the controls, individual outcomes, rejected candidates and reproduction details.

## Documentation

- [Qwen recipe](docs/recipes/README.md) and [Bolt recipe](docs/recipes/nemotron_bolt.md) — configuration, benchmarks and validation.
- [INT8 coverage audit](INT8_AUDIT_RESULTS.md) — precision choices and floating-point exceptions.
- [Experiment ledger](docs/recipes/surface_experiments_ledger.jsonl) — measured changes and adoption decisions.
- [Incremental builds](docs/contributing/incremental_build.md) — development workflow.
- [License](LICENSE) — Apache 2.0.
