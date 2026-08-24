# Agent Instructions for vllm-gfx908 (MI100 fork)

Operational guide for AI agents working in this repository. Read this before
building, serving, testing, or syncing. This file is fork-specific; the
upstream contribution rules (appendix below) still apply to anything destined
for vllm-project/vllm — but this repo is a deployment fork and is not
upstreamed directly.

## THE BASELINE (read first)

`docs/recipes/README.md` is the canonical baseline. Every feature listed there
(the two GS128 checkpoints, AITER W8A8 INT8 GEMMs everywhere, int8 KV/Mamba,
AITER unified attention, AITER custom all-reduce, fused
AR+RMSNorm+per-group INT8 quant-out, DFlash2, TP4, and C8) is mandatory. Do
not substitute W8A16, the fork-local Triton GEMM, TRITON_ATTN, RCCL, fp16 KV,
no-spec, another TP size, or another concurrency in the target recipe. Historical
experiments are archived (`~/archived-logs-20260822.tar.gz`, git history);
the experiment ledger lives on from this point under `logs/` fresh.

## The one paragraph you must internalize

This is the fastest vLLM branch for 4x AMD Instinct MI100 (gfx908 / CDNA1,
32GB, XGMI full mesh). MI100 int8 matrix rate is 2x every dtype except fp16
(equal) at half the bandwidth — so **int8 is the native dtype** of this stack:
GPTQ 8-bit weights (uint8b128, group_size 128) use AITER W8A8 INT8 GEMMs for
every decode and prefill shape, plus int8 **per-token-head** KV cache
(`--kv-cache-dtype int8_per_token_head`, block 32, f32 inline scales), int8
Mamba state, and AITER unified attention. The DFlash2 draft is also GPTQ INT8
GS128, and both target and draft KV use `int8_per_token_head`. Collectives use
AITER custom all-reduce with fused AR+RMSNorm+per-group INT8 quant-out. The
optimized configuration is the Qwen3.8 target + DFlash2 pair at TP4/C8 over XGMI;
do not derive an alternate production configuration from archival material.

## Repository layout (three sibling forks, all required)

```
~/vllm-gfx908     branch mi100-optimized-sync   (this repo; serving venv .venv/)
~/aiter           branch mi100-optimized-sync   (PYTHONPATH consumer, not pip-installed)
~/flash-attention branch gfx908-sync            (installed into .venv as flash-attn 2.8.4 python-only)
```

- `aiter` is consumed **from the checkout at runtime** via
  `PYTHONPATH=~/aiter` — never `pip install` it into `.venv` (that would
  shadow the checkout and its lazy JIT rebuild behavior).
- `flash-attention` is the interface layer only; its Triton AMD backend comes
  from its `third_party/aiter` submodule, which points at `../aiter-gfx908.git`
  (relative URL — resolves to the same org). The CK backend does not build on
  gfx908 (gfx90a+ ISA) — never build or enable it.
- Sync procedure: merge `upstream/main` into the `-sync` branches (vllm
  upstream = vllm-project/vllm, aiter upstream = ROCm/aiter, FA upstream =
  Dao-AILab/flash-attention). Production branches (`mi100-optimized`)
  advance only after E2E validation passes.
- Known root-owned dirs were worked around as `*.rootjunk` siblings in
  `~/aiter` and `.git/objects/.root-owned-*` here — they need a one-time
  `sudo rm`/`chown` by the human.

## Environment (do not deviate)

- All Python via `~/vllm-gfx908/.venv/bin/python`. Never system python, never
  bare pip against this venv without a `--dry-run` first — it pins
  torch 2.11.0+rocm7.1 + triton 3.6.0 and PyPI resolution would replace them
  with CUDA builds.
- Quantization uses a **separate venv** `~/quant-venv` (torch 2.13.0+rocm7.2,
  gptqmodel 7.3.4, transformers 5.15) so the serving venv stays pinned.
- ROCm env for any GPU work: `ROCM_PATH=/opt/rocm`,
  `LD_LIBRARY_PATH=/opt/rocm/lib`, `PYTORCH_ROCM_ARCH=gfx908`,
  `GPU_ARCHS=gfx908`, `VLLM_TARGET_DEVICE=rocm`.

## Build

```bash
export PYTORCH_ROCM_ARCH=gfx908 GPU_ARCHS=gfx908 VLLM_TARGET_DEVICE=rocm \
  ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm MAX_JOBS=32 \
  LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH \
  PATH="$PWD/.venv/bin:/opt/rocm/bin:$PATH"   # venv provides cmake+ninja
.venv/bin/python setup.py build_ext --inplace
```

~20-40 min; produces `vllm/_rocm_C.abi3.so`, `_C_stable_libtorch.abi3.so`
(carries the gptq repacked GEMM), and the rust tool parser. After any
rebuild, smoke-check:

```bash
.venv/bin/python -c "import vllm, vllm._rocm_C; from vllm.model_executor.kernels.linear.mixed_precision.aiter_w8a16 import AiterW8A16LinearKernel; print(AiterW8A16LinearKernel)"
```

aiter JIT modules (`~/aiter/aiter/jit/module_*.so`) rebuild lazily on first
use — but the JIT only triggers on `ModuleNotFoundError`, so after source
changes move the stale `.so` aside once.

## Serving

```bash
scripts/serve_direwolf_qwen38.sh {start|stop|restart|status|supervise}   # Qwen3.8 + DFlash2 (target config)
```

Non-negotiable settings in the qwen38 script:
`VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_CUSTOM_AR=1
VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`, AITER W8A8 for every GS128 GEMM,
`--tensor-parallel-size 4 --max-num-seqs 8
--kv-cache-dtype int8_per_token_head --mamba-ssm-cache-dtype int8`, compiler
pass `fuse_allreduce_rms=true`, and the speculative config
`{"method":"dflash","num_speculative_tokens":7,"kv_cache_dtype":"int8_per_token_head"}`.
The nested dtype is explicit and applies to the draft; the top-level
`--kv-cache-dtype int8_per_token_head` applies to the target. Do not replace
the draft dtype with float16. The non-causal INT8-PTH path is covered by
`scripts/test_int8_kv_micro.py`; E2E validation must also exercise this exact
all-INT8-KV serving configuration. `AiterW8A16LinearKernel` is a legacy class
name only; for these GS128 models it must dispatch AITER A8W8 for all M.
`TritonW8A16LinearKernel` is blocklisted by the launcher.

systemd: unit template at `scripts/vllm-openai-gfx908-qwen38.service` (`%h`
paths). Install: copy to `/etc/systemd/system/`, `daemon-reload`,
`enable --now`. It conflicts with the old qwen36 unit and the llama-api
service.

## Testing ladder (run in this order after any stack change)

1. **Micro** (idle GPU ok): `PYTHONPATH="$PWD:$HOME/aiter" .venv/bin/python scripts/test_int8_kv_micro.py`
   — int8 per-token-head attention vs fp16 reference.
2. **Battery** (idle GPU ok): `HIP_VISIBLE_DEVICES=<idle> .venv/bin/python scripts/battery_gfx908.py`
   — production shapes (hdim 256, GQA 6:1), FA 2.8.4 interface, boot import
   chain against merged aiter, AITER gemm whitelist. Expect `4/4 PASS`.
3. **Serving regression** (all 4 GPUs): boot the qwen38 production script and
   verify its startup report names AITER W8A8, AITER unified attention, AITER
   custom AR, both INT8 caches, INT8 Mamba, TP4, C8, and DFlash2.
4. **E2E**: qwen3.8 + DFlash2 via the serve script; check coherence, DFlash2
   acceptance length, and C8 throughput without changing the target contract.

Validation doctrine: `docs/recipes/README.md` is the baseline record for
this "get it working" pass; the historical A/B ledger was archived to git
history (see `logs/` before the fresh-slate commit). Optimization passes
should re-establish an A/B ledger there before tuning. No perf claim
without a reproducible A/B row.

## Quantizing a model

Recipe lives at `~/models/quantize_qwen38_27b_gptq8.py`. Params that matter:
`bits=8, group_size=128, sym, true_sequential, lm_head=False`, 512 mixed
evol-codealpaca + C4 samples binned 256-2048.
**Run mode**: single GPU (`HIP_VISIBLE_DEVICES=0`), `offload_to_disk=False`,
`auto_forward_data_parallel=False`. The other combos are known-bad here:
disk-offload crawls at ~60MB/s page-cache streaming; multi-GPU AFDP crashes
with `hipErrorIllegalAddress` in gptqmodel 7.3.4 on gfx908. Expect ~1-2h
total; verify output = 9-shard safetensors + quantize_config.json (bits 8,
gs 128) before serving.

## Model assets

- Published target: `curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128`, deployed at
  `~/models/Qwen3.8-27B-GPTQ-8bit-gs128`.
- Published DFlash2 companion: `curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128`,
  deployed at `~/.cache/huggingface/dflash2-int8/Qwen3.8-27B-DFlash2-GPTQ-8bit`.
  It is not standalone and is designed for the target above.

## What "done" means here

A task is done when: the battery is 4/4, the serve script boots the intended
model on all 4 GPUs, greedy outputs are sane, and any perf-affecting change
has a fresh A/B row in the ledger. Push only `-sync` branches; production
branch advancement is the human's call.

---

## Appendix: upstream vLLM contribution policy

The following applies only to changes intended for `vllm-project/vllm`:

### Duplicate-work checks

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.
- No low-value busywork PRs (single typo, isolated style change). Mechanical
  cleanups only when bundled with substantive work.
- Pure code-agent PRs are **not allowed**. A human submitter must review
  every changed line and run relevant tests. AI-assisted PR descriptions must
  state why it is not duplicating an existing PR, test commands and results,
  model eval results when output-affecting, and that AI assistance was used.
- Fail closed: if work is duplicate/trivial busywork, stop and explain.

### Upstream development workflow

- Never use system `python3` or bare `pip`; use `uv` and `.venv/bin/python`
  when working an upstream checkout.
- Tests: `uv pip install -r requirements/test/cuda.in`, then
  `.venv/bin/python -m pytest tests/path/to/test_file.py -v`. Design tests
  before writing them; reuse existing suites; no one-off kernel benchmarks in
  `tests/` (those go in `benchmarks/kernels/`); model-affecting changes need
  evals (`tests/evals/` or `vllm bench`) with results in the PR.
- Lint: `pre-commit run` (line length 88, Google-style docstrings).
- Commit trailers for AI assistance: `Co-authored-by:` / `Assisted-by:`.
