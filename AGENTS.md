# Agent Instructions for int8-vllm (MI100 fork)

Operational guide for AI agents working in this repository (the `vllm-gfx908`
checkout, remote `curvedinf/int8-vllm`, authoritative branch `main`).
Read this before
building, serving, testing, or syncing. This file is fork-specific; the
upstream contribution rules (appendix below) still apply to anything destined
for vllm-project/vllm — but this repo is a deployment fork and is not
upstreamed directly.

## THE BASELINE (read first)

`docs/recipes/README.md` is the canonical baseline. Every feature listed there
(the two GS128 checkpoints, AITER W8A8 INT8 GEMMs everywhere, int8
per-token-head KV on both target and draft, AITER unified attention, vLLM
CUSTOM all-reduce, DFlash2 with NS=15, TP4, C8, ACT_QUANT=round, and fp32
mamba state) is mandatory. The fused AR+RMSNorm+per-group INT8 quant-out
epilogue is OFF by default (an eager seam exists behind
`VLLM_GFX908_EAGER_EPILOGUE=1`; experiment E3 verdict was TPOT-neutral).
Do not substitute W8A16, the fork-local Triton GEMM, TRITON_ATTN, RCCL,
AITER CAR, fp16 KV, no-spec, another TP size, or another concurrency in the
target recipe. Historical
experiments are archived (archived logs tarball outside the repo, git history);
the A/B ledger lives at `docs/recipes/surface_experiments_ledger.jsonl`
(`logs/` is gitignored).

## The one paragraph you must internalize

This is the fastest vLLM branch for 4x AMD Instinct MI100 (gfx908 / CDNA1,
32GB, XGMI full mesh). MI100 int8 matrix rate is 2x every dtype except fp16
(equal) at half the bandwidth — so **int8 is the native dtype** of this stack:
GPTQ 8-bit weights (uint8b128, group_size 128) use AITER W8A8 INT8 GEMMs for
every decode and prefill shape, plus int8 **per-token-head** KV cache
(`--kv-cache-dtype int8_per_token_head`, block 32, f32 inline scales) and
AITER unified attention. The DFlash2 draft is also GPTQ INT8
GS128, and both target and draft KV use `int8_per_token_head`. Mamba state
stays fp32 (measured float exception — the int8 mamba experiment failed on
quality). Collectives use vLLM CUSTOM all-reduce; the fused
AR+RMSNorm+per-group INT8 quant-out epilogue is off. The
optimized configuration is the Qwen3.8 target + DFlash2 pair at TP4/C8 over XGMI;
do not derive an alternate production configuration from archival material.

## Repository layout (two sibling forks, both required)

```
<parent>/
├── vllm-gfx908/   branch main   (this repo — curvedinf/int8-vllm; serving venv .venv/)
└── aiter/         branch main   (curvedinf/int8-aiter; PYTHONPATH consumer, not pip-installed)
```

- `aiter` is consumed **from the sibling checkout at runtime** via
  `PYTHONPATH=../aiter` — never `pip install` it into `.venv` (that would
  shadow the checkout and its lazy JIT rebuild behavior). All attention and
  GEMM kernels come from this checkout; no separate attention package is part
  of the recipe.
- `main` is the authoritative branch in both repos. Sync procedure: merge
  `upstream/main` (vllm upstream = vllm-project/vllm, aiter upstream =
  ROCm/aiter) into `main`, then validate E2E before pushing.
- Known root-owned dirs were worked around as `*.rootjunk` siblings in the
  `../aiter` checkout and `.git/objects/.root-owned-*` here — they need a one-time
  `sudo rm`/`chown` by the human.

## Environment (do not deviate)

- All Python via `.venv/bin/python` in the repo root. Never system python, never
  bare pip against this venv without a `--dry-run` first — it pins
  torch 2.11.0+rocm7.1 + triton 3.6.0 and PyPI resolution would replace them
  with CUDA builds.
- Quantization uses a **separate quantization venv** outside the repo (torch 2.13.0+rocm7.2,
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

aiter JIT modules (`../aiter/aiter/jit/module_*.so`) rebuild lazily on first
use — but the JIT only triggers on `ModuleNotFoundError`, so after source
changes move the stale `.so` aside once.

## Serving

```bash
scripts/serve_recipe_qwen38.sh {start|stop|restart|status|supervise}   # Qwen3.8 + DFlash2 (target config)
```

Non-negotiable settings in the qwen38 script:
`VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
VLLM_GFX908_INT8_LM_HEAD=1`, AITER CK W8A8 for every GS128 GEMM,
`--tensor-parallel-size 4 --max-num-seqs 8
--kv-cache-dtype int8_per_token_head --mamba-ssm-cache-dtype float32`,
`VLLM_GFX908_ACT_QUANT=round` (fused round-to-nearest act quant — quality
default, see INT8_AUDIT_RESULTS.md), and the speculative config
`{"method":"dflash","num_speculative_tokens":15,"kv_cache_dtype":"int8_per_token_head"}`.
The nested dtype is explicit and applies to the draft; the top-level
`--kv-cache-dtype int8_per_token_head` applies to the target. The draft
MODEL dtype resolves from its checkpoint (bf16) — forcing the target's
fp16 onto it overflows its residual stream (fixed 2026-08-24; do not
regress). Measured float exceptions (do not quantize without re-gating):
selector codebooks, draft ctx-KV projection (bf16 > fp16 > int8 ladder),
GDN recurrent state (fp16 until a scaled kernel exists). All-reduce is
vLLM CUSTOM (`CAR=0 AR=1`) — measured fastest; AITER CAR is coherent and
available via `CAR=1` for tuning. Any dtype change on a draft surface
requires the acceptance gate: TP4/C8 greedy bench, acceptance within
±1pt of the current baseline (see INT8_AUDIT_RESULTS.md for method).
`AiterW8A16LinearKernel` is a legacy class name only; for these GS128
models it must dispatch AITER CK W8A8 for all M.
`TritonW8A16LinearKernel` is blocklisted by the launcher.

systemd: unit template at `scripts/vllm-openai-gfx908-qwen38.service` (`%h`
paths). Install: copy to `/etc/systemd/system/`, `daemon-reload`,
`enable --now`. It conflicts with the old qwen36 unit and the llama-api
service.

## Testing ladder (run in this order after any stack change)

1. **Micro** (idle GPU ok): `PYTHONPATH="$PWD:$PWD/../aiter" .venv/bin/python scripts/test_int8_kv_micro.py`
   — int8 per-token-head attention vs fp16 reference.
2. **Battery** (idle GPU ok): `HIP_VISIBLE_DEVICES=<idle> .venv/bin/python scripts/battery_gfx908.py`
   — production shapes (hdim 256, GQA 6:1), varlen causal/non-causal/sliding-window
   attention interface checks, boot import chain against merged aiter, AITER gemm
   whitelist. Expect `4/4 PASS`.
3. **Serving regression** (all 4 GPUs): boot the qwen38 production script and
   verify its startup report names AITER W8A8, AITER unified attention, vLLM
   custom AR, int8 PTH KV on both models, fp32 mamba, TP4, C8, and DFlash2
   NS=15.
4. **E2E**: qwen3.8 + DFlash2 via the serve script; check coherence, DFlash2
   acceptance length, and C8 throughput without changing the target contract.

Gate tooling: `scripts/kld_gate_boot.sh <tag>` (boot-gate harness),
`scripts/kld_probe_v2.py` (live diagnostic probe), and the A/B ledger at
`docs/recipes/surface_experiments_ledger.jsonl` (committed there because
`logs/` is gitignored).

Validation doctrine: `docs/recipes/README.md` is the baseline record for
this "get it working" pass; the historical A/B ledger was archived to git
history (see `logs/` before the fresh-slate commit). Optimization passes
should re-establish an A/B ledger at
`docs/recipes/surface_experiments_ledger.jsonl` before tuning. No perf claim
without a reproducible A/B row.

## Quantizing a model

Recipe lives outside the repo at `<models>/quantize_qwen38_27b_gptq8.py`. Params that matter:
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
  `<models>/Qwen3.8-27B-GPTQ-8bit-gs128`.
- Published DFlash2 companion: `curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128`,
  deployed at `<models>/dflash2-int8/Qwen3.8-27B-DFlash2-GPTQ-8bit`.
  It is not standalone and is designed for the target above.

## What "done" means here

A task is done when: the battery is 4/4, the serve script boots the intended
model on all 4 GPUs, greedy outputs are sane, and any perf-affecting change
has a fresh A/B row in the ledger. Push to `main`; it is the authoritative
branch in both forks.

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
