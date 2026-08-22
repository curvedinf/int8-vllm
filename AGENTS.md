# Agent Instructions for vllm-gfx908 (MI100 fork)

Operational guide for AI agents working in this repository. Read this before
building, serving, testing, or syncing. This file is fork-specific; the
upstream contribution rules (appendix below) still apply to anything destined
for vllm-project/vllm — but this repo is a deployment fork and is not
upstreamed directly.

## The one paragraph you must internalize

This is the fastest vLLM branch for 4x AMD Instinct MI100 (gfx908 / CDNA1,
32GB, XGMI full mesh). MI100 int8 matrix rate is 2x every dtype except fp16
(equal) at half the bandwidth — so **int8 is the native dtype** of this stack:
GPTQ 8-bit weights (uint8b128, group_size 32) through a fork-custom Triton
W8A16 kernel + repacked HIP GEMM, int8 **per-token-head** KV cache
(`--kv-cache-dtype int8_per_token_head`, block 32, f32 inline scales), int8
Q@K attention dot on the AITER path. fp16 activations between kernels (W8A8
was measured 2x slower at gs=32 — do not revisit without new data). The
**only sanctioned unquantized path** is the DFlash2 draft model (quantized
drafters are broken upstream, vllm#51581). The optimized configuration is
Qwen3.8-27B GPTQ8 + DFlash2 + FlashAttention 2 (Triton) + int8 KV + TP4/XGMI;
other combos work but are untuned.

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
.venv/bin/python -c "import vllm, vllm._rocm_C; from vllm import _custom_ops as o; print(o.gptq_w8a16_repacked_gemm)"
```

aiter JIT modules (`~/aiter/aiter/jit/module_*.so`) rebuild lazily on first
use — but the JIT only triggers on `ModuleNotFoundError`, so after source
changes move the stale `.so` aside once.

## Serving

```bash
scripts/serve_direwolf_qwen38.sh {start|stop|restart|status|supervise}   # Qwen3.8 + DFlash2 (target config)
scripts/serve_direwolf_qwen36.sh ...                                    # Qwen3.6 + MTP (legacy)
```

Non-negotiable flags in the qwen38 script (edit only with bench data):
`--tensor-parallel-size 4 --dtype half --attention-backend TRITON_ATTN
--kv-cache-dtype int8_per_token_head` and the speculative config
`{"method":"dflash2","num_speculative_tokens":7,"kv_cache_dtype":"fp16"}`
— the draft's KV **must** stay fp16 (non-causal attention has no
per-token-head int8 write path; inheriting int8 silently corrupts).
`VLLM_DISABLED_KERNELS=AiterW8A16LinearKernel` is required — the AITER W8A16
blockscale kernel garbles outputs on gfx908; the Triton W8A16 kernel is the
correct one. `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0` in prod (state
corruption after ~200 reqs); TRITON_ATTN is the stable default.

systemd: unit template at `scripts/vllm-openai-gfx908-qwen38.service` (`%h`
paths). Install: copy to `/etc/systemd/system/`, `daemon-reload`,
`enable --now`. It conflicts with the old qwen36 unit and the llama-api
service.

## Testing ladder (run in this order after any stack change)

1. **Micro** (idle GPU ok): `PYTHONPATH=~/aiter .venv/bin/python test_int8_kv_micro.py`
   — int8 per-token-head attention vs fp16 reference.
2. **Battery** (idle GPU ok): `HIP_VISIBLE_DEVICES=<idle> .venv/bin/python scripts/battery_gfx908.py`
   — production shapes (hdim 256, GQA 6:1), FA 2.8.4 interface, boot import
   chain against merged aiter, AITER gemm whitelist. Expect `4/4 PASS`.
3. **Serving regression** (all 4 GPUs): qwen3.6 model on the new stack via
   `scripts/bench_c8.py --kv-cache-dtype int8_per_token_head --mtp` — compare
   against ledger baselines before switching models.
4. **E2E**: qwen3.8 + DFlash2 via the serve script; greedy-equivalence vs
   no-spec; DFlash2 acceptance length; C8 throughput.

Validation doctrine: `logs/c8_optimization/experiments.md` is the single
source of truth for perf claims. Every optimization lands with an A/B row
there. Headline baselines: C8 fp8+no-MTP 112.35 → int8 KV 129.92 tok/s;
20:1 long-context TRITON_ATTN 28.09 out tok/s (TPOT 166.9 ms), AITER-UA int8
experimental track 54.81 (TPOT 75.7 ms). If a change has no A/B row, it did
not happen.

## Quantizing a model

Recipe lives at `~/models/quantize_qwen38_27b_gptq8.py`. Params that matter:
`bits=8, group_size=32, sym, true_sequential, lm_head=False` (lm_head stays
fp16 — also required for DFlash2's shared-head path), 512 mixed
evol-codealpaca + C4 samples binned 256-2048.
**Run mode**: single GPU (`HIP_VISIBLE_DEVICES=0`), `offload_to_disk=False`,
`auto_forward_data_parallel=False`. The other combos are known-bad here:
disk-offload crawls at ~60MB/s page-cache streaming; multi-GPU AFDP crashes
with `hipErrorIllegalAddress` in gptqmodel 7.3.4 on gfx908. Expect ~1-2h
total; verify output = 9-shard safetensors + quantize_config.json (bits 8,
gs 32) before serving.

## Model assets

- `Qwen/Qwen3.8-27B` base + `z-lab/Qwen3.8-27B-DFlash2` drafter: HF cache
  (the drafter is referenced by its **snapshot path** in the serve script —
  an HF cache repo root is not a model dir).
- GPTQ8 output: `~/models/Qwen3.8-27B-GPTQ-8bit` (HF upload target: see the
  README Models section placeholder).
- Previous prod model: `~/models/Qwen3.6-27B-GPTQ-8bit-MTP2`.

## gfx908 gotchas (learned the hard way)

- `hipErrorIllegalAddress` in gptqmodel multi-GPU forward (see above).
- AITER W8A16/a8w8 blockscale kernels garble output on gfx908 → disabled.
- AITER UA attention: fastest kernels but corrupts state after ~200 requests
  with prefix reuse → prod off; the int8 port lives in aiter's
  `ops/triton/attention/unified_attention.py` (scale caches + true-int8 Q@K).
- `import vllm` can succeed with stale compiled ops (lazy loading) — always
  rebuild before trusting a smoke test after merges.
- GPU node order in `rocm-smi` != HIP device order on this box (rocm-smi
  device 3 was HIP 0); trust `HIP_VISIBLE_DEVICES`, not labels.
- fp8 KV is NOT the int8 path — the plain `int8` alias was dropped in the
  2026-08 upstream sync; always `int8_per_token_head`.

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
