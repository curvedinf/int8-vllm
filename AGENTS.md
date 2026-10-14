# Agent Instructions for int8-vllm (MI100 fork)

Operational guide for AI agents working in this repository (the `int8-vllm`
checkout, remote `curvedinf/int8-vllm`, authoritative branch `main`).
Read this before
building, serving, testing, or syncing. This file is fork-specific; the
upstream contribution rules (appendix below) still apply to anything destined
for vllm-project/vllm — but this repo is a deployment fork and is not
upstreamed directly.

## THE BASELINE (read first)

`docs/recipes/README.md` is the canonical baseline. Every feature listed there
(the PTQR-retrained GS128 target + PTQR-retrained int8 DFlash2 draft (since
2026-09-06 — ledger PTQR_P2_R1DRAFT_FINAL: quantized fidelity == bf16, acceptance
3.83/13 at 42k), AITER W8A8 INT8 GEMMs everywhere, int8_block_g128 KV on both
target and draft (since 2026-09-04 — int8-first directive; halved KV noise vs
int8_per_token_head, acceptance 4.67/14 at 40k), AITER unified attention, vLLM
CUSTOM all-reduce, DFlash2 with NS=13, TP4, C8, ACT_QUANT=round, and fp32
mamba state) is mandatory. The fused AR+RMSNorm+per-group INT8 quant-out
epilogue is OFF by default (an eager seam exists behind
`VLLM_GFX908_EAGER_EPILOGUE=1`; experiment E3 verdict was TPOT-neutral).
Do not substitute W8A16, the fork-local Triton GEMM, TRITON_ATTN, RCCL,
AITER CAR, bf16/fp16 KV (fp16 draft KV breaks the draft; bf16 is banned by
the int8-first directive — int8_block_g is the sanctioned KV family),
no-spec, another TP size, or another concurrency in the
target recipe. Historical
experiments are archived (archived logs tarball outside the repo, git history);
the A/B ledger lives at `docs/recipes/surface_experiments_ledger.jsonl`
(`logs/` is gitignored).

## The one paragraph you must internalize

This is the fastest vLLM branch for 4x AMD Instinct MI100 (gfx908 / CDNA1,
32GB, XGMI full mesh). MI100 int8 matrix rate is 2x every dtype except fp16
(equal) at half the bandwidth — so **int8 is the native dtype** of this stack:
GPTQ 8-bit weights (uint8b128, group_size 128) use AITER W8A8 INT8 GEMMs for
every decode and prefill shape, plus int8_block_g128 KV cache
(`--kv-cache-dtype int8_block_g128`; PTQR-era, since 2026-09-06) and
AITER unified attention. The DFlash2 draft is PTQR-retrained int8 (rung-1,
`<models>/dflash2-ptqr-r1`: 35 G128 quant payloads + 12 dequant-grid bf16
surfaces on the deployed wrapper — quantized fidelity gate top-1 0.042 ==
bf16, acceptance 3.83/13 at 42k); the target is PTQR-retrained (Qwen3.8-27B-PTQR-R10S60: serving-gate
KLD 0.0069 vs 0.0110 deployed GPTQ, 42/52 greedy agreement vs 38/52,
acceptance 3.88/13 >= the bf16 baseline 3.67 — see ledger
PTQR_P1_R10S60_FINAL). Both target and draft KV use `int8_block_g128`. Mamba state
stays fp32 (measured float exception — the int8 mamba experiment failed on
quality). Collectives use vLLM CUSTOM all-reduce; the fused
AR+RMSNorm+per-group INT8 quant-out epilogue is off. The
optimized configuration is the Qwen3.8 target + DFlash2 pair at TP4/C8 over XGMI;
do not derive an alternate production configuration from archival material.

## Repository layout (two sibling forks, both required)

```
<parent>/
├── int8-vllm/      branch main   (this repo — curvedinf/int8-vllm; serving venv .venv/)
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
- The venv's ROCm *runtime* is deliberately patched ahead of its wheel:
  `torch/lib/libamdhip64.so` and `libhsa-runtime64.so` are the 7.2.53211
  builds (from the quant venv's torch 2.13 wheel; wheel originals kept as
  `*.wheel-rocm7.1`). The bundled 7.1 runtime leaks ~80 KiB of host AQL
  packet batches on every hipGraphLaunch replay
  (`VirtualGPU::dispatchAqlPacketBatch`, ~29 MB/min/worker under load →
  global host OOM kill every ~10 h; root-caused 2026-10-05, battery 4/4 and
  E2E golden re-verified on 7.2). Any venv reinstall must restore this
  override or re-validate the leak is fixed upstream.
- Quantization uses a **separate quantization venv** outside the repo (torch 2.13.0+rocm7.2,
  gptqmodel 7.3.4, transformers 5.15) so the serving venv stays pinned.
- ROCm env for any GPU work: `ROCM_PATH=/opt/rocm`,
  `LD_LIBRARY_PATH=/opt/rocm/lib`, `PYTORCH_ROCM_ARCH=gfx908`,
  `GPU_ARCHS=gfx908`, `VLLM_TARGET_DEVICE=rocm`.
- FS->GPU direct IO: `vllm/fs_gpu` wraps AMD Infinity Storage (hipFile
  0.3.0, ctypes — no new build deps). In-process it resolves to torch's
  bundled (patched 7.2) HIP runtime — one `libamdhip64.so.7`, verified by
  `scripts/fs_gpu_probe.py`. `VLLM_FS_GPU=auto` (default) uses AIS for
  weight loading whenever it is available; the serve-script `AISW` flag
  forces it for A/B boots. Loads go through the per-rank repack cache
  (`~/.cache/vllm/ais_repack/`, written on first boot,
  `scripts/fs_gpu_repack_check.py` is the byte gate); hipFile 0.3.0 runs
  ~one IO per process (threads/handles/async make no difference — probed),
  hence the cache's windowed sequential reads. Byte gate for the uncached
  path: `scripts/fs_gpu_weightcheck.py`.
- KV prefix-cache hierarchy: the recipe enables the stock
  `TieringOffloadingSpec` (VRAM -> 12 GiB pinned RAM LRU -> NVMe fs tier)
  via the serve-script `DISKTIER` lever (default 50 GiB, 0=off; root
  `~/.cache/vllm/kv_tier`). All tiering policy is the existing manager's
  (write-through cascade, staged promotion, CPU LRU); the only fork
  addition is `FileSystemTierManager.max_bytes` — oldest-file disk-budget
  reclamation, no scheduling.

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
--kv-cache-dtype auto --mamba-ssm-cache-dtype float32`,
`VLLM_GFX908_ACT_QUANT=round` (fused round-to-nearest act quant — quality
default, see INT8_AUDIT_RESULTS.md), and the speculative config
`{"method":"dflash","num_speculative_tokens":13,"kv_cache_dtype":"int8_block_g128"}`.
The nested dtype is explicit and applies to the draft; the top-level
`--kv-cache-dtype int8_block_g128` applies to the target. fp16 draft KV is
measured broken (mean_k=1.01, all drafts rejected) — do not use it; the `int8_block_g{4..128}` family (KV_DTYPE env) is the
long-context int8 fallback — see the table in docs/recipes/README.md.
The draft
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

systemd: unit template at `scripts/vllm.service` (unit name `vllm`)
(hardcoded `/home/curved` paths — `%h` resolves to `/root` in system units,
do not reintroduce it). Install: copy to `/etc/systemd/system/`, `daemon-reload`,
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
   NS=13.
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

- Published target (recipe default): PTQR-retrained export
  [`curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128-PTQR`](https://huggingface.co/curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128-PTQR),
  deployed at `<models>/Qwen3.8-27B-PTQR-R10S60`. Original one-shot
  checkpoint: `curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128`
  (`<models>/Qwen3.8-27B-GPTQ-8bit-gs128`) remains the wrapper
  source and rollback.
- DFlash2 draft (recipe default since 2026-09-06): PTQR rung-1 export
  [`curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128-PTQR`](https://huggingface.co/curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128-PTQR),
  deployed at `<models>/dflash2-ptqr-r1` (wrapper template: the original
  `curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128` gptqmodel checkpoint,
  cached at `~/.cache/huggingface/dflash2-int8/`; training source:
  `<models>/ptqr_draft_r1/draft_final.pt`). The bf16 draft
  (`<models>/dflash2-bf16-with-tokenizer`) remains the fidelity/rollback
  reference. It is not standalone and is designed for the target above.

## What "done" means here

A task is done when: the battery is 4/4, the serve script boots the intended
model on all 4 GPUs, greedy outputs are sane, and any perf-affecting change
has a fresh A/B row in the ledger. Push to `main`; it is the authoritative
branch in both forks.

## Agent skills (dev-skills)

Reusable process skills from the
[curvedinf/dev-skills](https://github.com/curvedinf/dev-skills) collection
(MIT; see `agents/skills/LICENSE`; installation record:
`agents/scratch/dev-skills-install.md`) live under
`agents/skills/<passive|activated|triggered>/<name>/SKILL.md`. **Passive**
skills are scoped persistent conventions; **triggered** skills fire on a
non-user event when their conditions hold; **activated** skills are explicitly
requested workflows. Route requests through the table below and follow the
explicit links — do not assume host-native discovery. Where generic skill
guidance conflicts with this file, this file wins: the testing ladder above
outranks `testing-doctrine` specifics, push-to-`main` outranks generic branch
flow, and the A/B-ledger doctrine stands unchanged.

### Request-term routing

- **Testing:** `testing-doctrine` governs cadence and coverage; `design-validation` designs the checks; `diagnose-and-repair` investigates a failing behavior.
- **Improvement:** `run-experiment` answers a bounded comparison; `indefinite-loop` repeats experiment batches only under a continuous request.
- **Documentation:** `documentation-style` shapes ordinary edits; `polish-docs` redesigns first-use experience.
- **Delivery:** `git-discipline` governs repository changes; `operate-workflow` handles external operations; `audit-and-resolve` assesses findings; `coordinate-work` manages assignments.

| Request terms | Read |
| --- | --- |
| install dev-skills, project-wide, agent-wide | [install-dev-skills](agents/skills/activated/install-dev-skills/SKILL.md) |
| claims, citations, uncertainty | [evidence-discipline](agents/skills/passive/evidence-discipline/SKILL.md) |
| scope, requirements, ownership | [contract-discipline](agents/skills/passive/contract-discipline/SKILL.md) |
| pre-commit, PID/PII, secrets, privacy leaks | [security-hygiene-check](agents/skills/triggered/security-hygiene-check/SKILL.md) |
| commit, push, merge, branches | [git-discipline](agents/skills/passive/git-discipline/SKILL.md) |
| test cadence, nightly, release gates | [testing-doctrine](agents/skills/passive/testing-doctrine/SKILL.md) |
| wording, terminology, examples | [documentation-style](agents/skills/passive/documentation-style/SKILL.md) |
| checkpoint, resume, history | [scratch-space](agents/skills/passive/scratch-space/SKILL.md) |
| periodic workflow, timed skill, shared timers | [design-time-triggered-skill](agents/skills/activated/design-time-triggered-skill/SKILL.md) |
| fresh session, weekly compaction | [maintain-scratch](agents/skills/triggered/maintain-scratch/SKILL.md) |
| create skill, author workflow | [design-skill](agents/skills/activated/design-skill/SKILL.md) |
| correction, workflow gap, refine, self-improvement | [refine-skill](agents/skills/triggered/refine-skill/SKILL.md) |
| README, onboarding, quickstart | [polish-docs](agents/skills/activated/polish-docs/SKILL.md) |
| debug, crash, regression | [diagnose-and-repair](agents/skills/activated/diagnose-and-repair/SKILL.md) |
| test plan, acceptance, coverage gaps | [design-validation](agents/skills/activated/design-validation/SKILL.md) |
| benchmark, compare, feasibility | [run-experiment](agents/skills/activated/run-experiment/SKILL.md) |
| continuous optimization, metrics | [indefinite-loop](agents/skills/activated/indefinite-loop/SKILL.md) |
| delegate, parallelize, handoff | [coordinate-work](agents/skills/activated/coordinate-work/SKILL.md) |
| review, audit, PR findings | [audit-and-resolve](agents/skills/activated/audit-and-resolve/SKILL.md) |
| deploy, migrate, resume job | [operate-workflow](agents/skills/activated/operate-workflow/SKILL.md) |

### Scratch space

**Fresh-session startup:** on the first project-context load after the initial
user prompt, read
[maintain-scratch](agents/skills/triggered/maintain-scratch/SKILL.md) and
check its entry in `agents/scratch/timers.json`; run it if due (seven-day
default). Check once per project per fresh session — not per task or
context-window compaction.

Use [scratch-space](agents/skills/passive/scratch-space/SKILL.md) for durable,
Git-versioned task notes in `agents/scratch/`. Never ignore that directory;
include relevant note updates when committing authorized work. Retain only
resumption state, costly evidence, and ongoing operational policy not
recoverable from canonical files or Git — no routine edit summaries or
completion receipts; delete obsolete records rather than archiving them.
`agents/tmp/` holds disposable files and stays gitignored.

### Pre-commit security gate

Before every project commit — including amendments and merge commits — run
[security-hygiene-check](agents/skills/triggered/security-hygiene-check/SKILL.md)
on the final candidate after all edits and staging. Apply the disclosure
policy in `agents/scratch/security-policy.md`; findings outside its approved
scope require an explicit user disposition before committing, even when
commit/push permission already exists. This gate is independent of
self-improvement enablement.

### Self-improvement

Self-improvement: enabled.

Always read and consider
[refine-skill](agents/skills/triggered/refine-skill/SKILL.md). When skill use
ends, assess its lessons once; apply an evidence-backed refinement when
enabled, before the final response.

### Authoring and references

Use [design-skill](agents/skills/activated/design-skill/SKILL.md) to create
skills and [refine-skill](agents/skills/triggered/refine-skill/SKILL.md) for
evidence-backed maintenance. Specify outcomes and consequential decision
criteria; delegate execution mechanics. Prefer consolidation over rule
accumulation; keep skills portable and self-contained; add supporting files
only for necessary task-specific material. Treat reference libraries as
inspiration, not requirements, and check licenses before copying:

| Library | Useful ideas to examine |
| --- | --- |
| [Anthropic skills](https://github.com/anthropics/skills) | Self-contained packaging, concise entrypoints, and supporting resources for substantial workflows. |
| [Obra Superpowers](https://github.com/obra/superpowers) | Composable development processes, explicit workflow transitions, and skill validation practices. |
| [Vercel agent-skills](https://github.com/vercel-labs/agent-skills) | Clear use conditions, impact-based prioritization, and separation of collections into focused skills. |
| [Hermes Agent bundled skills](https://github.com/NousResearch/hermes-agent/tree/main/skills) | Default skill architecture, capability-focused discovery, progressive disclosure, and evidence-driven refinement. |

The [Agent Skills specification](https://agentskills.io/specification) is a
format reference, not a skill library.

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
