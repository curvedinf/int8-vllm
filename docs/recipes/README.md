# Qwen3.8-27B Int8-Native Serving Recipe (gfx908 / 4x MI100)

**This is the canonical baseline.** Everything here is ON by intent. Do not
disable features listed here without explicit user instruction — they are
load-bearing for the int8-native doctrine and were each gated by KLD/bench
measurements. Historical experiments live in git history
(`logs/` was gitignored; the experiment narrative is preserved in commits
through 2026-08-22 and in `docs/recipes/archive-note.md`).

## The stack

| Repo | Branch | Role |
|---|---|---|
| `~/vllm-gfx908` | `mi100-optimized-sync` | serving engine, int8 kernels |
| `~/aiter` | `mi100-optimized-sync` | int8 unified-attention + gfx908 tuning (PYTHONPATH) |
| `~/flash-attention` | `gfx908-sync` | python-only 2.8.4, Triton AMD backend |

## The baseline configuration (all features ON)

```bash
scripts/serve_direwolf_qwen38.sh start    # or: supervise / restart / status
```

The script encodes the full intended feature set:

| Feature | Value | Why it's critical |
|---|---|---|
| Checkpoint | `~/models/Qwen3.8-27B-GPTQ-8bit-gs128` | GPTQ int8, gs=128 (enables W8A8) |
| Weights | int8 (Triton W8A16 decode / **W8A8 int8×int8 prefill**, M≥256 & N≥8192) | 2× MFMA rate on fat GEMMs |
| KV cache | `int8_per_token_head` (fp32 inline scales, block 32) | half KV bandwidth |
| Mamba/GDN state | `int8` (`--mamba-ssm-cache-dtype int8`) | 0.6% drift gate-passed |
| Embedding table | int8 gather (untied tables; tied lm_head stays fp16 — GEMM needs dense) | half gather bandwidth |
| Speculative decoding | **DFlash2, ns=7, int8 drafter, int8 draft KV — ON, non-negotiable** | user directive 2026-08-22 |
| Attention backend | `TRITON_ATTN` (aiter UA has a kernel-level-clean soak but stays off pending live soak) | stability |
| Kernel blocklist | `VLLM_DISABLED_KERNELS=AiterW8A16LinearKernel` | that kernel garbles on gfx908 |
| GPU util | 0.90 (spec drafter needs headroom) | |
| TP / graphs | TP4 XGMI, FULL_AND_PIECEWISE (forced FULL_DECODE_ONLY on gfx908) | |

systemd: `scripts/vllm-openai-gfx908-qwen38.service` (conflicts with the
retired qwen36 unit).

## Models

- Target: `~/models/Qwen3.8-27B-GPTQ-8bit-gs128` (30G; HF upload pending —
  card written at `README.md` inside the model dir)
- Drafter: `~/.cache/huggingface/dflash2-int8/Qwen3.8-27B-DFlash2-GPTQ-8bit`
  (true GPTQ int8; requires the post-bake remap below)

## Quantization recipes

- Target: `~/models/quantize_qwen38_27b_gptq8.py`
  `--group-size 128` (single GPU, ~3h; `--lm-head` OOMs on 32GB — deferred)
- Drafter: `~/models/quantize_dflash2_int8.py` (GPU0, ~3h) **then remap**
  (`docs/recipes/drafter_remap.py`): gptqmodel saves with a `model.` prefix,
  mangles the arch tag to `Qwen3ForCausalLM`, quantizes conv/selector
  params that must stay dense, and re-restores dense `.weight` for modules
  that carry GPTQ tensors (two loader sources). The remap fixes all four
  and restores `DFlash2DraftModel`.

## Validation gates (run these, in order, after any stack change)

1. `PYTHONPATH=~/aiter .venv/bin/python scripts/test_int8_kv_micro.py` — int8 PTH
   attention incl. non-causal (2.4e-4)
2. `HIP_VISIBLE_DEVICES=<idle> .venv/bin/python scripts/battery_gfx908.py`
   — prod shapes, FA interface, boot imports, gemm whitelist (4/4)
3. `scripts/kld_probe.py capture|compare` — real-artifact KLD gate
   (needs `NCCL_ALGO=Ring NCCL_PROTO=Simple` env; TP small when a server is up)
4. Coherence curl against the booted server
5. `.venv/bin/python scripts/ua_live_soak.py -n 500` — live correctness soak
   (2026-08-23: **500/500 coherent** on the SPEC=1 AR=0 baseline — true
   GPTQ-int8 drafter, float16 draft KV, RCCL all-reduce)

## Baseline status (2026-08-23)

The coherent spec-on int8 baseline is: target GPTQ-int8 gs128 + int8 PTH KV
+ int8 mamba state + DFlash2 spec (int8 drafter, float16 draft KV) + TP4
XGMI + AR via RCCL. Known-open items: aiter-CAR race (see landmines),
aiter-UA re-test on this clean baseline, FA2 fork wiring.

## KLD gate reference numbers (2026-08-22 sweep, teacher-forced)

| Config | KLD | agree@16 |
|---|---|---|
| gs128 weights | 0.0109 | 74% |
| + act quant | 0.0141 | 65% |
| + lm_head | 0.0142 | 65% |
| + embed | 0.0163 | 62% |
| mamba int8 state | 0.6% decode drift | — |

Gate: KLD ≤ 0.02 primary. 256-token free-rollout agreement is
non-discriminative at 27B scale — don't use it as the gate.

## Known landmines

- NCCL int8 AllGather requires Ring/Simple (Tree/LL has no int8 algo)
- gptqmodel multi-GPU AFDP crashes on gfx908 (hipErrorIllegalAddress);
  disk-offload crawls — always single-GPU
- Disk: quant saves need ~35G headroom
- The KLD offline engine cannot boot while a server holds GPUs
- DFlash2 spec throughput was 0.51× in the FIRST measurement — that path was
  unoptimized and the user has ruled it stays ON; the optimization pass is
  the next phase of work, not a reason to disable it

## Next-session optimization queue (do not lose)

1. **AITER_CUSTOM vs vLLM CustomAllreduce bench on ROCm 7.14**: the aiter
   fork build is DONE — `module_custom_all_reduce` builds on gfx908
   (bda53078d) plus fused AR+RMSNorm+per-group INT8 quant-out (527da7d1d,
   bit-exact vs vLLM `_quantize_activation_per_block` semantics; test in
   `op_tests/multigpu_tests/test_fused_ar_rms_int8_quant.py`). Remaining:
   wire the vLLM consumer + benchmark both ARs (the old -17% number is from
   ROCm 7.2, stale).
2. Spec-path tuning pass (ns sweep, verify-batch cost profile) — DFlash2
   stays ON regardless; make it win.
