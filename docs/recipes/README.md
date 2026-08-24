# Qwen3.8-27B + DFlash2 Int8-Native Serving Recipe (gfx908 / 4x MI100)

**This is the canonical baseline.** Everything here is ON by intent. Do not
disable features listed here without explicit user instruction — they are
load-bearing for the int8-native doctrine and were each gated by KLD/bench
measurements. Historical experiments live in git history; they are not
alternative production recipes.

## The stack

| Repo | Branch | Role |
|---|---|---|
| `~/vllm-gfx908` | `mi100-optimized-sync` | serving engine, int8 kernels |
| `~/aiter` | `mi100-optimized-sync` | int8 unified-attention + gfx908 tuning (PYTHONPATH) |
| `~/flash-attention` | `gfx908-sync` | python-only 2.8.4, Triton AMD backend |

## The baseline configuration (all features ON)

The published pair is designed to run together:

- Target: [`curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128`](https://huggingface.co/curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128)
- Draft: [`curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128`](https://huggingface.co/curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128)

The DFlash2 checkpoint is not standalone. It drafts speculative tokens for
the linked target model, which verifies them.

```bash
scripts/serve_direwolf_qwen38.sh start    # or: supervise / restart / status
```

The production script uses local copies of those artifacts. The model-facing
core of the same recipe is:

```bash
TARGET_MODEL=curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128
DRAFT_MODEL=curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128

VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_CUSTOM_AR=1 \
VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
VLLM_DISABLED_KERNELS=TritonW8A16LinearKernel \
.venv/bin/vllm serve "$TARGET_MODEL" \
  --tensor-parallel-size 4 \
  --max-num-seqs 8 \
  --dtype half \
  --kv-cache-dtype int8_per_token_head \
  --mamba-ssm-cache-dtype int8 \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","pass_config":{"fuse_allreduce_rms":true}}' \
  --speculative-config '{"method":"dflash","model":"'"$DRAFT_MODEL"'","num_speculative_tokens":7,"kv_cache_dtype":"int8_per_token_head"}'
```

The top-level KV flag configures the target. The `kv_cache_dtype` inside the
speculative JSON independently configures the DFlash2 draft; both are
deliberately `int8_per_token_head`.

The script encodes the full intended feature set:

| Feature | Value | Why it's critical |
|---|---|---|
| Checkpoint | [`curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128`](https://huggingface.co/curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128), deployed at `~/models/Qwen3.8-27B-GPTQ-8bit-gs128` | GPTQ int8, gs=128 (enables W8A8) |
| GEMMs | **AITER W8A8 INT8 everywhere**, decode and prefill | no W8A16 or fork-local Triton GEMM in the target run |
| KV cache | `int8_per_token_head` (fp32 inline scales, block 32) | half KV bandwidth |
| Mamba/GDN state | `int8` (`--mamba-ssm-cache-dtype int8`) | 0.6% drift gate-passed |
| Embedding lookup | int8 gather | half embedding bandwidth; it is not a GEMM exception |
| Speculative decoding | **DFlash2, ns=7, int8 drafter, int8 draft KV — ON, non-negotiable** | user directive 2026-08-22 |
| Attention backend | **AITER unified attention in INT8** | target and draft both use INT8-PTH KV |
| All-reduce | **AITER Custom All Reduce in INT8** | `VLLM_ROCM_USE_AITER_CUSTOM_AR=1`; no RCCL fallback in the recipe |
| Fused epilogue | **AR+RMSNorm+per-group INT8 quant-out** | `fuse_allreduce_rms=true`; feeds the next W8A8 consumer |
| GPU util | 0.86 (spec drafter and embedding dequant transient need headroom) | |
| TP / concurrency / graphs | **TP4, C8**, FULL_AND_PIECEWISE (forced FULL_DECODE_ONLY on gfx908) | `--tensor-parallel-size 4 --max-num-seqs 8` |

`AiterW8A16LinearKernel` is a compatibility name in the registry. For GS128
the implementation dispatches AITER A8W8 at every M. The launcher blocklists
`TritonW8A16LinearKernel` to prevent silently reverting to the old path.

systemd: `scripts/vllm-openai-gfx908-qwen38.service` (conflicts with the
retired qwen36 unit).

## Models

- Target: [`curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128`](https://huggingface.co/curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128), deployed at
  `~/models/Qwen3.8-27B-GPTQ-8bit-gs128` (30G)
- Drafter: [`curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128`](https://huggingface.co/curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128), deployed at
  `~/.cache/huggingface/dflash2-int8/Qwen3.8-27B-DFlash2-GPTQ-8bit`
  (true GPTQ int8 GS128; requires the post-bake remap below when rebuilt)

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

1. `PYTHONPATH="$PWD:$HOME/aiter" .venv/bin/python scripts/test_int8_kv_micro.py` — int8 PTH
   attention incl. non-causal (2.4e-4)
2. `HIP_VISIBLE_DEVICES=<idle> .venv/bin/python scripts/battery_gfx908.py`
   — prod shapes, FA interface, boot imports, gemm whitelist (4/4)
3. `scripts/kld_probe.py capture|compare` — real-artifact KLD gate
   (needs `NCCL_ALGO=Ring NCCL_PROTO=Simple` env; TP small when a server is up)
4. Coherence curl against the booted server
5. `.venv/bin/python scripts/ua_live_soak.py -n 500` — verify the exact
   published model pair and require AITER W8A8, AITER unified attention,
   AITER custom AR, fused INT8 quant-out, both INT8-PTH caches, INT8 Mamba,
   TP4, and C8 to remain enabled.

## Baseline status (2026-08-23)

The required int8 baseline is the two published GS128 models together with
AITER W8A8 INT8 GEMMs everywhere, INT8-PTH target and draft KV, INT8 Mamba,
AITER unified attention in INT8, AITER custom all-reduce in INT8, fused
AR+RMSNorm+per-group INT8 quant-out, DFlash2, TP4, and C8. Validation must
exercise this exact contract without a reduced fallback configuration.

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
