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
VLLM_ROCM_USE_AITER_CUSTOM_AR=0 \
VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
VLLM_GFX908_ACT_QUANT=round \
VLLM_DISABLED_KERNELS=TritonW8A16LinearKernel \
.venv/bin/vllm serve "$TARGET_MODEL" \
  --tensor-parallel-size 4 \
  --max-num-seqs 8 \
  --dtype half \
  --kv-cache-dtype int8_per_token_head \
  --mamba-ssm-cache-dtype float32 \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","pass_config":{"fuse_allreduce_rms":false}}' \
  --speculative-config '{"method":"dflash","model":"'"$DRAFT_MODEL"'","num_speculative_tokens":15,"kv_cache_dtype":"int8_per_token_head"}'
```

The top-level KV flag configures the target. The `kv_cache_dtype` inside the
speculative JSON independently configures the DFlash2 draft; both are
deliberately `int8_per_token_head`.

The script encodes the full intended feature set:

| Feature | Value | Why it's critical |
|---|---|---|
| Checkpoint | [`curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128`](https://huggingface.co/curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128), deployed at `~/models/Qwen3.8-27B-GPTQ-8bit-gs128` | GPTQ int8, gs=128 (enables W8A8) |
| GEMMs | **AITER W8A8 INT8 everywhere**, decode and prefill | no W8A16 or fork-local Triton GEMM in the target run |
| KV cache | `int8_per_token_head` (fp32 inline scales, block 32) | half KV bandwidth; replay-measured 0.85% per token-head — normal int8 SNR |
| Mamba/GDN state | `float32` (`--mamba-ssm-cache-dtype float32`) | REQUIRED: the fp16 state round-trip broke delta-rule cancellation and blew states to 63k (4% under fp16 ceiling) — the KLD-tail generator; int8 state corrupts in the int8-KV combo (bisect 2026-08-25); fp32 is the checkpoint's own declared dtype |
| Act quantizer | round-to-nearest (`VLLM_GFX908_ACT_QUANT=round`, fused Triton kernel) | halved the dominant 10-15% act-quant error leg; fixed 10x first-token-stop inflation (empty responses); also faster than the 4-pass eager aiter chain |
| Embedding lookup | int8 gather | half embedding bandwidth; it is not a GEMM exception |
| Speculative decoding | **DFlash2, ns=15, int8 drafter, int8 draft KV — ON, non-negotiable** | NS=15 per the 2026-08-24/25 sweep (NS=17 collapses: 29.7% acceptance) |
| Attention backend | **AITER unified attention in INT8** | target and draft both use INT8-PTH KV |
| All-reduce | **vLLM CUSTOM all-reduce** (`VLLM_ROCM_USE_AITER_CUSTOM_AR=0`) | audited TP4/C8 rerun: vLLM CUSTOM 63.49 tok/s beats AITER CAR 58.34 and PYNCCL 53.04; AITER CAR gfx908 forces the naive kernel until tuned — CAR stays a tuning lever, not the default |
| Fused epilogue | OFF (`fuse_allreduce_rms=false`) | the fused INT8 epilogue path is implemented but inactive; enable only after the gfx908 graph integration work lands |
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
3. `scripts/kld_probe_v2.py` — real-artifact KLD gate; the boot-gate
   harness `scripts/kld_gate_boot.sh <tag>` wraps the boot + probe cycle
4. Coherence curl against the booted server
5. `.venv/bin/python scripts/ua_live_soak.py -n 500` — verify the exact
   published model pair and require AITER W8A8, AITER unified attention,
   vLLM CUSTOM all-reduce (CAR=0), fused epilogue OFF, both INT8-PTH
   caches, float32 Mamba state, TP4, C8, and DFlash2 NS=15 to remain
   enabled.

## AR+RMS+per-token-int8-quant fused epilogue (2026-08-24, status: kernel DONE, production enable BLOCKED)

**aiter `ec90fc933`**: `fused_ar_rms_int8_per_token_quant` — AR + residual +
RMSNorm + per-token int8 quant producing exactly the CK W8A8 activation
format (int8 [M,K] + fp32 [M,1] scales, `pertoken_quant` numerics with
exact-rsqrts and trunc-toward-zero payload). 2-rank suite
(`op_tests/multigpu_tests/test_fused_ar_rms_int8_quant.py --suite
per_token`): payload **bit-exact** vs the unfused chain at every production
shape; scales within reduction-order ULP ties only (≤3.5e-07 rel).

**vLLM `89e808d3e`**: custom ops (`rocm_aiter_pertoken_quant_int8`,
`rocm_aiter_fused_allreduce_rmsnorm_quant_per_token_int8`) + AR-fusion
patterns registered ahead of the FP8 variants, gated on
`supports_per_token_int8_quant`. gfx908 capture uses the pool-staging path
(`registered=False` under capture), per the graph-coherence fix.

**Why it is NOT enabled in production**: the fusion runs only under
torch.compile, and this session bisected **torch.compile itself as an
output corruptor** on this stack (greedy smoke: factual self-contradictions,
degenerate repetition) with CAR off, ARFUSE off, RCCL — i.e. independent of
everything new. That is why the platform hook (rocm.py:1052) disables
compile on gfx908; the July cache predates the current stack. Boot blockers
on the compile path were fixed and landed (`199db5876`: CK gemm custom op —
aiter-JIT config lookup was dynamo-skip-marked; `input_guard` device_index
nullcontext) so the day Inductor-on-gfx908 is fixed, the fusion activates
with zero further work (set `VLLM_MI100_TORCH_COMPILE=1 CAR=1 ARFUSE=true
CGMODE=FULL_DECODE_ONLY`; PIECEWISE graphs still hang at TP>1).

Eager activation route (built, experiment E3): an eager AR+norm+quant seam
now exists behind `VLLM_GFX908_EAGER_EPILOGUE=1`. Verdict: neutral on TPOT
and prefill-only by design, so production keeps the epilogue OFF
(`fuse_allreduce_rms=false`).

Per-group int8 variant (fp16-scale format for the Triton blockscale path)
remains available for the fallback path only.

## Current production status (2026-08-25, fp32-state + round-quant stack)

THE canonical component/accuracy/perf state. Other docs (README.md,
AGENTS.md, INT8_AUDIT_RESULTS.md) link here; they do not restate this table.

| Component | State |
|---|---|
| Target/draft GEMMs | CK W8A8 (per-channel weights, per-token activations) |
| Act quantizer | round-to-nearest fused Triton kernel (`VLLM_GFX908_ACT_QUANT=round`, default; aiter trunc via `=aiter`) |
| lm_head | int8 W8A8 (`VLLM_GFX908_INT8_LM_HEAD=1`) |
| Target + draft KV | int8-PTH both |
| GDN state | **float32** — REQUIRED (int8 corrupts with int8 KV; fp16 round-trip blew states to 63k = KLD-tail generator; fp32 = checkpoint-declared dtype) |
| DFlash2 surfaces | conv + selector projections W8A8; codebooks bf16 (audited) |
| Draft ctx-KV projection | bf16 dense (dtype ladder: 71.2 > 69.2 > 66.0% acceptance) |
| Fused norm+quant | landed, default OFF (acceptance gate 71.2 -> 46.5%; numerics fix future) |
| All-reduce | vLLM CUSTOM (63.49 tok/s beats AITER CAR 58.34, PYNCCL 53.04 at TP4/C8) |
| NS | 15 |

Accuracy (52-prompt KLD gate vs BF16 ref; method in INT8_AUDIT_RESULTS.md):
median 0.0153, greedy agreement 38/52, first-token stop prob at BF16 parity
(0.38% vs 0.36%). Perf (TP4/C8, fast-regime boots): TPOT 12.52 ms, TTFT
~219 ms. NUMBER DEFINITIONS (use these exactly; the project bans
whole-request mixing):
- **Steady-state TG rate** = concurrency / mean TPOT = 8 / 12.52 ms =
  **639 tok/s** — the decode-only figure while all 8 streams are past TTFT.
  This is the project's primary throughput metric.
- **Wall-clock output throughput** (what `vllm bench serve` prints as
  "Output token throughput": total tokens / total wall time incl. the
  staggered finish as requests hit EOS) = **348 tok/s** on the same boot.
  Lower by construction (concurrency decays after the first finisher);
  comparable only against other wall-clock numbers.
Older numbers elsewhere (554 tok/s / 14.44 ms / 770 tok/s regimes) predate
the corruption bisect and the accuracy program — superseded; the audit
doc's history section explains why.
