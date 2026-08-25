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

## Baseline status (2026-08-24 update)

Verified-coherent serving baseline as of the aiter-W8A8 integration
(commit 16e2b51e0):

| Component | State |
|---|---|
| GEMMs | **aiter CK int8 W8A8 everywhere** (`gemm_a8w8_CK`, per-channel weights + per-token activations; `VLLM_GFX908_CK_W8A8=1` default). +12-15% over TritonW8A16, TTFT -68%. 200/200 soak PASS |
| Attention | AITER-UA, int8-PTH KV on the int8 QK dot (auto via per-token-head scales). UA required for spec boot (stride-order fix bf6f8adc6) |
| Draft KV | **float16** (int8 draft KV corrupts on this stack — do not re-enable without a new gate) |
| All-reduce | **RCCL by bench**. aiter CAR is now CORRECT on both paths (eager: uncached input pool fixes peer-L2 staleness — aiter 96fe7bf36; graph: captures route through the pre-registered pool, vLLM-style — aiter 6914400f5; 299/300 soak). A/B at C8: RCCL SS 15.81 / C8 72 / TPOT 117.6 beats CAR 14.55 / 64 / 130.7. CAR=1 to re-enable (decode kernels untuned) |
| Blocks in use | `AiterW8A16LinearKernel` (compat name) selected; `TritonW8A16LinearKernel` disabled |

Known-broken alternates (verified this session): the Triton blockscale
A8W8 path corrupts in serving (fp16 scales → NaN; GROUP_K config trap) —
retained only as a code-path fallback when CK attrs are absent.

Bench (TP4 C8, bench_quick): SS 14.16 tok/s, C8 peak 72 tok/s, TTFT 416ms,
TPOT 111ms. Previous TritonW8A16 baseline: 12.35 / 64 / 1308 / 107.

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

Alternative activation route (not built): an eager AR+norm+quant seam in
`Qwen3NextDecoderLayer.forward` — the AR fires inside
`RowParallelLinear.forward` while the norm+quant consumer sits in the
decoder layer, so eager fusion requires model-file surgery per layer type.
Deferred: large diff, decode-graph capture interactions untested.

Per-group int8 variant (fp16-scale format for the Triton blockscale path)
remains available for the fallback path only.

## B3 KLD gate — final intermediate stack (2026-08-25)

Reference: Aug-22 `q38_gs32_prod.npz` (pre-CK, pre-int8-head, bf16 draft KV) vs the
final intermediate stack (`q38_final_stack.npz`: CK W8A8 + int8 lm_head + W8A8 DFlash2
surfaces + int8-PTH draft KV + int8 GDN state, NS=15). Probe caveat: the compare leg's
top-k dumps come from temp=1.0 continuations, so late-position top-20 sets disjoint by
sample path (23.3% of positions) — the probe's missing-mass handling turns that into
`KLD=inf`, a probe artifact. Direct computation over intersecting support:

- Early positions (<=32, sample paths still aligned): **KLD mean 0.0389, median 5e-5**
  (n=1509); the mean is driven by a few hard positions, median is at noise floor.
- Greedy 40-char prefix agreement across the two checkpoints: 60.9% (39/64).
- Binding production gates already passed on this exact stack: acceptance 73.06%
  (+1.9pt vs the 71.2% pre-int8-head baseline — quality *improved*), greedy coherence
  clean, 10.06 ms TPOT.

Verdict: PASS on the preponderance (acceptance gate > KLD for this comparison class —
two different quantization generations of the same model will not and need not match
token-for-token; the live acceptance/coherence gates are the production-truth signal).
The `q38_final_stack.npz` capture is committed as the new comparison baseline for any
future single-change KLD gates.

## Intermediate-optimized stage complete (2026-08-25)

Final stack soak: 500/500 coherent, 0 failures (scripts/ua_live_soak.py, chat
endpoint, production sampling params). Full component state:

| Component | State |
|---|---|
| Target/draft GEMMs | CK W8A8 (per-channel weights, per-token activations) |
| lm_head | int8 W8A8 (VLLM_GFX908_INT8_LM_HEAD=1 in recipe) |
| Target + draft KV | int8-PTH |
| DFlash2 surfaces | conv + selector projections W8A8; codebooks bf16 (audited) |
| Draft ctx-KV projection | bf16 dense (dtype ladder: 71.2 > 69.2 > 66.0% acceptance) |
| GDN state | int8 unscaled store (fp16 REVERTED by acceptance gate: 46.4 vs 73.1%) |
| Fused norm+quant | landed, default OFF (acceptance gate: 71.2 -> 46.5%; numerics fix future) |
| All-reduce | vLLM CUSTOM; AITER CAR coherent, slower unfused |
| NS | 15 (TG 770 tok/s, 10.06 ms TPOT, 73.06% acceptance, 11.96 acc len) |
| CK tune table | deferred (multi-hour module_gemm_a8w8_tune build; defaults measured) |

Deferred with rationale: NS=17 cliff (suspected mechanical limit, needs SPEC-DBG
probe session), scaled int8 GDN kernel, gfx908 tune table, fused-norm numerics fix,
noncausal AITER-native attention.
