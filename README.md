<!-- markdownlint-disable MD001 MD041 -->
# vLLM for AMD Instinct MI100 (gfx908)

A high-performance fork of [vLLM](https://github.com/vllm-project/vllm) tuned
specifically for 4x AMD Instinct MI100 (gfx908 / CDNA1) over XGMI. MI100 gets
2x int8 rate vs any dtype other than FP16 (equal), at half the memory
bandwidth — so this branch treats **int8 as the native dtype** across the
whole model surface and exploits it everywhere it wins.

> **Optimized configuration:** [Qwen3.8-27B GPTQ INT8 W8A8 GS128](https://huggingface.co/curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128)
> + its matching [DFlash2 GPTQ INT8 W8A8 GS128 draft](https://huggingface.co/curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128)
> + AITER CK W8A8 INT8 GEMMs at every decode/prefill shape + INT8 lm_head
> + AITER unified attention in INT8 + INT8 per-token-head target and draft
> KV + DFlash2 speculative decoding at NS=15 + TP4/C8 over XGMI. This
> complete pair and runtime contract is the sole production recipe;
> DFlash2 speculative decoding is always enabled.

## Performance

C8 serving benchmark — 8 concurrent requests, TP4, input 32 / output 1000,
greedy, `vllm bench serve` fresh-boot legs. TPOT-derived TG rate is the
primary decode metric (8 streams ÷ mean TPOT); acceptance from the server's
spec-decode counters. Current stack measured 2026-08-24/25.

**Current production stack** (DFlash2 W8A8, NS=15, int8 lm_head + draft KV):

| Metric | Value |
|---|---|
| TG rate (decode-only, 8 streams) | **~770 tok/s** |
| Mean / median TPOT | 10.06 / 9.61 ms |
| Draft acceptance | 73.1% (11.96 accepted of 15) |
| TTFT (32-token input) | ~440 ms |

Cumulative timeline (whole-request output tok/s unless noted; pre-CK-era
steps from `logs/c8_optimization/experiments.md`, W8A8-era from the
2026-08-24/25 gate benches):

| Step | Config delta | Output tok/s | Notes |
|---|---|---:|---|
| Baseline | fp8 KV, no spec, MI300X GEMM configs | 112.35 | |
| +1 | AITER MI100 small-M GEMM configs | 113.74 | +1.2% |
| +2 | MTP k=2 | 117.63 | +3.4% |
| +3 | int8-PTH KV cache | 129.92 | +10.4% |
| +4 | AITER CK W8A8 GEMMs (vs valid W8A16) | 63.49 median | audit-era 3-repeat leg; +31.8% vs 48.17 |
| +5 | **DFlash2 bf16-dtype fix** (was zero-acceptance) | 226.9 | 3.6× — the fp16-overflow root cause |
| +6 | lm_head + conv/selector W8A8, NS=15 | 331.9 whole-req / **770 TG** | acceptance 65.6→76.3% through the sweep |
| +7 | draft KV int8-PTH (UA noncausal fix) | final stack | 73.1% acc, 10.06 ms TPOT, 500/500 soak |

Reference points: native MTP2 one-run (audit): 252.2 whole-request tok/s /
28.18 ms TPOT — DFlash2 at NS=15 beats it 2.6× on TPOT. No-spec
historical: ~128 tok/s. MTP2's TPOT-derived TG ≈ 284 tok/s vs DFlash2's 770.

Kernel-level gains recorded during tuning: attention `waves_per_eu=1` +8.7%
output, decode `num_warps=4/num_stages=1` −6.2% TPOT, AITER lm_head GEMM
small-M config 1.74x vs rocBLAS.

## Int8 doctrine — what runs where

Measured as of 2026-08-24 (INT8_AUDIT_RESULTS.md reconciled). The recipe is
int8-native everywhere the acceptance gate passes; every exception below is a
measured decision, not an aspiration.

| Component | dtype | Notes |
|---|---|---|
| Target GEMMs | **W8A8 int8 (CK)** | per-channel weight requant at load + per-token activation quant; `gemm_a8w8_CK` |
| lm_head | **W8A8 int8 (CK)** | `VLLM_GFX908_INT8_LM_HEAD=1`; gated 65.6→65.6% acceptance (neutral), halves head memory |
| KV cache (target) | **int8 per-token-head** | f32 inline scales, block 32 |
| Draft KV | **int8 per-token-head** | after the UA noncausal `kv_quant_mode` fix; acceptance 76.3→79.7% vs bf16 |
| DFlash2 drafter | **GPTQ int8 GS128 → CK W8A8** | conv `kernel_projection` + selector `hidden_projection` W8A8 (`VLLM_GFX908_DF2_W8A8`) |
| Attention P@V | fp16 | V dequantized with scale folded into P |
| Selector codebooks | bf16 | audited exception: candidate-row gathers, ranking-sensitive |
| Draft ctx-KV projection | **bf16 dense** | dtype ladder measured: bf16 71.2% > fp16 69.2% > int8 66.0% acceptance |
| Mamba/GDN recurrent state | **int8 (unscaled store)** | REVERTED from fp16 by acceptance gate (46.4% -> 73.1%): checkpoints distilled/served with this store; fp16 shifts recurrent dynamics and breaks draft agreement. A properly scaled int8 kernel remains future work |
| All-reduce | vLLM CUSTOM (fp16 payload) | AITER CAR is coherent but 8.8% slower unfused; fused epilogue blocked by Inductor corruption |
| Tensor parallel / concurrency | **TP4 / C8** | `--tensor-parallel-size 4 --max-num-seqs 8` |

Both target and DFlash2 draft use `int8_per_token_head` KV. The draft setting
is explicit inside the speculative-config JSON so it cannot silently inherit
a different target setting.

## Dependencies

All three repos are maintained as sibling forks, synced to upstream:

| Repo | Branch | Base |
|---|---|---|
| `<your-org>/vllm-gfx908` | `mi100-optimized` (prod), `mi100-optimized-sync` (this update) | vllm-project/vllm main @ 2026-08 |
| `<your-org>/aiter-gfx908` | `mi100-optimized-sync` (`e0b64a642`) | ROCm/aiter main @ 2026-08; carries int8 unified-attention + gfx908 tunings |
| `<your-org>/flash-attention` | `gfx908-sync` (`d279da4`) | Dao-AILab/flash-attention main @ 2026-08; `third_party/aiter` submodule → the aiter fork above |

The aiter checkout is consumed at runtime via `PYTHONPATH=~/aiter`;
flash-attention 2.8.4 (pure-python, Triton AMD backend from the aiter fork) is
installed in the serving venv. The CK FA backend does not apply to gfx908
(uses gfx90a+ ISA) and is not built.

Fork-specific code that matters: the compatibility-named
`AiterW8A16LinearKernel` selects AITER A8W8 for every GS128 shape
(`vllm/model_executor/kernels/linear/mixed_precision/aiter_w8a16.py`), gfx908
attention tuning
(TILE_SIZE=32 decode, adaptive flash-decoding split-K) in
`vllm/v1/attention/ops/triton_unified_attention.py`, gfx908 env defaults in
`vllm/platforms/rocm.py`, and the int8 per-token-head port in aiter's
`unified_attention`.

## Models

- **[Qwen3.8-27B-GPTQ-INT8-W8A8-GS128](https://huggingface.co/curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128)**
  (matching target model; local deployment: `~/models/Qwen3.8-27B-GPTQ-8bit-gs128`)
- **[Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128](https://huggingface.co/curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128)**
  (required speculative-decoding companion; not standalone; local deployment:
  `~/.cache/huggingface/dflash2-int8/Qwen3.8-27B-DFlash2-GPTQ-8bit`)
- Quantization recipe: `~/models/quantize_qwen38_27b_gptq8.py`
  (GPTQModel 7.3.4; bits=8, group_size=128, sym, true-sequential; 512 mixed
  code+C4 calibration samples binned 256–2048)

## Serving

```bash
# The production script is fixed to the complete AITER INT8 TP4/C8 contract.
scripts/serve_direwolf_qwen38.sh {start|stop|restart|status}

# systemd
sudo cp scripts/vllm-openai-gfx908-qwen38.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now vllm-openai-gfx908-qwen38
```

The launcher fixes `VLLM_ROCM_USE_AITER=1`,
`VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`, and
`VLLM_GFX908_INT8_LM_HEAD=1`; it blocks the fork-local Triton
mixed-precision selector so the GS128 weights use AITER W8A8 at every M.
The all-reduce is vLLM `CUSTOM` (`CAR=0 AR=1`) — measured fastest
sustained; AITER CAR remains available via `CAR=1` for tuning. Key flags
are `--tensor-parallel-size 4 --max-num-seqs 8
--kv-cache-dtype int8_per_token_head --mamba-ssm-cache-dtype float16` and
`--speculative-config
'{"method":"dflash","num_speculative_tokens":15,"kv_cache_dtype":"int8_per_token_head"}'`
(NS=15 measured best TG; NS=17 collapses — see INT8_AUDIT_RESULTS.md).
The draft dtype resolves from its checkpoint (bf16) — forcing the target's
fp16 overflows the draft residual stream (fixed 2026-08-24). The
production script has no no-spec, RCCL, TRITON_ATTN, or W8A16 mode.

## Hardware / software

| Component | Value |
|---|---|
| GPU | 4x AMD Instinct MI100 32GB (gfx908), XGMI full mesh |
| ROCm | 7.x (`/opt/rocm`) |
| PyTorch | 2.11.0+rocm7.1 (serving venv) / 2.13.0+rocm7.2 (quantization venv) |
| Triton | 3.6.0 |
| Build | `PYTORCH_ROCM_ARCH=gfx908 VLLM_TARGET_DEVICE=rocm`, see `docs/contributing/incremental_build.md` |

## Testing

```bash
# int8 KV micro-correctness (AITER unified-attention path)
PYTHONPATH="$PWD:$HOME/aiter" .venv/bin/python scripts/test_int8_kv_micro.py

# post-sync battery: int8 KV at production shapes (hdim 256, GQA 6:1),
# FA 2.8.4 interface, boot-import chain, GEMM whitelist
HIP_VISIBLE_DEVICES=1 .venv/bin/python scripts/battery_gfx908.py
```

Optimization history with full A/B data lives in
`logs/c8_optimization/experiments.md`; per-experiment server logs in
`logs/`.
