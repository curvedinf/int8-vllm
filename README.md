<!-- markdownlint-disable MD001 MD041 -->
# vLLM for AMD Instinct MI100 (gfx908)

A high-performance fork of [vLLM](https://github.com/vllm-project/vllm) tuned
specifically for 4x AMD Instinct MI100 (gfx908 / CDNA1) over XGMI. MI100 gets
2x int8 rate vs any dtype other than FP16 (equal), at half the memory
bandwidth — so this branch treats **int8 as the native dtype** across the
whole model surface and exploits it everywhere it wins.

> **Optimized configuration:** [Qwen3.8-27B GPTQ INT8 W8A8 GS128](https://huggingface.co/curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128)
> + its matching [DFlash2 GPTQ INT8 W8A8 GS128 draft](https://huggingface.co/curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128)
> + AITER W8A8 INT8 GEMMs at every decode/prefill shape + AITER unified
> attention in INT8 + INT8 Mamba state + INT8 per-token-head target and draft
> KV + AITER custom all-reduce with fused AR+RMSNorm+per-group INT8 quant-out
> + TP4/C8 over XGMI. This complete pair and runtime contract is the sole
> production recipe; DFlash2 speculative decoding is always enabled.

## Performance

C8 serving benchmark — 8 concurrent requests, TP4, input 32 / output 1000,
`vllm bench serve` via `scripts/bench_quick.sh`. Cumulative optimization timeline
from `logs/c8_optimization/experiments.md`:

| Step | Config delta | Output tok/s | Δ |
|---|---|---:|---:|
| Baseline | fp8 KV, no MTP, MI300X GEMM configs | 112.35 | — |
| +1 | AITER MI100 small-M GEMM configs | 113.74 | +1.2% |
| +2 | MTP speculative decoding (k=2) | 117.63 | +3.4% |
| +3 | **int8 per-token-head KV cache** | **129.92** | **+10.4%** |

Kernel-level gains recorded during tuning: attention `waves_per_eu=1` +8.7%
output, decode `num_warps=4/num_stages=1` −6.2% TPOT, AITER lm_head GEMM
small-M config 1.74x vs rocBLAS.

## Int8 doctrine — what runs where

| Component | dtype | Notes |
|---|---|---|
| GEMMs | **AITER W8A8 INT8** | GPTQ 8-bit weights (uint8b128, group_size 128) and dynamically quantized INT8 activations at every decode and prefill M; W8A16 is not part of the target run |
| KV cache | **int8** | per-token-head quantized, f32 inline scales, block 32 |
| Attention | **AITER unified attention, int8** | Q quantized per-row in-kernel; K/V use the int8 per-token-head cache |
| Mamba/GDN state | **int8** | `--mamba-ssm-cache-dtype int8` |
| Attention P@V | fp16 | V dequantized with scale folded into P |
| DFlash2 drafter | **GPTQ int8 GS128** | matching published DFlash2 checkpoint; int8 per-token-head draft KV |
| Tensor parallel / concurrency | **TP4 / C8** | `--tensor-parallel-size 4 --max-num-seqs 8` |
| Collective epilogue | **AITER INT8** | AITER custom all-reduce plus fused AR+RMSNorm+per-group INT8 quant-out |

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
`VLLM_ROCM_USE_AITER_CUSTOM_AR=1`, and
`VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`; it blocks the fork-local Triton
mixed-precision selector so the GS128 weights use AITER W8A8 at every M. Key
flags are `--tensor-parallel-size 4 --max-num-seqs 8
--kv-cache-dtype int8_per_token_head --mamba-ssm-cache-dtype int8
--compilation-config '{"pass_config":{"fuse_allreduce_rms":true},...}'` and
`--speculative-config
'{"method":"dflash","model":"curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128","num_speculative_tokens":7,"kv_cache_dtype":"int8_per_token_head"}'`.
The top-level flag applies to the target; the nested field applies to the
draft, so both KV caches use int8 per-token-head. The production script has
no no-spec, RCCL, TRITON_ATTN, W8A16, or fusion-off mode.

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
