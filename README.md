<!-- markdownlint-disable MD001 MD041 -->
# vLLM for AMD Instinct MI100 (gfx908)

A high-performance fork of [vLLM](https://github.com/vllm-project/vllm) tuned
specifically for 4x AMD Instinct MI100 (gfx908 / CDNA1) over XGMI. MI100 gets
2x int8 rate vs any dtype other than FP16 (equal), at half the memory
bandwidth — so this branch treats **int8 as the native dtype** across the
whole model surface and exploits it everywhere it wins.

> **Optimized configuration:** Qwen3.8-27B (GPTQ int8) + DFlash 2 speculative
> decoding + FlashAttention 2 (Triton) + int8 per-token-head KV cache + TP4
> with XGMI custom all-reduce. Most other combinations (other models, TP1/TP2,
> fp16/bf16 KV, no speculation, MTP) work but are not tuned; only the
> configuration above is optimized.

## Performance

C8 serving benchmark — 8 concurrent requests, TP4, input 32 / output 1000,
`vllm bench serve` via `scripts/bench_c8.py`. Cumulative optimization timeline
from `logs/c8_optimization/experiments.md`:

| Step | Config delta | Output tok/s | Δ |
|---|---|---:|---:|
| Baseline | fp8 KV, no MTP, MI300X GEMM configs | 112.35 | — |
| +1 | AITER MI100 small-M GEMM configs | 113.74 | +1.2% |
| +2 | MTP speculative decoding (k=2) | 117.63 | +3.4% |
| +3 | **int8 per-token-head KV cache** | **129.92** | **+10.4%** |

Long-context 20:1 prompt:generation (input 5000 / output 250, MTP-2, int8 KV,
block 32), same ledger:

| Attention path | Output tok/s | TPOT |
|---|---:|---:|
| TRITON_ATTN (production default) | 28.09 | 166.9 ms |
| AITER unified-attn int8 (experimental track, full tuning) | 54.81 | 75.7 ms |

AITER-UA int8 is the faster kernel path but is disabled in production
(`VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0`) pending a state-corruption
investigation after ~200 requests; TRITON_ATTN is the stable default.

Kernel-level gains recorded during tuning: attention `waves_per_eu=1` +8.7%
output, decode `num_warps=4/num_stages=1` −6.2% TPOT, AITER lm_head GEMM
small-M config 1.74x vs rocBLAS.

### Qwen3.8-27B + DFlash 2 (this branch's target config)

| Workload | Output tok/s | Notes |
|---|---:|---|
| C8 (8 conc, in 32 / out 1000) | *pending* | post-upstream-sync re-measure |
| Acceptance length (DFlash2, k=7) | *pending* | vs MTP-2 ≈ 2.0 |
| 20:1 PP:TG (in 5000 / out 250) | *pending* | |

*(Rows will be filled from `scripts/bench_c8.py` A/B vs the Qwen3.6 service.)*

## Int8 doctrine — what runs where

| Component | dtype | Notes |
|---|---|---|
| GEMM weights | **int8** | GPTQ 8-bit (uint8b128, group_size 32), Triton W8A16 kernel |
| KV cache | **int8** | per-token-head quantized, f32 inline scales, block 32 |
| Attention Q@K dot (AITER-UA path) | **int8** | Q quantized per-row in-kernel; K read from int8 cache |
| GEMM activations | fp16 | W8A8 tried and rejected for gs=32 (2x slower; see ledger 2026-07-02) |
| Attention P@V | fp16 | V dequantized with scale folded into P |
| DFlash2 drafter | unquantized (fp16 runtime) | quantized drafters are broken upstream (vllm#51581) |

Sanctioned exception: the DFlash2 draft model stays unquantized; everything
on the target path remains int8-native.

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

Fork-specific code that matters: Triton W8A16 kernel + repacked HIP GEMM
(`vllm/model_executor/kernels/linear/mixed_precision/triton_w8a16.py`,
`csrc/libtorch_stable/quantization/gptq/q_gemm.cu`), gfx908 attention tuning
(TILE_SIZE=32 decode, adaptive flash-decoding split-K) in
`vllm/v1/attention/ops/triton_unified_attention.py`, gfx908 env defaults in
`vllm/platforms/rocm.py`, and the int8 per-token-head port in aiter's
`unified_attention`.

## Models

- **Qwen3.8-27B-GPTQ-8bit** (int8, this fork's target model):
  **[HF upload — coming soon]** <!-- TODO: replace with the HF model URL after upload -->
- Qwen3.6-27B-GPTQ-8bit-MTP2 (previous production model): local
  `~/models/Qwen3.6-27B-GPTQ-8bit-MTP2`
- DFlash 2 drafter: `z-lab/Qwen3.8-27B-DFlash2` (BF16, ungated)
- Quantization recipe: `~/models/quantize_qwen38_27b_gptq8.py`
  (GPTQModel 7.3.4; bits=8, group_size=32, sym, true-sequential; 512 mixed
  code+C4 calibration samples binned 256–2048)

## Serving

```bash
# Qwen3.8-27B int8 + DFlash2 (target config)
scripts/serve_direwolf_qwen38.sh {start|stop|restart|status}

# systemd
sudo cp scripts/vllm-openai-gfx908-qwen38.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now vllm-openai-gfx908-qwen38
```

Key server flags: `--tensor-parallel-size 4 --dtype half --attention-backend
TRITON_ATTN --kv-cache-dtype int8_per_token_head --speculative-config
'{"method":"dflash2","num_speculative_tokens":7,"kv_cache_dtype":"fp16"}'`
(draft KV stays fp16 — it is non-causal and unquantized; the target KV cache
is int8 per-token-head).

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
PYTHONPATH=~/aiter .venv/bin/python test_int8_kv_micro.py

# post-sync battery: int8 KV at production shapes (hdim 256, GQA 6:1),
# FA 2.8.4 interface, boot-import chain, GEMM whitelist
HIP_VISIBLE_DEVICES=1 .venv/bin/python scripts/battery_gfx908.py
```

Optimization history with full A/B data lives in
`logs/c8_optimization/experiments.md`; per-experiment server logs in
`logs/`.
