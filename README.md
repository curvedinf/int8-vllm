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

**Current production stack** (DFlash2 W8A8, NS=15, fp32 GDN state +
round act quant). Current numbers live in `docs/recipes/README.md`
"Current production status" — the one source of truth for this table;
they are not restated here.

TG rate = concurrency ÷ mean TPOT: pure decode throughput, prefill excluded.
That is the project's one throughput metric. "Whole-request output
throughput" (all output tokens ÷ total wall time incl. TTFT) is BANNED — it
mixes prefill and decode into one meaningless number and has poisoned
comparisons in this repo; do not compute or cite it.

Optimization history and the full component-level dtype table are NOT
restated here — `docs/recipes/README.md` (component table + current status)
and `INT8_AUDIT_RESULTS.md` (measurement program) are the sources of truth.
Highlights of the current stack only:

- Int8-native: W8A8 GEMMs (CK), int8 lm_head, int8-PTH KV (target + draft),
  int8 embedding, round-to-nearest per-token act quant.
- Measured float exceptions (never quantize without re-gating): GDN state
  stays fp32 (int8 corrupts with int8 KV; fp16 blew states to the fp16
  ceiling), selector codebooks bf16, draft ctx-KV projection bf16.
- TP4 / C8, DFlash2 NS=15, vLLM CUSTOM all-reduce.

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

The complete flag contract (env pins, dtypes, spec config, NS, levers) is
maintained in exactly one place: `docs/recipes/README.md`. The launcher
`scripts/serve_direwolf_qwen38.sh` is its executable form.

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
