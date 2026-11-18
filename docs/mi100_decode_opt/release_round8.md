# Round-8 release — Qwen3.5-122B-A10B-GPTQ-4bit MoE config tune

## TL;DR

`btbtyler09/vllm-rocm-gfx908:v0.20.0rc1.dev` now ships a tuned MoE config
for **Qwen3.5-122B-A10B-GPTQ-4bit on TP=4** (E=256 routed experts,
intermediate/TP=128 packed → file lookup name N=256). Result:

| Scenario | Default heuristic | Round-8 B4 tune | Δ |
|---|---:|---:|---:|
| Single User Latency c=1 | 53.15 tok/s | **59.69 tok/s** | **+12.3%** |
| Decode Stress c=1 | 55.58 tok/s | **62.86 tok/s** | **+13.1%** |
| Concurrency Scaling c=32 | 294.98 tok/s | **389.67 tok/s** | **+32.1%** |
| Concurrency Scaling c=64 | 487.17 tok/s | 496.44 tok/s | +1.9% |
| Concurrency Scaling c=128 | 603.25 tok/s | 615.79 tok/s | +2.1% |

**Net positive for the typical MI100 deployment** (single-user / interactive
chat). Wins also at c=32 and ≥c=64.

## Tradeoff (read this if you serve at c=4–c=16)

The same explicit Triton metadata that wins at c=1 (waves_per_eu=2, kpack=1,
matrix_instr_nonkdim=16, num_warps=4, num_stages=2) regresses moderate-m
batches:

| Scenario | Default | Round-8 B4 | Δ |
|---|---:|---:|---:|
| Concurrency Scaling c=2 | 93.54 | 90.09 | -3.7% |
| Concurrency Scaling c=4 | 148.39 | 132.20 | -10.9% |
| Concurrency Scaling c=8 | 206.21 | 177.23 | -14.0% |
| Concurrency Scaling c=16 | 300.20 | 239.38 | **-20.3%** |

Why: at m=128 (c=16 with top_k=8), the default heuristic passes only tile
sizes to Triton, letting Triton's compile-time autotuner pick num_warps,
num_stages, kpack, matrix_instr_nonkdim. Our explicit values are tuned for
small-m and lock Triton out of the auto-selection that's better for big-m.

## Opt-out (disable the tune for batch serving)

If you serve 122B at c=4-16 and care about throughput more than per-user
latency, opt out one of two ways:

**Option 1**: point `VLLM_TUNED_CONFIG_FOLDER` to a directory without this
file:
```
mkdir -p /tmp/empty_moe_configs
docker run ... -e VLLM_TUNED_CONFIG_FOLDER=/tmp/empty_moe_configs \
  -v /tmp/empty_moe_configs:/tmp/empty_moe_configs:ro \
  btbtyler09/vllm-rocm-gfx908:v0.20.0rc1.dev vllm serve ...
```

**Option 2**: bind-mount over the file with an empty/absent path:
```
docker run ... -v /dev/null:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/E=256,N=256,device_name=Arcturus_GL-XL_[Instinct_MI100],dtype=int4_w4a16.json:ro
```
(vLLM falls back to default heuristic when the file fails to parse.)

## What else is in this image (cumulative shipped optimizations)

This image is the v0.20 baseline with all prior round wins baked in:

- Round-3 5h: custom-op CAR escape hatch routes captured all-reduce
  through gfx908's uncached buffer_ptrs (TP-layout-dependent, applies to
  all multi-GPU models)
- Round-3 5j: NCCL Tree+LL env defaults
- Round-4 Phase 2 E: Mori-pattern persistent-handle CAR (replaced the
  stage-1 CAR for captured paths)
- Round-4 Phase 3 B1': tuned MoE config for Qwen3.6-35B-A3B-GPTQ-8bit
  (E=256, N=128, int8_w8a16)
- Round-5: GPTQ q_gemm BLOCK_KN_SIZE=128→256 + launch_bounds
- Round-6: GPTQ q_gemm v_dot2c-conditional dispatch (m_count>=2 path)
- **Round-8 (this release)**: tuned MoE config for Qwen3.5-122B-A10B-GPTQ-4bit
  (E=256, N=256, int4_w4a16)

## Per-model performance reference (4×MI100, c=1)

| Model | TPOT | tok/s |
|---|---:|---:|
| Qwen3.6-35B-A3B-GPTQ-8bit (round-4) | 9.59 ms | 104 |
| Qwen3.5-122B-A10B-GPTQ-4bit (round-8 B4) | 15.98 ms | 60 |
| Qwen3.6-27B-GPTQ-8bit (round-6) | 16.26 ms | 53 |

122B leapfrogged 27B in c=1 throughput while running the largest model.

## Verification

Full 12-tier reports:
- Round-8 ship (B4): `~/mi100-llm-testing/Model_Reports/benchmark_Qwen3.5-122B-A10B-GPTQ-4bit_round8.md`
  (NOTE: the file in this repo's CI was actually the Phase-0 default config
  baseline — see the round-8 commit for context)
- Phase-0 default-config baseline (used as comparison point):
  `~/mi100-llm-testing/Model_Reports/benchmark_Qwen3.5-122B-A10B-GPTQ-4bit_round8_phase0_baseline.md`

The historical round-8 Docker build is preserved in commit `ec452b82ac`.
For the current MI100 image, see the [installation guide](../recipes/mi100-install.md).
