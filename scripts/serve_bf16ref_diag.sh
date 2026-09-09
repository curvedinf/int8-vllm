#!/usr/bin/env bash
# Diagnostic bf16-reference server (NOT the recipe): serves the unquantized
# reference checkpoint with the LAYERPROBE env passed the same way the recipe
# script passes env to workers (env-block on the launch line).
set -euo pipefail
LOG_DIR="${HOME}/vllm-gfx908/logs/serve_recipe_qwen38"
PROBE_DIR="$(cat "${LOG_DIR}/LAYERPROBE" 2>/dev/null | tr -d '[:space:]')"
NCCL_ALGO=Ring NCCL_PROTO=Simple \
VLLM_LAYERPROBE="${PROBE_DIR}" \
"${HOME}/vllm-gfx908/.venv/bin/vllm" serve \
  "${HOME}/models/Qwen3.8-27B-bf16-ref" \
  --served-model-name qwen3.8-27b-bf16ref \
  --host 127.0.0.1 --port 8021 --tensor-parallel-size 4 \
  --dtype bfloat16 --max-model-len 262144 --max-num-seqs 6 \
  --gpu-memory-utilization 0.92 --disable-log-stats \
  --max-num-batched-tokens 2048 --api-key bf16ref-test
