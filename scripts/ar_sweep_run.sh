#!/usr/bin/env bash
# E2: sweep vLLM CUSTOM AR geometries on 4 GPUs. One combo at a time,
# 4 ranks launched concurrently, CSV on stdout.
set -uo pipefail
cd "${HOME}/vllm-gfx908"
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
export GPU_ARCHS=gfx908 PYTORCH_ROCM_ARCH=gfx908
export MASTER_PORT=29532

COMBOS=("512 16" "256 16" "512 32" "256 32" "512 8" "256 8" "512 36" "256 24")
for combo in "${COMBOS[@]}"; do
  set -- $combo
  PIDS=()
  for r in 0 1 2 3; do
    RANK=$r .venv/bin/python scripts/ar_sweep_child.py "$1" "$2" >> /tmp/ar_sweep.csv 2>/dev/null &
    PIDS+=($!)
  done
  wait "${PIDS[@]}" 2>/dev/null
done
echo "DONE rows=$(grep -c ',' /tmp/ar_sweep.csv)"
