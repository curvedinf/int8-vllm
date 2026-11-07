#!/usr/bin/env bash
# Flint PTQR rung ladder (mirrors the Qwen R10S60 program).
# Usage: scripts/ptqr_flint_rungs.sh <rung> [extra trainer args...]
set -euo pipefail
cd "$(dirname "$0")/.."

RUNG="${1:?usage: ptqr_flint_rungs.sh <rung> [args...]}"
shift || true
MODEL=/home/curved/models/flint-bf16
DATA=/home/curved/SDGraft/data
OUT=/home/curved/models/flint-ptqr-r${RUNG}
STEPS=(60 120 200)
LRS=(1e-5 7e-6 4e-6)
IDX=$((RUNG - 1)); (( IDX < 0 )) && IDX=0

export ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=4

exec .venv/bin/python -m torch.distributed.run --nproc_per_node=4 \
  --master_port=2984${RUNG} scripts/ptqr_train_flint.py \
  --model "${MODEL}" --teacher_stream \
  --data_dir "${DATA}" --data_name flint_longctx/train_tokens.pt \
  --val_name flint_longctx/val_tokens.pt \
  --seq_len 2048 --batch_size 1 \
  --max_steps "${STEPS[$IDX]}" --lr "${LRS[$IDX]}" \
  --logits_chunk 128 --save_every 20 --eval_every 20 \
  --out_dir "${OUT}" "$@"
