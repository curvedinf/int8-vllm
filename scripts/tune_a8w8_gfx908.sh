#!/usr/bin/env bash
# Tune aiter CK W8A8 GEMMs for the gfx908 production shapes and merge into
# the serving table (aiter/configs/a8w8_tuned_gemm.csv).
#
# The engine path (fp16 activations + fp16/bf16 out + fp32 per-channel
# scales -> gemm_a8w8_CK) reads this table; missing shapes fall back to a
# default CK config that runs at ~15% of HBM bandwidth at decode Ms
# (measured 2026-09-11: M=84 qkvz 0.345ms = 180 GB/s effective).
#
# Usage: tune_a8w8_gfx908.sh [gpu_id]   (default GPU 1; server must be down)
set -euo pipefail
GPU="${1:-1}"
AITER=/home/curved/aiter
PY=/home/curved/vllm-gfx908/.venv/bin/python
SHAPES=$(mktemp /tmp/untune_shapes.XXXXXX.csv)

cat > "$SHAPES" <<'EOF'
M,N,K,q_dtype_w
84,12032,5120,torch.int8
84,5120,5120,torch.int8
84,8704,5120,torch.int8
84,5120,4352,torch.int8
14,12032,5120,torch.int8
14,5120,5120,torch.int8
14,8704,5120,torch.int8
14,5120,4352,torch.int8
1,12032,5120,torch.int8
1,5120,5120,torch.int8
1,8704,5120,torch.int8
1,5120,4352,torch.int8
84,1536,5120,torch.int8
84,5120,1024,torch.int8
14,1536,5120,torch.int8
14,5120,1024,torch.int8
2048,12032,5120,torch.int8
2048,5120,5120,torch.int8
2048,8704,5120,torch.int8
2048,5120,4352,torch.int8
4096,1536,5120,torch.int8
4096,5120,1024,torch.int8
EOF

echo "tuning $(($(wc -l < "$SHAPES") - 1)) shapes on GPU $GPU (fp16 out / fp32 scales)"
cd "$AITER"
PYTHONPATH="$AITER" HIP_VISIBLE_DEVICES="$GPU" "$PY" \
  csrc/ck_gemm_a8w8/gemm_a8w8_tune.py \
  -i "$SHAPES" \
  -o "$AITER/aiter/configs/a8w8_tuned_gemm.csv" \
  -k --out_dtype fp16 --scale_dtype fp32 --mp 1 --warmup 5 --iters 50
rm -f "$SHAPES"
echo "done — new rows merged into a8w8_tuned_gemm.csv"
