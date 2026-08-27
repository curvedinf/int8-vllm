#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/curved/vllm-gfx908
OUT=/home/curved/models/kld/dflash_audit
BF16=/home/curved/models/dflash2-bf16-with-tokenizer
INT8=/home/curved/.cache/huggingface/dflash2-int8/Qwen3.8-27B-DFlash2-GPTQ-8bit
KEY="$(tr -d '\r\n' </etc/llama/llama-api.key)"

systemctl stop vllm-openai-gfx908-qwen38.service
install -d -o curved -g curved "$OUT"

stop_server() {
  runuser -u curved -- env HOME=/home/curved "$ROOT/scripts/serve_recipe_qwen38.sh" stop || true
}
trap stop_server EXIT

run_leg() {
  local tag=$1 draft=$2 head=$3 surfaces=$4 kv=$5
  rm -rf "$OUT/$tag"
  install -d -o curved -g curved "$OUT/$tag"
  runuser -u curved -- env \
    HOME=/home/curved \
    VLLM_API_KEY="$KEY" \
    VLLM_QUANT_AUDIT="$OUT/$tag" VLLM_QUANT_AUDIT_MAX=3 \
    VLLM_QUANT_AUDIT_DFLASH_ONLY=1 \
    VLLM_DFLASH_AUDIT=1 VLLM_DFLASH_DRAFT_EAGER=1 \
    DRAFT_MODEL_DIR="$draft" NS=7 CMODE=0 CGMODE=NONE LOGSTATS=1 \
    INT8HEAD="$head" DF2W8A8="$surfaces" DRAFT_KV_DTYPE="$kv" \
    "$ROOT/scripts/serve_recipe_qwen38.sh" start
  runuser -u curved -- env \
    HOME=/home/curved DFLASH_KEY="$KEY" DFLASH_AUDIT_DIR="$OUT" \
    "$ROOT/.venv/bin/python" "$ROOT/scripts/dflash_audit.py" capture --tag "$tag"
  stop_server
}

run_leg bf16_auto_kv "$BF16" 1 1 auto
run_leg bf16_int8_kv "$BF16" 1 1 int8_per_token_head
run_leg int8_prod_sync "$INT8" 1 1 int8_per_token_head
run_leg bf16_dense_head "$BF16" 0 1 int8_per_token_head

runuser -u curved -- env HOME=/home/curved DFLASH_AUDIT_DIR="$OUT" \
  "$ROOT/.venv/bin/python" "$ROOT/scripts/dflash_audit.py" compare \
  --reference bf16_auto_kv --test bf16_int8_kv
runuser -u curved -- env HOME=/home/curved DFLASH_AUDIT_DIR="$OUT" \
  "$ROOT/.venv/bin/python" "$ROOT/scripts/dflash_audit.py" compare \
  --reference bf16_int8_kv --test int8_prod_sync

chown -R curved:curved "$OUT"
trap - EXIT
stop_server
