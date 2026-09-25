#!/usr/bin/env bash
# Interleaved-boot paired A/B harness (GOALOPT).
# Alternates baseline/candidate boots in tight pairs (AB then BA) so host
# drift hits both arms equally; each leg = 1 short-C6 bench + 1 warm-prefix
# 32k/800 steady leg. Writes per-leg metrics to a CSV for paired stats.
#
# Usage: ab_interleaved.sh <pairs> <tag> <result_csv>
# Candidate env: AITER_CONFIG_GEMM_A8W8=.../a8w8_tuned_gemm_msmall.csv
set -u
ROOT=/home/curved/vllm-gfx908
PAIRS="${1:-3}"
TAG="${2:-abgemm}"
CSV="${3:-$ROOT/logs/goal_opt/${TAG}_pairs.csv}"
KEY="$(cat /tmp/q38_apikey)"
CAND_CSV=${CAND_CSV:-/home/curved/aiter/aiter/configs/a8w8_tuned_gemm_msmall.csv}

echo "arm,pair,tok_s,tpot_ms,steady_agg,steady_overlap" > "$CSV"

boot_leg () {
  local arm="$1" pair="$2" extra_env="$3"
  "$ROOT/scripts/serve_recipe_qwen38.sh" stop >/dev/null 2>&1
  sleep 5
  if [ -n "$extra_env" ]; then
    env VLLM_API_KEY="$KEY" $extra_env "$ROOT/scripts/serve_recipe_qwen38.sh" start >/dev/null 2>&1
  else
    env VLLM_API_KEY="$KEY" "$ROOT/scripts/serve_recipe_qwen38.sh" start >/dev/null 2>&1
  fi
  local i
  for i in $(seq 1 90); do
    curl -fsS --max-time 2 http://127.0.0.1:8020/health >/dev/null 2>&1 && break
    sleep 10
  done
  curl -fsS --max-time 2 http://127.0.0.1:8020/health >/dev/null 2>&1 || { echo "$arm,$pair,BOOT_FAIL,,," >> "$CSV"; return; }
  export OPENAI_API_KEY="$KEY" VLLM_API_KEY="$KEY"
  local bench="$ROOT/logs/goal_opt/${TAG}_${arm}_p${pair}.log"
  "$ROOT/.venv/bin/vllm" bench serve \
    --base-url http://127.0.0.1:8020 --model qwen3.8-27b-gptq8 \
    --tokenizer /home/curved/models/Qwen3.8-27B-PTQR-R10S60 \
    --dataset-name random --num-prompts 32 --max-concurrency 6 \
    --random-input-len 32 --random-output-len 1000 \
    --endpoint /v1/completions --skip-chat-template --temperature 0 \
    --num-warmups 2 > "$bench" 2>&1
  local tok tpot
  tok=$(grep -a 'Output token throughput' "$bench" | grep -o '[0-9.]*')
  tpot=$(grep -a 'Mean TPOT' "$bench" | grep -o '[0-9.]*')
  local st="$ROOT/logs/goal_opt/${TAG}_${arm}_p${pair}_steady.log"
  ( cd "$ROOT/scripts" && STEADY_CTX=32000 STEADY_STREAM=1 \
    STEADY_SEED_BASE=$((900000 + pair * 100 + 7)) \
    ../.venv/bin/python c6_steady.py 6 800 > "$st" 2>&1 )
  local agg ovl
  agg=$(grep -a AGGREGATE "$st" | grep -o '[0-9.]*' | head -1)
  ovl=$(grep -a 'decode overlap' "$st" | grep -o '= [0-9.]*' | grep -o '[0-9.]*')
  echo "$arm,$pair,$tok,$tpot,$agg,$ovl" >> "$CSV"
  echo "$(date +%T) $arm p$pair: tok=$tok tpot=$tpot agg=$agg ovl=$ovl"
}

for p in $(seq 1 "$PAIRS"); do
  if [ $((p % 2)) -eq 1 ]; then order="base cand"; else order="cand base"; fi
  for arm in $order; do
    if [ "$arm" = "cand" ]; then
      boot_leg cand "$p" "AITER_CONFIG_GEMM_A8W8=$CAND_CSV"
    else
      boot_leg base "$p" ""
    fi
  done
done
"$ROOT/scripts/serve_recipe_qwen38.sh" stop >/dev/null 2>&1
echo "PAIRS_DONE $CSV"
