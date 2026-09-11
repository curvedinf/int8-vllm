#!/usr/bin/env bash
# Post-tune validation chain: verify the tuned W8A8 configs end-to-end.
# 1) microbench M=84 GEMM (expect << 38.7ms/36-layer step)
# 2) boot the production server (JIT reads the tuned table at first use)
# 3) steady C6/2k + C6/20k + ramp C6 tests vs banked 126 / 24-31 / 94.8
set -u
cd /home/curved/vllm-gfx908
K="$(cat /tmp/.vk)"
export VLLM_API_KEY="$K"

echo "=== 1) GEMM microbench (tuned)"
HIP_VISIBLE_DEVICES=1 .venv/bin/python /tmp/bench_gemm_m84.py 2>&1 | tail -3

echo "=== 2) boot server"
for i in 1 2 3 4 5 6; do
  echo "--- boot attempt $i $(date +%H:%M:%S)"
  scripts/serve_recipe_qwen38.sh start >/dev/null 2>&1
  ok=0
  for j in $(seq 1 30); do
    sleep 20
    if curl -sf -H "Authorization: Bearer $K" http://127.0.0.1:8020/v1/models >/dev/null 2>&1; then ok=1; break; fi
    if ! scripts/serve_recipe_qwen38.sh status 2>/dev/null | grep -q "^running"; then break; fi
  done
  [ "$ok" = "1" ] && { echo "SERVER UP after attempt $i at $(date +%H:%M:%S)"; break; }
  scripts/serve_recipe_qwen38.sh stop >/dev/null 2>&1; sleep 10
done

echo "=== 3a) steady C6/2k (banked 125.98)"
STEADY_CTX=2000 .venv/bin/python scripts/c6_steady.py 6 400 2>&1 | grep -aE "AGGREGATE|ERROR"
echo "=== 3b) steady C6/20k (banked 23.8-31.3)"
STEADY_CTX=20000 .venv/bin/python scripts/c6_steady.py 6 400 2>&1 | grep -aE "AGGREGATE|ERROR"
echo "=== 3c) ramp C6 (banked 94.75@2k, 24.53@20k)"
.venv/bin/python scripts/c6_throughput.py 6 2>&1 | grep -aE "AGGREGATE|ERROR"
echo "CHAIN-DONE $(date +%H:%M:%S)"
