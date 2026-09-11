#!/usr/bin/env bash
# A/B phase 2: boot with VLLM_SCHED_NO_MIX=1 and run the same C6 tests.
set -u
K="$(cat /tmp/.vk)"
export VLLM_API_KEY="$K" VLLM_STEPPHASE=1 VLLM_SCHED_NO_MIX=1
cd /home/curved/vllm-gfx908

scripts/serve_recipe_qwen38.sh stop >/dev/null 2>&1
sleep 8

for i in 1 2 3 4 5 6; do
  echo "=== nomix-boot attempt $i $(date +%H:%M:%S)"
  scripts/serve_recipe_qwen38.sh start >/dev/null 2>&1
  ok=0
  for j in $(seq 1 26); do
    sleep 20
    if curl -sf -H "Authorization: Bearer $K" http://127.0.0.1:8020/v1/models >/dev/null 2>&1; then
      ok=1; break
    fi
    if ! scripts/serve_recipe_qwen38.sh status 2>/dev/null | grep -q "^running"; then
      break
    fi
  done
  if [ "$ok" = "1" ]; then
    echo "NOMIX SERVER UP after attempt $i at $(date +%H:%M:%S)"
    break
  fi
  echo "attempt $i failed"
  scripts/serve_recipe_qwen38.sh stop >/dev/null 2>&1
  sleep 10
done

M=$(wc -l < logs/serve_recipe_qwen38/server.log)
echo "$M" > /tmp/ab_nomix_mark.txt
.venv/bin/python scripts/c6_throughput.py 6 2>&1 | grep -aE "AGGREGATE|Error"
