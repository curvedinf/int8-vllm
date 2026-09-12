#!/usr/bin/env bash
# GS32 requant (user-approved 2026-09-10: "For quality, lets do GS32 like Q8_0")
# + full validation chain. ONE command; stops prod, quantizes (~2h, GPU 0),
# boots the GS32 checkpoint, runs the speed A/B and the unseeded full-read
# quality gate (outputs saved for manual reading).
#
# Usage: gs32_requant_and_validate.sh
set -euo pipefail
cd /home/curved/vllm-gfx908
K="$(cat /tmp/.vk)"
export VLLM_API_KEY="$K"
QUANT_PY=/home/curved/models/quantize_qwen38_27b_gptq8.py
QUANT_VENV="${QUANT_VENV:-}"   # set if the quant venv python isn't on PATH
GS32_DIR=/home/curved/models/Qwen3.8-27B-PTQR-R10S60-GS32

echo "=== 1) stop prod"
scripts/serve_recipe_qwen38.sh stop || true
sleep 10

echo "=== 2) quantize GS32 ($(date +%H:%M:%S); expect ~2h)"
if [ -n "$QUANT_VENV" ]; then
  PY="$QUANT_VENV"
else
  PY=python3
fi
ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH \
PYTORCH_ROCM_ARCH=gfx908 GPU_ARCHS=gfx908 VLLM_TARGET_DEVICE=rocm \
HIP_VISIBLE_DEVICES=0 \
"$PY" "$QUANT_PY" --group-size 32 \
  --output-dir "$GS32_DIR" 2>&1 | tee logs/gs32_quant.log | tail -5

echo "=== 3) verify shards"
ls "$GS32_DIR" | head -12
test -f "$GS32_DIR/quantize_config.json" || { echo "MISSING quantize_config.json"; exit 1; }
grep -o '"group_size": *[0-9]*' "$GS32_DIR/quantize_config.json"

echo "=== 4) boot GS32 server"
sed -e "s|Qwen3.8-27B-PTQR-R10S60|$(basename "$GS32_DIR")|" \
    scripts/serve_recipe_qwen38.sh > scripts/serve_recipe_qwen38_gs32.sh
chmod +x scripts/serve_recipe_qwen38_gs32.sh
for i in 1 2 3 4 5 6; do
  scripts/serve_recipe_qwen38_gs32.sh start >/dev/null 2>&1
  ok=0
  for j in $(seq 1 36); do
    sleep 20
    if curl -sf -H "Authorization: Bearer $K" http://127.0.0.1:8020/v1/models >/dev/null 2>&1; then ok=1; break; fi
    if ! scripts/serve_recipe_qwen38_gs32.sh status 2>/dev/null | grep -q "^running"; then break; fi
  done
  [ "$ok" = "1" ] && { echo "GS32 SERVER UP attempt $i"; break; }
  scripts/serve_recipe_qwen38_gs32.sh stop >/dev/null 2>&1; sleep 10
done

echo "=== 5) speed A/B (banked: steady 120-128@2k, 24-31@20k; ramp 94.8@2k)"
PORT=8020 STEADY_CTX=2000 .venv/bin/python scripts/c6_steady.py 6 400 2>&1 | grep -aE "AGGREGATE|ERROR"
PORT=8020 STEADY_CTX=20000 .venv/bin/python scripts/c6_steady.py 6 400 2>&1 | grep -aE "AGGREGATE|ERROR"
PORT=8020 .venv/bin/python scripts/c6_throughput.py 6 2>&1 | grep -aE "AGGREGATE|ERROR"

echo "=== 6) unseeded full-read quality gate (long-in/long-out)"
mkdir -p logs/gs32_readgate
PORT=8020 .venv/bin/python - <<'EOF'
import json, urllib.request, os, sys
sys.path.insert(0, "scripts")
from garble_repro import build_prompt
KEY = os.environ["VLLM_API_KEY"]
corpus = build_prompt(40000, f"readgate-{int(__import__('time').time())}")
body = {"model": "qwen3.8-27b-gptq8", "messages": [{"role": "user", "content":
    "You are given reference notes. Write a long, coherent chronological essay "
    "synthesizing them. Write as much as possible; do not stop early.\n\n" + corpus}],
    "max_tokens": 4000, "temperature": 1.0, "top_p": 0.95, "top_k": 20}
req = urllib.request.Request("http://127.0.0.1:8020/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
r = json.load(urllib.request.urlopen(req, timeout=3600))
txt = r["choices"][0]["message"]["content"]
open("logs/gs32_readgate/output_40k_4k.txt", "w").write(txt)
print(f"readgate saved: {len(txt)} chars, {r['usage']['completion_tokens']} tokens")
EOF
echo "READ the file: logs/gs32_readgate/output_40k_4k.txt (manual gate)"
echo "GS32-CHAIN-DONE $(date +%H:%M:%S)"
