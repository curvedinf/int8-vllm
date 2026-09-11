#!/usr/bin/env bash
# A/B bench for the drafter FULL-cudagraph fix (fix-on legs).
# Usage: ab_drafter_graph.sh <tag>
set -u
TAG="${1:-ab}"
KEY="$(cat /tmp/.vk)"
export VLLM_API_KEY="$KEY"
LOG="logs/serve_recipe_qwen38/ab_${TAG}_$(date +%H%M%S).log"

metrics() {
  curl -sf -H "Authorization: Bearer $KEY" http://127.0.0.1:8020/metrics 2>/dev/null \
    | grep -iE "spec_decode|acceptance|draft" | grep -v "^#" >>"$LOG"
  echo "--- metrics snapshot $(date +%H:%M:%S)" >>"$LOG"
}

{
echo "=== A/B drafter-graph fix tag=$TAG $(date)"
metrics
} >>"$LOG"

.venv/bin/python - <<'PY' >>"$LOG" 2>&1
import sys
sys.path.insert(0, 'scripts')
from tpot_decomp2 import decomp
# Pre-drift short legs (matched vs banked baseline: 76 / 41 / 21 tok/s)
for ctx in (2000, 8000, 20000):
    decomp(ctx, out_tokens=800, tag=f'fix{ctx}')
# Long legs at 20k (vs pre-fix 118.1ms leg 1)
for i in range(2):
    print(f'--- long leg {i+1}/2', flush=True)
    decomp(20000, out_tokens=4000, tag=f'fixlong{i}')
PY

metrics
echo "DONE $(date)" >>"$LOG"
tail -5 "$LOG"
