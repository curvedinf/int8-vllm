#!/usr/bin/env bash
# Quick perf gate: single-stream + C8 against the canonical RUNNING Direwolf
# server on :8020. Results count only when that server was launched by
# serve_direwolf_qwen38.sh with the target+DFlash2 GS128 pair, AITER W8A8
# everywhere, INT8 KV/Mamba/UA/custom-AR/quant-out, TP4, and max-num-seqs=8.
# Usage: scripts/bench_quick.sh <model_dir> <tag> [api_key]
set -euo pipefail

ROOT="${HOME}/vllm-gfx908"
VENV="${ROOT}/.venv"
MODEL_DIR="${1:?model dir}"
TAG="${2:?tag}"
KEY="${3:-test-key-local-only}"
OUT="${ROOT}/logs/c8_optimization/${TAG}"
mkdir -p "${OUT}"

# Single-stream: 3 realistic prompts x 256 tokens, greedy
"${VENV}/bin/python" - "$MODEL_DIR" "$KEY" "$OUT" <<'EOF'
import json, sys, time, urllib.request
model_dir, key, out = sys.argv[1], sys.argv[2], sys.argv[3]
URL = "http://127.0.0.1:8020/v1/completions"
HDR = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
PROMPTS = [
    "Write a Python function that checks if a string is a palindrome, ignoring punctuation and case:",
    "Explain the difference between TCP and UDP, and when each is appropriate:",
    "A train leaves at 3pm traveling 60 km/h. Another leaves at 4pm at 80 km/h on the same track. When does the second catch the first?",
]
rates = []
for p in PROMPTS:
    body = json.dumps({"model": "qwen3.8-27b-gptq8", "prompt": p,
                       "max_tokens": 256, "temperature": 0}).encode()
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(
        urllib.request.Request(URL, body, HDR), timeout=600).read())
    dt = time.time() - t0
    n = r["usage"]["completion_tokens"]
    rates.append(n / dt)
    print(f"  single-stream: {n} tok / {dt:.1f}s = {n/dt:.2f} tok/s")
import statistics
print(f"  MEAN single-stream: {statistics.mean(rates):.2f} tok/s")
json.dump({"single_stream_tok_s": rates, "mean": statistics.mean(rates)},
          open(f"{out}/single_stream.json", "w"))
EOF

# C8: 8 concurrent random 32-in/1000-out
OPENAI_API_KEY="${KEY}" "${VENV}/bin/vllm" bench serve \
  --base-url http://127.0.0.1:8020 --model qwen3.8-27b-gptq8 \
  --tokenizer "${MODEL_DIR}" --dataset-name random --num-prompts 8 \
  --max-concurrency 8 --random-input-len 32 --random-output-len 1000 \
  --endpoint /v1/completions --no-stream --skip-chat-template \
  > "${OUT}/c8.log" 2>&1
grep -aE 'output token throughput|total token throughput|Mean TTFT|Mean TPOT|Successful' \
  "${OUT}/c8.log" | tee "${OUT}/c8_metrics.txt"
echo "RESULTS IN ${OUT}"
