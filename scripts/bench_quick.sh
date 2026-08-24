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

PID_FILE="${ROOT}/logs/serve_direwolf_qwen38/server.pid"
SERVER_LOG="${ROOT}/logs/serve_direwolf_qwen38/server.log"
[[ -r "${PID_FILE}" ]] || { printf 'missing server pid file: %s\n' "${PID_FILE}" >&2; exit 1; }
SERVER_PID="$(cat "${PID_FILE}")"
kill -0 "${SERVER_PID}" 2>/dev/null || { printf 'server pid is not running: %s\n' "${SERVER_PID}" >&2; exit 1; }

# Make every result self-identifying without leaking the API key.
{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'tag=%s\n' "${TAG}"
  printf 'server_pid=%s\n' "${SERVER_PID}"
  printf 'vllm_sha=%s\n' "$(git -C "${ROOT}" rev-parse HEAD)"
  printf 'aiter_sha=%s\n' "$(git -C "${HOME}/aiter" rev-parse HEAD)"
  printf 'server_cmdline='
  tr '\0' ' ' < "/proc/${SERVER_PID}/cmdline"
  printf '\nserver_environment:\n'
  tr '\0' '\n' < "/proc/${SERVER_PID}/environ" | grep -E \
    '^(CAR|AR|ARFUSE|CGMODE|VLLM_GFX908_CK_W8A8|VLLM_DISABLED_KERNELS|VLLM_ROCM_USE_AITER_CUSTOM_AR|VLLM_ROCM_USE_AITER|VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION|HIP_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES|ROCR_VISIBLE_DEVICES|NCCL_ALGO|NCCL_PROTO)=' | sort
} > "${OUT}/manifest.txt"

grep -aE \
  "non-default args|EngineCoreConfig|compilation_config|all.?reduce|AITER_CUSTOM|'CUSTOM'|'PYNCCL'|gemm_a8w8|tuned config|fuse_allreduce_rms|CUDAGraphMode" \
  "${SERVER_LOG}" > "${OUT}/server_config.log" 2>/dev/null || true
rocm-smi --showmeminfo vram --showuse > "${OUT}/gpu_before.log" 2>&1 || true

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
samples = []
for p in PROMPTS:
    body = json.dumps({"model": "qwen3.8-27b-gptq8", "prompt": p,
                       "max_tokens": 256, "temperature": 0}).encode()
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(
        urllib.request.Request(URL, body, HDR), timeout=600).read())
    dt = time.time() - t0
    n = r["usage"]["completion_tokens"]
    rates.append(n / dt)
    samples.append({"prompt": p, "text": r["choices"][0]["text"],
                    "completion_tokens": n, "seconds": dt})
    print(f"  single-stream: {n} tok / {dt:.1f}s = {n/dt:.2f} tok/s")
import statistics
print(f"  MEAN single-stream: {statistics.mean(rates):.2f} tok/s")
json.dump({"single_stream_tok_s": rates, "mean": statistics.mean(rates),
           "samples": samples},
          open(f"{out}/single_stream.json", "w"))
EOF

# C8: 8 concurrent random 32-in/1000-out
OPENAI_API_KEY="${KEY}" "${VENV}/bin/vllm" bench serve \
  --base-url http://127.0.0.1:8020 --model qwen3.8-27b-gptq8 \
  --tokenizer "${MODEL_DIR}" --dataset-name random --num-prompts 8 \
  --max-concurrency 8 --random-input-len 32 --random-output-len 1000 \
  --endpoint /v1/completions --no-stream --skip-chat-template \
  --temperature 0 --num-warmups 2 \
  > "${OUT}/c8.log" 2>&1
grep -aiE 'output token throughput|total token throughput|Mean TTFT|Mean TPOT|Successful' \
  "${OUT}/c8.log" | tee "${OUT}/c8_metrics.txt"
echo "RESULTS IN ${OUT}"
