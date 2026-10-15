#!/usr/bin/env bash
# Official packed Bolt + LiLiCorr; peer recipe to serve_recipe_qwen38.sh.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${HOME}/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB}"
DRAFT_MODEL_DIR="${DRAFT_MODEL_DIR:-${HOME}/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/serve_recipe_bolt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8020}"
mkdir -p "${LOG_DIR}"
PID_FILE="${LOG_DIR}/supervisor.pid"

alive() {
  [[ -r "${PID_FILE}" ]] || return 1
  local pid
  pid="$(cat "${PID_FILE}")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/${pid}/cmdline" | rg -q 'serve_recipe_bolt.sh supervise'
}

serve() {
  export VLLM_API_KEY="${VLLM_API_KEY:-${LLAMA_API_KEY:-}}"
  if [[ -z "${VLLM_API_KEY}" ]]; then
    local key_file="${API_KEY_FILE:-/etc/llama/llama-api.key}"
    [[ -r "$key_file" ]] || { echo "Cannot read API key file: $key_file" >&2; return 1; }
    VLLM_API_KEY="$(tr -d '\r\n' < "$key_file")"
  fi
  [[ -n "${VLLM_API_KEY}" ]] || { echo 'API key is empty' >&2; return 1; }
  export PATH="${ROOT_DIR}/.venv/bin:/opt/rocm/bin:${PATH}"
  export PYTHONPATH="${ROOT_DIR}/python_startup:${ROOT_DIR}:${ROOT_DIR}/../aiter:${PYTHONPATH:-}"
  export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export GPU_ARCHS=gfx908 PYTORCH_ROCM_ARCH=gfx908 VLLM_TARGET_DEVICE=rocm
  export HIP_VISIBLE_DEVICES=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3
  export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
  export VLLM_ROCM_USE_AITER_CUSTOM_AR=0 VLLM_GFX908_ACT_QUANT=round
  export VLLM_TP_SAMPLE_SYNC=1
  # Reuse the Qwen mixed-load policy with Bolt's resolved hybrid page size.
  export VLLM_GFX908_MIXED_PREFILL_TOKENS="${VLLM_GFX908_MIXED_PREFILL_TOKENS:-4256}"
  export VLLM_GFX908_MIXED_PREFILL_INTERVAL="${VLLM_GFX908_MIXED_PREFILL_INTERVAL:-16}"
  export VLLM_GFX908_PREFIX_FAIR_ADMISSION=1
  # Qwen's specialized HD256/GQA6 attention switches do not cover Bolt's
  # HD128/GQA8 layout. Use the already validated generic unified path.
  export NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_P2P_DISABLE=0 NCCL_DMABUF_ENABLE=0
  export HSA_ENABLE_IPC_MODE_LEGACY=0
  # First-cache construction has thousands of small expert writes. Buffer
  # those through the host; AIS weight reads and packed storage are unchanged.
  export VLLM_AIS_REPACK_CPU_WRITE=1
  # Preserve the official excluded draft surfaces; the Qwen conv calibration
  # is not a calibration of this draft.
  export VLLM_GFX908_DF2_W8A8=0
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  local spec
  spec="$("${ROOT_DIR}/.venv/bin/python" -c 'import json,sys; print(json.dumps(dict(method="dflash",model=sys.argv[1],num_speculative_tokens=7,kv_cache_dtype="int8_block_g128")))' "$DRAFT_MODEL_DIR")"
  local args=(
    -m vllm.entrypoints.openai.api_server
    --model "$MODEL_DIR" --served-model-name nemotron-bolt
    --host "$HOST" --port "$PORT" --trust-remote-code
    --enable-auto-tool-choice --tool-call-parser qwen3_coder
    --reasoning-parser-plugin "${MODEL_DIR}/ultra_v3_reasoning_parser.py"
    --reasoning-parser ultra_v3
    --limit-mm-per-prompt '{"image":1,"video":1}'
    --video-pruning-rate 0.5
    --media-io-kwargs '{"video":{"fps":2,"num_frames":256}}'
    --tensor-parallel-size 4 --max-num-seqs 8 --dtype bfloat16
    --max-model-len "${MAX_MODEL_LEN:-32768}"
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-8512}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
    --kv-cache-dtype int8_block_g128 --mamba-ssm-cache-dtype float32
    --speculative-config "$spec"
    --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592}}'
  )
  if [[ "${EAGER:-0}" == 1 ]]; then
    args+=(--enforce-eager)
  else
    args+=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8,16,24,32,40,48,56,64]}')
  fi
  "${ROOT_DIR}/.venv/bin/python" "${args[@]}" &
  child=$!
  wait "$child"
}

case "${1:-status}" in
  supervise)
    if alive; then echo 'Bolt supervisor is already running' >&2; exit 1; fi
    echo "$$" > "$PID_FILE"
    child=''
    trap 'if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; rm -f "$PID_FILE"; exit' TERM INT
    trap 'rm -f "$PID_FILE"' EXIT
    serve
    ;;
  start)
    if alive; then echo 'Bolt supervisor is already running'; exit 0; fi
    nohup setsid bash "$0" supervise >> "${LOG_DIR}/server.log" 2>&1 < /dev/null &
    for _ in $(seq 1 100); do
      if alive; then
        echo "Bolt starting; log: ${LOG_DIR}/server.log"
        exit 0
      fi
      sleep 0.1
    done
    echo "Bolt supervisor failed to start; see ${LOG_DIR}/server.log" >&2
    exit 1
    ;;
  stop)
    if alive; then
      pid="$(cat "$PID_FILE")"
      # `start` creates a private session. Include booting workers, before
      # the API server has installed its normal graceful-shutdown handlers.
      if [[ "$(ps -o pgid= -p "$pid" | tr -d ' ')" == "$pid" ]]; then
        kill -TERM -- "-$pid"
      else
        kill -TERM "$pid"
      fi
      for _ in $(seq 1 60); do alive || break; sleep 1; done
      if alive; then echo 'Supervisor has not exited' >&2; exit 1; fi
    fi
    ;;
  restart) bash "$0" stop; bash "$0" start ;;
  status)
    if alive; then
      echo "Bolt supervisor running (PID $(cat "$PID_FILE"))"
      curl --fail --silent --show-error --max-time 3 "http://${HOST}:${PORT}/health"
    else echo 'Bolt stopped'; exit 1; fi
    ;;
  *) echo "Usage: $0 {start|stop|restart|status|supervise}" >&2; exit 2 ;;
esac
