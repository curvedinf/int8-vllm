#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${HOME}/vllm-gfx908"
VENV="${ROOT_DIR}/.venv"
MODEL_DIR="${HOME}/models/Qwen3.8-27B-GPTQ-8bit"
SERVED_MODEL_NAME="qwen3.8-27b-gptq8"
LOG_DIR="${ROOT_DIR}/logs/serve_direwolf_qwen38"
PID_FILE="${LOG_DIR}/server.pid"
API_KEY_FILE="/etc/llama/llama-api.key"
WORKDIR="/tmp"

HOST="127.0.0.1"
PORT=8020
STARTUP_TIMEOUT=900
STOP_TIMEOUT=20
CPUSET="0-47"

COMMON_ENV=(
  PATH="${VENV}/bin:${PATH:-}"
  ROCM_PATH="/opt/rocm"
  HIP_PATH="/opt/rocm"
  GPU_ARCHS="gfx908"
  PYTORCH_ROCM_ARCH="gfx908"
  BUILD_TARGET="rocm"
  MAX_JOBS="48"
  LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  PYTHONPATH="${ROOT_DIR}/python_startup:${ROOT_DIR}:${HOME}/aiter:${PYTHONPATH:-}"
  HF_HOME="${HOME}/.cache/huggingface"
  OMP_NUM_THREADS="48"
  MKL_NUM_THREADS="48"
  OPENBLAS_NUM_THREADS="48"
  NUMEXPR_NUM_THREADS="48"
  HIP_VISIBLE_DEVICES="0,1,2,3"
  CUDA_VISIBLE_DEVICES="0,1,2,3"
  ROCR_VISIBLE_DEVICES="0,1,2,3"
  HSA_ENABLE_IPC_MODE_LEGACY="0"
  VLLM_TARGET_DEVICE="rocm"
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="1800"
  VLLM_ROCM_USE_AITER="1"
  VLLM_ROCM_USE_AITER_TRITON_GEMM="1"
  VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION="0"
  # AITER a16w8_blockscale/a8w8_blockscale for GPTQ 8-bit on gfx908 produces
  # garbled / truncated outputs. Force the Triton W8A16 (A16W8) kernel instead.
  VLLM_DISABLED_KERNELS="AiterW8A16LinearKernel"
  NCCL_ALGO="Ring"
  NCCL_PROTO="Simple"
  NCCL_P2P_DISABLE="0"
  NCCL_DMABUF_ENABLE="0"
  NCCL_DEBUG="INFO"
  RCCL_LOG_LEVEL="INFO"
)

DRAFT_MODEL_DIR="${HOME}/.cache/huggingface/dflash2-int8/Qwen3.8-27B-DFlash2-GPTQ-8bit"

ARGS=(
  serve "${MODEL_DIR}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size 4
  --dtype half
  --max-model-len 65536
  --max-num-seqs 8
  --gpu-memory-utilization 0.90
  --attention-backend TRITON_ATTN
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["+gemma_rms_norm","+silu_and_mul","+rms_norm_gated","+rotary_embedding","+apply_rotary_emb","none"]}'
  --language-model-only
  --skip-mm-profiling
  --disable-custom-all-reduce
  --disable-log-stats
  --disable-uvicorn-access-log
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --default-chat-template-kwargs '{"enable_thinking":false}'
  --override-generation-config '{"temperature":0.7,"top_p":0.80,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0}'
  --kv-cache-dtype int8_per_token_head --mamba-ssm-cache-dtype int8
  # Target: int8 KV + int8 mamba state. DFlash2 spec ON (int8 drafter, int8 draft KV):
  # the earlier 0.5x was an unoptimized path; spec gets its own tuning pass.
  --speculative-config '{"method":"dflash","model":"'"${DRAFT_MODEL_DIR}"'","num_speculative_tokens":7,"kv_cache_dtype":"int8_per_token_head"}'
)

usage() {
  printf 'usage: %s {start|stop|restart|status|supervise}\n' "$0"
}

is_pid_running() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

is_pgrp_running() {
  local pgid="${1:-}"
  [[ -n "${pgid}" ]] && kill -0 "-${pgid}" 2>/dev/null
}

is_running() {
  [[ -f "${PID_FILE}" ]] && is_pid_running "$(cat "${PID_FILE}")"
}

read_api_key() {
  if [[ -n "${VLLM_API_KEY:-}" ]]; then
    printf '%s' "${VLLM_API_KEY}"
    return 0
  fi

  if [[ -n "${LLAMA_API_KEY:-}" ]]; then
    printf '%s' "${LLAMA_API_KEY}"
    return 0
  fi

  [[ -r "${API_KEY_FILE}" ]] || { printf 'missing API key file: %s\n' "${API_KEY_FILE}" >&2; exit 1; }

  local api_key
  api_key="$(tr -d '\r\n' < "${API_KEY_FILE}")"
  [[ -n "${api_key}" ]] || { printf 'empty API key file: %s\n' "${API_KEY_FILE}" >&2; exit 1; }
  printf '%s' "${api_key}"
}

rotate_log() {
  mkdir -p "${LOG_DIR}"
  if [[ -s "${LOG_DIR}/server.log" ]]; then
    mv "${LOG_DIR}/server.log" "${LOG_DIR}/server.$(date -u +%Y%m%dT%H%M%SZ).log"
    ls -1t "${LOG_DIR}"/server.*.log 2>/dev/null | tail -n +21 | xargs -r rm -f
  fi
}

start_server() {
  [[ -x "${VENV}/bin/vllm" ]] || { printf 'missing vLLM executable: %s\n' "${VENV}/bin/vllm" >&2; exit 1; }
  [[ -d "${MODEL_DIR}" ]] || { printf 'missing model directory: %s\n' "${MODEL_DIR}" >&2; exit 1; }

  if is_running; then
    printf 'already running pid=%s url=http://%s:%s log=%s/server.log\n' \
      "$(cat "${PID_FILE}")" "${HOST}" "${PORT}" "${LOG_DIR}"
    return 0
  fi

  mkdir -p "${LOG_DIR}"
  rm -f "${PID_FILE}"
  rotate_log

  printf 'starting direwolf Qwen3.8 server: url=http://%s:%s cpuset=%s log=%s/server.log\n' \
    "${HOST}" "${PORT}" "${CPUSET}" "${LOG_DIR}"

  local api_key
  api_key="$(read_api_key)"

  (
    cd "${WORKDIR}"
    exec setsid taskset -c "${CPUSET}" env -u HSA_OVERRIDE_GFX_VERSION \
      VLLM_API_KEY="${api_key}" \
      "${COMMON_ENV[@]}" \
      "${VENV}/bin/vllm" "${ARGS[@]}"
  ) >"${LOG_DIR}/server.log" 2>&1 </dev/null &

  printf '%s\n' "$!" >"${PID_FILE}"
}

wait_ready() {
  for _ in $(seq 1 "${STARTUP_TIMEOUT}"); do
    if ! is_running; then
      printf 'server exited during startup; see %s/server.log\n' "${LOG_DIR}" >&2
      return 1
    fi
    if curl -fsS --max-time 2 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      printf 'ready: http://%s:%s\n' "${HOST}" "${PORT}"
      return 0
    fi
    if curl -fsS --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
      printf 'ready: http://%s:%s\n' "${HOST}" "${PORT}"
      return 0
    fi
    sleep 1
  done

  printf 'server did not become ready within %ss; see %s/server.log\n' \
    "${STARTUP_TIMEOUT}" "${LOG_DIR}" >&2
  return 1
}

stop_server() {
  if [[ ! -f "${PID_FILE}" ]]; then
    printf 'stopped\n'
    return 0
  fi

  local pid
  pid="$(cat "${PID_FILE}")"
  if is_pid_running "${pid}" || is_pgrp_running "${pid}"; then
    printf 'stopping pid=%s\n' "${pid}"
    kill -TERM "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    for _ in $(seq 1 "${STOP_TIMEOUT}"); do
      if ! is_pid_running "${pid}" && ! is_pgrp_running "${pid}"; then
        rm -f "${PID_FILE}"
        return 0
      fi
      sleep 1
    done
    kill -KILL "-${pid}" 2>/dev/null || true
    kill -KILL "${pid}" 2>/dev/null || true
  else
    printf 'stopped\n'
  fi

  rm -f "${PID_FILE}"
}

status_server() {
  if is_running; then
    printf 'running pid=%s url=http://%s:%s model=%s log=%s/server.log\n' \
      "$(cat "${PID_FILE}")" "${HOST}" "${PORT}" "${SERVED_MODEL_NAME}" "${LOG_DIR}"
    rg -n 'Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency for 65,536|via P2P/IPC|Uvicorn running|dflash|DFlash|spec' \
      "${LOG_DIR}/server.log" 2>/dev/null | tail -20 || true
  else
    printf 'stopped url=http://%s:%s model=%s log=%s/server.log\n' \
      "${HOST}" "${PORT}" "${SERVED_MODEL_NAME}" "${LOG_DIR}"
  fi
}

supervise_server() {
  trap 'stop_server; exit 0' INT TERM
  start_server
  wait_ready
  while true; do
    if ! is_running; then
      printf 'server exited\n' >&2
      stop_server
      exit 1
    fi
    sleep 5
  done
}

case "${1:-}" in
  start)
    start_server
    wait_ready
    ;;
  stop)
    stop_server
    ;;
  restart)
    stop_server
    start_server
    wait_ready
    ;;
  status)
    status_server
    ;;
  supervise)
    supervise_server
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
