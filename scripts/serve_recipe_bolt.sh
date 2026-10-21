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
  local pid argument previous=''
  pid="$(cat "${PID_FILE}")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  while IFS= read -r -d '' argument; do
    if [[ "$previous" == */serve_recipe_bolt.sh || "$previous" == serve_recipe_bolt.sh ]]; then
      [[ "$argument" == supervise ]] && return 0
    fi
    previous="$argument"
  done < "/proc/${pid}/cmdline"
  return 1
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
  # Prefill router-gate GEMM on int8 (reuses the fused-op activation quant);
  # +3.8% prefill median, decode CI spans zero, acceptance unchanged
  # (ledger BOLT_LOOP2_B21_E5_GATE_I8_CONFIRM_20261018).
  export VLLM_GFX908_GATE_I8_PREFILL="${VLLM_GFX908_GATE_I8_PREFILL:-1}"
  # Standard combination: native MoE norm+GS128 quantization, page-outer
  # attention, route64 expert prefill and weighted expert reduction. These
  # dispatch automatically for supported shapes; keep the legacy quantizer off.
  export VLLM_GFX908_FUSED_NORM_QUANT=0
  # Direct GS128 decoder path; validated separately from KV.
  export VLLM_GFX908_RESIDUAL_INT8="${VLLM_GFX908_RESIDUAL_INT8:-1}"
  if [[ "$VLLM_GFX908_RESIDUAL_INT8" == 1 ]]; then
    export VLLM_RESIDUAL_G128_LIBRARY="${VLLM_RESIDUAL_G128_LIBRARY:-$ROOT_DIR/vllm/_residual_g128.so}"
    # 64 CTAs measured 42.1 vs 35.6 tok/s at C8 (2026-10-21); the kernel
    # supports up to 64 and falls back to the payload-sized grid.
    export VLLM_GFX908_G128_AR_BLOCKS="${VLLM_GFX908_G128_AR_BLOCKS:-64}"
    if [[ ! -f "$VLLM_RESIDUAL_G128_LIBRARY" ]]; then
      printf '%s\n' 'Build the decoder collective with scripts/build_residual_g128.sh first.' >&2
      return 1
    fi
  fi
  # Decode-shape dense GEMM dispatch (split-K for M<=16, BM tiers), page
  # prefill MMA. The earlier warmup crash was an unmasked scale load in the
  # split-K K tail (K=1344); fixed 2026-10-21 and quality-gated 52/52.
  export VLLM_GFX908_DENSE_DECODE_DISPATCH="${VLLM_GFX908_DENSE_DECODE_DISPATCH:-1}"
  # The side-stream shared/routed fork is capture-safe after the split-K
  # K-tail fix: healthy 8-size capture and 52/52 quality; C8 bench 45.4 vs
  # 42.1 tok/s vs the serial fork (2026-10-21).
  export VLLM_GFX908_SHARED_OVERLAP="${VLLM_GFX908_SHARED_OVERLAP:-1}"
  export VLLM_G128_PAGE_MMA="${VLLM_G128_PAGE_MMA:-fp16}"
  export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-0}"
  export VLLM_BOLT_KERNEL_TUNING="${VLLM_BOLT_KERNEL_TUNING:-0}"
  export VLLM_BOLT_DENSE_CACHE_GIB="${VLLM_BOLT_DENSE_CACHE_GIB:-3}"
  export VLLM_BOLT_DENSE_CACHE_DROP_PACKED="${VLLM_BOLT_DENSE_CACHE_DROP_PACKED:-1}"
  export VLLM_BOLT_EXPERT_CACHE_GIB="${VLLM_BOLT_EXPERT_CACHE_GIB:-11}"
  export VLLM_BOLT_EXPERT_CACHE_SLOTS="${VLLM_BOLT_EXPERT_CACHE_SLOTS:-auto}"
  export VLLM_BOLT_EXPERT_CACHE_REFRESH="${VLLM_BOLT_EXPERT_CACHE_REFRESH:-128}"
  # Prefill-informed expert admission is a decode-priority lever: it lifts
  # decode ~+3% but costs ~4% prefill via per-forward rerank churn. Off by
  # default under the prefill-focus directive; set 1 to trade the other way.
  export VLLM_BOLT_PREFILL_ADMIT="${VLLM_BOLT_PREFILL_ADMIT:-0}"
  export VLLM_ROCM_PIN_LOCAL_OFFLOAD=1
  # Reuse the Qwen mixed-load policy with Bolt's resolved hybrid page size.
  export VLLM_GFX908_MIXED_PREFILL_TOKENS="${VLLM_GFX908_MIXED_PREFILL_TOKENS:-4256}"
  export VLLM_GFX908_MIXED_PREFILL_INTERVAL="${VLLM_GFX908_MIXED_PREFILL_INTERVAL:-16}"
  export VLLM_GFX908_PREFIX_FAIR_ADMISSION="${VLLM_GFX908_PREFIX_FAIR_ADMISSION:-1}"
  export VLLM_BOLT_SHORT_PREFILL_COHORT="${VLLM_BOLT_SHORT_PREFILL_COHORT:-1}"
  # AITER unified attention selects Bolt's HD128/GQA8, 4256-token page
  # prefill specialization automatically, with the generic path as fallback.
  export NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_P2P_DISABLE=0 NCCL_DMABUF_ENABLE=0
  export HSA_ENABLE_IPC_MODE_LEGACY=0
  # First-cache construction has thousands of small expert writes. Buffer
  # those through the host; AIS weight reads and packed storage are unchanged.
  export VLLM_AIS_REPACK_CPU_WRITE=1
  # bcache's sequential_cutoff is 4 MiB: 32 MiB loader windows bypass the SSD
  # and stream from the HDD. Keep every read under the cutoff.
  export VLLM_AIS_REPACK_WINDOW_MB="${VLLM_AIS_REPACK_WINDOW_MB:-2}"
  # Preserve the official excluded draft surfaces; the Qwen conv calibration
  # is not a calibration of this draft.
  export VLLM_GFX908_DF2_W8A8=0
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  # EngineCore/worker processes do not reliably inherit the fork-local
  # VLLM_* switches; publish them for both entry points to apply.
  "${ROOT_DIR}/.venv/bin/python" - <<'PY'
import json, os
data = {
    k: v
    for k, v in os.environ.items()
    if k.startswith("VLLM_") and k != "VLLM_API_KEY"
}
with open("/dev/shm/vllm_bolt_runtime_env.json", "w") as fh:
    json.dump(data, fh)
PY
  local spec
  spec="$("${ROOT_DIR}/.venv/bin/python" -c 'import json,sys; print(json.dumps(dict(method="dflash",model=sys.argv[1],num_speculative_tokens=7,kv_cache_dtype="int8_block_g128")))' "$DRAFT_MODEL_DIR")"
  local args=(
    -m vllm.entrypoints.openai.api_server
    --model "$MODEL_DIR" --served-model-name nemotron-bolt
    --enable-prompt-tokens-details
    --override-generation-config '{"temperature":1.0,"top_p":0.95,"max_new_tokens":32768}'
    --host "$HOST" --port "$PORT" --trust-remote-code
    --enable-auto-tool-choice --tool-call-parser qwen3_coder
    --reasoning-parser-plugin "${MODEL_DIR}/ultra_v3_reasoning_parser.py"
    --reasoning-parser ultra_v3
    --limit-mm-per-prompt '{"image":1,"video":1}'
    --video-pruning-rate 0.5
    --media-io-kwargs '{"video":{"fps":2,"num_frames":256}}'
    --tensor-parallel-size 4 --max-num-seqs 8 --dtype bfloat16
    --max-model-len "${MAX_MODEL_LEN:-262144}"
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-8512}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
    # Fixed arena follows the Qwen recipe; leave space for resident INT8 copies.
    # 8.5 GB reserves ~7.92 GiB (1,993,332 tokens = 7.60x 256k, 6x256k plus
    # 1.6x GPU prefix) and passed C8 + prefill with ~0.8 GiB workspace.
    # 8.9 GB previously exhausted workspace under C8; re-test before raising.
    --kv-cache-memory "${KVMEM:-8500000000}"
    --kv-cache-dtype int8_block_g128 --mamba-ssm-cache-dtype float32
    --speculative-config "$spec"
    --kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${KV_OFFLOAD_BYTES:-25769803776}}}"
  )
  # The pipeline shares ~247 tokens before its per-page URL/HTML. Physical
  # 4256-token pages cannot match that prefix. Keep bounded early states and
  # the normal replay tail; 8-token hashes include the draft lookahead proof.
  # Set PREFIX_CHECKPOINTS=0 to restore whole-page matching for an A/B boot.
  if [[ "${PREFIX_CHECKPOINTS:-64 128 232}" != 0 ]]; then
    local -a checkpoints
    read -r -a checkpoints <<< "${PREFIX_CHECKPOINTS:-64 128 232}"
    args+=(--prefix-match-unit 8 --prefix-cache-checkpoint-tokens "${checkpoints[@]}")
  fi
  if [[ "${EAGER:-0}" == 1 ]]; then
    args+=(--enforce-eager)
  else
    # Capture only the sizes the C6/C8 recipes use; vLLM pads smaller
    # batches up to the next captured size. Override with CUDAGRAPH_SIZES.
    local cg_sizes="${CUDAGRAPH_SIZES:-16,32,48,64}"
    args+=(--compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${cg_sizes}]}")
  fi
  # Warm the page cache with the full ~30 GiB weight working set while the
  # engine and four workers import. The loaders wait for the sentinel, so no
  # byte is read twice; they then read from RAM (>1 GiB/s) instead of the HDD
  # (~50 MiB/s). bcache does not retain enough of the set; see
  # scripts/prefetch_bolt_cache.py.
  local prefetch_pid=''
  export VLLM_BOLT_PREFETCH_DONE="${VLLM_BOLT_PREFETCH_DONE:-/dev/shm/vllm_bolt_prefetch.done}"
  if [[ "${PREFETCH:-1}" == 1 ]]; then
    export VLLM_BOLT_WAIT_PREFETCH=1
    "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/prefetch_bolt_cache.py" \
      --model-dir "${MODEL_DIR}" --tp-size 4 \
      --threads "${PREFETCH_THREADS:-8}" \
      --seconds "${PREFETCH_SECONDS:-300}" \
      --done-file "${VLLM_BOLT_PREFETCH_DONE}" >> "${LOG_DIR}/prefetch.log" 2>&1 &
    prefetch_pid=$!
  else
    unset VLLM_BOLT_WAIT_PREFETCH
  fi
  "${ROOT_DIR}/.venv/bin/python" "${args[@]}" &
  child=$!
  wait "$child"
  if [[ -n "${prefetch_pid}" ]]; then
    kill "${prefetch_pid}" 2>/dev/null || true
    wait "${prefetch_pid}" 2>/dev/null || true
  fi
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
    # A crashed worker leaves a multi-GiB /dev/shm offload mmap and psm
    # segments behind. Clear them only when no supervisor is alive.
    rm -f /dev/shm/vllm_offload_*.mmap /dev/shm/psm_* \
      "${VLLM_BOLT_PREFETCH_DONE:-/dev/shm/vllm_bolt_prefetch.done}" \
      /dev/shm/vllm_bolt_runtime_env.json 2>/dev/null || true
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
      for _ in $(seq 1 180); do alive || break; sleep 1; done
      if alive; then
        # Post-bench teardown (large KV/IPC unmaps) can outrun TERM grace.
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        for _ in $(seq 1 30); do alive || break; sleep 1; done
      fi
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
