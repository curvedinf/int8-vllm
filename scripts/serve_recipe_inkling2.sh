#!/usr/bin/env bash
# Inkling-2-Mini PTQR W4A8 on four MI100s.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
launcher="$root/scripts/serve_recipe_inkling2.sh"
served_name=inkling-2-mini
recipe_label=Inkling-2-Mini
python="${VLLM_PYTHON:-$root/.venv/bin/python}"
model="${MODEL_DIR:-$HOME/models/Inkling-2-Mini-PTQR-W4A8-G128}"
logs="${LOG_DIR:-$root/logs/serve_recipe_inkling2}"
pid_file="$logs/server.pid"

serve() {
  if [[ ! -r "$model/model.safetensors.index.json" ]]; then
    echo "Inkling-2-Mini checkpoint missing at $model" >&2
    return 1
  fi
  export ROCM_PATH="${ROCM_PATH:-/opt/rocm}" HIP_PATH="${HIP_PATH:-/opt/rocm}"
  export PATH="$(dirname -- "$python"):$ROCM_PATH/bin:$PATH"
  export PYTHONPATH="$root/python_startup:$root:${AITER_DIR:-$root/../int8-aiter}${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export GPU_ARCHS=gfx908 PYTORCH_ROCM_ARCH=gfx908 VLLM_TARGET_DEVICE=rocm
  export HIP_VISIBLE_DEVICES="${MI100_DEVICES:-0,1,2,3}" CUDA_VISIBLE_DEVICES="${MI100_DEVICES:-0,1,2,3}"
  export FLYDSL_GPU_ARCH=gfx908 VLLM_USE_V2_MODEL_RUNNER=1
  export VLLM_AIS_REPACK_WINDOW_MB="${VLLM_AIS_REPACK_WINDOW_MB:-16}"
  export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
  export VLLM_ROCM_USE_AITER_CUSTOM_AR="${VLLM_ROCM_USE_AITER_CUSTOM_AR:-0}"
  if [[ -z "${VLLM_API_KEY:-}" && -n "${API_KEY_FILE:-}" ]]; then
    VLLM_API_KEY="$(tr -d '\r\n' <"$API_KEY_FILE")"
    export VLLM_API_KEY
  fi
  source "$root/scripts/mi100-cpu-map.sh"
  local runtime_gib=14
  if [[ "${MI100_XGMI:-on}" == off ]]; then runtime_gib=18; fi
  local host_kv_gib
  host_kv_gib="$("$python" -S -m scripts.mi100_host_kv \
    --ram-reserve-gib "${INKLING_HOST_RAM_RESERVE_GIB:-8}" \
    --runtime-gib "${INKLING_HOST_RUNTIME_GIB:-$runtime_gib}" \
    --host-kv-gib "${INKLING_HOST_KV_GIB:-auto}")"
  local -a args=(
    -m vllm.entrypoints.openai.api_server --model "$model" --served-model-name "$served_name"
    "${cpu_bind_args[@]}"
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8020}" --tensor-parallel-size "${TPS:-4}"
    --dtype float16 --load-format "${LOAD_FORMAT:-ais}" --kv-cache-dtype int8_block_g128
    --max-model-len "${MAXLEN:-262144}" --max-num-seqs "${C:-8}"
    --block-size "${BLOCK_SIZE:-16}" --max-num-batched-tokens "${MNBT:-1024}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}"
    --mm-encoder-tp-mode data --tokenizer-mode inkling --reasoning-parser inkling
    --enable-auto-tool-choice --tool-call-parser inkling
    --override-generation-config '{"temperature":1.0}'
  )
  if [[ "${EAGER:-0}" == 1 ]]; then
    args+=(--enforce-eager)
  else
    local graph='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}'
    args+=(--compilation-config "${COMPILATION_CONFIG:-$graph}")
  fi
  [[ -z "${ADDITIONAL_CONFIG:-}" ]] || args+=(--additional-config "$ADDITIONAL_CONFIG")
  [[ -z "${PROFILER_CONFIG:-}" ]] || args+=(--profiler-config "$PROFILER_CONFIG")
  if (( host_kv_gib > 0 )); then
    args+=(--kv-offloading-backend native --kv-offloading-size "$host_kv_gib")
  fi
  case "${MI100_XGMI:-on}" in
    on) export VLLM_TP_CAR_MAX_BYTES="${VLLM_TP_CAR_MAX_BYTES:-67108864}" ;;
    off)
      export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
      export VLLM_TP_AR_FP32="${VLLM_TP_AR_FP32:-gather}"
      args+=(--disable-custom-all-reduce) ;;
    *) echo "MI100_XGMI must be on or off" >&2; return 2 ;;
  esac
  exec "$python" "${args[@]}" "$@"
}

source "$root/scripts/mi100-recipe-lifecycle.sh"
