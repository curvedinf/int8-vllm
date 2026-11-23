#!/usr/bin/env bash
# Inkling-2-Mini PTQR W4A8 on four MI100s.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
launcher="$root/scripts/serve_recipe_inkling2mini.sh"
served_name=inkling-2-mini
recipe_label=Inkling-2-Mini
python="${VLLM_PYTHON:-$root/.venv/bin/python}"
models="${MODELS_DIR:-$HOME/models}"
model="${MODEL_DIR:-$models/Inkling-2-Mini-PTQR-W4A8-Vision-PTQR-INT8}"
if [[ -z "${MODEL_DIR:-}" && ! -r "$model/model.safetensors.index.json" ]]; then
  model="$models/Inkling-2-Mini-PTQR-W4A8-G128"
fi
logs="${LOG_DIR:-$root/logs/serve_recipe_inkling2mini}"
pid_file="$logs/server.pid"

serve() {
  if [[ ! -r "$model/model.safetensors.index.json" ]]; then
    echo "Inkling-2-Mini checkpoint missing at $model" >&2
    return 1
  fi
  if [[ "${INKLING_WARM_CHECKPOINT:-0}" == 1 ]]; then
    "$python" -S "$root/scripts/warm_hy4_engram.py" "$model" --checkpoint --auto \
      --workers 1 --threshold-ms 1
  fi
  export ROCM_PATH="${ROCM_PATH:-/opt/rocm}" HIP_PATH="${HIP_PATH:-/opt/rocm}"
  export PATH="$(dirname -- "$python"):$ROCM_PATH/bin:$PATH"
  export PYTHONPATH="$root/python_startup:$root:${AITER_DIR:-$root/../int8-aiter}${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export GPU_ARCHS=gfx908 PYTORCH_ROCM_ARCH=gfx908 VLLM_TARGET_DEVICE=rocm
  export HIP_VISIBLE_DEVICES="${MI100_DEVICES:-0,1,2,3}" CUDA_VISIBLE_DEVICES="${MI100_DEVICES:-0,1,2,3}"
  export FLYDSL_GPU_ARCH=gfx908 VLLM_USE_V2_MODEL_RUNNER=1
  export VLLM_AIS_REPACK_WINDOW_MB="${VLLM_AIS_REPACK_WINDOW_MB:-2}"
  export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
  export VLLM_ROCM_USE_AITER_CUSTOM_AR="${VLLM_ROCM_USE_AITER_CUSTOM_AR:-0}"
  if [[ -z "${VLLM_API_KEY:-}" && -n "${API_KEY_FILE:-}" ]]; then
    VLLM_API_KEY="$(tr -d '\r\n' <"$API_KEY_FILE")"
    export VLLM_API_KEY
  fi
  source "$root/scripts/mi100-cpu-map.sh"
  local draft="${DRAFT_MODEL_DIR:-$models/Inkling-2-Mini-DSpark-PTQR-W8A8-G128}"
  if [[ -z "${DRAFT_MODEL_DIR:-}" && ! -r "$draft/mtp.safetensors" ]]; then
    draft="$models/Inkling-2-Mini-source"
  fi
  local use_draft=0
  if [[ -n "${DRAFT_MODEL_DIR:-}" || -r "$draft/mtp.safetensors" ]]; then use_draft=1; fi
  local runtime_gib=14
  if [[ "${MI100_XGMI:-on}" == off ]]; then runtime_gib=18; fi
  # The native draft and its verification graphs increase the TP4 host footprint.
  if [[ "${SPEC:-$use_draft}" == 1 ]]; then runtime_gib=$((runtime_gib + 6)); fi
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
    --max-model-len "${MAXLEN:-1048576}" --max-num-seqs "${C:-12}"
    --block-size "${BLOCK_SIZE:-16}" --max-num-batched-tokens "${MNBT:-1024}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}"
    --tokenizer-mode inkling --reasoning-parser inkling
    --enable-auto-tool-choice --tool-call-parser inkling
    --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20}'
  )
  local capture_sizes='[1,2,4,8,12]'
  if [[ "${SPEC:-$use_draft}" == 1 ]]; then
    if [[ ! -r "$draft/mtp.safetensors" || ! -r "$draft/config.json" ]]; then
      echo "Missing Inkling-2-Mini DSpark checkpoint at $draft" >&2
      return 1
    fi
    local spec
    spec="$("$python" -S -c '
import json, sys
print(json.dumps(dict(method="dspark", model=sys.argv[1], num_speculative_tokens=7,
                      kv_cache_dtype="int8_block_g128",
                      draft_load_config={"load_format": sys.argv[2]})))' "$draft" "${DRAFT_LOAD_FORMAT:-ais}")"
    args+=(--speculative-config "$spec")
    capture_sizes='[8,16,32,64,96]'
  fi
  if [[ "${EAGER:-0}" == 1 ]]; then
    args+=(--enforce-eager)
  else
    local graph='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":'"${capture_sizes}"'}'
    args+=(--compilation-config "${COMPILATION_CONFIG:-$graph}")
  fi
  local additional
  additional="$("$python" -S -c '
import json, sys
config = dict(
    inkling_merge_query_gate=False,
    inkling_merge_rel_sink=False,
    inkling_share_attention_input=True,
    inkling_dspark_fused_residual=True,
    inkling_fused_prefill_epilogue=False,
    inkling_fused_expert_activation=False,
    inkling_fused_router=False,
    inkling_fused_loop_mix=False,
    inkling_fused_message_gate=False,
    inkling_batch_message_max_tokens=0,
    hy4_direct_fp16_output=True,
)
config.update(json.loads(sys.argv[1] or "{}"))
print(json.dumps(config))
' "${ADDITIONAL_CONFIG:-}")"
  args+=(--additional-config "$additional")
  [[ -z "${PROFILER_CONFIG:-}" ]] || args+=(--profiler-config "$PROFILER_CONFIG")
  if (( host_kv_gib > 0 )); then
    # Inkling's host KV has several large tensors per rank. Power-of-two
    # rounding of each pinned tensor can exceed the reserve by many GiB.
    local alloc_conf="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_HIP_ALLOC_CONF:-}}}"
    if [[ "$alloc_conf" != *pinned_max_round_threshold_mb:* ]]; then
      export PYTORCH_ALLOC_CONF="${alloc_conf:+$alloc_conf,}pinned_max_round_threshold_mb:16"
    fi
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
