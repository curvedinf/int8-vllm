#!/usr/bin/env bash
# HY4-Flash NeUQI W4A8 target, PTQR W8A8 DFly2 and ViT, packed Q4 Engram.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python="${VLLM_PYTHON:-$root/.venv/bin/python}"
model="${MODEL_DIR:-$HOME/models/Hy4-flash-NeUQI-W4A8}"
draft="${DRAFT_MODEL_DIR:-$HOME/models/Hy4-flash-DFly2-PTQR-INT8}"
logs="${LOG_DIR:-$root/logs/serve_recipe_hy4}"
pid_file="$logs/server.pid"

launcher="$root/scripts/serve_recipe_hy4.sh"
served_name=hy4-flash
recipe_label=HY4

serve() {
  local tp="${TPS:-4}" kv="${KV_DTYPE:-int8_block_g128}"
  local mode="${ENGRAM_MODE:-bcache}" load="${LOAD_FORMAT:-ais}"
  local devices="${MI100_DEVICES:-0,1,2,3}" default_ar=1
  if [[ "${MI100_XGMI:-on}" == off ]]; then
    default_ar=0
    export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
    export VLLM_TP_AR_FP32="${VLLM_TP_AR_FP32:-gather}"
  else
    export VLLM_TP_CAR_MAX_BYTES="${VLLM_TP_CAR_MAX_BYTES:-67108864}"
  fi
  if [[ ! -r "$model/model.safetensors.index.json" ]]; then
    echo "HY4 target checkpoint missing at $model" >&2
    return 1
  fi
  export ROCM_PATH="${ROCM_PATH:-/opt/rocm}" HIP_PATH="${HIP_PATH:-/opt/rocm}"
  export PATH="$(dirname -- "$python"):$ROCM_PATH/bin:$PATH"
  export PYTHONPATH="$root/python_startup:$root:${AITER_DIR:-$root/../int8-aiter}${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export GPU_ARCHS=gfx908 PYTORCH_ROCM_ARCH=gfx908 VLLM_TARGET_DEVICE=rocm
  export FLYDSL_GPU_ARCH="${FLYDSL_GPU_ARCH:-gfx908}"
  export VLLM_AIS_REPACK_WINDOW_MB="${VLLM_AIS_REPACK_WINDOW_MB:-16}"
  export HIP_VISIBLE_DEVICES="$devices" CUDA_VISIBLE_DEVICES="$devices"
  export VLLM_USE_V2_MODEL_RUNNER=1
  export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
  export VLLM_ROCM_USE_AITER_CUSTOM_AR="${VLLM_ROCM_USE_AITER_CUSTOM_AR:-0}"
  export VLLM_GFX908_INT8_EMBEDDING="${VLLM_GFX908_INT8_EMBEDDING:-1}"
  export VLLM_GFX908_INT8_LM_HEAD="${VLLM_GFX908_INT8_LM_HEAD:-1}"
  # A set-but-empty VLLM_API_KEY serves without authentication; the key
  # file is only read when the variable is unset.
  if [[ ! ${VLLM_API_KEY+x} && -n "${API_KEY_FILE:-}" ]]; then
    VLLM_API_KEY="$(tr -d '\r\n' <"$API_KEY_FILE")"
    export VLLM_API_KEY
  else
    export VLLM_API_KEY
  fi
  if [[ "$load" == hy4_direct || "$mode" == direct ]]; then
    export HIPFILE_ALLOW_COMPAT_MODE=false HIPFILE_FORCE_COMPAT_MODE=false
  fi
  local table_dir="${ENGRAM_DIR:-$model/engram}"
  # Host memory growth from launch to peak C8 load, rounded up to whole GiB.
  local runtime_gib host_kv_gib
  case "${MI100_CONTAINER:-0}-${MI100_XGMI:-on}" in
    0-on) runtime_gib=14 ;;
    0-off) runtime_gib=18 ;;
    1-on) runtime_gib=19 ;;
    *) runtime_gib=20 ;;
  esac
  host_kv_gib="$("$python" "$root/scripts/warm_hy4_engram.py" "$table_dir" \
    --check-ram "$mode" --ram-reserve-gib "${HY4_HOST_RAM_RESERVE_GIB:-8}" \
    --runtime-gib "$runtime_gib" --host-kv-gib "${HY4_HOST_KV_GIB:-auto}")"
  if [[ "$mode" == bcache ]]; then
    local -a prime=()
    case "${ENGRAM_PRIME:-auto}" in
      auto) prime+=(--auto) ;;
      on) ;;
      off) ;;
      *) echo "ENGRAM_PRIME must be auto, on, or off" >&2; return 2 ;;
    esac
    if [[ "${ENGRAM_PRIME:-auto}" != off ]]; then
      "$python" "$root/scripts/warm_hy4_engram.py" "$table_dir" "${prime[@]}" &
      local prime_pid=$!
      trap 'kill "$prime_pid" 2>/dev/null || true; wait "$prime_pid" 2>/dev/null || true; exit 143' TERM INT
      wait "$prime_pid"
      trap - TERM INT
    fi
  fi
  local engram spec mm_default='{"image":1,"video":1}'
  local mm_processing='{"images_kwargs":{"max_pixels":1048576},"videos_kwargs":{"max_pixels":4194304,"max_frames":32}}'
  engram="$("$python" -c 'import json,sys; print(json.dumps(dict(packed_table=sys.argv[1], table_directory=sys.argv[2], page_cache_mib=int(sys.argv[3]), cpu_offload=False)))' "$mode" "$table_dir" "${ENGRAM_CACHE_MIB:-32}")"
  source "$root/scripts/mi100-cpu-map.sh"
  local -a args=(
    -m vllm.entrypoints.openai.api_server --model "$model" --served-model-name hy4-flash
    "${cpu_bind_args[@]}"
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8020}" --tensor-parallel-size "$tp"
    --dtype float16 --load-format "$load" --kv-cache-dtype "$kv" --engram-config "$engram"
    --max-model-len "${MAXLEN:-262144}" --max-num-seqs "${C:-8}"
    --max-num-batched-tokens "${MNBT:-2048}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}"
    --mm-encoder-tp-mode data --limit-mm-per-prompt "${MM_LIMIT:-$mm_default}"
    --mm-processor-kwargs "${MM_PROCESSOR_KWARGS:-$mm_processing}"
    --tokenizer-mode hy_v4 --reasoning-parser hy_v4 --enable-auto-tool-choice --tool-call-parser hy_v4
    --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20}'
  )
  if (( host_kv_gib > 0 )); then
    args+=(--kv-offloading-backend native --kv-offloading-size "$host_kv_gib")
  fi
  if [[ "${SPEC:-1}" == 1 ]]; then
    if [[ ! -r "$draft/model.safetensors.index.json" ]]; then
      echo "HY4 DFly2 checkpoint missing at $draft" >&2
      return 1
    fi
    spec="$("$python" - "$draft" "${NS:-7}" "$kv" "${FORCE_REJECT:-0}" <<'PY'
import json, sys
width = int(sys.argv[2])
spec = dict(method="dfly",model=sys.argv[1],num_speculative_tokens=width,
            kv_cache_dtype=sys.argv[3],draft_sample_method="probabilistic")
if sys.argv[4] == "1":
    spec.update(rejection_sample_method="synthetic",synthetic_acceptance_rates=[0.]*width)
print(json.dumps(spec))
PY
)"
    args+=(--speculative-config "$spec")
  fi
  [[ "${AR:-$default_ar}" == 1 ]] || args+=(--disable-custom-all-reduce)
  [[ "${EAGER:-0}" == 0 ]] || args+=(--enforce-eager)
  local additional
  additional="$("$python" - "${ADDITIONAL_CONFIG:-}" <<'PY'
import json
import sys

defaults = {
    "hy4_causal_long_chunk_value_tile": 256,
    "hy4_causal_long_chunk_page_pair": True,
    "hy4_index_tp_min_width": 16384,
    "hy4_share_weight_reads": True,
    "hy4_rank_cache": True,
    "hy4_engram_cache_rows": True,
    "hy4_engram_prefetch_pages": 64,
    "hy4_fused_hashes": True,
    "hy4_flydsl_gemm": True,
    "hy4_flydsl_mla_scores": True,
    "hy4_flydsl_mla_values": True,
    "hy4_share_attention_input": True,
    "hy4_fused_group_selection": True,
    "hy4_dynamic_empty_tiles": False,
    "hy4_sparse_score_tile": -1,
    "hy4_pair_threshold": 768,
    "hy4_causal_verify_query_tile": 4,
    "hy4_flydsl_engram": True,
    "hy4_engram_fused_residual": True,
    "hy4_pack_verify_query": True,
    "hy4_engram_full_proj_min": 137,
    "hy4_flydsl_ihc": True,
    "hy4_flydsl_vision_mlp": True,
    "hy4_flydsl_router": False,
    "hy4_flydsl_dfly_norm": True,
    "hy4_rope_group": 64,
}
overrides = json.loads(sys.argv[1] or "{}")
if not isinstance(overrides, dict):
    sys.exit("ADDITIONAL_CONFIG must be a JSON object")
defaults.update(overrides)
print(json.dumps(defaults))
PY
)"
  args+=(--additional-config "$additional")
  [[ -z "${COMPILATION_CONFIG:-}" ]] || args+=(--compilation-config "$COMPILATION_CONFIG")
  exec "$python" "${args[@]}" "$@"
}

source "$root/scripts/mi100-recipe-lifecycle.sh"
