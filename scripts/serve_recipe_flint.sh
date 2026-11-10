#!/usr/bin/env bash
# Official packed Flint + Flint-DFlash2 PTQR recipe; peer script to
# serve_recipe_qwen38.sh / serve_recipe_bolt.sh.
#
# Stack (all mandatory, see docs/recipes/README.md and the ledger
# agents/scratch/flint-int8-recipe.md):
#   - target: curvedinf/Flint-GPTQ-INT8-W8A8-GS128-PTQR (PTQR rung 1 s40)
#   - draft:  curvedinf/Flint-DFlash2-GPTQ-INT8-W8A8-GS128-PTQR
#   - AITER W8A8 INT8 GEMMs for every GS128 dense linear
#   - routed 256-expert MoE on the gfx908 grouped a8w8-blockscale kernel
#     (vllm/model_executor/layers/quantization/gptq_moe_a8w8.py)
#   - AITER unified attention, vLLM CUSTOM all-reduce
#   - int8_block_g128 KV on BOTH target and draft
#   - TP4 / C12 (twelve concurrent sequences), DFlash2 NS=7
#   - 512k per-request cap (Flint announcement; card recommends <=256k)
# No 16-bit fallback: the a8w8 expert path is the production path. Setting
# VLLM_GFX908_MOE_A8W8=0 (WNA16 experts) is a diagnostic lever only and is
# NOT a supported recipe configuration.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${HOME}/models/Flint-GPTQ-INT8-W8A8-GS128-PTQR-r1s40-512k}"  # retrained on the 512k-scope corpus (rollback: r1s40-v2)
DRAFT_MODEL_DIR="${DRAFT_MODEL_DIR:-${HOME}/models/flint-dflash2-ptqr-r1-int8}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/serve_recipe_flint}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8030}"
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
    if [[ "$previous" == */serve_recipe_flint.sh || "$previous" == serve_recipe_flint.sh ]]; then
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
  export PYTHONPATH="${ROOT_DIR}/python_startup:${ROOT_DIR}:${ROOT_DIR}/../int8-aiter:${PYTHONPATH:-}"
  export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export GPU_ARCHS=gfx908 PYTORCH_ROCM_ARCH=gfx908 VLLM_TARGET_DEVICE=rocm
  export HIP_VISIBLE_DEVICES=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3
  export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
  # vLLM CUSTOM all-reduce is the measured fastest on this mesh; AITER CAR is
  # coherent and remains available via VLLM_ROCM_USE_AITER_CUSTOM_AR=1.
  export VLLM_ROCM_USE_AITER_CUSTOM_AR=0
  # Fused round-to-nearest activation quant (quality default, see
  # INT8_AUDIT_RESULTS.md).
  export VLLM_GFX908_ACT_QUANT=round
  # Deployed decode band (M 65..128, i.e. C12 with NS=7) runs the tuned
  # Triton a8w8-blockscale GEMM off the retained GS128 originals; prefill
  # stays on CK. Measured at M=96: qkv 95.3 -> 34.7us end to end (quant +
  # gemm), gate_up gemm 70.4 -> 52.9. =0 restores the all-CK dispatch and
  # the GS128 free.
  export VLLM_GFX908_DECODE_BLOCKSCALE="${VLLM_GFX908_DECODE_BLOCKSCALE:-1}"
  # LM head: untied per-channel INT8 weights + per-token activation quant
  # through the CK W8A8 GEMM (gfx908 int8 doctrine). Micro at TP4 vocab
  # shard N=37536, K=2048: M=96 bf16 1775us -> int8 310us; M=12 214 -> 144.
  # The draft proposes through the same head, so verify and propose both
  # benefit. Default on; =0 restores the bf16 head for rollback.
  export VLLM_GFX908_INT8_LM_HEAD="${VLLM_GFX908_INT8_LM_HEAD:-1}"
  # Draft MLP: fused gate_up+SwiGLU+GS128 kernel (int8-only draft activations)
  export VLLM_GFX908_DRAFT_GATED_I8="${VLLM_GFX908_DRAFT_GATED_I8:-1}"
  # Routed experts: GPTQ int8-group -> aiter grouped a8w8-blockscale GEMM.
  # This is the production path (W8A8, int8 activations). Do not disable.
  export VLLM_GFX908_MOE_A8W8="${VLLM_GFX908_MOE_A8W8:-1}"
  # Direct FS->GPU weight loading through AMD Infinity Storage. The flint
  # export was fixed for AIS 2026-11-08 (full 38.5 GiB byte gate PASS; the
  # earlier hipFileRead InvalidValue was a stale/partial repack cache).
  # First boot writes the per-rank repack cache (~15 min); later boots hit
  # it. VLLM_FS_GPU=0 remains the plain-safetensors fallback.
  export VLLM_FS_GPU="${VLLM_FS_GPU:-auto}"
  # Flint's checkpoint documents 262,144 but is released with 512k support
  # (recursive.com announcement; card recommends staying <=256k). The guard
  # must be lifted for --max-model-len > the config's max_position_embeddings;
  # the rope tables are extended to the runtime length in the model code.
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
  # INT8 payload boundary at the attention o_proj: the blockscale GEMM emits
  # codes+GS128 scales, the GS128 collective reduces them, and the fused
  # residual+norm+quant seam consumes the payload. Requires the socket-IPC
  # library (built on demand).
  export VLLM_GFX908_O_PROJ_I8="${VLLM_GFX908_O_PROJ_I8:-1}"
  if [[ "$VLLM_GFX908_O_PROJ_I8" == 1 ]]; then
    export VLLM_GFX908_RESIDUAL_INT8="${VLLM_GFX908_RESIDUAL_INT8:-1}"
    export VLLM_RESIDUAL_G128_LIBRARY="${VLLM_RESIDUAL_G128_LIBRARY:-${ROOT_DIR}/vllm/_residual_g128.so}"
    if [[ ! -f "$VLLM_RESIDUAL_G128_LIBRARY" ]]; then
      bash "${ROOT_DIR}/scripts/build_residual_g128.sh" "$VLLM_RESIDUAL_G128_LIBRARY" || {
        echo 'Failed to build the GS128 collective; disabling payload mode' >&2
        export VLLM_GFX908_O_PROJ_I8=0 VLLM_GFX908_RESIDUAL_INT8=0
      }
    fi
  fi
  export NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_P2P_DISABLE=0 NCCL_DMABUF_ENABLE=0
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

  local spec
  spec="$("${ROOT_DIR}/.venv/bin/python" -c 'import json,sys; print(json.dumps(dict(method="dflash",model=sys.argv[1],num_speculative_tokens=int(sys.argv[2]),kv_cache_dtype="int8_block_g128")))' "$DRAFT_MODEL_DIR" "${NS:-7}")"
  local args=(
    -m vllm.entrypoints.openai.api_server
    --model "$MODEL_DIR" --served-model-name flint
    --host "$HOST" --port "$PORT"
    --tensor-parallel-size "${TPS:-4}"
    --max-num-seqs "${C:-12}"
    --dtype bfloat16
    --max-model-len "${MAXLEN:-524288}"
    --max-num-batched-tokens "${MNBT:-2048}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
    # PTQR-era int8_block_g128 is the gated KV dtype for target AND draft
    # (fp16 draft KV is measured broken; bf16 is banned by int8-first).
    --kv-cache-dtype int8_block_g128
    --speculative-config "$spec"
    # Flint's template renders <think> reasoning blocks; the qwen3 engine
    # parser splits them into reasoning_content. No tool-call parser: the
    # checkpoint's XML tool format has no vLLM parser yet.
    --reasoning-parser qwen3
    --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20}'
  )
  if [[ -n "${PROFILE_DIR:-}" ]]; then
    # torch-profiler diagnostic boot: exposes /start_profile + /stop_profile.
    args+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_DIR}\"}")
  fi
  # Mamba prefix-cache mode lever (FlashStates-style recurrent-state reuse).
  # "align" turns on MambaManager's checkpoint machinery so a prefix hit can
  # restore the recurrent state instead of rebuilding it.
  # Default on: without it mamba groups never cache and a warm prefix hit
  # rebuilds the recurrent state (warm TTFT 1165ms vs 391ms for a miss).
  export MAMBA_MODE="${MAMBA_MODE:-align}"
  if [[ -n "${MAMBA_MODE}" ]]; then
    args+=(--mamba-cache-mode "${MAMBA_MODE}")
  fi
  # Mamba group block size. It must exceed the prefix hash unit (64) for
  # partial (sub-block) Mamba hits, i.e. for a warm hit to resume at 64-token
  # granularity instead of a 128-token boundary.
  if [[ -n "${MAMBA_BLOCK:-}" ]]; then
    args+=(--mamba-block-size "${MAMBA_BLOCK}")
  fi
  # Prefix hash unit. Must stay below the Mamba kernel block size (64) for
  # partial (sub-block) Mamba hits -> warm hits resume at this granularity.
  if [[ -n "${MATCH_UNIT:-}" ]]; then
    args+=(--prefix-match-unit "${MATCH_UNIT}")
  fi
  # Async-scheduling A/B lever (None=default). 0 forces sync, 1 forces on.
  if [[ "${ASYNC_SCHED:-}" == "0" ]]; then
    args+=(--no-async-scheduling)
  elif [[ "${ASYNC_SCHED:-}" == "1" ]]; then
    args+=(--async-scheduling)
  fi
  # Batch-sharded sampling: each rank runs the int8 head only for the rows it
  # owns and an all-to-all exchanges vocab shards, replacing the per-step
  # full-vocab logits all-gather. Requires compute_logits_local (Flint has it).
  if [[ "${BATCH_SHARDED:-0}" == "1" ]]; then
    args+=(--enable-batch-sharded-sampling)
  fi
  if [[ "${EAGER:-0}" == 1 ]]; then
    args+=(--enforce-eager)
  else
    # FULL_DECODE_ONLY graphs; the a8w8 MoE path is capture-safe (padding
    # expert ids -1 are zero-weighted, not wrapped to expert 255).
    args+=(--compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CUDAGRAPH_SIZES:-16,32,48,64,96,128}]}")
  fi
  local _launch=()
  if [[ "${ROCPROF:-0}" == "1" ]]; then
    # rocprofv3 boot: kernel dispatch trace for whole-process-tree (graphs
    # included; HIP graph replay dispatches are traced individually). CSV is
    # written incrementally so a SIGTERM stop still leaves usable data.
    local _pdir="${ROCPROF_DIR:-${LOG_DIR}/rocprof}"
    mkdir -p "$_pdir"
    local _period=()
    if [[ -n "${ROCPROF_PERIOD:-}" ]]; then
      # e.g. ROCPROF_PERIOD=420:90:0 -> collect from t=420s for 90s, once.
      # Keeps boot untraced (tracing during weight load wedges the workers).
      _period=(--collection-period "${ROCPROF_PERIOD}")
    fi
    _launch=(rocprofv3 --kernel-trace -f csv -d "$_pdir" "${_period[@]}" --)
  fi
  "${_launch[@]}" "${ROOT_DIR}/.venv/bin/python" "${args[@]}" &
  child=$!
  wait "$child"
}

case "${1:-status}" in
  supervise)
    if alive; then echo 'Flint supervisor is already running' >&2; exit 1; fi
    echo "$$" > "$PID_FILE"
    child=''
    trap 'if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; rm -f "$PID_FILE"; exit' TERM INT
    trap 'rm -f "$PID_FILE"' EXIT
    serve
    ;;
  start)
    if alive; then echo 'Flint supervisor is already running'; exit 0; fi
    nohup setsid bash "$0" supervise >> "${LOG_DIR}/server.log" 2>&1 < /dev/null &
    for _ in $(seq 1 100); do
      if alive; then
        echo "Flint starting; log: ${LOG_DIR}/server.log"
        exit 0
      fi
      sleep 0.1
    done
    echo "Flint supervisor failed to start; see ${LOG_DIR}/server.log" >&2
    exit 1
    ;;
  stop)
    if alive; then
      pid="$(cat "$PID_FILE")"
      if [[ "$(ps -o pgid= -p "$pid" | tr -d ' ')" == "$pid" ]]; then
        kill -TERM -- "-$pid"
      else
        kill -TERM "$pid"
      fi
      for _ in $(seq 1 180); do alive || break; sleep 1; done
      if alive; then
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        for _ in $(seq 1 30); do alive || break; sleep 1; done
      fi
      if alive; then echo 'Supervisor has not exited' >&2; exit 1; fi
    fi
    ;;
  restart) bash "$0" stop; bash "$0" start ;;
  status)
    if alive; then
      echo "Flint supervisor running (PID $(cat "$PID_FILE"))"
      curl --fail --silent --show-error --max-time 3 "http://${HOST}:${PORT}/health"
    else echo 'Flint stopped'; exit 1; fi
    ;;
  *) echo "Usage: $0 {start|stop|restart|status|supervise}" >&2; exit 2 ;;
esac
