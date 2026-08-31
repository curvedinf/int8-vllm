#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${HOME}/vllm-gfx908"
VENV="${ROOT_DIR}/.venv"
MODEL_DIR="${HOME}/models/Qwen3.8-27B-GPTQ-8bit-gs128"
SERVED_MODEL_NAME="qwen3.8-27b-gptq8"
LOG_DIR="${ROOT_DIR}/logs/serve_recipe_qwen38"
# The OffloadingConnector's CPU tier mmaps /dev/shm; an unclean kill leaks
# the 12 GiB buffer and (with psm_* churn segments) exhausts the tmpfs,
# crash-looping every subsequent boot during KV-cache init. Clean orphans
# before starting (safe: no live server exists at this point).
find /dev/shm -maxdepth 1 -user "$(id -un)" \( -name 'vllm_offload_*.mmap' -o -name 'psm_*' \) -delete 2>/dev/null || true
PID_FILE="${LOG_DIR}/server.pid"
API_KEY_FILE="/etc/llama/llama-api.key"
WORKDIR="/tmp"

HOST="127.0.0.1"
PORT=8020
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-2400}"
STOP_TIMEOUT=20
CPUSET="0-47"

# UA flag file (diagnostic lever for systemd-driven boots: UA=0 selects the
# Triton unified-attention path instead of AITER's). Must resolve BEFORE
# COMMON_ENV, which consumes ${UA:-1}.
_ua_flag="${LOG_DIR}/UA"
if [[ -f "${_ua_flag}" ]]; then
  UA="$(tr -d '[:space:]' < "${_ua_flag}")"
fi

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
  # CAR default OFF after an auditable 2026-08-24 TP4/C8 rerun. Median
  # sustained output throughput over three deterministic repeats:
  #   vLLM CUSTOM (CAR=0, AR=1): 63.49 tok/s, TPOT 105.00 ms
  #   AITER CAR   (CAR=1, AR=1): 58.34 tok/s, TPOT 114.27 ms
  #   PYNCCL      (CAR=0, AR=0): 53.04 tok/s, TPOT 133.89 ms
  # AITER CAR is coherent after the graph repair, but its gfx908 path still
  # forces the naive kernel with generic launch defaults. CAR=1 remains a
  # tuning control; CAR=0/AR=1 selects the current fastest vLLM CUSTOM path.
  VLLM_ROCM_USE_AITER_CUSTOM_AR="${CAR:-0}"
  VLLM_ROCM_USE_AITER_TRITON_GEMM="1"
  VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION="${UA:-1}"
  # Full W8A8 doctrine, gated pieces: int8 lm_head (A1 gate-passed, neutral
  # acceptance, halves head memory) and DFlash2 conv/selector projections
  # (A2 gate-passed, acceptance +10.7pt). Fused norm+quant stays OFF
  # (B1 gate-failed — see envs.py comment).
  VLLM_GFX908_INT8_LM_HEAD="${INT8HEAD:-1}"
  VLLM_GFX908_DF2_W8A8="${DF2W8A8:-1}"
  # Pin the remaining W8A8-stack defaults explicitly so a future env default
  # change cannot silently regress the recipe: CK GEMM path on, dead GS128
  # weight copies freed, int8 embedding gather.
  VLLM_GFX908_CK_W8A8="${CKW8A8:-1}"
  VLLM_GFX908_CK_FREE_GS128="${FREEGS128:-1}"
  VLLM_GFX908_INT8_EMBEDDING="${INT8EMB:-1}"
  # The compatibility-named AiterW8A16LinearKernel is the GS128 selector;
  # this branch routes every GS128 shape through its AITER A8W8 INT8 path.
  VLLM_DISABLED_KERNELS="${VLLM_DISABLED_KERNELS:-TritonW8A16LinearKernel}"
  NCCL_ALGO="Ring"
  NCCL_PROTO="Simple"
  NCCL_P2P_DISABLE="0"
  NCCL_DMABUF_ENABLE="0"
  NCCL_DEBUG="INFO"
  RCCL_LOG_LEVEL="INFO"
)

DRAFT_MODEL_DIR="${DRAFT_MODEL_DIR:-${HOME}/.cache/huggingface/dflash2-int8/Qwen3.8-27B-DFlash2-GPTQ-8bit}"

ARGS=(
  serve "${MODEL_DIR}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size 4
  --dtype "${DTYPE:-half}"
  --max-model-len 262144
  --max-num-seqs 6
  --gpu-memory-utilization 0.92
  # KVMEM flag file / env: pin the KV arena size in bytes (gpu_worker reports
  # ~1.4GiB/GPU unused headroom at 0.92 util; a pinned --kv-cache-memory
  # both recovers it and makes the pool deterministic across boots).
  # ${LOG_DIR}/KVMEM (e.g. "19000000000") overrides the env for systemd boots.
  --compilation-config '{"mode":'"${CMODE:-3}"',"cudagraph_mode":"'"${CGMODE:-FULL_AND_PIECEWISE}"'","custom_ops":["+gemma_rms_norm","+silu_and_mul","+rms_norm_gated","+rotary_embedding","+apply_rotary_emb","none"],"pass_config":{"fuse_allreduce_rms":'"${ARFUSE:-false}"'}}'
  --language-model-only
  --skip-mm-profiling
  --disable-uvicorn-access-log
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  # The chat template opens <think> on every assistant turn; without a
  # reasoning parser the think block leaks into content and clients read the
  # scratchpad as the answer (looks like total instruction-following failure).
  # The qwen3 engine parser splits it into reasoning_content.
  --reasoning-parser qwen3
  # Qwen3.8-27B model-card defaults (HF): temperature 1.0 / top_p 0.95 /
  # top_k 20, no repetition or presence penalties. Thinking LOW: this
  # checkpoint's non-thinking mode is the weak one — the template's
  # reasoning_effort=low keeps brief thinking on (its supported levels are
  # xhigh/medium/low; xhigh is the template default).
  --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"low"}'
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}'
  # GDN recurrent state: fp32 — REQUIRED for quality. Phase-1 replay convicted
  # the fp16 state round-trip: decode re-stores h in fp16 EVERY token
  # (fused_sigmoid_gating_delta_rule_update), and once |h|*2^-11 > beta*|v| the
  # delta-rule cancellation fails and |h| grows multiplicatively — recorded
  # bootA states hit 63,245 (4% under the fp16 ceiling 65,504) while the BF16
  # reference stays ~1.2 (early layers carry exp(A) up to 60). This is the
  # generator of the KLD tail (recipe-vs-BF16 p95 11.36). The checkpoint
  # itself declares mamba_ssm_dtype float32; int8 state remains banned
  # (corruption bisect 2026-08-25) until a scaled-int8 kernel exists.
  # MAMBADT env remains the bisect lever.
  --kv-cache-dtype "${KV_DTYPE:-int8_per_token_head}" --mamba-ssm-cache-dtype "${MAMBADT:-float32}"
  # NS=13 default per the 2026-08-26 tuned-aiter sweep (see docs/recipes
  # README history): best measured TPOT 12.34 ms / TG 639-equivalent regime.
  # NS=15 prior default (2026-08-24 sweep) measured 18.89 ms same-session;
  # NS=17 collapses (29.7% acceptance — under investigation).
  # Draft KV int8-PTH: full-W8A8 doctrine.
  # SPECOFF=1 drops the draft entirely (diagnostic target-only legs).
  # The speculative-config is appended conditionally after ARGS below.
  # CPU KV second tier: 12 GiB total (cross-worker) host DRAM via the native
  # OffloadingConnector. Blocks are copied as raw bytes, so int8-PTH inline
  # scales and the fp32 mamba state pages transfer dtype-safely. L2 reuse
  # cache only — the live arena stays on-GPU.
  # 2026-08-30: the connector's dflash draft-group misclassification was
  # fixed (eagle catch-all flagged every group incl. mamba -> misaligned
  # resume boundaries -> concurrent garble/wedges; see offloading/scheduler.py
  # and logs/garble/NOTES.md). Tier ON per the recipe.
)
if [[ "${OFFLOAD:-1}" == "1" ]]; then
  ARGS+=(--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":12884901888}}')
fi
# OFFLOAD flag file (diagnostic lever for systemd-driven restarts).
_offload_flag="${LOG_DIR}/OFFLOAD"
if [[ -f "${_offload_flag}" ]]; then
  _offload_value="$(tr -d '[:space:]' < "${_offload_flag}")"
  if [[ "${_offload_value}" != "1" ]]; then
    _filtered=()
    _skip_next=0
    for _a in "${ARGS[@]}"; do
      if [[ "${_a}" == --kv-transfer-config* ]]; then
        # The value is a separate argv element unless '='-joined.
        _skip_next=1
        continue
      fi
      if [[ "${_skip_next}" == 1 ]]; then
        _skip_next=0
        continue
      fi
      _filtered+=("$_a")
    done
    ARGS=("${_filtered[@]}")
  fi
fi

# Draft KV int8-PTH: full-W8A8 doctrine. SPECOFF=1 drops the draft for
# diagnostic target-only legs. Flag-file override mirrors the levers above
# (systemd-driven restarts cannot pass per-boot env).
_spec_flag="${LOG_DIR}/SPECOFF"
if [[ -f "${_spec_flag}" ]]; then
  _spec_value="$(tr -d '[:space:]' < "${_spec_flag}")"
else
  _spec_value="${SPECOFF:-0}"
fi
# NS flag file (diagnostic lever for systemd-driven boots, e.g. NS=2 window test)
_ns_flag="${LOG_DIR}/NS"
if [[ -f "${_ns_flag}" ]]; then
  NS="$(tr -d '[:space:]' < "${_ns_flag}")"
fi
if [[ "${_spec_value}" != "1" ]]; then
  ARGS+=(--speculative-config '{"method":"dflash","model":"'"${DRAFT_MODEL_DIR}"'","num_speculative_tokens":'"${NS:-13}"',"kv_cache_dtype":"'"${DRAFT_KV_DTYPE:-int8_per_token_head}"'"}')
fi

# LOGSTATS=1 enables periodic engine/spec-decode stat logging
# (default off: --disable-log-stats).
if [[ "${LOGSTATS:-0}" != "1" ]]; then
  ARGS+=(--disable-log-stats)
fi

# PREFIXCACHE=0 disables prefix caching (diagnostic lever for the garble
# bisect: isolates the hybrid mamba-align prefix-hit machinery). Flag-file
# override mirrors OFFLOAD above for systemd-driven restarts.
_prefix_flag="${LOG_DIR}/PREFIXCACHE"
if [[ -f "${_prefix_flag}" ]]; then
  _prefix_value="$(tr -d '[:space:]' < "${_prefix_flag}")"
else
  _prefix_value="${PREFIXCACHE:-1}"
fi
if [[ "${_prefix_value}" == "0" ]]; then
  ARGS+=(--no-enable-prefix-caching)
fi

# C6 means six concurrent sequences. The target and DFlash2 draft are both
# GPTQ INT8 GS128. The actual runtime selections are reported in the startup
# log; do not infer AITER CAR, fused quant-out, or draft INT8 KV from this
# contract comment.
# Spec-decode token budget; MNBT and NS remain tuning controls only.
MNBT="${MNBT:-2048}"
ARGS+=(--max-num-batched-tokens "${MNBT}")
# KVMEM: optional pinned KV cache size in bytes (flag file or env).
# Default 20.2 GiB (2026-08-30): recovers the ~1.4 GiB/GPU the 0.92-util
# profiler leaves unused -> 1,031,145-token arena (was 982,523; +4.9%
# capacity, 3.93x max-len). Concurrent-round stable, 0 collapse windows.
_kvmem_flag="${LOG_DIR}/KVMEM"
if [[ -f "${_kvmem_flag}" ]]; then
  KVMEM="$(tr -d '[:space:]' < "${_kvmem_flag}")"
fi
KVMEM="${KVMEM:-20200000000}"
ARGS+=(--kv-cache-memory "${KVMEM}")
# GDNDUMP: optional dir for the spec-rewind audit lever (gdn_attn.py dumps
# per-token state indices + accepted counts per step when non-empty).
_gdndump_flag="${LOG_DIR}/GDNDUMP"
if [[ -f "${_gdndump_flag}" ]]; then
  VLLM_GDN_DUMP_DIR="$(tr -d '[:space:]' < "${_gdndump_flag}")"
fi
# GDNRING: zero-overhead ring capture (no device sync; dumped at process
# exit). Prefer this over GDNDUMP — the per-step dump's sync masks the garble.
_gdnring_flag="${LOG_DIR}/GDNRING"
if [[ -f "${_gdnring_flag}" ]]; then
  VLLM_GDN_RING="$(tr -d '[:space:]' < "${_gdnring_flag}")"
fi
# CANDRING flag file: DFlash2 candidate ring (per-round ids/scores/drafts).
_candring_flag="${LOG_DIR}/CANDRING"
if [[ -f "${_candring_flag}" ]]; then
  VLLM_CAND_RING="$(tr -d '[:space:]' < "${_candring_flag}")"
fi

# SPECDBG flag file: per-round speculator candidate/debug prints
# (SPEC-DBG3/4) into server.log.
_specdbg_flag="${LOG_DIR}/SPECDBG"
if [[ -f "${_specdbg_flag}" ]]; then
  VLLM_SPEC_DEBUG_DUMP=1
fi

# ASMRING flag file: verify-input assembly audit ring (dumped at exit).
_asmring_flag="${LOG_DIR}/ASMRING"
if [[ -f "${_asmring_flag}" ]]; then
  VLLM_ASM_RING="$(tr -d '[:space:]' < "${_asmring_flag}")"
fi

# NOLOADS flag file: bypass tier load serving (diagnostic lever).
_noloads_flag="${LOG_DIR}/NOLOADS"
if [[ -f "${_noloads_flag}" ]]; then
  VLLM_OFFLOAD_NO_LOADS=1
fi
# GDNBISECT: race-bisect lever injected into the GDN metadata build —
# one of: sleep | stream | device (see gdn_attn.py).
_gdnbisect_flag="${LOG_DIR}/GDNBISECT"
if [[ -f "${_gdnbisect_flag}" ]]; then
  VLLM_GDN_BISECT="$(tr -d '[:space:]' < "${_gdnbisect_flag}")"
fi
# AR=0 fully disables custom all-reduce (PYNCCL/RCCL path); AR=1 leaves the
# selected custom backend enabled. With the defaults CAR=0/AR=1, vLLM CUSTOM
# is selected ahead of PYNCCL.
AR="${AR:-1}"
if [[ "${AR}" != "1" ]]; then
  ARGS+=(--disable-custom-all-reduce)
fi

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

  printf 'starting recipe Qwen3.8 server: url=http://%s:%s cpuset=%s log=%s/server.log\n' \
    "${HOST}" "${PORT}" "${CPUSET}" "${LOG_DIR}"
  printf '%s\n' 'contract: target+DFlash2 GS128; AITER W8A8/UA/custom-AR; INT8 KV/Mamba/quant-out; TP4/C6; 12GiB CPU KV tier'

  local api_key
  api_key="$(read_api_key)"

  (
    cd "${WORKDIR}"
    exec setsid taskset -c "${CPUSET}" env -u HSA_OVERRIDE_GFX_VERSION \
      VLLM_API_KEY="${api_key}" \
      "${COMMON_ENV[@]}" \
  VLLM_SPEC_DEBUG_DUMP="${VLLM_SPEC_DEBUG_DUMP:-}" \
  VLLM_CAND_RING="${VLLM_CAND_RING:-}" \
  VLLM_GFX908_ACT_QUANT="${VLLM_GFX908_ACT_QUANT:-round}" \
  VLLM_GDN_DUMP_DIR="${VLLM_GDN_DUMP_DIR:-}" \
  VLLM_GDN_RING="${VLLM_GDN_RING:-}" \
  VLLM_GDN_BISECT="${VLLM_GDN_BISECT:-}" \
  VLLM_ASM_RING="${VLLM_ASM_RING:-}" \
  VLLM_SPEC_DEBUG_DUMP="${VLLM_SPEC_DEBUG_DUMP:-}" \
  VLLM_CAND_RING="${VLLM_CAND_RING:-}" \
  VLLM_OFFLOAD_NO_LOADS="${VLLM_OFFLOAD_NO_LOADS:-}" \
  VLLM_DFLASH_DRAFT_EAGER="${VLLM_DFLASH_DRAFT_EAGER:-}" \
      VLLM_DFLASH_AUDIT="${VLLM_DFLASH_AUDIT:-}" \
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
    rg -n 'Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency|via P2P/IPC|Uvicorn running|dflash|DFlash|spec|ffload' \
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
