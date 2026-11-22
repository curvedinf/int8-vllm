#!/usr/bin/env bash
# Single-host MI100 recipe entry point for native and container installs.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cards=${MI100_CARDS:-4}
link=${MI100_XGMI:-on}
recipe=${MI100_RECIPE:-qwen38}
action=${1:-serve}
case "$action" in
  serve|check|start|stop|status|restart|supervise) ;;
  *) echo "Usage: $0 [serve|check|start|stop|status|restart|supervise]" >&2; exit 2 ;;
esac
pp=1
tp=$cards
case "$cards" in
  2) devices=0,1 ;;
  3) devices=0,1,2; if [[ "$recipe" != hy4 ]]; then tp=1; pp=3; fi ;;
  4) devices=0,1,2,3 ;;
  *) echo "MI100_CARDS must be 2, 3 or 4." >&2; exit 2 ;;
esac
case "$link" in
  on) ;;
  off)
    # Use topology-aware RCCL. PCIe peer access remains available when
    # the host supports it. These switches do not rewire physical links.
    export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
    export AR=0 ;;
  *) echo "MI100_XGMI must be on or off." >&2; exit 2 ;;
esac
export TPS="$tp" MI100_PP_SIZE="$pp"
# hipFile synchronizes each small cache write on some filesystems. Use
# buffered writes and one final fsync when constructing the rank cache.
# The read path, including subsequent direct-to-GPU boots, stays the same.
export VLLM_AIS_REPACK_CPU_WRITE="${VLLM_AIS_REPACK_CPU_WRITE:-1}"
if [[ "$pp" == 3 ]]; then
  # All draft query-head counts are 32, so TP3 is impossible. Partition
  # target layers instead and place the complete drafter on the last stage.
  export AR=0 VLLM_TP_AR_FP32=0
fi
if [[ "$recipe" != flint && ( "$link" == off || "$pp" == 3 ) ]]; then
  export VLLM_GFX908_O_PROJ_I8=0 VLLM_GFX908_RESIDUAL_INT8=0
fi
if [[ "$cards" == 2 ]]; then
  # Keep different TP shards out of the four-card production repack cache.
  export VLLM_AIS_REPACK_DIR="${VLLM_AIS_REPACK_DIR:-${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}/ais_repack_tp2}"
fi
export LOG_DIR="${LOG_DIR:-$root/logs/mi100/$recipe-$cards-$link}"
case "$recipe" in
  qwen38)
    target=Qwen3.8-27B-PTQR-R10S60
    draft=dflash2-ptqr-r1
    target_repo=curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128-PTQR
    target_rev=6f1c4a384442bc4e6dc7df0a8a99bd82c3ab775f
    draft_repo=curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128-PTQR
    draft_rev=7db6ca87ff6c68a77fc220f62bb1166666449055
    export C="${C:-8}" DISKTIER="${DISKTIER:-0}"
    if [[ "$cards-$link" == 4-off ]]; then
      # Four RCCL ranks need a fixed FP32 summation order across prefill
      # chunk sizes. The XGMI profile provides that order in custom AR.
      export VLLM_TP_AR_FP32="${VLLM_TP_AR_FP32:-gather}"
    elif [[ "$link" == off ]]; then
      export VLLM_TP_AR_FP32="${VLLM_TP_AR_FP32:-1}"
    else
      export VLLM_TP_AR_FP32="${VLLM_TP_AR_FP32:-car1}"
    fi
    export VLLM_GFX908_DF2_CODEBOOK_I8="${VLLM_GFX908_DF2_CODEBOOK_I8:-0}"
    # Qwen uses CK W8A8 for both prefill and decode on every layout.
    export VLLM_GFX908_DECODE_BLOCKSCALE="${VLLM_GFX908_DECODE_BLOCKSCALE:-0}"
    if [[ "$cards" == 4 ]]; then
      export VLLM_GDN_PREFILL_EXACT="${VLLM_GDN_PREFILL_EXACT:-1}"
    fi
    if [[ "$cards-$link" == 2-off ]]; then
      # RCCL 2.32 can leave 29.40 GiB free per 31.98 GiB card after TP2
      # initialization, just below the 29.43 GiB requested at 92%.
      export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
    fi
    export KVMEM="${KVMEM:-auto}"
    if [[ "$cards" != 4 ]]; then export MAXLEN="${MAXLEN:-auto}"; fi ;;
  flint)
    # Flint's MoE passes GS128 payloads between layers even on PCIe.
    # The collective reduces their FP32 values on non-mesh topologies.
    export VLLM_GFX908_RESIDUAL_INT8=1
    target=Flint-GPTQ-INT8-W8A8-GS128-PTQR-r1s40-512k
    draft=flint-dflash2-ptqr-r1-int8
    target_repo=voodooquant/Flint-GPTQ-INT8-PTQR
    target_rev=4174bc9971eaa1e6a8a5b49e185e084a449b0872
    draft_repo=voodooquant/Flint-DFlash2-GPTQ-INT8-PTQR
    draft_rev=6148b0a4f3df2b732c96289e4333b9b66741e222
    if [[ "$cards" != 4 ]]; then export MAXLEN="${MAXLEN:-auto}"; fi ;;
  bolt)
    target=NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB
    draft=NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr
    if [[ "$cards" != 4 ]]; then
      export VLLM_GFX908_RESIDUAL_INT8=0
    fi
    if [[ "$pp" == 3 ]]; then export PREFETCH=0; fi
    target_repo=nvidia/$target
    target_rev=a326822af111290f2af5ac59ef4b5040347dabc4
    draft_repo=nvidia/$draft
    draft_rev=4c6a5613b6d887194a3a5493e3a18a5b3930bb86
    if [[ "$cards" != 4 ]]; then
      export VLLM_BOLT_EXPERT_CACHE_GIB="${VLLM_BOLT_EXPERT_CACHE_GIB:-3}" KVMEM="${KVMEM:-4000000000}"
      export MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"
    fi ;;
  hy4)
    if [[ "${MI100_CONTAINER:-0}" == 1 ]]; then
      # Bound RCCL staging so the four ROCm 10.2 workers retain room for KV.
      export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-1048576}"
      export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
    fi
    target=Hy4-flash-NeUQI-W4A8
    draft=Hy4-flash-DFly2-PTQR-INT8
    target_repo=voodooquant/Hy4-flash-NeUQI-INT4-W4A8-GS128
    target_rev=6bcd3e95f31ff6a23201e99273ddf5ab134faec0
    draft_repo=voodooquant/Hy4-flash-DFly2-PTQR-INT8-W8A8-GS128
    draft_rev=026c6c08dc582c7dca03b4db4c14dcf85a9d006a
    export VLLM_API_KEY="${VLLM_API_KEY:-${LLAMA_API_KEY:-}}"
    if [[ -z "${API_KEY_FILE:-}" && -r /etc/llama/llama-api.key ]]; then
      export API_KEY_FILE=/etc/llama/llama-api.key
    fi ;;
  *) echo "MI100_RECIPE must be qwen38, flint, bolt or hy4." >&2; exit 2 ;;
esac
launcher="$root/scripts/serve_recipe_$recipe.sh"
if [[ "$action" == status || "$action" == stop ]]; then
  exec "$launcher" "$action"
fi
resolve_model() {
  local path=$1 repo=$2 explicit=$3 revision=$4
  if [[ -r "$path/config.json" ]]; then
    printf '%s\n' "$path"
  elif [[ -n "$explicit" ]]; then
    echo "Missing checkpoint config: $path/config.json" >&2
    return 1
  else
    "$root/.venv/bin/python" - "$repo" "$revision" <<'PYTHON'
import sys
from huggingface_hub import snapshot_download
repo, revision = sys.argv[1:]
print(f"Resolving {repo} at {revision} in the Hugging Face cache", file=sys.stderr, flush=True)
path = snapshot_download(repo, revision=revision)
print(f"Checkpoint ready {repo} at {revision}", file=sys.stderr, flush=True)
print(path)
PYTHON
  fi
}
visible=${MI100_DEVICES:-${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-$devices}}}
IFS=, read -ra gpu_ids <<< "$visible"
if (( ${#gpu_ids[@]} < cards )); then
  echo "Requested $cards cards but the GPU visibility list contains ${#gpu_ids[@]}." >&2
  exit 2
fi
devices=$(IFS=,; echo "${gpu_ids[*]:0:cards}")
export MI100_DEVICES="$devices" HIP_VISIBLE_DEVICES="$devices" CUDA_VISIBLE_DEVICES="$devices"
printf 'MI100 recipe=%s HIP devices=%s XGMI=%s TP=%s PP=%s\n' \
  "$recipe" "$devices" "$link" "$tp" "$pp" >&2
unset _MI100_CPU_BIND_FLAGS
if [[ -n "${MI100_CPU_MAP:-}" ]]; then
  source "$root/scripts/mi100-cpu-map.sh"
  export _MI100_CPU_BIND_FLAGS
fi
# Fail before a large checkpoint download on a misconfigured host.
# Return the shared CU count separately from library diagnostics.
reported_cus=$("$root/.venv/bin/python" - "$cards" "$link" "$action" 3>&1 1>&2 <<'PYTHON'
import os
import sys
import torch
from vllm.platforms import current_platform

cards = int(sys.argv[1])
if not current_platform.is_rocm() or torch.cuda.device_count() < cards:
    sys.exit(f"Need {cards} visible ROCm devices. Check /dev/kfd and /dev/dri.")
properties = [torch.cuda.get_device_properties(i) for i in range(cards)]
for index, gpu in enumerate(properties):
    arch = gpu.gcnArchName.split(":")[0]
    if arch != "gfx908":
        sys.exit(f"GPU {index} reports {arch}. This recipe requires MI100 (gfx908).")
missing_peers = [
    f"{source}->{dest}" for source in range(cards) for dest in range(cards)
    if source != dest and not torch.cuda.can_device_access_peer(source, dest)
]
if missing_peers:
    print("HIP peer access unavailable for " + ", ".join(missing_peers)
          + ". Transfers may be staged through host memory. "
          "See Common issues in docs/recipes/mi100-install.md.", file=sys.stderr)
if sys.argv[2] == "on":
    import amdsmi

    # HIP and AMD SMI can enumerate the same MI100s in different orders.
    amdsmi.amdsmi_init()
    try:
        by_bdf = {
            amdsmi.amdsmi_get_gpu_device_bdf(handle).lower(): i
            for i, handle in enumerate(amdsmi.amdsmi_get_processor_handles())
        }
        indices = [
            by_bdf[f"{g.pci_domain_id:04x}:{g.pci_bus_id:02x}:{g.pci_device_id:02x}.0"]
            for g in properties
        ]
    finally:
        amdsmi.amdsmi_shut_down()
    if not current_platform.is_fully_connected(indices):
        sys.exit("Selected GPUs do not report a full XGMI mesh. Check rocm-smi --showtopotype or select MI100_XGMI=off.")
if sys.argv[3] == "check":
    for source, gpu in enumerate(properties):
        print(f"GPU {source}: {gpu.name}, {gpu.multi_processor_count} CUs", flush=True)
        expected = torch.arange(1 << 20, dtype=torch.int32, device=f"cuda:{source}")
        for dest in range(cards):
            if source == dest:
                continue
            received = expected.to(f"cuda:{dest}").to(expected.device)
            if not torch.equal(received, expected):
                sys.exit(f"Incorrect round-trip copy from GPU {source} to GPU {dest}.")
            peer = torch.cuda.can_device_access_peer(source, dest)
            print(f"{source}->{dest}: round-trip copy PASS, HIP peer access={peer}")
cu_counts = {gpu.multi_processor_count for gpu in properties}
if len(cu_counts) == 1:
    os.write(3, str(cu_counts.pop()).encode())
PYTHON
)
if [[ "$action" == check ]]; then exit 0; fi
if [[ -n "$reported_cus" ]]; then export CU_NUM="${CU_NUM:-$reported_cus}"; fi
models=${MODELS_DIR:-$HOME/models}
MODEL_DIR=$(resolve_model "${MODEL_DIR:-$models/$target}" "$target_repo" "${MODEL_DIR:-}" "${MODEL_REVISION:-$target_rev}")
DRAFT_MODEL_DIR=$(resolve_model "${DRAFT_MODEL_DIR:-$models/$draft}" "$draft_repo" "${DRAFT_MODEL_DIR:-}" "${DRAFT_MODEL_REVISION:-$draft_rev}")
export MODEL_DIR DRAFT_MODEL_DIR
if [[ -z "${VLLM_API_KEY:-}${LLAMA_API_KEY:-}" && ! -r "${API_KEY_FILE:-/etc/llama/llama-api.key}" ]]; then
  key_dir="${XDG_CACHE_HOME:-$HOME/.cache}/mi100"
  mkdir -p "$key_dir"
  export API_KEY_FILE="$key_dir/api.key"
  # Publish a complete key even when two launchers start simultaneously.
  tmp=$(mktemp "$key_dir/.key.XXXXXX")
  "$root/.venv/bin/python" -c 'import secrets; print(secrets.token_urlsafe(32))' > "$tmp"
  ln "$tmp" "$API_KEY_FILE" 2>/dev/null || [[ -s "$API_KEY_FILE" ]]
  rm -f "$tmp"
  echo "API key file: $API_KEY_FILE" >&2
fi
if [[ "$action" == serve && "$recipe" != qwen38 ]]; then
  action=supervise
fi
exec "$launcher" "$action"
