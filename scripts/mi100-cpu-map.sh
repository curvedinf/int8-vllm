# Sourced by the native recipe scripts and the MI100 hardware check.
cpu_bind_args=()
if [[ -n "${MI100_CPU_MAP:-}" ]]; then
  # ROCr otherwise lets its helper threads run outside the worker CPU list.
  export HSA_OVERRIDE_CPU_AFFINITY_DEBUG="${HSA_OVERRIDE_CPU_AFFINITY_DEBUG:-0}"
  # mi100-serve validates before downloading models or starting a supervisor.
  # Direct recipe and systemd launches resolve the same flags here.
  if [[ -z "${_MI100_CPU_BIND_FLAGS:-}" ]]; then
    _mi100_scripts=$(dirname -- "${BASH_SOURCE[0]}")
    _MI100_CPU_BIND_FLAGS=$("$_mi100_scripts/../.venv/bin/python" "$_mi100_scripts/mi100_cpu_map.py")
    unset _mi100_scripts
  fi
  read -r -a cpu_bind_args <<< "$_MI100_CPU_BIND_FLAGS"
fi
