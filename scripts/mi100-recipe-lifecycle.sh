# Sourced after the recipe defines serve(), launcher, served_name, logs and recipe_label.
is_running() {
  [[ -r "$pid_file" ]] || return 1
  local pid arg previous='' api=0
  read -r pid <"$pid_file"
  [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || return 1
  while IFS= read -r -d '' arg; do
    if [[ "$previous" == "$launcher" && "$arg" == serve ]]; then return 0; fi
    if [[ "$arg" == vllm.entrypoints.openai.api_server ]]; then api=1; fi
    if [[ "$previous" == --served-model-name && "$arg" == "$served_name" && "$api" == 1 ]]; then return 0; fi
    previous="$arg"
  done <"/proc/$pid/cmdline"
  return 1
}

case "${1:-serve}" in
  serve|supervise) [[ $# -eq 0 ]] || shift; serve "$@" ;;
  start)
    mkdir -p -- "$logs"
    exec 9>"$logs/recipe.lock"
    flock 9
    if is_running; then echo "$recipe_label is already running"; exit 0; fi
    shift
    setsid "$launcher" serve "$@" </dev/null >>"$logs/server.log" 2>&1 9>&- &
    pid=$!
    printf '%s\n' "$pid" >"$pid_file"
    # Keep the start lock through the short fork/exec interval. Other callers
    # can identify the process once it enters the launcher or API server.
    for ((attempt=0; attempt<200; attempt++)); do
      is_running && break
      kill -0 "$pid" 2>/dev/null || { rm -f -- "$pid_file"; wait "$pid"; exit 1; }
      sleep 0.01
    done
    if ! is_running; then echo "$recipe_label launch could not be verified. See $logs/server.log" >&2; exit 1; fi
    echo "$recipe_label starting, log at $logs/server.log"
    ;;
  stop)
    [[ -d "$logs" ]] || exit 0
    exec 9>"$logs/recipe.lock"
    flock 9
    if is_running; then
      read -r pid <"$pid_file"
      # A startup failure can let the API process exit before its workers
      # finish loading. Signal the dedicated group while its leader is known.
      if [[ "$(ps -o sid= -p "$pid" | tr -d ' ')" == "$pid" ]]; then
        kill -TERM -- "-$pid" 2>/dev/null || true
      else
        kill -TERM "$pid" 2>/dev/null || true
      fi
      while is_running; do sleep 1; done
      while ps -s "$pid" -o stat= | grep -Eq '^[[:blank:]]*[^Z[:space:]]'; do sleep 1; done
    fi
    rm -f -- "$pid_file"
    ;;
  restart) shift; "$launcher" stop; exec "$launcher" start "$@" ;;
  status) if is_running; then echo "$recipe_label is running"; else echo "$recipe_label is stopped"; exit 1; fi ;;
  *) echo "Usage: $0 {serve|start|stop|restart|status} [vLLM options]" >&2; exit 2 ;;
esac
