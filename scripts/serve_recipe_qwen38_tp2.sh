#!/usr/bin/env bash
# Preserve the two-card service interface, port and PID/log location.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
case "${1:-}" in
  start|stop|restart|status|supervise) ;;
  *) echo "Usage: $0 {start|stop|restart|status|supervise}" >&2; exit 2 ;;
esac
export MI100_RECIPE=qwen38 MI100_CARDS=2 MI100_XGMI="${MI100_XGMI:-off}"
export PORT="${PORT:-8021}" LOG_DIR="${LOG_DIR:-$root/logs/serve_recipe_qwen38_tp2}"
exec "$root/scripts/mi100-serve.sh" "$1"
