#!/usr/bin/env bash
# Adaptive ninja supervisor (GOALOPT): ramp -j during the build.
# Usage: ninja_ramp.sh <build_dir> <log_file>
# Every 45s: avail > 35 GB and jobs < max  -> kill ninja, relaunch at next level.
#            avail < 12 GB                  -> kill ninja, drop one level.
#            otherwise                      -> keep running.
# Ninja resumes incrementally; killing only loses in-flight compiles.
set -u
BD="$1"; LOG="$2"
cd "$BD" || exit 2
export PATH="/home/curved/vllm-gfx908/.venv/bin:/opt/rocm/bin:$PATH"

avail_mb() { awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo; }
rss_mb() {
  ps -eo rss,comm | awk '$2 ~ /hipcc|clang|cc1|ninja/ {s+=$1} END {printf "%d", s/1024}'
}

LEVELS=(8 12 16 20 24 32)
idx=0
NJA=""

log() { echo "$(date +%T) $*" >> "$LOG"; }

while :; do
  if ! ninja -n >/dev/null 2>&1; then
    log "BUILD COMPLETE"
    break
  fi
  [ -f module_gemm_a8w8_tune.so ] && { log "SO_LINKED"; break; }
  j=${LEVELS[$idx]}
  log "launch -j$j (avail=$(avail_mb)MB rss=$(rss_mb)MB)"
  ninja -j"$j" >> "$LOG" 2>&1 &
  NJA=$!
  while kill -0 "$NJA" 2>/dev/null; do
    sleep 45
    A=$(avail_mb)
    log "  -j$j avail=${A}MB rss=$(rss_mb)MB"
    if [ "$A" -lt 12000 ] && [ "$idx" -gt 0 ]; then
      log "  LOW MEMORY -> drop to ${LEVELS[$((idx - 1))]}"
      kill -9 "$NJA" 2>/dev/null
      pkill -9 -P "$NJA" 2>/dev/null
      sleep 8
      idx=$((idx - 1))
      break
    elif [ "$A" -gt 35000 ] && [ "$idx" -lt $((${#LEVELS[@]} - 1)) ]; then
      log "  HEADROOM -> raise to ${LEVELS[$((idx + 1))]}"
      kill -9 "$NJA" 2>/dev/null
      pkill -9 -P "$NJA" 2>/dev/null
      sleep 3
      idx=$((idx + 1))
      break
    fi
  done
  wait "$NJA" 2>/dev/null
  if ! kill -0 "$NJA" 2>/dev/null && ninja -n >/dev/null 2>&1; then
    :
  fi
done
ls -la module_gemm_a8w8_tune.so 2>/dev/null >> "$LOG"
