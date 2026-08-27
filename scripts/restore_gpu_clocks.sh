#!/usr/bin/env bash
# Restore MI100 clock/power state after the 2026-08-26 tuner pin.
# Idempotent; safe to run repeatedly. Requires root to change anything.
set -u

ok=1
for d in /sys/class/drm/card*/device; do
  lvl="$(cat "$d/power_dpm_force_performance_level" 2>/dev/null || echo unknown)"
  [[ "${lvl}" != "auto" ]] && ok=0
done
for h in /sys/class/hwmon/hwmon*/power1_cap; do
  cap="$(cat "$h" 2>/dev/null || echo 0)"
  (( cap < 290000000 )) && ok=0
done

if (( ok )); then
  echo "GPU clock state OK (DPM=auto, cap=290W)"
  exit 0
fi

if [[ "$(id -u)" != "0" ]]; then
  echo "GPUs are throttled (DPM manual @ sclk level 0 = 300MHz, power cap 105W)." >&2
  echo "Re-run this script as root:" >&2
  echo "  sudo $0" >&2
  exit 1
fi

for d in /sys/class/drm/card*/device; do
  echo auto > "$d/power_dpm_force_performance_level" 2>/dev/null || true
done
rocm-smi --setpoweroverdrive 290 >/dev/null 2>&1 || true

ok=1
for d in /sys/class/drm/card*/device; do
  lvl="$(cat "$d/power_dpm_force_performance_level" 2>/dev/null || echo unknown)"
  [[ "${lvl}" != "auto" ]] && ok=0
done
for h in /sys/class/hwmon/hwmon*/power1_cap; do
  cap="$(cat "$h" 2>/dev/null || echo 0)"
  (( cap < 290000000 )) && ok=0
done
if (( ok )); then
  echo "restored: DPM=auto, power cap=290W"
else
  echo "WARN: still throttled after restore attempt" >&2
  exit 1
fi
