#!/usr/bin/python3
# SPDX-License-Identifier: Apache-2.0
"""MI100 power-only thermal control. See --help; --check reports the policy.

The boot script owns the fixed 90 W boot setting. This daemon only changes live
caps. A clean stop, including a restart, sets every card to 105 W. Starting the
daemon preserves the live cap and restarts the five-minute warmup; it never
restores a learned cap across a reboot. Power increases require
every card's current GFX utilization to exceed 95%. Thermal reductions do not
depend on load: above 90 C core / 89 C memory reduces the cap by 1 W per five
minutes; above 95 C core / 90 C memory cuts it by 20% immediately, repeating
after ten seconds if still hot. Every reduction respects the minimum cap.
Faults lower caps without stopping any workload. Power bounds reload live.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shlex
import signal
import socket
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path

CONFIG_FILE = Path("/etc/default/rocm-downclock")
RUNTIME_DIR = Path("/run/mi100-thermal-controller")
STATUS_FILE = RUNTIME_DIR / "status.json"
STOP_POWER_W = 105


class SafetyFault(RuntimeError):
    pass


def log(event, **fields):
    print(
        json.dumps(
            {
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": event,
                **fields,
            },
            allow_nan=False,
        ),
        flush=True,
    )


@dataclass(frozen=True)
class Config:
    startup_w: int = 90
    minimum_w: int = 60
    maximum_w: int = 145
    expected_cards: int = 4
    poll_seconds: float = 1.0
    step_w: int = 1
    step_seconds: float = 300.0
    average_seconds: float = 300.0
    peak_seconds: float = 60.0
    core_limit_c: float = 95.0
    memory_limit_c: float = 90.0
    average_limit_c: float = 90.0
    average_target_c: float = 89.0
    average_deadband_c: float = 0.5
    core_soft_c: float = 90.0
    memory_soft_c: float = 89.0
    memory_increase_max_c: float = 88.0
    emergency_fraction: float = 0.20
    emergency_hot_seconds: float = 10.0
    emergency_rearm_seconds: float = 60.0
    maximum_sample_gap_seconds: float = 3.0
    load_minimum_percent: float = 95.0

    @classmethod
    def load(cls, path=CONFIG_FILE, use_environment=True):
        names = {
            "MI100_CAP_W": "startup_w",
            "MI100_THERMAL_MIN_W": "minimum_w",
            "MI100_THERMAL_MAX_W": "maximum_w",
            "MI100_THERMAL_EXPECTED_CARDS": "expected_cards",
        }
        values = {}
        if path.exists():
            for line in path.read_text().splitlines():
                key, sep, value = line.strip().partition("=")
                if sep and key in names:
                    words = shlex.split(value, comments=True)
                    if len(words) != 1:
                        raise SafetyFault(f"Invalid {key} in {path}")
                    values[key] = words[0]
        if use_environment:
            values.update({key: os.environ[key] for key in names if key in os.environ})
        kwargs = {names[key]: int(value) for key, value in values.items()}
        cfg = cls(**kwargs)
        if cfg.startup_w != 90:
            raise SafetyFault("MI100_CAP_W must remain 90 for the fixed boot policy")
        if not 1 <= cfg.minimum_w <= cfg.startup_w <= cfg.maximum_w <= 290:
            raise SafetyFault("Invalid minimum/startup/maximum MI100 power bounds")
        if cfg.expected_cards < 1:
            raise SafetyFault("Expected card count must be positive")
        return cfg


class Window:
    """Time-weighted, piecewise-constant samples using a monotonic clock."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.samples = deque()

    def add(self, now, value):
        if self.samples and now <= self.samples[-1][0]:
            raise SafetyFault("Sensor timestamps must strictly increase")
        self.samples.append((now, value))
        cutoff = now - self.seconds
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()

    def ready(self, now):
        return bool(self.samples and self.samples[0][0] <= now - self.seconds)

    def mean(self, now, require_full=True):
        if not self.samples or (require_full and not self.ready(now)):
            return None
        start = max(now - self.seconds, self.samples[0][0])
        if now <= start:
            return self.samples[-1][1] if not require_full else None
        total = 0.0
        samples = list(self.samples)
        for index, (stamp, value) in enumerate(samples):
            end = samples[index + 1][0] if index + 1 < len(samples) else now
            total += value * max(0.0, min(end, now) - max(stamp, start))
        return total / (now - start)

    def peak(self):
        return max(value for _, value in self.samples)

    def slope(self):
        # A full minute limits sensitivity to individual one-degree readings.
        if len(self.samples) < 2:
            return 0.0
        first, last = self.samples[0][0], self.samples[-1][0]
        if last - first < self.seconds:
            return 0.0
        xs = [stamp - first for stamp, _ in self.samples]
        ys = [value for _, value in self.samples]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        denominator = sum((x - mx) ** 2 for x in xs)
        return (
            sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denominator
            if denominator
            else 0.0
        )


@dataclass
class Sample:
    pci: str
    cap_w: float
    temperatures: dict[str, float]
    busy_percent: float | None


class Hardware:
    """Use PCI identities rather than unstable DRM or rocm-smi indices."""

    def __init__(self, cfg, root=Path("/sys/bus/pci/devices"), strict=True):
        self.cfg, self.root, self.cards = cfg, root, {}
        for dev in sorted(root.iterdir()):
            try:
                matches = (dev / "vendor").read_text().strip() == "0x1002" and (
                    dev / "device"
                ).read_text().strip() == "0x738c"
            except FileNotFoundError:
                continue
            if not matches:
                continue
            hwmons = [
                hw
                for hw in (dev / "hwmon").glob("hwmon*")
                if (hw / "power1_cap").is_file()
            ]
            if len(hwmons) != 1:
                if strict:
                    raise SafetyFault(
                        f"{dev.name}: missing or ambiguous power interface"
                    )
                continue
            self.cards[dev.name] = (dev, hwmons[0])
        if strict and len(self.cards) != cfg.expected_cards:
            raise SafetyFault(
                f"Expected {cfg.expected_cards} MI100s, found {len(self.cards)}"
            )
        if strict:
            self.validate_bounds(cfg)

    def validate_bounds(self, cfg):
        for pci, (_, hw) in self.cards.items():
            minimum = int((hw / "power1_cap_min").read_text()) / 1e6
            maximum = int((hw / "power1_cap_max").read_text()) / 1e6
            if cfg.minimum_w < minimum or cfg.maximum_w > maximum:
                raise SafetyFault(
                    f"{pci}: configured bounds exceed driver range "
                    f"{minimum}..{maximum} W"
                )

    def read(self):
        result = []
        for pci, (dev, hw) in self.cards.items():
            temperatures = {}
            for path in sorted(hw.glob("temp*_input")):
                label = (
                    (hw / (path.name.removesuffix("_input") + "_label"))
                    .read_text()
                    .strip()
                )
                if label in temperatures:
                    raise SafetyFault(f"{pci}: duplicate temperature label {label}")
                value = int(path.read_text()) / 1000
                if not -40 <= value <= 150:
                    raise SafetyFault(f"{pci}/{label}: invalid temperature {value}")
                temperatures[label] = value
            if not {"edge", "junction", "mem"} <= temperatures.keys():
                raise SafetyFault(
                    f"{pci}: missing required edge/junction/memory sensor"
                )
            busy_path = dev / "gpu_busy_percent"
            busy = float(busy_path.read_text()) if busy_path.exists() else None
            if busy is not None and not 0 <= busy <= 100:
                raise SafetyFault(f"{pci}: invalid GPU utilization")
            result.append(
                Sample(
                    pci, int((hw / "power1_cap").read_text()) / 1e6, temperatures, busy
                )
            )
        return result

    def apply(self, watts):
        if not self.cfg.minimum_w <= watts <= self.cfg.maximum_w:
            raise SafetyFault(f"Refusing out-of-range power cap {watts} W")
        expected = round(watts * 1e6)
        # Errors propagate to the failsafe, which handles partial writes too.
        for _, hw in self.cards.values():
            (hw / "power1_cap").write_text(f"{expected}\n")
        for pci, (_, hw) in self.cards.items():
            actual = int((hw / "power1_cap").read_text())
            if actual != expected:
                raise SafetyFault(
                    f"{pci}: cap readback {actual} != {expected} microwatts"
                )


@dataclass
class Decision:
    reason: str
    power_w: int | None = None
    emergency: bool = False


class Controller:
    def __init__(self, cfg, cap_w, now):
        self.cfg, self.cap_w = cfg, cap_w
        self.started = self.last_change = now
        self.last_sample = None
        self.recent, self.averages = {}, {}
        self.gfx_utilization = {}
        self.load_eligible = False
        self.emergency_started = None
        self.last_emergency_cut = None
        self.cool_since = None
        self.stats = {}
        self.last_reason = "warming_up"

    def step(self, samples, now):
        cfg = self.cfg
        if (
            self.last_sample is not None
            and now - self.last_sample > cfg.maximum_sample_gap_seconds
        ):
            raise SafetyFault("Temperature sampling gap exceeded three seconds")
        if (
            len(samples) != cfg.expected_cards
            or len({s.pci for s in samples}) != cfg.expected_cards
        ):
            raise SafetyFault("MI100 card set changed")
        if any(abs(s.cap_w - self.cap_w) > 0.000001 for s in samples):
            raise SafetyFault("Live cap changed outside the controller")
        values = {
            (s.pci, label): value
            for s in samples
            for label, value in s.temperatures.items()
        }
        if self.recent and values.keys() != self.recent.keys():
            raise SafetyFault("MI100 temperature sensor set changed")
        self.last_sample = now
        stats, hot, soft = {}, [], []
        for key, value in values.items():
            memory = (
                key[1] == "mem" or "memory" in key[1].lower() or "hbm" in key[1].lower()
            )
            recent = self.recent.setdefault(key, Window(cfg.peak_seconds))
            recent.add(now, value)
            limit = cfg.memory_limit_c if memory else cfg.core_limit_c
            forecast = value + max(0.0, recent.slope()) * cfg.step_seconds
            soft_limit = cfg.memory_soft_c if memory else cfg.core_soft_c
            row = {
                "temperature_c": value,
                "limit_c": limit,
                "soft_limit_c": soft_limit,
                "maximum_60s_c": recent.peak(),
                "forecast_300s_c": round(forecast, 3),
                "memory": memory,
            }
            if not memory:
                average = self.averages.setdefault(key, Window(cfg.average_seconds))
                average.add(now, value)
                row["average_300s_c"] = average.mean(now)
                mean = row["average_300s_c"]
                if mean is not None and mean > cfg.average_limit_c:
                    soft.append(f"{key[0]}/{key[1]} average {mean:.2f} C")
            if value > limit:
                hot.append((key, value, value - limit))
            if value > soft_limit:
                soft.append(f"{key[0]}/{key[1]} current {value:.1f} C")
            stats[key] = row
        self.stats = stats
        self.gfx_utilization = {s.pci: s.busy_percent for s in samples}
        loaded = all(
            value is not None and value > cfg.load_minimum_percent
            for value in self.gfx_utilization.values()
        )
        self.load_eligible = loaded
        if hot and (
            self.last_emergency_cut is None
            or now - self.last_emergency_cut >= cfg.emergency_hot_seconds
        ):
            if self.emergency_started is None:
                self.emergency_started = now
            self.last_emergency_cut = now
            self.cool_since = None
            requested = math.floor(self.cap_w * (1 - cfg.emergency_fraction))
            target = max(cfg.minimum_w, requested)
            reason = "instant_limit: " + ", ".join(
                f"{pci}/{label}={value:g}C" for (pci, label), value, _ in hot
            )
            if target < self.cap_w:
                self.cap_w, self.last_change = target, now
            else:
                target = None
                reason = "minimum_power_bound; " + reason
            self.last_reason = reason
            return Decision(reason, target, True)

        if self.emergency_started is not None:
            if hot:
                self.cool_since = None
            elif all(
                row["temperature_c"] <= row["limit_c"] - 2 for row in stats.values()
            ):
                if self.cool_since is None:
                    self.cool_since = now
                if now - self.cool_since >= cfg.emergency_rearm_seconds:
                    self.emergency_started = None
                    self.last_emergency_cut = None
                    self.cool_since = None
            else:
                self.cool_since = None

        ready = bool(self.averages) and all(
            window.ready(now) for window in self.averages.values()
        )
        due = now - self.last_change >= cfg.step_seconds
        reason = (
            "holding_for_gfx_load"
            if not loaded
            else "warming_up"
            if not ready
            else "settling"
            if not due
            else "holding_temperature"
        )
        target = None
        if due and soft and self.cap_w > cfg.minimum_w:
            target = max(cfg.minimum_w, self.cap_w - cfg.step_w)
            reason = "reduce: " + "; ".join(soft)
        elif soft and self.cap_w <= cfg.minimum_w:
            reason = "minimum_power_bound"
        elif due and loaded and ready and not soft and self.emergency_started is None:
            averages_cool = all(
                window.mean(now) < cfg.average_target_c - cfg.average_deadband_c
                for window in self.averages.values()
            )
            peaks_cool = all(
                row["maximum_60s_c"] <= cfg.memory_increase_max_c
                and row["forecast_300s_c"] <= cfg.memory_soft_c
                if row["memory"]
                else row["maximum_60s_c"] < cfg.core_soft_c
                and row["forecast_300s_c"] < cfg.core_limit_c - 1
                for row in stats.values()
            )
            if averages_cool and peaks_cool and self.cap_w < cfg.maximum_w:
                target = self.cap_w + cfg.step_w
                reason = "increase: averages/peaks below targets; every card GFX >95%"
            elif self.cap_w >= cfg.maximum_w:
                reason = "maximum_power_bound"
        if target is not None:
            self.cap_w, self.last_change = target, now
        self.last_reason = reason
        return Decision(reason, target)

    def status(self, now):
        return {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "power_cap_w": self.cap_w,
            "boot_power_w": self.cfg.startup_w,
            "minimum_power_w": self.cfg.minimum_w,
            "maximum_power_w": self.cfg.maximum_w,
            "gfx_utilization_percent": self.gfx_utilization,
            "power_increase_load_eligible": self.load_eligible,
            "uptime_seconds": round(now - self.started, 1),
            "warmup_remaining_seconds": round(
                max(0, self.cfg.average_seconds - (now - self.started)), 1
            ),
            "next_normal_adjustment_seconds": round(
                max(0, self.cfg.step_seconds - (now - self.last_change)), 1
            ),
            "reason": self.last_reason,
            "emergency_latched": self.emergency_started is not None,
            "power_only": True,
            "sensors": {
                f"{pci}/{label}": row for (pci, label), row in self.stats.items()
            },
        }


def notify(message):
    address = os.getenv("NOTIFY_SOCKET")
    if address:
        if address.startswith("@"):
            address = "\0" + address[1:]
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.sendto(message.encode(), address)


def write_status(status):
    RUNTIME_DIR.mkdir(mode=0o755, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=RUNTIME_DIR, delete=False) as stream:
        temp = Path(stream.name)
        json.dump(status, stream, allow_nan=False)
        stream.write("\n")
    temp.chmod(0o644)
    temp.replace(STATUS_FILE)


def failsafe(cfg, reason):
    log("failsafe", reason=reason)
    try:
        hardware = Hardware(cfg, strict=False)
        for pci, (_, hw) in hardware.cards.items():
            try:
                cap = hw / "power1_cap"
                target = min(int(cap.read_text()), cfg.minimum_w * 1000000)
                if target <= 0:
                    raise SafetyFault("Invalid existing cap")
                cap.write_text(f"{target}\n")
                if int(cap.read_text()) != target:
                    raise SafetyFault("Failsafe cap readback failed")
                log("failsafe_cap", pci=pci, power_w=target / 1e6)
            except Exception as exc:
                log("failsafe_cap_failed", pci=pci, error=str(exc))
    except Exception as exc:
        log("failsafe_discovery_failed", error=str(exc))


def reload_bounds(cfg, engine, hardware, path):
    # EnvironmentFile belongs to the process's startup environment. Reading
    # the file directly avoids restoring a stale inherited bound on reload.
    candidate = Config.load(path, use_environment=False)
    if replace(candidate, minimum_w=cfg.minimum_w, maximum_w=cfg.maximum_w) != cfg:
        raise SafetyFault("Only minimum/maximum bounds support live reload")
    if not candidate.minimum_w <= engine.cap_w <= candidate.maximum_w:
        raise SafetyFault("Reloaded bounds must contain the current live cap")
    hardware.validate_bounds(candidate)
    hardware.cfg = engine.cfg = candidate
    return candidate


def run(cfg, config_path=CONFIG_FILE):
    RUNTIME_DIR.mkdir(mode=0o755, exist_ok=True)
    # Hold the descriptor for the lifetime of the process. A second invocation
    # must not interfere with the running controller or invoke its failsafe.
    with (RUNTIME_DIR / "controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("already_running")
            return 2
        quit_event = threading.Event()
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: quit_event.set())
        try:
            hardware = Hardware(cfg)
            samples = hardware.read()
            # The ordered boot script sets 90 W on every boot. A daemon-only
            # handover must preserve the live cap, including caps above 90 W.
            initial_cap = min(cfg.maximum_w, math.floor(min(s.cap_w for s in samples)))
            hardware.apply(initial_cap)
            engine = Controller(cfg, initial_cap, time.monotonic())
            next_status = next_log = 0.0
            next_config_check = 0.0
            config_mtime = config_path.stat().st_mtime_ns
            first = True
            while not quit_event.is_set():
                now = time.monotonic()
                if now >= next_config_check:
                    try:
                        current_mtime = config_path.stat().st_mtime_ns
                        if current_mtime != config_mtime:
                            config_mtime = current_mtime
                            cfg = reload_bounds(cfg, engine, hardware, config_path)
                            next_status = 0.0
                            log(
                                "configuration_reloaded",
                                minimum_w=cfg.minimum_w,
                                maximum_w=cfg.maximum_w,
                            )
                    except Exception as exc:
                        log("configuration_reload_rejected", error=str(exc))
                    next_config_check = now + 5
                samples = hardware.read()
                previous_cap = engine.cap_w
                decision = engine.step(samples, now)
                if decision.power_w is not None:
                    hardware.apply(decision.power_w)
                    log(
                        "emergency_cut" if decision.emergency else "power_change",
                        previous_power_w=previous_cap,
                        power_w=decision.power_w,
                        reason=decision.reason,
                    )
                status = engine.status(now)
                if now >= next_status or decision.power_w is not None:
                    write_status(status)
                    next_status = now + 5
                if now >= next_log:
                    log("temperatures", **status)
                    next_log = now + 60
                if first:
                    log("started", policy=asdict(cfg), power_w=initial_cap)
                    notify("READY=1")
                    first = False
                notify(f"WATCHDOG=1\nSTATUS={engine.cap_w} W; {engine.last_reason}")
                quit_event.wait(max(0, now + cfg.poll_seconds - time.monotonic()))
            notify("STOPPING=1")
            log("stopped", power_w=engine.cap_w)
            return 0
        except Exception as exc:
            log("controller_fault", error=str(exc))
            failsafe(cfg, str(exc))
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--check",
        action="store_true",
        help="Read and validate configuration and live sensors; make no changes",
    )
    modes.add_argument(
        "--status", action="store_true", help="Print the running controller status"
    )
    modes.add_argument(
        "--failsafe",
        action="store_true",
        help="Systemd stop handler: set 105 W on success; lower caps on failure",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    args = parser.parse_args()
    if args.status:
        print(json.dumps(json.loads(STATUS_FILE.read_text()), indent=2))
        return 0
    if args.failsafe and os.getenv("SERVICE_RESULT") == "success":
        try:
            # The fixed stop policy is independent of live exploration bounds
            # and must still work if the configuration file has become invalid.
            hardware = Hardware(Config(minimum_w=STOP_POWER_W, maximum_w=STOP_POWER_W))
            hardware.apply(STOP_POWER_W)
            log("stop_cap", power_w=STOP_POWER_W, cards=list(hardware.cards))
            return 0
        except Exception as exc:
            log("stop_cap_failed", error=str(exc))
            return 1
    try:
        cfg = Config.load(args.config)
    except Exception as exc:
        log("configuration_error", error=str(exc))
        if not args.check:
            failsafe(Config(), "invalid configuration")
        return 1
    if args.check:
        try:
            hardware = Hardware(cfg)
            print(
                json.dumps(
                    {
                        "policy": asdict(cfg),
                        "cards": [asdict(sample) for sample in hardware.read()],
                    },
                    indent=2,
                )
            )
            return 0
        except Exception as exc:
            log("check_failed", error=str(exc))
            return 1
    if args.failsafe:
        failsafe(cfg, os.getenv("SERVICE_RESULT", "explicit failsafe"))
        return 0
    return run(cfg, args.config)


if __name__ == "__main__":
    sys.exit(main())
