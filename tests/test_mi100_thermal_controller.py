# SPDX-License-Identifier: Apache-2.0
"""Power policy and faults must leave workloads running."""

import os
import signal
import subprocess
from dataclasses import asdict, replace

import pytest

from scripts import mi100_thermal_controller as thermal


def readings(engine, core=70, memory=70, busy=0):
    return [
        thermal.Sample(
            str(index),
            engine.cap_w,
            {"edge": 60, "junction": core, "mem": memory},
            busy,
        )
        for index in range(engine.cfg.expected_cards)
    ]


def advance(engine, start, end, **kwargs):
    for now in range(start, end + 1):
        decision = engine.step(readings(engine, **kwargs), now)
    return decision


@pytest.fixture(autouse=True)
def forbid_workload_control(monkeypatch):
    """Neither normal decisions nor error handlers may control processes."""

    def forbidden(*args, **kwargs):
        pytest.fail("Thermal controller attempted process or service control")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setattr(os, "kill", forbidden)
    monkeypatch.setattr(os, "killpg", forbidden)


@pytest.mark.parametrize(
    "core,memory,power,emergency",
    [
        (90, 70, None, False),
        (90.01, 70, 102, False),
        (95, 70, 102, False),
        (95.01, 70, 82, True),
        (110, 70, 82, True),
        (70, 89, None, False),
        (70, 89.01, 102, False),
        (70, 90, 102, False),
        (70, 90.01, 82, True),
        (70, 110, 82, True),
        (96, 91, 82, True),
    ],
)
def test_independent_thresholds(core, memory, power, emergency):
    engine = thermal.Controller(thermal.Config(), 103, 0)
    # Only the emergency rule bypasses the five-minute step interval.
    engine.last_change = -300
    decision = engine.step(readings(engine, core, memory), 0)
    assert decision.power_w == power
    assert decision.emergency is emergency
    assert "stop_workload" not in asdict(decision)
    assert engine.status(0)["power_only"] is True


def test_original_five_minute_average_shutdown_becomes_one_watt_cut():
    engine = thermal.Controller(thermal.Config(), 103, 0)
    decision = advance(engine, 0, 300, core=91)
    assert decision.power_w == 102
    assert not decision.emergency
    assert any(row.get("average_300s_c") == 91 for row in engine.stats.values())
    assert "stop" not in decision.reason
    assert advance(engine, 301, 599, core=91).power_w is None
    assert advance(engine, 600, 600, core=91).power_w == 101


def test_one_watt_memory_reduction_does_not_require_gpu_load():
    engine = thermal.Controller(thermal.Config(), 103, 0)
    assert advance(engine, 0, 299, memory=89.5).power_w is None
    assert advance(engine, 300, 300, memory=89.5).power_w == 102


def test_single_hot_sensor_on_one_idle_card_protects_all_cards():
    engine = thermal.Controller(thermal.Config(), 103, 0)
    samples = readings(engine)
    samples[2].temperatures["mem"] = 91
    assert engine.step(samples, 0).power_w == 82


def test_persistent_emergency_reduces_power_repeatedly_and_holds_at_minimum():
    engine = thermal.Controller(thermal.Config(), 103, 0)
    assert advance(engine, 0, 0, core=96).power_w == 82
    assert advance(engine, 1, 9, core=96).power_w is None
    assert advance(engine, 10, 10, core=96).power_w == 65
    assert advance(engine, 11, 20, core=96).power_w == 60
    advance(engine, 21, 29, core=110, memory=110)
    decision = advance(engine, 30, 30, core=110, memory=110)
    assert decision.power_w is None
    assert engine.cap_w == 60
    assert "minimum_power_bound" in decision.reason


def test_soft_cut_at_minimum_does_not_increase_or_stop():
    engine = thermal.Controller(thermal.Config(), 60, 0)
    decision = advance(engine, 0, 300, core=91)
    assert engine.cap_w == 60
    assert decision.power_w is None
    assert decision.reason == "minimum_power_bound"


def test_cooling_rearms_emergency_without_restoring_previous_cap():
    engine = thermal.Controller(thermal.Config(), 103, 0)
    assert advance(engine, 0, 0, core=96).power_w == 82
    advance(engine, 1, 61)
    assert engine.emergency_started is None
    assert engine.cap_w == 82
    assert advance(engine, 62, 62, memory=91).power_w == 65


@pytest.mark.parametrize("busy,expected", [(0, None), (95, None), (96, 104)])
def test_power_increase_requires_full_warmup_and_high_load(busy, expected):
    engine = thermal.Controller(thermal.Config(), 103, 0)
    assert advance(engine, 0, 299, busy=busy).power_w is None
    assert advance(engine, 300, 300, busy=busy).power_w == expected


def test_one_idle_card_prevents_power_increase():
    engine = thermal.Controller(thermal.Config(), 103, 0)
    advance(engine, 0, 299, busy=100)
    samples = readings(engine, busy=100)
    samples[1].busy_percent = 0
    assert engine.step(samples, 300).power_w is None


def test_power_increase_is_bounded():
    engine = thermal.Controller(thermal.Config(), 145, 0)
    assert advance(engine, 0, 300, busy=100).power_w is None
    assert engine.last_reason == "maximum_power_bound"


def test_sampling_fault_fails_instead_of_issuing_workload_action():
    engine = thermal.Controller(thermal.Config(), 103, 0)
    engine.step(readings(engine), 0)
    with pytest.raises(thermal.SafetyFault, match="sampling gap"):
        engine.step(readings(engine), 4)


def test_legacy_workload_setting_is_ignored(tmp_path, monkeypatch):
    path = tmp_path / "config"
    path.write_text("MI100_THERMAL_WORKLOAD_SERVICE=legacy.service\n")
    monkeypatch.setenv("MI100_THERMAL_WORKLOAD_SERVICE", "another.service")
    cfg = thermal.Config.load(path)
    assert "workload_service" not in asdict(cfg)
    assert (cfg.core_soft_c, cfg.core_limit_c) == (90, 95)
    assert (cfg.memory_soft_c, cfg.memory_limit_c) == (89, 90)


@pytest.mark.parametrize("cap", [103, 50])
def test_failsafe_lowers_caps_only(tmp_path, monkeypatch, cap):
    hw = tmp_path / "hwmon"
    hw.mkdir()
    path = hw / "power1_cap"
    path.write_text(str(cap * 1_000_000))

    class FakeHardware:
        def __init__(self, cfg, strict):
            assert strict is False
            self.cards = {"fake-pci": (tmp_path, hw)}

    monkeypatch.setattr(thermal, "Hardware", FakeHardware)
    thermal.failsafe(thermal.Config(), "sensor failure")
    assert int(path.read_text()) == min(cap, 60) * 1_000_000


def test_failsafe_discovery_failure_never_controls_workload(monkeypatch, capsys):
    def fail_discovery(*args, **kwargs):
        raise OSError("driver unavailable")

    monkeypatch.setattr(thermal, "Hardware", fail_discovery)
    thermal.failsafe(thermal.Config(), "driver failure")
    assert "failsafe_discovery_failed" in capsys.readouterr().out


@pytest.mark.parametrize("cap_w", [60, 145])
def test_successful_service_stop_sets_every_card_to_105_w(tmp_path, monkeypatch, cap_w):
    root = tmp_path / "pci"
    caps = []
    for index in range(4):
        dev = root / str(index)
        hw = dev / "hwmon" / "hwmon0"
        hw.mkdir(parents=True)
        (dev / "vendor").write_text("0x1002")
        (dev / "device").write_text("0x738c")
        (hw / "power1_cap_min").write_text("1000000")
        (hw / "power1_cap_max").write_text("290000000")
        cap = hw / "power1_cap"
        cap.write_text(str(cap_w * 1_000_000))
        caps.append(cap)

    # A clean stop must work without sensors or a valid runtime configuration.
    config_path = tmp_path / "config"
    config_path.write_text("MI100_THERMAL_MAX_W=invalid\n")
    monkeypatch.setenv("SERVICE_RESULT", "success")
    monkeypatch.setenv("MI100_THERMAL_MAX_W", "90")
    monkeypatch.setattr(
        thermal.sys,
        "argv",
        ["controller", "--failsafe", "--config", str(config_path)],
    )
    hardware_class = thermal.Hardware
    monkeypatch.setattr(thermal, "Hardware", lambda cfg: hardware_class(cfg, root=root))

    assert thermal.main() == 0
    assert [int(cap.read_text()) for cap in caps] == [105_000_000] * 4


def test_service_stop_reports_power_write_failure(monkeypatch, capsys):
    def fail_discovery(*args, **kwargs):
        raise OSError("driver unavailable")

    monkeypatch.setenv("SERVICE_RESULT", "success")
    monkeypatch.setattr(thermal.sys, "argv", ["controller", "--failsafe"])
    monkeypatch.setattr(thermal, "Hardware", fail_discovery)
    assert thermal.main() == 1
    assert "stop_cap_failed" in capsys.readouterr().out


def test_service_failure_keeps_power_only_failsafe(tmp_path, monkeypatch):
    config_path = tmp_path / "config"
    config_path.write_text("")
    calls = []
    monkeypatch.setenv("SERVICE_RESULT", "watchdog")
    monkeypatch.setattr(
        thermal.sys,
        "argv",
        ["controller", "--failsafe", "--config", str(config_path)],
    )
    monkeypatch.setattr(thermal, "failsafe", lambda cfg, reason: calls.append(reason))
    assert thermal.main() == 0
    assert calls == ["watchdog"]


def test_run_fault_invokes_power_only_failsafe(tmp_path, monkeypatch):
    config_path = tmp_path / "config"
    config_path.write_text("")
    monkeypatch.setattr(thermal, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(signal, "signal", lambda *args: None)

    class FakeHardware:
        cards = {}

        def __init__(self, cfg, strict=True):
            pass

        def read(self):
            raise OSError("sensor unavailable")

    monkeypatch.setattr(thermal, "Hardware", FakeHardware)
    assert thermal.run(thermal.Config(), config_path) == 1


def test_bounds_reload_does_not_change_live_cap(tmp_path):
    cfg = thermal.Config()
    engine = thermal.Controller(cfg, 103, 0)
    path = tmp_path / "config"
    path.write_text("MI100_THERMAL_MAX_W=110\n")

    class FakeHardware:
        def validate_bounds(self, candidate):
            assert candidate.maximum_w == 110

    hardware = FakeHardware()
    assert thermal.reload_bounds(cfg, engine, hardware, path) == replace(
        cfg, maximum_w=110
    )
    assert engine.cap_w == 103
