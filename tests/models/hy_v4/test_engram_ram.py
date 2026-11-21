# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from scripts import warm_hy4_engram as warmer


def test_available_ram_uses_tightest_ancestor_and_reclaimable_file_cache(tmp_path):
    proc = tmp_path / "proc"
    root = tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/parent/worker\n")
    (proc / "meminfo").write_text("MemFree: 128 kB\nMemAvailable: 2048 kB\n")
    (root / "parent/worker").mkdir(parents=True)
    (root / "parent/worker/memory.max").write_text("max\n")
    (root / "parent/memory.max").write_text("1048576\n")
    (root / "parent/memory.current").write_text("786432\n")
    (root / "parent/memory.stat").write_text("inactive_file 131072\n")
    assert warmer.available_ram_bytes(proc, root) == 393216
    (root / "parent/memory.max").write_text("max\n")
    assert warmer.available_ram_bytes(proc, root) == 2 << 20


def test_ssd_and_pinned_engram_share_the_same_host_reserve(tmp_path, monkeypatch):
    (tmp_path / "engram-2.q4_0").write_bytes(bytes(64))
    (tmp_path / "engram-12.q4_0").write_bytes(bytes(64))
    reserve = 8 << 30
    monkeypatch.setattr(warmer, "available_ram_bytes", lambda: reserve + 128)
    warmer.check_ram(tmp_path, "ram", 8)
    monkeypatch.setattr(warmer, "available_ram_bytes", lambda: reserve + 127)
    warmer.check_ram(tmp_path, "bcache", 8)
    with pytest.raises(ValueError, match="host reserve is 8 GiB"):
        warmer.check_ram(tmp_path, "ram", 8)
    monkeypatch.setattr(warmer, "available_ram_bytes", lambda: reserve - 1)
    with pytest.raises(ValueError, match="host reserve is 8 GiB"):
        warmer.check_ram(tmp_path, "bcache", 8)
