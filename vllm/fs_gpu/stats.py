# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Counters for the FS->GPU layer (surfaced by probes and gates)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class FsGpuStats:
    reads: int = 0
    writes: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    read_seconds: float = 0.0
    write_seconds: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    def snapshot(self) -> str:
        rb = self.bytes_read / (1 << 30)
        wb = self.bytes_written / (1 << 30)
        rmb = rb / self.read_seconds if self.read_seconds else 0.0
        return (
            f"fs_gpu: reads={self.reads} ({rb:.2f} GiB, {rmb:.2f} GiB/s) "
            f"writes={self.writes} ({wb:.2f} GiB) "
            f"elapsed={time.monotonic() - self.started_at:.1f}s"
        )


fs_gpu_stats = FsGpuStats()
