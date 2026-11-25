# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in in-graph timeline: clock-stamp marks at model stages.

With ``VLLM_INKLING_GRAPH_TIMELINE=1`` a model drops a stamp kernel at each
mark; captured into its decode graph, each stamp writes the GPU real-time
clock (40 ns ticks on gfx908) into ``sink.timeline_buf[slot]``. HIP events
recorded through graph nodes cannot be read back on this stack.
"""

import os
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    import numpy as np

# "1": fixed slot per mark (read back after every step); "ring": every stamp
# takes the next slot of a ring, so many steps can be reconstructed at once.
_MODE = os.environ.get("VLLM_INKLING_GRAPH_TIMELINE", "0")
ENABLED = _MODE in ("1", "ring")
RING = _MODE == "ring"
_WARM_STAMPS = 3  # the first stamp of a replay can read a stale clock

_sink = None


@triton.jit
def _clock_stamp(out_ptr, slot, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    t = tl.inline_asm_elementwise(
        "s_memrealtime $0",
        "=s,r",
        [offs.to(tl.uint32)],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )
    tl.store(out_ptr + slot * BLOCK + offs, t)


RING_SLOTS = 1 << 20


@triton.jit
def _clock_ring(Out, Counter, SLOTS: tl.constexpr):
    offs = tl.arange(0, 1)
    # The counter stays monotonic; only the address wraps, so a long run
    # cannot write past the buffer.
    slot = tl.atomic_add(Counter, 1) & (SLOTS - 1)
    t = tl.inline_asm_elementwise(
        "s_memrealtime $0",
        "=s,r",
        [offs.to(tl.uint32)],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )
    tl.store(Out + slot + offs, t)


class TimelineSink:
    """Stamp slots and labels of the last capture of one graph."""

    def __init__(self) -> None:
        self.timeline_buf = torch.zeros(64 * 1024, dtype=torch.int64, device="cuda")
        self.timeline_labels: list[tuple[str, int]] = []
        self.timeline_count = 0
        self._timeline: list[tuple[str, int]] = []
        self.counter = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.ring_labels: list[str] = []
        self.ring_buf = (
            torch.zeros(RING_SLOTS, dtype=torch.int64, device="cuda") if RING else None
        )

    def begin(self) -> None:
        """Start a forward: later marks append after the warm-up stamps."""
        self._timeline = []
        for _ in range(_WARM_STAMPS):
            self.mark("warm", -1)

    def mark(self, label: str, visit: int = -2) -> None:
        self._timeline.append((label, visit))
        if RING:
            self.ring_labels.append(label)
            _clock_ring[(1,)](
                self.ring_buf, self.counter, SLOTS=RING_SLOTS, num_warps=1
            )
        else:
            _clock_stamp[(1,)](
                self.timeline_buf, len(self._timeline) - 1, BLOCK=64, num_warps=1
            )

    def finish(self) -> None:
        self.timeline_labels = self._timeline
        self.timeline_count = len(self._timeline)

    def dump(self) -> "tuple[np.ndarray, int]":
        """Chronological ticks of the ring and the total stamp count."""
        total = int(self.counter.item())
        ticks = self.ring_buf.cpu().numpy()
        if total <= RING_SLOTS:
            return ticks[:total], total
        start = total & (RING_SLOTS - 1)
        return np.concatenate((ticks[start:], ticks[:start])), total


def dump_rings(sinks, out: str, step: int, device_index: int) -> None:
    """VLLM_INKLING_RING_DUMP=<prefix>: one .npz per sink for a live worker."""
    import numpy as np

    for name, sink in sinks:
        if sink is None or sink.ring_buf is None:
            continue
        ticks, total = sink.dump()
        labels = np.array(sink.ring_labels[-4096:], dtype=object)
        np.savez(
            f"{out}.{device_index}.{name}.{step}.npz",
            ticks=ticks,
            total=total,
            labels=labels,
            allow_pickle=True,
        )


def set_sink(sink: TimelineSink | None) -> None:
    global _sink
    _sink = sink


def mark(label: str) -> None:
    if _sink is not None:
        _sink.mark(label)
