# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registered files, registered device buffers, and the GPU staging pool.

Alignment model (v1): every AIS transfer targets a single registered,
4096-byte-aligned staging region on the device; payload then moves to its
final tensor with a device-to-device copy on the current stream. This keeps
exactly one buffer registration per pool (not per tensor) and sidesteps the
O_DIRECT alignment requirements of arbitrary safetensors offsets. The
extra D2D hop costs ~20 ms per 30 GiB of weights on MI100 — noise next to
the NVMe IO that dominates.
"""

from __future__ import annotations

import ctypes
import os

import torch

from vllm.fs_gpu.ais import FsGpuError, get_ais

ALIGN = 4096


def _align_down(x: int) -> int:
    return x & ~(ALIGN - 1)


def _align_up(x: int) -> int:
    return (x + ALIGN - 1) & ~(ALIGN - 1)


class RegisteredFile:
    """An O_DIRECT-opened file registered for AIS IO."""

    def __init__(self, path: str, write: bool = False):
        self.path = path
        self.ais = get_ais()
        flags = os.O_DIRECT | (os.O_RDWR if write else os.O_RDONLY)
        try:
            self.fd = os.open(path, flags)
        except OSError as e:
            # e.g. tmpfs or a fs without O_DIRECT support
            raise FsGpuError(f"open({path!r}, O_DIRECT)", 0, e.errno) from e
        self.size = os.fstat(self.fd).st_size
        self.fh = self.ais.handle_register(self.fd)

    def close(self) -> None:
        if self.fh is not None:
            self.ais.handle_deregister(self.fh)
            self.fh = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self) -> RegisteredFile:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class GpuStagingPool:
    """A registered, 4K-aligned device arena used as the AIS landing zone.

    The backing tensor is a plain ``torch`` uint8 allocation; the AIS base
    pointer is the first 4096-aligned address inside it. ``capacity`` is the
    usable byte count from that aligned base.
    """

    def __init__(self, min_bytes: int, device: str | torch.device = "cuda"):
        self.device = torch.device(device)
        want = _align_up(max(min_bytes, ALIGN)) + ALIGN
        self.tensor = torch.empty(want, dtype=torch.uint8, device=self.device)
        raw = self.tensor.data_ptr()
        self.base = _align_up(raw)
        self.capacity = want - (self.base - raw) - ALIGN
        self.ais = get_ais()
        self.ais.buf_register(self.base, self.capacity)

    def grow(self, min_bytes: int) -> None:
        """Re-register a larger arena (frees the old registration)."""
        if min_bytes <= self.capacity:
            return
        self.ais.buf_deregister(self.base)
        want = _align_up(min_bytes) + ALIGN
        self.tensor = torch.empty(want, dtype=torch.uint8, device=self.device)
        raw = self.tensor.data_ptr()
        self.base = _align_up(raw)
        self.capacity = want - (self.base - raw) - ALIGN
        self.ais.buf_register(self.base, self.capacity)

    def close(self) -> None:
        self.ais.buf_deregister(self.base)
        self.tensor = None  # type: ignore[assignment]
        self.base = 0
        self.capacity = 0

    def view(self, offset: int, length: int) -> torch.Tensor:
        """A uint8 device view of [offset, offset+length) in the arena."""
        raw_off = self.base - self.tensor.data_ptr() + offset
        return self.tensor[raw_off : raw_off + length]
