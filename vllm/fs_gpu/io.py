# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-level FS->GPU transfers on top of the AIS bindings.

:func:`read_into_tensor` moves ``nbytes`` at ``file_offset`` of an already
registered file into any contiguous device tensor, routing through the
staging pool in 4K-aligned chunks. :func:`write_from_tensor` is the mirror
for GPU->file stores (aligned extents only in v1 — enough for probes and
future KV pages, whose layout we control).
"""

from __future__ import annotations

import torch

from vllm.fs_gpu.ais import FsGpuError
from vllm.fs_gpu.region import ALIGN, GpuStagingPool, RegisteredFile, _align_up
from vllm.fs_gpu.stats import fs_gpu_stats


def _byte_view(t: torch.Tensor) -> torch.Tensor:
    if t.numel() == 0:
        raise FsGpuError("zero-length tensor", 5000 + 21)
    # Flatten first: .view(torch.uint8) on an [a, b] int32 tensor yields
    # [a, b*4], and byte-offset slicing must operate on the flat form.
    return t.reshape(-1).view(torch.uint8)


def read_into_tensor(
    f: RegisteredFile,
    pool: GpuStagingPool,
    tensor: torch.Tensor,
    file_offset: int,
    chunk_hint: int = 32 << 20,
) -> int:
    """Read ``tensor.nbytes`` from ``file_offset`` into the device tensor.

    The copy into ``tensor`` is enqueued on the current stream; AIS reads are
    synchronous, so the bytes are device-visible before it returns. Returns
    the number of payload bytes read.
    """
    if not tensor.is_contiguous():
        raise FsGpuError("read_into_tensor: non-contiguous destination", 5000 + 22)
    dst = _byte_view(tensor)
    total = dst.numel()
    done = 0
    while done < total:
        payload = min(total - done, chunk_hint, pool.capacity - ALIGN)
        # Enclose [file_offset+done, +payload) in a 4K-aligned extent,
        # clamped to EOF — the last tensor in a shard may end mid-block.
        start = file_offset + done
        head = start & (ALIGN - 1)
        extent = _align_up(head + payload)
        avail = f.size - (start - head)
        if extent > avail:
            extent = max(int(avail), 1)
        f.ais.read(f.fh, pool.base, extent, start - head, 0)
        dst[done : done + payload].copy_(pool.view(head, payload))
        done += payload
        fs_gpu_stats.reads += 1
    fs_gpu_stats.bytes_read += total
    return total


def write_from_tensor(
    f: RegisteredFile,
    pool: GpuStagingPool,
    tensor: torch.Tensor,
    file_offset: int,
    chunk_hint: int = 32 << 20,
) -> int:
    """Write ``tensor.nbytes`` to ``file_offset`` (extent must be 4K-aligned).

    v1 contract: ``file_offset`` and ``tensor.nbytes`` must already be
    4K-aligned (true for probe scratch files and future KV page layouts,
    both under our control). Arbitrary unaligned writes would need
    read-modify-write padding and are deliberately not supported yet.
    """
    if not tensor.is_contiguous():
        raise FsGpuError("write_from_tensor: non-contiguous source", 5000 + 22)
    src = _byte_view(tensor)
    total = src.numel()
    if file_offset & (ALIGN - 1) or total & (ALIGN - 1):
        raise FsGpuError(
            "write_from_tensor: offset/size must be 4K-aligned "
            f"(offset={file_offset}, size={total})",
            5000 + 22,
        )
    done = 0
    while done < total:
        n = min(total - done, chunk_hint, pool.capacity)
        pool.view(0, n).copy_(src[done : done + n])
        f.ais.write(f.fh, pool.base, n, file_offset + done, 0)
        done += n
        fs_gpu_stats.writes += 1
    fs_gpu_stats.bytes_written += total
    return total
