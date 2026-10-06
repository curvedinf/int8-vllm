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

from concurrent.futures import ThreadPoolExecutor

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


def merge_ranges(
    ranges: list[tuple[int, int, int]],
    max_gap: int,
    max_run: int,
) -> list[tuple[int, int, int]]:
    """Merge file-adjacent (tensor_off, file_off, nbytes) ranges.

    hipFile 0.3.0 executes ~one IO per process (~9k IOPS regardless of
    threads/handles/async — probed), so strided shard reads are
    latency-bound, not bandwidth-bound. Merging a rank's range with its
    neighbors' gaps trades extra bytes (which the NVMe has headroom for)
    for far fewer, larger IOs. ``max_gap`` caps the wasteful gap willing
    to be read; ``max_run`` caps the merged IO size.
    """
    if not ranges:
        return ranges
    out = [ranges[0]]
    for bo, fo, n in ranges[1:]:
        cur_bo, cur_fo, cur_n = out[-1]
        gap = fo - (cur_fo + cur_n)
        if 0 <= gap <= max_gap and cur_n + gap + n <= max_run:
            out[-1] = (cur_bo, cur_fo, cur_n + gap + n)
        else:
            out.append((bo, fo, n))
    return out


def fill_ranges(
    f: RegisteredFile,
    tensor: torch.Tensor,
    ranges: list[tuple[int, int, int]],
    executor: ThreadPoolExecutor | None = None,
) -> None:
    """Read checkpoint byte ranges directly into a device tensor.

    ``ranges`` are ``(tensor_byte_offset, file_offset, nbytes)`` triples.
    The tensor's base is registered for the duration, and each range is
    issued as one raw AIS read with an explicit buffer offset — unaligned
    file/buffer offsets and sizes are fine (probe-verified: hipFile bounces
    internally). With an executor, reads are fanned out across its threads;
    ctypes releases the GIL, so they genuinely overlap.
    """
    if not tensor.is_contiguous():
        raise FsGpuError("fill_ranges: non-contiguous destination", 5000 + 22)
    base = tensor.data_ptr()
    nbytes = tensor.numel() * tensor.element_size()
    f.ais.buf_register(base, nbytes)
    try:

        def _one(r: tuple[int, int, int]) -> None:
            buf_off, file_off, n = r
            f.ais.read(f.fh, base, n, file_off, buf_off)

        if executor is not None and len(ranges) > 16:
            list(executor.map(_one, ranges))
        else:
            for r in ranges:
                _one(r)
    finally:
        f.ais.buf_deregister(base)
    read = sum(n for _, _, n in ranges)
    fs_gpu_stats.bytes_read += read
    fs_gpu_stats.reads += len(ranges)


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
