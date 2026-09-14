# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel + tuned constants for the ``swap_blocks_batch`` fast path."""

from __future__ import annotations

import ctypes
import os

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import tl, triton

# Constants tuned empirically on H100 (PCIe Gen5):
#   NUM_SMS         - smallest SM slice within 5% of peak bandwidth at the
#                     8-32 KB block sizes that matter in practice
#   THRESHOLD_BYTES - max payload per descriptor where Triton beats DMA; above
#                     this the C++ cuMemcpyBatchAsync path takes the lead
#   MIN_N           - minimum batch size where Triton's per-launch cost is
#                     amortized; below this DMA wins
NUM_SMS = 12
THRESHOLD_BYTES = 28 * 1024
MIN_N = 16

_FORCE_NO_FALLBACK: bool | None = None

_HIP_LIB = None


def _hip_lib():
    global _HIP_LIB
    if _HIP_LIB is None:
        _HIP_LIB = ctypes.CDLL("libamdhip64.so")
        _HIP_LIB.hipMemcpyAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_void_p,
        ]
        _HIP_LIB.hipMemcpyAsync.restype = ctypes.c_int
    return _HIP_LIB


def swap_blocks_classic(
    src_addrs: torch.Tensor,
    dst_addrs: torch.Tensor,
    sizes: torch.Tensor,
    is_src_access_order_any: bool = False,
    *,
    device_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Classic executor: one plain hipMemcpyAsync per descriptor, enqueued
    on the CURRENT stream (callers run inside their transfer-stream
    context). The boring per-copy API that upstream vLLM used for years —
    neither the hipMemcpyBatchAsync driver path (races CUDA-graph replay on
    gfx908) nor shader stores to host pointers. hipMemcpyDeviceToHost=2.
    """
    hip = _hip_lib()
    for sp, dp, n in zip(
        src_addrs.tolist(), dst_addrs.tolist(), sizes.tolist()
    ):
        rc = hip.hipMemcpyAsync(
            ctypes.c_void_p(dp), ctypes.c_void_p(sp),
            ctypes.c_size_t(n), 2, ctypes.c_void_p(0),
        )
        if rc != 0:
            raise RuntimeError(f"hipMemcpyAsync failed rc={rc}")


@triton.jit
def _swap_blocks_kernel(
    src_addrs,
    dst_addrs,
    sizes,
    n_jobs,  # type: ignore[name-defined]
    BYTES_PER_CHUNK: tl.constexpr,  # type: ignore[name-defined]
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    WORDS_PER_CHUNK: tl.constexpr = BYTES_PER_CHUNK // 8
    offsets = tl.arange(0, WORDS_PER_CHUNK)
    job = pid
    while job < n_jobs:
        src = tl.load(src_addrs + job).to(tl.pointer_type(tl.int64))
        dst = tl.load(dst_addrs + job).to(tl.pointer_type(tl.int64))
        words = tl.load(sizes + job) // 8
        for start in range(0, words, WORDS_PER_CHUNK):
            idx = start + offsets
            mask = idx < words
            data = tl.load(src + idx, mask=mask, other=0)
            tl.store(dst + idx, data, mask=mask)
        job += num_progs


def swap_blocks_batch(
    src_addrs: torch.Tensor,
    dst_addrs: torch.Tensor,
    sizes: torch.Tensor,
    is_src_access_order_any: bool = False,
    *,
    bytes_per_chunk: int,
    device_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Triton implementation of ``swap_blocks_batch`` for small CPU->GPU batches.

    ``device_buffers``: optional persistent device-side staging tensors owned
    by the caller. Without them the descriptors are staged via ``.to()``,
    whose transient allocations are freed back to the caching allocator
    while the H2D copy + kernel are still queued — a block then reused by a
    *different* stream gets overwritten by the late descriptor copy
    (cross-stream unsafe free; observed as deterministic device-memory
    corruption on gfx908, 2026-09-14). Persistent buffers close that hole.
    """
    global _FORCE_NO_FALLBACK
    if _FORCE_NO_FALLBACK is None:
        _FORCE_NO_FALLBACK = (
            os.environ.get("VLLM_OFFLOAD_TRITON_STORES") == "force"
        )
    n = src_addrs.numel()
    # Too few descriptors to amortize Triton's launch cost.
    # Under =force we still take the Triton path: small decode-time batches
    # going back to the driver batch API reopens the hipMemcpyBatchAsync
    # race against CUDA-graph replay (the 2026-09-14 drift repro).
    if n < MIN_N and not _FORCE_NO_FALLBACK:
        ops.swap_blocks_batch(
            src_addrs,
            dst_addrs,
            sizes,
            is_src_access_order_any=is_src_access_order_any,
        )
        return
    if device_buffers is not None:
        dev_src, dev_dst, dev_sizes = device_buffers
        src_d = dev_src[:n]
        dst_d = dev_dst[:n]
        sizes_d = dev_sizes[:n]
        src_d.copy_(src_addrs, non_blocking=True)
        dst_d.copy_(dst_addrs, non_blocking=True)
        sizes_d.copy_(sizes, non_blocking=True)
    else:
        src_d = src_addrs.to("cuda", non_blocking=True)
        dst_d = dst_addrs.to("cuda", non_blocking=True)
        sizes_d = sizes.to("cuda", non_blocking=True)
    _swap_blocks_kernel[(min(NUM_SMS, n),)](
        src_d,
        dst_d,
        sizes_d,
        n,
        BYTES_PER_CHUNK=bytes_per_chunk,
    )
