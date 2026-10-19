# SPDX-License-Identifier: Apache-2.0
"""Explicit GS128 payloads over the native TP4 custom-AR IPC allocator."""

import ctypes
import os
from pathlib import Path

import torch
import torch.distributed as dist


class ResidualG128Allreduce:
    # Chunk prefill without reserving another full activation-sized peer buffer.
    chunk_values = 1024 * 1024

    def __init__(self, native):
        if native.disabled or native.world_size != 4 or not torch.version.hip:
            raise RuntimeError("Decoder GS128 transport requires ROCm TP4 custom AR")
        path = os.environ.get("VLLM_RESIDUAL_G128_LIBRARY")
        if not path:
            raise RuntimeError("Set VLLM_RESIDUAL_G128_LIBRARY before enabling INT8")
        self.lib = ctypes.CDLL(str(Path(path).resolve(strict=True)))
        self.native = native
        self.buffers = self.signals = None
        self.buffers = native.create_shared_buffer(
            self.chunk_values * 33 // 32, group=native.group, uncached=True
        )
        self.signals = native.create_shared_buffer(
            4096, group=native.group, uncached=True
        )
        self.bp = (ctypes.c_void_p * 4)(*self.buffers)
        self.sp = (ctypes.c_void_p * 4)(*self.signals)
        self.fn = self.lib.residual_g128_reduce
        self.fn.restype = ctypes.c_int
        self.fn.argtypes = (
            [ctypes.c_void_p] * 4
            + [ctypes.c_int] * 2
            + [ctypes.POINTER(ctypes.c_void_p)] * 2
            + [ctypes.c_void_p]
        )
        zero = self.lib.residual_g128_zero
        zero.restype = ctypes.c_int
        zero.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        rc = zero(
            self.signals[native.rank],
            torch.cuda.current_stream(native.device).cuda_stream,
        )
        if rc:
            raise RuntimeError(f"Decoder GS128 signal initialization failed: {rc}")
        torch.cuda.synchronize(native.device)
        dist.barrier(group=native.group)

    def close(self):
        for name in ("buffers", "signals"):
            pointers = getattr(self, name, None)
            if pointers is not None:
                self.native.free_shared_buffer(pointers, rank=self.native.rank)
                setattr(self, name, None)

    def all_reduce(self, value):
        from vllm.model_executor.layers.residual_int8 import (
            ResidualInt8,
            _validate,
        )

        _validate(value)
        q, s = value.codes, value.scales
        if not q.is_cuda or q.shape[-1] % 128:
            raise ValueError("Decoder GS128 collective requires aligned GPU payloads")
        out, scales = torch.empty_like(q), torch.empty_like(s)
        stream = torch.cuda.current_stream(q.device).cuda_stream
        for start in range(0, q.numel(), self.chunk_values):
            n = min(self.chunk_values, q.numel() - start)
            rc = self.fn(
                q.data_ptr() + start,
                s.data_ptr() + start // 128 * 4,
                out.data_ptr() + start,
                scales.data_ptr() + start // 128 * 4,
                n,
                self.native.rank,
                self.bp,
                self.sp,
                stream,
            )
            if rc:
                raise RuntimeError(f"Decoder GS128 collective launch failed: {rc}")
        return ResidualInt8(out, scales, value.compute_dtype)


def all_reduce(value):
    from vllm.distributed import get_tp_group

    native = get_tp_group().device_communicator.ca_comm
    if native is None or native.residual_g128 is None:
        raise RuntimeError("Decoder GS128 collective was not initialized")
    return native.residual_g128.all_reduce(value)
