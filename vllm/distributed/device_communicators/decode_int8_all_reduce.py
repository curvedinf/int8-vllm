# SPDX-License-Identifier: Apache-2.0
"""Private uncached TP4 transport for decode int8 intermediates."""

import ctypes
import os
from pathlib import Path

import torch
import torch.distributed as dist


class DecodeInt8Allreduce:
    def __init__(self, native):
        if native.disabled or native.world_size != 4 or not torch.version.hip:
            raise RuntimeError("Decode INT8 requires ROCm TP4 native CAR")
        path = os.environ.get("VLLM_DECODE_INT8_LIBRARY")
        if not path:
            raise RuntimeError("Set VLLM_DECODE_INT8_LIBRARY before boot")
        self.lib = ctypes.CDLL(str(Path(path).resolve(strict=True)))
        self.native = native
        self.buffers = self.signals = None
        self.buffers = native.create_shared_buffer(
            64 * 5120 + 64 * 40 * 4, group=native.group, uncached=True
        )
        self.signals = native.create_shared_buffer(
            4096, group=native.group, uncached=True
        )
        self.bp = (ctypes.c_void_p * 4)(*self.buffers)
        self.sp = (ctypes.c_void_p * 4)(*self.signals)
        self.fn = self.lib.decode_int8_norm
        self.fn.restype = ctypes.c_int
        self.fn.argtypes = (
            [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_float]
            + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
            + [ctypes.c_int] * 2
            + [ctypes.POINTER(ctypes.c_void_p)] * 2 + [ctypes.c_void_p]
        )
        zero = self.lib.decode_int8_zero
        zero.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        zero.restype = ctypes.c_int
        rc = zero(self.signals[native.rank],
                  torch.cuda.current_stream(native.device).cuda_stream)
        if rc:
            raise RuntimeError(f"Decode INT8 signal initialization failed: {rc}")
        torch.cuda.synchronize(native.device)
        dist.barrier(group=native.group)

    def close(self):
        for name in ("buffers", "signals"):
            pointers = getattr(self, name, None)
            if pointers is not None:
                self.native.free_shared_buffer(pointers, rank=self.native.rank)
                setattr(self, name, None)

    def norm(self, q, s, residual, weight, epsilon, gemma,
             delta, base, block, group):
        rows, hidden = q.shape
        if (hidden != 5120 or not 0 < rows <= 64 or q.dtype != torch.int8
                or s.dtype != torch.float32 or s.shape != (rows, 40)
                or residual.dtype != torch.bfloat16
                or residual.shape != q.shape or weight.dtype != torch.bfloat16
                or weight.numel() not in (0, 5120)):
            raise ValueError("Unsupported decode INT8 residual/norm contract")
        tensors = [q, s, residual, weight]
        if delta is not None:
            if (group <= 0 or 5120 % group or block <= 0
                    or base is None or base.shape != (2, 5120)
                    or delta.shape != (rows, 2, 5120 // group)
                    or delta.dtype != torch.bfloat16
                    or base.dtype != torch.bfloat16):
                raise ValueError("Only the deployed two-tap DFlash2 conv is supported")
            tensors += [delta, base]
        if not all(t.is_cuda and t.is_contiguous() for t in tensors):
            raise ValueError("Decode INT8 requires contiguous GPU tensors")
        out = torch.empty_like(residual)
        res_out = torch.empty_like(residual)
        rc = self.fn(
            q.data_ptr(), s.data_ptr(), residual.data_ptr(), weight.data_ptr(),
            out.data_ptr(), res_out.data_ptr(), rows, epsilon, self.native.rank,
            gemma, weight.numel() != 0,
            delta.data_ptr() if delta is not None else 0,
            base.data_ptr() if base is not None else 0,
            block, group, self.bp, self.sp,
            torch.cuda.current_stream(q.device).cuda_stream,
        )
        if rc:
            raise RuntimeError(f"Decode INT8 CAR/RMSNorm HIP error: {rc}")
        return out, res_out
