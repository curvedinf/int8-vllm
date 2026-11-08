# SPDX-License-Identifier: Apache-2.0
"""Fused TP4 collective: AllReduce + residual + RMSNorm + GS128 quant.

Wraps csrc/custom_all_reduce_rms_g128.hip over the native custom-AR IPC
buffers. One launch replaces the custom all-reduce followed by the
residual+RMSNorm+group-quant prologue kernel; numerics mirror the unfused
chain (AR rounded to bf16, residual add in fp32 rounded to bf16, fp32 norm,
amax/127 round-half-even group quant).
"""

import ctypes
import os
from pathlib import Path

import torch
import torch.distributed as dist

_DEFAULT_LIB = Path(__file__).resolve().parents[3] / "vllm" / "_ar_rms_g128.so"


class ArRmsG128:
    def __init__(self, native, max_rows: int, hidden: int):
        if native.disabled or native.world_size != 4 or not torch.version.hip:
            raise RuntimeError("Fused AR+RMSNorm+quant requires ROCm TP4 custom AR")
        path = os.environ.get("VLLM_AR_RMS_G128_LIBRARY", str(_DEFAULT_LIB))
        self.lib = ctypes.CDLL(str(Path(path).resolve(strict=True)))
        self.native = native
        self.rank = native.rank
        self.max_rows = max_rows
        self.hidden = hidden
        self.buffers = native.create_shared_buffer(
            max_rows * hidden * 2, group=native.group, uncached=True)
        self.signals = native.create_shared_buffer(
            16384, group=native.group, uncached=True)
        self.bp = (ctypes.c_void_p * 4)(*self.buffers)
        self.sp = (ctypes.c_void_p * 4)(*self.signals)
        self.fn = self.lib.ar_rms_g128_launch
        self.fn.restype = ctypes.c_int
        self.fn.argtypes = (
            [ctypes.c_void_p] * 7
            + [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
            + [ctypes.POINTER(ctypes.c_void_p)] * 2
            + [ctypes.c_void_p]
        )
        zero = self.lib.ar_rms_g128_zero
        zero.restype = ctypes.c_int
        zero.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        rc = zero(self.signals[native.rank],
                  torch.cuda.current_stream(native.device).cuda_stream)
        if rc:
            raise RuntimeError(f"AR+RMSNorm signal init failed: {rc}")
        torch.cuda.synchronize(native.device)
        dist.barrier(group=native.group)

    def close(self):
        for name in ("buffers", "signals"):
            pointers = getattr(self, name, None)
            if pointers is not None:
                self.native.free_shared_buffer(pointers, rank=self.native.rank)
                setattr(self, name, None)

    def fused(self, partials, residual, weight, eps):
        """Partials [M,K] bf16 -> (normed, residual_out, q int8, scales fp32)."""
        m, k = partials.shape
        if m > self.max_rows or k != self.hidden:
            raise ValueError(f"AR+RMSNorm payload {m}x{k} exceeds "
                             f"{self.max_rows}x{self.hidden}")
        if not partials.is_contiguous():
            partials = partials.contiguous()
        if not residual.is_contiguous():
            residual = residual.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        normed = torch.empty_like(partials)
        res_out = torch.empty_like(residual)
        q = torch.empty((m, k), dtype=torch.int8, device=partials.device)
        s = torch.empty((m, k // 128), dtype=torch.float32,
                        device=partials.device)
        rc = self.fn(
            partials.data_ptr(), residual.data_ptr(), weight.data_ptr(),
            normed.data_ptr(), res_out.data_ptr(), q.data_ptr(), s.data_ptr(),
            m, k, float(eps), self.rank, self.bp, self.sp,
            torch.cuda.current_stream(partials.device).cuda_stream)
        if rc:
            raise RuntimeError(f"AR+RMSNorm launch failed: {rc}")
        return normed, res_out, q, s


_INSTANCE: ArRmsG128 | None = None


def get_ar_rms(max_rows: int, hidden: int) -> "ArRmsG128 | None":
    """Process-wide singleton; None when the fused collective is unavailable."""
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE
    try:
        from vllm.distributed import get_tp_group

        native = get_tp_group().device_communicator.ca_comm
        if native is None or native.disabled or native.world_size != 4:
            return None
        _INSTANCE = ArRmsG128(native, max_rows, hidden)
        from vllm.logger import init_logger

        init_logger(__name__).info_once(
            "flint: fused custom-AR + residual + RMSNorm + GS128 quant "
            "collective active (TP4)")
        return _INSTANCE
    except Exception:
        return None
