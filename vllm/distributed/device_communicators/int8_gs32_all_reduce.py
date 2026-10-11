# SPDX-License-Identifier: Apache-2.0
"""Opt-in MLP-only TP4 INT8 transport, FP32 GS32 scales and accumulation.

Build csrc/custom_all_reduce_int8_gs32.hip to the path named by
VLLM_MLP_CAR_INT8_LIBRARY before starting workers. No lazy builds during capture.
"""

import ctypes
import os
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.utils.torch_utils import direct_register_custom_op


def selected_mlp(prefix: str) -> bool:
    return prefix.endswith(".mlp.down_proj")


class Int8Gs32Allreduce:
    def __init__(self, native):
        if native.disabled or native.world_size != 4 or not torch.version.hip:
            raise RuntimeError("MLP INT8 GS32 CAR requires enabled ROCm TP4 CAR")
        from vllm import _custom_ops as ops

        if ops.meta_size() != 4864:
            raise RuntimeError("MLP INT8 GS32 CAR signal ABI mismatch")
        path = os.environ.get("VLLM_MLP_CAR_INT8_LIBRARY")
        if not path:
            raise RuntimeError("Set VLLM_MLP_CAR_INT8_LIBRARY to the built HIP library")
        self.lib = ctypes.CDLL(str(Path(path).resolve(strict=True)))
        self.fn = self.lib.mlp_car_int8_gs32
        self.fn.restype = ctypes.c_int
        self.fn.argtypes = (
            [ctypes.c_int] + [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2
            + [ctypes.POINTER(ctypes.c_void_p)] * 2 + [ctypes.c_void_p]
        )
        self.rank = native.rank
        self.max_bytes = native.max_size
        self.native = native
        self.buffers = self.signals = None
        # Separate storage and epochs: native collectives may be interleaved.
        self.buffers = native.create_shared_buffer(
            self.max_bytes, group=native.group, uncached=True
        )
        self.signals = native.create_shared_buffer(
            ops.meta_size() + self.max_bytes, group=native.group, uncached=True
        )
        self.buffer_ptrs = (ctypes.c_void_p * 4)(*self.buffers)
        self.signal_ptrs = (ctypes.c_void_p * 4)(*self.signals)
        zero = self.lib.mlp_car_int8_zero_meta
        zero.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        zero.restype = ctypes.c_int
        rc = zero(self.signals[self.rank],
                  torch.cuda.current_stream(native.device).cuda_stream)
        if rc:
            raise RuntimeError(
                f"MLP INT8 GS32 CAR metadata initialization failed: {rc}"
            )
        torch.cuda.synchronize(native.device)
        dist.barrier(group=native.group)

    def close(self):
        for name in ("buffers", "signals"):
            pointers = getattr(self, name, None)
            if pointers is not None:
                self.native.free_shared_buffer(pointers, rank=self.rank)
                setattr(self, name, None)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        n = x.numel()
        if n == 0:
            return torch.empty_like(x)
        if (x.dtype not in (torch.bfloat16, torch.float16)
                or not x.is_cuda or not x.is_contiguous()
                or n % 256 or n * x.element_size() > self.max_bytes):
            raise RuntimeError(f"Unsupported MLP GS32 CAR tensor: {x.shape}, {x.dtype}")
        out = torch.empty_like(x)
        rc = self.fn(
            int(x.dtype == torch.float16), x.data_ptr(), out.data_ptr(), n,
            self.rank, self.buffer_ptrs, self.signal_ptrs,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        if rc:
            raise RuntimeError(f"MLP INT8 GS32 CAR HIP launch failed: {rc}")
        return out


def _dispatch(x: torch.Tensor) -> torch.Tensor:
    from vllm.distributed import get_tp_group

    native = get_tp_group().device_communicator.ca_comm
    if native is None or native.int8_gs32 is None:
        raise RuntimeError("MLP GS32 CAR was not initialized before graph capture")
    return native.int8_gs32.all_reduce(x)


def _fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="mlp_car_int8_gs32", op_func=_dispatch, fake_impl=_fake
)


def mlp_car_int8_gs32(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.mlp_car_int8_gs32(x)
