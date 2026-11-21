# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal safetensors header parser for direct-to-GPU reads.

Replaces ``safetensors.safe_open`` for the AIS path: we only need the
(name -> dtype, shape, byte range) table, never a CPU-materialized tensor.
Data offsets are relative to the start of the data region (8 + header_len),
and the data region is 8-byte aligned per the format — the AIS reader
handles the 4K alignment of the enclosing extents.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass

import torch

MAX_HEADER_SIZE = 100_000_000

_DTYPES: dict[str, torch.dtype] = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    start: int  # byte offset relative to the data region
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


class SafetensorFile:
    """Parsed safetensors index for one shard file."""

    def __init__(self, path: str, header_bytes: bytes | None = None):
        self.path = path
        if header_bytes is None:
            with open(path, "rb", buffering=0) as f:
                prefix = f.read(8)
                if len(prefix) != 8:
                    raise ValueError(f"{path}: incomplete safetensors header length")
                (header_len,) = struct.unpack("<Q", prefix)
                if not 1 <= header_len <= MAX_HEADER_SIZE:
                    raise ValueError(f"{path}: invalid safetensors header length")
                header_bytes = f.read(header_len)
                if len(header_bytes) != header_len:
                    raise ValueError(f"{path}: incomplete safetensors header")
        header_len = len(header_bytes)
        header = json.loads(header_bytes)
        if not isinstance(header, dict):
            raise ValueError(f"{path}: expected a safetensors header object")
        self.data_begin = 8 + header_len
        self.metadata = header.get("__metadata__", {})
        if not isinstance(self.metadata, dict):
            raise ValueError(f"{path}: invalid safetensors metadata")
        self.tensors: dict[str, TensorSpec] = {}
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(entry, dict) or not isinstance(entry.get("dtype"), str):
                raise ValueError(f"{path}: invalid tensor header for {name!r}")
            dtype = _DTYPES.get(entry["dtype"])
            if dtype is None:
                raise ValueError(
                    f"{path}: tensor {name!r} has unsupported dtype "
                    f"{entry['dtype']!r}"
                )
            shape = entry.get("shape")
            offsets = entry.get("data_offsets")
            if not isinstance(shape, list) or any(
                type(dim) is not int or dim < 0 for dim in shape
            ):
                raise ValueError(f"{path}: invalid tensor shape for {name!r}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(n) is not int or n < 0 for n in offsets)
                or offsets[1] - offsets[0] != math.prod(shape) * dtype.itemsize
            ):
                raise ValueError(f"{path}: invalid tensor extent for {name!r}")
            self.tensors[name] = TensorSpec(name, dtype, tuple(shape), *offsets)

    def __getitem__(self, name: str) -> TensorSpec:
        return self.tensors[name]

    def __iter__(self):
        return iter(self.tensors.values())

    def __len__(self) -> int:
        return len(self.tensors)
