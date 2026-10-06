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
import struct
from dataclasses import dataclass

import torch

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

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        self.data_begin = 8 + header_len
        self.metadata = header.get("__metadata__", {})
        self.tensors: dict[str, TensorSpec] = {}
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            dtype = _DTYPES.get(entry["dtype"])
            if dtype is None:
                raise ValueError(
                    f"{path}: tensor {name!r} has unsupported dtype "
                    f"{entry['dtype']!r}"
                )
            self.tensors[name] = TensorSpec(
                name=name,
                dtype=dtype,
                shape=tuple(entry["shape"]),
                start=int(entry["data_offsets"][0]),
                end=int(entry["data_offsets"][1]),
            )

    def __getitem__(self, name: str) -> TensorSpec:
        return self.tensors[name]

    def __iter__(self):
        return iter(self.tensors.values())

    def __len__(self) -> int:
        return len(self.tensors)
