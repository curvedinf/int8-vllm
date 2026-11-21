# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write standard safetensors with a page-aligned header and GPU-only payloads."""

import json
import math
import os
import struct
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from vllm.fs_gpu.direct import DirectStorage, allocate_file
from vllm.fs_gpu.region import ALIGN, RegisteredFile, _align_up
from vllm.fs_gpu.safetensors import _DTYPES


class DirectTensorWriter:
    """Accept successive chunks of each named tensor, in declaration order."""

    def __init__(
        self,
        path: Path,
        tensors: dict,
        io: DirectStorage,
        metadata=None,
        *,
        buffer_bytes=ALIGN,
    ):
        if buffer_bytes < ALIGN or buffer_bytes % ALIGN:
            raise ValueError("GPU write buffer must contain whole pages")
        if not tensors:
            raise ValueError("Empty checkpoint shard")
        header = {"__metadata__": metadata or {"format": "pt"}}
        dtype_name = {dtype: name for name, dtype in _DTYPES.items()}
        self.entries = []
        length = 0
        for name, (shape, dtype) in tensors.items():
            size = math.prod(shape) * dtype.itemsize
            if name == "__metadata__" or size <= 0:
                raise ValueError(f"Invalid tensor specification for {name}")
            header[name] = dict(
                dtype=dtype_name[dtype],
                shape=shape,
                data_offsets=[length, length + size],
            )
            self.entries.append((name, dtype, size))
            length += size
        encoded = json.dumps(header, separators=(",", ":")).encode()
        self.begin = _align_up(8 + len(encoded))
        encoded = encoded.ljust(self.begin - 8, b" ")
        self.path, self.io = path, io
        self.temp = TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent)
        self.partial = Path(self.temp.name) / "shard"
        self.total = self.begin + length
        self.file = None
        try:
            allocate_file(self.partial, _align_up(self.total))
            with self.partial.open("r+b", buffering=0) as f:
                f.write(struct.pack("<Q", len(encoded)) + encoded)
            self.file = RegisteredFile(str(self.partial), write=True, direct=True)
            self.tail = io.pool.tensor.new_empty(min(buffer_bytes, _align_up(length)))
        except BaseException:
            if self.file is not None:
                self.file.close()
            self.temp.cleanup()
            raise
        self.used = self.index = self.written = 0
        self.offset = self.begin

    def append(self, name, chunk):
        if self.index >= len(self.entries):
            raise ValueError(f"Unexpected data after last tensor: {name}")
        expected, dtype, size = self.entries[self.index]
        if (
            name != expected
            or chunk.dtype != dtype
            or not chunk.is_contiguous()
            or chunk.device != self.tail.device
        ):
            raise ValueError(f"Expected contiguous GPU data for {expected} ({dtype})")
        src = chunk.reshape(-1).view(torch.uint8)
        if self.written + len(src) > size:
            raise ValueError(f"Too many bytes for {name}")
        self.written += len(src)
        capacity = self.tail.numel()
        if self.used:
            n = min(len(src), capacity - self.used)
            self.tail[self.used : self.used + n].copy_(src[:n])
            self.used += n
            src = src[n:]
            if self.used == capacity:
                self.io.write(self.file, self.offset, self.tail)
                self.offset += capacity
                self.used = 0
        full = len(src) // capacity * capacity
        if full:
            self.io.write(self.file, self.offset, src[:full])
            self.offset += full
        if len(src) > full:
            self.used = len(src) - full
            self.tail[: self.used].copy_(src[full:])
        if self.written == size:
            self.index += 1
            self.written = 0

    def __enter__(self):
        return self

    def __exit__(self, ex_type, *_):
        try:
            if ex_type is None:
                if self.index != len(self.entries):
                    raise ValueError(f"Missing data for {self.entries[self.index][0]}")
                if self.used:
                    padded = _align_up(self.used)
                    self.tail[self.used : padded].zero_()
                    self.io.write(self.file, self.offset, self.tail[:padded])
                os.ftruncate(self.file.fd, self.total)
                os.fsync(self.file.fd)
            self.file.close()
            if ex_type is None:
                os.replace(self.partial, self.path)
        finally:
            if self.file.fd is not None:
                self.file.close()
            self.temp.cleanup()
