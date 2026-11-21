# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named GPU-only reads from indexed safetensors, including slices of large tensors."""

import json
import math
from contextlib import ExitStack
from pathlib import Path

import torch

from vllm.fs_gpu.direct import DirectStorage
from vllm.fs_gpu.region import RegisteredFile


class DirectCheckpoint:
    def __init__(self, directory: Path, io: DirectStorage):
        self.directory = Path(directory)
        self.io = io
        self.device = io.pool.device
        self.index = json.loads(
            (self.directory / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.files = {}
        self.context = ExitStack()

    def _entry(self, name):
        filename = self.index[name]
        if Path(filename).name != filename:
            raise ValueError(f"Invalid safetensors shard name: {filename}")
        if filename not in self.files:
            file = self.context.enter_context(
                RegisteredFile(str(self.directory / filename), direct=True)
            )
            header = self.io.safetensor_header(file)
            self.files[filename] = file, header
        file, header = self.files[filename]
        spec = header[name]
        if math.prod(spec.shape) * spec.dtype.itemsize != spec.nbytes:
            raise ValueError(f"Incorrect tensor bounds: {name}")
        return file, header.data_begin + spec.start, spec

    def spec(self, name):
        return self._entry(name)[2]

    def read(self, name, *, start=0, shape=None):
        file, origin, spec = self._entry(name)
        shape = spec.shape if shape is None else tuple(shape)
        if start < 0 or (start + math.prod(shape)) * spec.dtype.itemsize > spec.nbytes:
            raise ValueError(f"Tensor slice exceeds {name}")
        data = torch.empty(shape, dtype=spec.dtype, device=self.device)
        self.io.read(file, origin + start * spec.dtype.itemsize, data)
        return data

    def read_rows(self, name, rows):
        """Read the distinct embedding rows requested by calibration into VRAM."""
        file, origin, spec = self._entry(name)
        if len(spec.shape) != 2:
            raise ValueError(f"Expected matrix for row lookup: {name}")
        unique, inverse = rows.flatten().unique(sorted=True, return_inverse=True)
        # Only row addresses reach the host. File data goes to device storage.
        indices = unique.tolist()
        width = spec.shape[1]
        stride = width * spec.dtype.itemsize
        packed = torch.empty(
            (len(indices), width), device=self.device, dtype=spec.dtype
        )
        first = 0
        while first < len(indices):
            row = indices[first]
            stop = first + 1
            while stop < len(indices) and indices[stop] == row + stop - first:
                stop += 1
            if not 0 <= row <= indices[stop - 1] < spec.shape[0]:
                raise IndexError(f"Embedding row outside {name}")
            self.io.read(file, origin + row * stride, packed[first:stop])
            first = stop
        return packed[inverse].reshape(*rows.shape, width)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.context.close()
