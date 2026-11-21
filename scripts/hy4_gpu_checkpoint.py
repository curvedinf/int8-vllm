# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep complete calibrated safetensor shards on peer GPUs until export."""

import json
import math
import struct
from contextlib import contextmanager
from pathlib import Path

import torch

from scripts.hy4_archive import archive_gpu_bytes
from vllm.fs_gpu.direct import DirectStorage
from vllm.fs_gpu.region import _align_up
from vllm.fs_gpu.safetensors import _DTYPES


class GPUCheckpoint:
    def __init__(self, device, peers, workspace):
        self.source = device
        self.devices = [torch.device("cuda", i) for i in peers]
        if not self.devices:
            raise ValueError("The GPU checkpoint needs at least one storage GPU")
        for dest in self.devices:
            if dest != device and not torch.cuda.can_device_access_peer(
                device.index, dest.index
            ):
                raise ValueError(f"No peer path between {device} and {dest}")
        self.workspace = workspace
        self.shards = []

    @contextmanager
    def writer(self, path, specs, io):
        dtype_names = {dtype: key for key, dtype in _DTYPES.items()}
        header = {"__metadata__": {"format": "pt"}}
        offset = 0
        for name, (shape, dtype) in specs.items():
            size = math.prod(shape) * dtype.itemsize
            header[name] = {
                "dtype": dtype_names[dtype],
                "shape": shape,
                "data_offsets": [offset, offset + size],
            }
            offset += size
        body = json.dumps(header, separators=(",", ":")).encode()
        begin = _align_up(len(body) + 8)
        body = struct.pack("<Q", begin - 8) + body.ljust(begin - 8, b" ")
        free, device = max(
            ((torch.cuda.mem_get_info(d)[0], d) for d in self.devices),
            key=lambda pair: pair[0],
        )
        if free < begin + offset + (256 << 20):
            raise MemoryError(f"Insufficient peer VRAM to hold {path.name}")
        data = torch.empty(begin + offset, dtype=torch.uint8, device=device)
        data[:begin].copy_(torch.frombuffer(bytearray(body), dtype=torch.uint8))
        cursor = begin
        names = iter(specs)
        current = next(names)

        def append(name, chunk):
            nonlocal cursor, current
            if (
                name != current
                or chunk.dtype != specs[name][1]
                or chunk.device != self.source
            ):
                raise ValueError(f"Invalid GPU checkpoint data for {name}")
            stop = cursor + chunk.nbytes
            end = begin + header[name]["data_offsets"][1]
            if stop > end:
                raise ValueError(f"Oversized tensor {name}")
            data[cursor:stop].copy_(chunk.contiguous().reshape(-1).view(torch.uint8))
            cursor = stop
            if stop == end:
                current = next(names, None)

        from types import SimpleNamespace

        yield SimpleNamespace(append=append)
        if current is not None or cursor != data.numel():
            raise ValueError(f"Incomplete GPU shard {path.name}")
        self.shards.append((Path(path), data))

    def export(self):
        # Deferred until calibration and conversion have completed on all layers.
        for device in self.devices:
            with torch.cuda.device(device), DirectStorage(device) as io:
                for index, entry in enumerate(self.shards):
                    if entry is None or entry[1].device != device:
                        continue
                    path, data = entry
                    archive_gpu_bytes(data, path, self.workspace, io)
                    self.shards[index] = None
                    del data
            print(f"Exported completed checkpoint shards on {device}", flush=True)
