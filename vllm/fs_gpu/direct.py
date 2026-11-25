# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded VRAM staging for strict disk-to-device and device-to-disk transfers."""

import os
import struct

import torch

from vllm.fs_gpu.ais import get_ais
from vllm.fs_gpu.region import ALIGN, GpuStagingPool, RegisteredFile, _align_up
from vllm.fs_gpu.safetensors import MAX_HEADER_SIZE, SafetensorFile


class DirectStorage:
    """Use one device and stream. Quantization requires the physical P2P path."""

    def __init__(
        self, device: torch.device, window: int = 16 << 20, *, require_fastpath=True
    ):
        # VLLM_FS_GPU_ALLOW_FALLBACK=1: accept hipFile's host-bounce backend
        # (same bytes, host-read speed). The physical P2P path is unavailable
        # for files whose block device has no PCI parent (e.g. bcache).
        allow_fallback = os.environ.get("VLLM_FS_GPU_ALLOW_FALLBACK") == "1"
        if require_fastpath and not get_ais().fastpath_only and not allow_fallback:
            raise RuntimeError(
                "Set HIPFILE_ALLOW_COMPAT_MODE=false and "
                "HIPFILE_FORCE_COMPAT_MODE=false before starting the process."
            )
        if window < ALIGN or window % ALIGN:
            raise ValueError("Direct storage window must be a multiple of 4096")
        self.pool = GpuStagingPool(window, device)
        self.window = window
        self.copied = torch.cuda.Event()
        self.pending = False

    def read(self, file: RegisteredFile, offset: int, out: torch.Tensor) -> None:
        """Read a contiguous slice, including unaligned safetensors payloads."""
        dest = self._bytes(out)
        if not file.direct:
            raise ValueError("Direct storage requires an O_DIRECT file handle")
        if offset < 0 or offset + dest.numel() > file.size:
            raise ValueError(f"Read outside {file.path}")
        pos = 0
        while pos < dest.numel():
            self._wait()
            start = offset + pos
            prefix = start % ALIGN
            take = min(dest.numel() - pos, self.window - prefix)
            request = _align_up(prefix + take)
            got = file.ais.read_once(file.fh, self.pool.base, request, start - prefix)
            if got < prefix + take:
                raise OSError(f"Short hipFile read from {file.path} at {start}: {got}")
            dest[pos : pos + take].copy_(self.pool.view(prefix, take))
            self.copied.record(torch.cuda.current_stream(self.pool.device))
            self.pending = True
            pos += take

    def write(self, file: RegisteredFile, offset: int, data: torch.Tensor) -> None:
        """Write full pages from GPU, including format-defined final padding."""
        src = self._bytes(data)
        if not file.direct:
            raise ValueError("Direct storage requires an O_DIRECT file handle")
        if offset < 0 or offset % ALIGN or src.numel() % ALIGN:
            raise ValueError("Direct writes must span whole, aligned pages")
        self._wait()
        for pos in range(0, src.numel(), self.window):
            take = min(src.numel() - pos, self.window)
            self.pool.view(0, take).copy_(src[pos : pos + take])
            self.copied.record(torch.cuda.current_stream(self.pool.device))
            self.copied.synchronize()
            file.ais.write(file.fh, self.pool.base, take, offset + pos)

    def safetensor_header(self, file: RegisteredFile) -> SafetensorFile:
        """Expose only the header to the host, even on a shared page."""
        device = self.pool.tensor.device
        length = torch.empty(8, device=device, dtype=torch.uint8)
        self.read(file, 0, length)
        header_len = struct.unpack("<Q", length.cpu().numpy().tobytes())[0]
        if not 1 <= header_len <= min(MAX_HEADER_SIZE, file.size - 8):
            raise ValueError(f"Invalid safetensors header length in {file.path}")
        header = torch.empty(header_len, device=device, dtype=torch.uint8)
        self.read(file, 8, header)
        return SafetensorFile(file.path, header.cpu().numpy().tobytes())

    def _bytes(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device != self.pool.tensor.device or not tensor.is_contiguous():
            raise ValueError("Direct IO requires a contiguous tensor on the pool GPU")
        return tensor.reshape(-1).view(torch.uint8)

    def _wait(self) -> None:
        if self.pending:
            self.copied.synchronize()
            self.pending = False

    def close(self) -> None:
        self._wait()
        self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def allocate_file(path: str | os.PathLike, size: int) -> None:
    """Reserve space before direct IO, so a partial conversion fails early."""
    if size <= 0 or size % ALIGN:
        raise ValueError("Output size must be a positive multiple of 4096")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
    try:
        os.posix_fallocate(fd, 0, size)
    finally:
        os.close(fd)
