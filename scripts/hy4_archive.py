# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Archive finished GPU checkpoint objects through a bounded native-NVMe spool."""

import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from vllm.fs_gpu.direct import allocate_file
from vllm.fs_gpu.region import ALIGN, RegisteredFile, _align_up


def archive_gpu_bytes(data, destination, workspace, io, chunk_mib=32):
    """Serialize an immutable GPU artifact by strict hipFile, then install its bytes.

    The downstream file copy is ordinary filesystem transport, which can use the
    kernel page cache on a non-P2P volume. It never supplies data to quantization.
    """
    if data.device != io.pool.device or not data.is_contiguous():
        raise ValueError("Expected the finished artifact on the active IO GPU")
    if chunk_mib < 1:
        raise ValueError("The direct export window must be positive")
    data = data.reshape(-1).view(torch.uint8)
    destination = Path(destination)
    chunk_size = chunk_mib << 20
    with (
        TemporaryDirectory(prefix=".install-", dir=destination.parent) as out,
        TemporaryDirectory(prefix="hy4-export-", dir=workspace) as spool,
    ):
        piece = Path(spool) / "completed.part"
        complete = Path(out) / "artifact"
        with complete.open("xb", buffering=0) as archival_fd:
            for pos in range(0, data.numel(), chunk_size):
                count = min(chunk_size, data.numel() - pos)
                padded = _align_up(count)
                if shutil.disk_usage(spool).free < padded + (192 << 20):
                    raise OSError(
                        "Insufficient native NVMe space for the bounded export"
                    )
                allocate_file(piece, padded)
                with RegisteredFile(str(piece), write=True, direct=True) as direct:
                    span = data[pos : pos + count]
                    if count % ALIGN:
                        last = span.new_zeros(padded)
                        last[:count].copy_(span)
                        span = last
                    io.write(direct, 0, span)
                    os.ftruncate(direct.fd, count)
                    os.fsync(direct.fd)
                # External artifact installation consumes only completed bytes
                # already serialized to native NVMe. No host tensor is created.
                subprocess.run(["cat", "--", piece], stdout=archival_fd, check=True)
                piece.unlink()
            os.fsync(archival_fd.fileno())
        if complete.stat().st_size != data.numel():
            raise OSError(f"Incomplete archive of {destination.name}")
        os.replace(complete, destination)
