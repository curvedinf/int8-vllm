# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Download immutable Hub byte ranges to a bounded native NVMe workspace.

The HTTP client acquires source artifacts. The converter reads every downloaded
payload with strict hipFile and never copies a local source file through host RAM.
"""

import json
import math
import re
import struct
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote
from urllib.request import Request, urlopen

import torch

from vllm.fs_gpu.region import RegisteredFile
from vllm.fs_gpu.safetensors import MAX_HEADER_SIZE, SafetensorFile


class HubDirectSource:
    def __init__(self, directory, io, workspace, repo, revision, window_mib=32):
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise ValueError("Pin a Hub commit hash for range downloads")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or window_mib < 1:
            raise ValueError("Invalid source repository or window size")
        self.io = io
        self.device = io.pool.device
        self.directory = Path(directory)
        self.index = json.loads(
            (self.directory / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.base = f"https://huggingface.co/{repo}/resolve/{revision}"
        self.window = window_mib << 20
        self.temp = TemporaryDirectory(prefix="hy4-download-", dir=workspace)
        self.path = Path(self.temp.name) / "source.range"
        self.headers = {}
        self.cached = None
        self.file = None
        self.embedding = None
        self.layer_cache = {}
        self.cache_prefix = None
        self.calibration_dir = None
        self.corpus_identity = None
        self.engram_rows = {}

    def _url(self, filename, first, last):
        if Path(filename).name != filename:
            raise ValueError(f"Invalid upstream shard name: {filename}")
        return f"{self.base}/{quote(filename)}?hy4_range={first}-{last}"

    def _metadata(self, filename, first, last):
        request = Request(
            self._url(filename, first, last), headers={"Range": f"bytes={first}-{last}"}
        )
        with urlopen(request, timeout=60) as response:
            if response.status != 206 or not response.headers[
                "Content-Range"
            ].startswith(f"bytes {first}-{last}/"):
                raise OSError(
                    f"Hub returned incorrect safetensor header range for {filename}"
                )
            result = response.read(last - first + 1)
            if len(result) != last - first + 1:
                raise OSError("Incomplete Hub metadata")
            return result

    def _entry(self, name):
        filename = self.index[name]
        if filename not in self.headers:
            local = self.directory / filename
            if local.is_file():
                # Only the upstream safetensor length and JSON table are read.
                header = SafetensorFile(str(local))
            else:
                (length,) = struct.unpack("<Q", self._metadata(filename, 0, 7))
                if not 1 <= length <= MAX_HEADER_SIZE:
                    raise ValueError(f"Invalid Hub safetensors header length: {length}")
                header = SafetensorFile(
                    filename, self._metadata(filename, 8, 7 + length)
                )
            self.headers[filename] = header
        header = self.headers[filename]
        spec = header[name]
        if math.prod(spec.shape) * spec.dtype.itemsize != spec.nbytes:
            raise ValueError(f"Invalid source tensor extent: {name}")
        return filename, header.data_begin + spec.start, spec

    def file_order(self, name):
        filename, origin, _ = self._entry(name)
        return filename, origin

    def spec(self, name):
        return self._entry(name)[2]

    def _download(self, filename, start, end):
        if self.file is not None:
            self.file.close()
            self.file = None
        self.cached = None
        headers = self.path.with_suffix(".headers")
        subprocess.run(
            [
                "curl",
                "--silent",
                "--show-error",
                "--fail",
                "--location",
                "--retry",
                "4",
                "--max-filesize",
                str(end - start),
                "--max-time",
                "300",
                "--header",
                "Accept-Encoding: identity",
                "--range",
                f"{start}-{end - 1}",
                "--dump-header",
                str(headers),
                "--output",
                str(self.path),
                self._url(filename, start, end - 1),
            ],
            check=True,
        )
        received = re.findall(
            r"(?im)^content-range:\s*bytes (\d+)-(\d+)/\d+\s*$", headers.read_text()
        )
        if self.path.stat().st_size != end - start or received[-1:] != [
            (str(start), str(end - 1))
        ]:
            raise OSError(f"Incomplete or mismatched Hub byte range for {filename}")
        self.file = RegisteredFile(str(self.path), direct=True)
        self.cached = filename, start, end

    def read(self, name, *, start=0, shape=None):
        if name in self.layer_cache:
            value = self.layer_cache[name]
            if shape is None and not start:
                return value
            return value.reshape(-1).narrow(0, start, math.prod(shape)).view(shape)
        filename, origin, spec = self._entry(name)
        shape = spec.shape if shape is None else tuple(shape)
        end = start + math.prod(shape)
        if start < 0 or end * spec.dtype.itemsize > spec.nbytes:
            raise ValueError(f"Tensor range exceeds {name}")
        output = torch.empty(shape, device=self.device, dtype=spec.dtype)
        dest = output.reshape(-1).view(torch.uint8)
        begin = origin + start * spec.dtype.itemsize
        pos = 0
        while pos < dest.numel():
            at = begin + pos
            if (
                not self.cached
                or self.cached[0] != filename
                or not self.cached[1] <= at < self.cached[2]
            ):
                header = self.headers[filename]
                end_of_file = header.data_begin + max(s.end for s in header)
                stop = min(at + self.window, end_of_file)
                self._download(filename, at, stop)
            _, low, high = self.cached
            count = min(len(dest) - pos, high - at)
            self.io.read(self.file, at - low, dest[pos : pos + count])
            pos += count
        if (
            self.cache_prefix
            and name.startswith(self.cache_prefix)
            and start == 0
            and shape == spec.shape
        ):
            self.layer_cache[name] = output
        return output

    def begin_layer(self, idx):
        self.layer_cache.clear()
        self.engram_rows.clear()
        self.cache_prefix = f"model.layers.{idx}."

    def read_rows(self, name, rows):
        if name == "model.embed_tokens.weight":
            if self.embedding is None:
                self.embedding = self.read(name)
            return self.embedding[rows]
        if self.calibration_dir is None or ".engram_layer." not in name:
            raise ValueError(f"No GPU row source for {name}")
        if name not in self.engram_rows:
            import time

            layer = int(name.split(".")[2]) + 1
            path = self.calibration_dir / f"calibration-engram-{layer}.safetensors"
            if not path.exists():
                print(
                    f"Waiting for calibration rows from packed Engram layer {layer}",
                    flush=True,
                )
            while not path.is_file():
                time.sleep(3)
            with RegisteredFile(str(path), direct=True) as file:
                header = self.io.safetensor_header(file)
                if header.metadata != {"source": name, "corpus": self.corpus_identity}:
                    raise ValueError(
                        f"Engram calibration source or corpus differs for {name}"
                    )
                tensors = []
                for key in ("row_ids", "source_rows"):
                    spec = header[key]
                    value = torch.empty(
                        spec.shape, dtype=spec.dtype, device=self.device
                    )
                    self.io.read(file, header.data_begin + spec.start, value)
                    tensors.append(value)
            self.engram_rows[name] = tuple(tensors)
        selected, values = self.engram_rows[name]
        indices = torch.searchsorted(selected, rows.contiguous())
        if bool((indices >= len(selected)).any()) or not torch.equal(
            selected[indices], rows
        ):
            raise ValueError(f"Calibration asks for an unobserved Engram row in {name}")
        return values[indices]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.file is not None:
            self.file.close()
        self.temp.cleanup()
