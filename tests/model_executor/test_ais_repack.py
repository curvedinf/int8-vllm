# SPDX-License-Identifier: Apache-2.0
"""Packed-byte and publication contract for the optional CPU cache writer."""

import json
import os
import struct
from pathlib import Path

import pytest
import torch

from vllm.fs_gpu.repack import RepackCache


def test_cpu_repack_short_writes_and_alignment(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_AIS_REPACK_CPU_WRITE", "1")

    def unexpected_ais():
        raise AssertionError("CPU cache construction must not initialize AIS")

    monkeypatch.setattr("vllm.fs_gpu.repack.get_ais", unexpected_ais)
    source = tmp_path / "source.safetensors"
    source.write_bytes(b"source fingerprint")
    cache = RepackCache(str(tmp_path / "cache"), 4, 0)
    writer = cache.writer([str(source)])
    pwrite = os.pwrite
    monkeypatch.setattr(
        os, "pwrite", lambda fd, data, offset: pwrite(fd, data[:3], offset)
    )
    writer.append(
        "odd",
        torch.uint8,
        (5,),
        torch.tensor([0, 127, 128, 254, 255], dtype=torch.uint8),
        [(0, 5)],
    )
    writer.append(
        "next",
        torch.uint8,
        (4,),
        torch.tensor([4, 3, 2, 1], dtype=torch.uint8),
        [(0, 4)],
    )
    assert not Path(cache.meta_path).exists()
    assert writer.commit()
    assert cache.validate([str(source)])
    payload = Path(cache.bin_path).read_bytes()
    assert payload[:5] == bytes([0, 127, 128, 254, 255])
    assert payload[8:12] == bytes([4, 3, 2, 1])
    assert cache.tensors[1].block_off == 8
    source.write_bytes(b"changed source")
    assert not cache.validate([str(source)])


@pytest.mark.parametrize(
    "contents", [b"\x01", struct.pack("<Q", 1 << 63), struct.pack("<Q", 8) + b"{}"]
)
def test_incomplete_rank_cache_header_reports_invalid_file(tmp_path, contents):
    from vllm.fs_gpu.safetensors import SafetensorFile

    path = tmp_path / "rank.safetensors"
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="header"):
        SafetensorFile(str(path))


@pytest.mark.parametrize(
    "header",
    [
        {"__metadata__": []},
        {"w": {"dtype": "U8", "shape": [1], "data_offsets": [0]}},
        {"w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 2]}},
    ],
)
def test_invalid_rank_cache_tensor_directory(header):
    from vllm.fs_gpu.safetensors import SafetensorFile

    with pytest.raises(ValueError):
        SafetensorFile("rank.safetensors", json.dumps(header).encode())
