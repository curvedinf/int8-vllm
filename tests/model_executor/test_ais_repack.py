# SPDX-License-Identifier: Apache-2.0
"""Packed-byte and publication contract for the optional CPU cache writer."""

import os
from pathlib import Path

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
