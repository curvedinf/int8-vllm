#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pack Inkling PTQR export pieces into balanced safetensors release shards."""

import argparse
import json
import os
import shutil
import struct
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm.fs_gpu.region import _align_up
from vllm.fs_gpu.safetensors import _DTYPES, SafetensorFile

MIN_SHARD_BYTES = 2_000_000_000
TARGET_SHARD_BYTES = 4 << 30
INDEX_NAME = "model.safetensors.index.json"


def _plan_shards(source: Path, weight_map: dict[str, str]):
    expected = defaultdict(set)
    for tensor, filename in weight_map.items():
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError(f"Invalid checkpoint piece {filename!r}")
        expected[filename].add(tensor)

    pieces = []
    for filename, tensors in expected.items():
        header = SafetensorFile(str(source / filename))
        if tensors != header.tensors.keys():
            raise ValueError(f"Index disagrees with {filename}")
        end = 0
        for spec in sorted(header, key=lambda spec: spec.start):
            if spec.start != end:
                raise ValueError(f"Noncontiguous tensor {spec.name} in {filename}")
            end = spec.end
        if (source / filename).stat().st_size != header.data_begin + end:
            raise ValueError(f"Incomplete checkpoint piece {filename}")
        pieces.append((end, header))

    total = sum(size for size, _ in pieces)
    if total < MIN_SHARD_BYTES:
        raise ValueError("Inkling export is too small for a complete release")
    max_count = min(
        len(pieces),
        total // MIN_SHARD_BYTES,
        (total + TARGET_SHARD_BYTES - 1) // TARGET_SHARD_BYTES,
    )
    # Group whole pieces so assembly can copy contiguous file extents.
    for count in range(max_count, 0, -1):
        groups = [[] for _ in range(count)]
        sizes = [0] * count
        for size, header in sorted(pieces, key=lambda piece: -piece[0]):
            index = min(range(count), key=sizes.__getitem__)
            groups[index].append((size, header))
            sizes[index] += size
        if min(sizes) >= MIN_SHARD_BYTES:
            return groups, total
    raise ValueError("Could not assemble a complete release")


def _write_shard(path: Path, pieces):
    # A checkpoint piece already resides on disk. Only its header goes
    # through Python. The kernel copies the serialized tensor data.
    dtype_names = {dtype: name for name, dtype in _DTYPES.items()}
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for size, piece in pieces:
        for spec in piece:
            header[spec.name] = {
                "dtype": dtype_names[spec.dtype],
                "shape": spec.shape,
                "data_offsets": [offset + spec.start, offset + spec.end],
            }
        offset += size

    body = json.dumps(header, separators=(",", ":")).encode()
    header_size = _align_up(8 + len(body)) - 8
    with path.open("xb", buffering=0) as dest:
        dest.write(struct.pack("<Q", header_size) + body.ljust(header_size, b" "))
        for size, piece in pieces:
            with open(piece.path, "rb", buffering=0) as src:
                src.seek(piece.data_begin)
                remaining = size
                while remaining:
                    count = os.copy_file_range(
                        src.fileno(), dest.fileno(), min(remaining, 64 << 20)
                    )
                    if not count:
                        raise OSError(f"Short copy from {piece.path}")
                    remaining -= count
        os.fsync(dest.fileno())
    if path.stat().st_size != 8 + header_size + offset:
        raise OSError(f"Incomplete release shard {path}")
    return header.keys() - {"__metadata__"}


def copy_assets(source: Path, destination: Path):
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
        "processor_config.json",
    ):
        shutil.copyfile(source / name, destination / name)
    shutil.copytree(source / "tiktoken", destination / "tiktoken")


def assemble(source: Path, destination: Path):
    if source.resolve() == destination.resolve():
        raise ValueError("Use a separate destination for the assembled checkpoint")
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise FileExistsError(f"Expected an empty release directory at {destination}")
    groups, total = _plan_shards(
        source, json.loads((source / INDEX_NAME).read_text())["weight_map"]
    )
    with TemporaryDirectory(dir=destination.parent) as folder:
        stage = Path(folder) / "release"
        stage.mkdir()
        weight_map = {}
        for index, group in enumerate(groups, 1):
            filename = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
            for name in sorted(_write_shard(stage / filename, group)):
                weight_map[name] = filename
        shutil.copyfile(source / "config.json", stage / "config.json")
        copy_assets(source, stage)
        (stage / INDEX_NAME).write_text(
            json.dumps(
                {"metadata": {"total_size": total}, "weight_map": weight_map},
                indent=2,
            )
            + "\n"
        )
        stage.rename(destination)
    print(
        f"Assembled {len(groups)} release shards with {total:,} payload bytes",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    assemble(args.source, args.destination)
