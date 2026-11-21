#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert HY4 Engram tables from BF16 safetensors to aligned GPU-packed Q4_0."""

import argparse
import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from vllm.fs_gpu.direct import DirectStorage, allocate_file
from vllm.fs_gpu.region import ALIGN, RegisteredFile
from vllm.models.hy_v4_flash.engram_q4 import (
    ROWS_PER_PAGE,
    WIDTH,
    encode_rows,
    n_pages,
    page_rows,
    table_path,
)

TABLE = re.compile(
    r"model\.layers\.(\d+)\.(engram|engram_layer)\.multi_head_embedding\.embedding\.weight"
)


def convert_table(
    source: Path,
    tensor_name: str,
    output: Path,
    io: DirectStorage,
    chunk_pages: int = 256,
) -> None:
    if chunk_pages < 1:
        raise ValueError("chunk_pages must be positive")
    with RegisteredFile(str(source), direct=True) as meta:
        header = io.safetensor_header(meta)
    spec = header[tensor_name]
    if (
        len(spec.shape) != 2
        or spec.shape[1] != WIDTH
        or spec.dtype not in (torch.bfloat16, torch.float16)
    ):
        raise ValueError(f"Expected 96-wide floating-point Engram rows, found {spec}")
    nrows = spec.shape[0]
    size = n_pages(nrows) * ALIGN
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        partial = Path(temp) / "table"
        allocate_file(partial, size)
        device = io.pool.tensor.device
        rows = torch.empty(
            (chunk_pages * ROWS_PER_PAGE, WIDTH), device=device, dtype=spec.dtype
        )
        packed = torch.empty((chunk_pages, ALIGN), device=device, dtype=torch.uint8)
        with (
            RegisteredFile(str(source), direct=True) as src,
            RegisteredFile(str(partial), write=True, direct=True) as dst,
        ):
            for first_page in range(0, n_pages(nrows), chunk_pages):
                first_row = first_page * ROWS_PER_PAGE
                take = min(len(rows), nrows - first_row)
                source_offset = (
                    header.data_begin
                    + spec.start
                    + first_row * WIDTH * rows.element_size()
                )
                io.read(src, source_offset, rows[:take])
                chunk = packed[: n_pages(take)]
                chunk.zero_()
                encoded = encode_rows(rows[:take])
                full, tail = divmod(take, ROWS_PER_PAGE)
                view = page_rows(chunk)
                if full:
                    view[:full].copy_(
                        encoded[: full * ROWS_PER_PAGE].view(full, ROWS_PER_PAGE, -1)
                    )
                if tail:
                    view[full, :tail].copy_(encoded[-tail:])
                io.write(dst, first_page * ALIGN, chunk)
            os.fsync(dst.fd)
        os.replace(partial, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--layer",
        type=int,
        action="append",
        help="One-based HY4 Engram layer, repeatable",
    )
    parser.add_argument("--chunk-pages", type=int, default=256)
    args = parser.parse_args()
    os.environ.setdefault("HIPFILE_ALLOW_COMPAT_MODE", "false")
    os.environ.setdefault("HIPFILE_FORCE_COMPAT_MODE", "false")
    index = json.loads((args.source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    config = json.loads((args.source / "config.json").read_text())
    expected = set(config.get("text_config", config)["engram_layer_ids"])
    wanted = set(args.layer) if args.layer else expected
    if not wanted or not wanted <= expected:
        parser.error(f"Choose Engram layers from {sorted(expected)}")
    names = {}
    for name, shard in index.items():
        if match := TABLE.fullmatch(name):
            layer_id = int(match[1]) + (match[2] == "engram_layer")
            if layer_id in names:
                raise ValueError(f"Duplicate Engram table for layer {layer_id}")
            names[layer_id] = (name, shard)
    if set(names) != expected:
        raise ValueError(
            f"Expected Engram tables for {sorted(expected)}, found {sorted(names)}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.device)
    with DirectStorage(torch.device("cuda", args.device)) as io:
        for layer in sorted(wanted):
            name, shard = names[layer]
            dest = table_path(args.output, layer)
            print(f"Packing {name} into {dest}", flush=True)
            convert_table(args.source / shard, name, dest, io, args.chunk_pages)


if __name__ == "__main__":
    main()
