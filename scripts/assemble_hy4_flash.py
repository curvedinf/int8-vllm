#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Assemble and check the deployable HY4 VL target and its separate DFly2 companion."""

import argparse
import json
import math
import os
import shutil
import struct
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from vllm.fs_gpu.region import ALIGN, _align_up
from vllm.fs_gpu.safetensors import SafetensorFile
from vllm.models.hy_v4_flash.engram_q4 import (
    ROW_BYTES,
    ROWS_PER_PAGE,
    n_pages,
    table_path,
)

DRAFT_ID = "voodooquant/Hy4-flash-DFly2-PTQR-INT8-W8A8-GS128"
MODIFICATION = "Modified by voodooquant from Tencent Hy4-flash. See NOTICE and LICENSE."


def read(path):
    return json.loads(path.read_text())


def store(path, data):
    with NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        json.dump(data, tmp, indent=2)
        tmp.write("\n")
    os.replace(tmp.name, path)


def mark_shard(path):
    """Attach the derivative notice to the header while preserving every tensor byte."""
    with path.open("r+b", buffering=0) as file:
        original = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(original))
        meta = header.setdefault("__metadata__", {})
        if meta.get("modification") == MODIFICATION:
            return
        meta["modification"] = MODIFICATION
        body = json.dumps(header, separators=(",", ":")).encode()
        if len(body) <= original:
            file.seek(8)
            file.write(body.ljust(original, b" "))
            os.fsync(file.fileno())
            return
        # A nearly full metadata page may need one more page. Copy the
        # serialized payload within the filesystem, without decoding it.
        size = _align_up(8 + len(body)) - 8
        with TemporaryDirectory(dir=path.parent) as folder:
            destination = Path(folder) / path.name
            with destination.open("wb", buffering=0) as updated:
                updated.write(struct.pack("<Q", size) + body.ljust(size, b" "))
                remaining = os.fstat(file.fileno()).st_size - original - 8
                while remaining:
                    count = os.copy_file_range(
                        file.fileno(), updated.fileno(), min(remaining, 64 << 20)
                    )
                    if count == 0:
                        raise OSError(f"Short copy while labeling {path.name}")
                    remaining -= count
                os.fsync(updated.fileno())
            os.replace(destination, path)


def check_index(directory, table):
    files = defaultdict(set)
    for name, filename in table.items():
        if Path(filename).name != filename:
            raise ValueError(f"Invalid checkpoint shard {filename}")
        files[filename].add(name)
    total = 0
    for filename, keys in files.items():
        path = directory / filename
        header = SafetensorFile(str(path))
        if set(header.tensors) != keys:
            raise ValueError(f"Index disagrees with {filename}")
        end = 0
        for spec in sorted(header, key=lambda s: s.start):
            if (
                spec.start != end
                or math.prod(spec.shape) * spec.dtype.itemsize != spec.nbytes
            ):
                raise ValueError(f"Incorrect extent of {spec.name}")
            end = spec.end
        if path.stat().st_size != header.data_begin + end:
            raise ValueError(f"Truncated checkpoint shard {filename}")
        mark_shard(path)
        total += end
    return total


def copy_assets(source, dest, vision):
    files = (
        "LICENSE",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    )
    for name in files + (("processor_config.json",) if vision else ()):
        shutil.copyfile(source / name, dest / name)
    surfaces = (
        "The layer shards carry NeUQI INT4 language projections and experts. "
        "The vision shard carries PTQR INT8 projections. The Engram files contain "
        "new Q4_0 tables with 75 rows per page."
        if vision
        else "The DFly2 shards carry PTQR INT8 projections for the original drafter."
    )
    (dest / "NOTICE").write_text(
        "Tencent Hy4-flash\n\n" + MODIFICATION + "\n" + surfaces + "\n"
        "The configuration and weight index describe the derivative layout.\n"
        "The original distribution's Tencent Hy License, Version 1.0, is in LICENSE.\n"
    )


def target(source, directory, ns, companion):
    original = read(source / "config.json")
    text = original["text_config"]
    for layer in text["engram_layer_ids"]:
        count = sum(
            sum(order) for order in text["engram_layer_head_vocab_sizes"][str(layer)]
        )
        path = table_path(directory / "engram", layer)
        if path.stat().st_size != n_pages(count) * ALIGN:
            raise ValueError(f"Incomplete packed Engram table for layer {layer}")
        with path.open("r+b", buffering=0) as table:
            # The last 46 bytes of every 4 KiB page lie outside the 75 rows.
            table.seek(ROWS_PER_PAGE * ROW_BYTES)
            table.write(b"Modified by voodooquant: Engram Q4_0")
            os.fsync(table.fileno())
    weights = read(directory / "backbone.index.json")["weight_map"]
    vision = read(directory / "vision.index.json")["weight_map"]
    if weights.keys() & vision.keys() or not vision:
        raise ValueError("Wrong target or vision index")
    weights.update(vision)
    quant = read(directory / "backbone.quant.json")
    vision_quant = read(directory / "vision.quant.json")["quantized_modules"]
    if set(vision_quant.values()) != {8} or set(
        quant["quantized_modules"].values()
    ) != {4}:
        raise ValueError(
            "Backbone must be NeUQI W4 and the trained vision projections must be W8"
        )
    audit = read(directory / "calibration.json")
    if [row["layer"] for row in audit["layers"]] != list(
        range(text["num_hidden_layers"])
    ):
        raise ValueError("Incomplete language calibration")
    original_map = read(source / "model.safetensors.index.json")["weight_map"]
    for name in original_map:
        if name.startswith("model.dfly_layers.") or name.endswith(
            "multi_head_embedding.embedding.weight"
        ):
            continue
        if name.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")):
            leaf, proj = name.rsplit(".", 1)
            for expert in range(text["n_routed_experts"]):
                for suffix in ("weight", "weight_scale", "weight_zero"):
                    if f"{leaf}.{expert}.{proj}.{suffix}" not in weights:
                        raise ValueError(
                            f"Missing expert payload for {name} expert {expert}"
                        )
        elif name not in weights:
            raise ValueError(f"Missing source tensor {name}")
    quant["quantized_modules"].update(vision_quant)
    original.update(
        dtype="float16", quantization_config=quant, modification_notice=MODIFICATION
    )
    text.update(dtype="float16", quantization_config=quant)
    original["speculative_config"] = {
        "method": "dfly",
        "model": companion,
        "num_speculative_tokens": ns,
        "draft_sample_method": "probabilistic",
    }
    total = check_index(directory, weights)
    copy_assets(source, directory, True)
    store(
        directory / "model.safetensors.index.json",
        {
            "metadata": {"total_size": total, "modification": MODIFICATION},
            "weight_map": weights,
        },
    )
    store(directory / "config.json", original)
    print(
        f"Validated {len(weights):,} target/vision tensors, "
        f"{total:,} payload bytes and four packed tables."
    )


def draft(source, directory):
    from vllm.models.hy_v4_flash.configs import HYV4FlashVLConfig
    from vllm.models.hy_v4_flash.dfly_config import HYV4FlashDFlySpeculatorConfig

    original = read(source / "config.json")
    runs = read(directory / "dfly-training.json")
    original["dfly_intermediate_size"] = runs["dfly_intermediate_size"]
    parent = HYV4FlashVLConfig.from_dict(original)
    cfg = HYV4FlashDFlySpeculatorConfig.derive_draft_hf_config(
        parent, target_config=parent
    ).to_dict()
    cfg["modification_notice"] = MODIFICATION
    cfg.update(
        model_type="hy_v4_dfly",
        architectures=["HYV4DFlyModel"],
        dtype="float16",
        quantization_config=read(directory / "dfly.quant.json"),
    )
    weights = read(directory / "model.safetensors.index.json")["weight_map"]
    expected = {
        n
        for n in read(source / "model.safetensors.index.json")["weight_map"]
        if n.startswith("model.dfly_layers.")
    }
    if not expected <= weights.keys() or set(
        cfg["quantization_config"]["quantized_modules"].values()
    ) != {8}:
        raise ValueError("Incomplete DFly2 PTQR checkpoint")
    total = check_index(directory, weights)
    copy_assets(source, directory, False)
    store(
        directory / "model.safetensors.index.json",
        {
            "metadata": {"total_size": total, "modification": MODIFICATION},
            "weight_map": weights,
        },
    )
    store(directory / "config.json", cfg)
    print(f"Validated {len(weights):,} DFly tensors and {total:,} payload bytes.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--target", type=Path)
    p.add_argument("--draft", type=Path)
    p.add_argument("--num-speculative-tokens", type=int, default=7)
    p.add_argument("--companion-id", default=DRAFT_ID)
    args = p.parse_args()
    if not (args.target or args.draft) or args.num_speculative_tokens < 1:
        p.error("Specify a target and/or draft output and a positive draft width")
    if args.target:
        target(args.source, args.target, args.num_speculative_tokens, args.companion_id)
    if args.draft:
        draft(args.source, args.draft)


if __name__ == "__main__":
    main()
