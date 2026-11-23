# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refit Inkling-2-Mini vision at W8A8 and assemble it with the trained W4 text."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from scripts.assemble_inkling2 import INDEX_NAME, assemble, copy_assets
from scripts.hy4_gpu_checkpoint import GPUCheckpoint
from scripts.hy4_hub_source import HubDirectSource
from scripts.inkling2_teacher import SOURCE_REPO, SOURCE_REVISION
from scripts.ptqr_inkling2 import fit_projection, load_inputs, source_weight
from vllm.fs_gpu.checkpoint import DirectCheckpoint
from vllm.fs_gpu.direct import DirectStorage

PATCH_WEIGHT = "model.visual.backbone.embeddings.patch_embedding.weight"


def patch_samples(source, images, seed):
    """Collect the teacher's normalized RGB patches on the same 48/12 image split."""
    from PIL import Image
    from transformers import AutoProcessor

    from scripts.calibrate_inkling2 import InputSamples

    paths = sorted(
        path
        for path in images.rglob("*")
        if path.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if len(paths) < 60:
        raise ValueError(
            "Patch calibration requires 48 training images and 12 held-out images"
        )
    processor = AutoProcessor.from_pretrained(
        source, local_files_only=True
    ).image_processor
    config = json.loads((source / "config.json").read_text())["vision_config"]
    patch, merge = config["native_patch_size"], config["adapter_merge_size"]
    results = []
    for group in (paths[:48], paths[48:60]):
        sample = InputSamples(seed, 768, 0)
        for path in group:
            with Image.open(path) as im:
                img = im.convert("RGB")
                img.thumbnail((640, 640))
                pixels = processor(images=img, return_tensors="pt")["pixel_values"]
            rows = (
                pixels.reshape(-1, merge, patch, merge, patch, 3)
                .transpose(2, 3)
                .reshape(-1, patch * patch * 3)
            )
            sample.observe(PATCH_WEIGHT, (rows.half(),))
        priority, values = sample.rows[PATCH_WEIGHT]
        results.append(values[priority >= 0])
    return results


def train(args):
    device = torch.device("cuda", 0)
    torch.cuda.manual_seed(args.seed)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    index = json.loads((args.target_pieces / INDEX_NAME).read_text())["weight_map"]
    config = json.loads((args.target_pieces / "config.json").read_text())
    manifest = config["quantization_config"]
    expected = {
        key for key in manifest["quantized_modules"] if key.startswith("model.visual.")
    }
    calibrated = {
        source_weight(path.stem): path.stem
        for path in (args.samples / "train").glob("model.vision_tower.*.pt")
    }
    if expected != calibrated.keys():
        raise ValueError(f"Vision calibration mismatch: {expected ^ calibrated.keys()}")
    expected.add(PATCH_WEIGHT)
    patches = patch_samples(args.source, args.images, args.seed)
    vision_piece = {index[key] for key in expected}
    if len(vision_piece) != 1:
        raise ValueError(
            "Pass the original PTQR training pieces to reuse the text payload"
        )
    filename = vision_piece.pop()
    visual_names = {k for k, v in index.items() if v == filename}
    if any(not k.startswith("model.visual.") for k in visual_names):
        raise ValueError("Vision piece also contains text parameters")
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        stage = Path(tmp)
        copy_assets(args.target_pieces, stage)
        for other in set(index.values()) - {filename}:
            os.link(args.target_pieces / other, stage / other)
        payload, report = {}, {}
        with (
            DirectStorage(device) as io,
            (
                HubDirectSource(
                    args.source,
                    io,
                    args.workspace,
                    SOURCE_REPO,
                    SOURCE_REVISION,
                    window_mib=128,
                )
                if args.hub_source
                else DirectCheckpoint(args.source, io)
            ) as source,
        ):
            for name in sorted(
                (n for n in visual_names if not n.endswith(("_scale", "_zero"))),
                key=source.file_order if args.hub_source else None,
            ):
                original = source.read(name)
                if name not in expected:
                    payload[name] = original
                    continue
                if original.dtype != torch.bfloat16:
                    raise ValueError(
                        f"Expected an original BF16 vision matrix at {name}"
                    )
                if name == PATCH_WEIGHT:
                    padding = -original.shape[1] % 128
                    original = torch.nn.functional.pad(original, (0, padding))
                    train_rows, val_rows = [
                        torch.nn.functional.pad(x.to(device), (0, padding))
                        for x in patches
                    ]
                else:
                    key = calibrated[name]
                    (train_rows,) = load_inputs(args.samples, "train", key, device)
                    (val_rows,) = load_inputs(args.samples, "eval", key, device)
                fields, errors = fit_projection(
                    original, train_rows, val_rows, args, rng, bits=8
                )
                payload[name], payload[name + "_scale"] = fields
                report[name] = dict(
                    train_rows=len(train_rows),
                    validation_rows=len(val_rows),
                    nearest_and_trained_relative_error=errors,
                )
            specs = {k: (v.shape, v.dtype) for k, v in payload.items()}
            bank = GPUCheckpoint(device, [0], args.workspace)
            with bank.writer(stage / filename, specs, io) as writer:
                for key, value in payload.items():
                    writer.append(key, value)
            bank.export()
        index = {k: v for k, v in index.items() if v != filename}
        index.update(dict.fromkeys(payload, filename))
        manifest["vision_bits"] = 8
        manifest["quantized_modules"] = sorted(
            set(manifest["quantized_modules"]) | {PATCH_WEIGHT}
        )
        (stage / INDEX_NAME).write_text(json.dumps({"weight_map": index}))
        (stage / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        assemble(stage, args.output)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "target_pieces", "samples", "workspace", "output", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument(
        "--hub-source",
        action="store_true",
        help="Stage immutable source ranges on the native AIS workspace",
    )
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--row-tile", type=int, default=768)
    parser.add_argument("--seed", type=int, default=721)
    train(parser.parse_args())
