#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert the HY4 text backbone to calibrated NeUQI G128 directly on a GPU."""

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from vllm.fs_gpu.checkpoint import DirectCheckpoint
from vllm.fs_gpu.direct import DirectStorage
from vllm.fs_gpu.safetensor_writer import DirectTensorWriter
from vllm.models.hy_v4_flash.calibration import (
    calibrate_layer,
    load_teacher_layer,
    target_matrix,
)
from vllm.models.hy_v4_flash.neuqi import quantize_rows

ENGRAM = "engram_layer.multi_head_embedding.embedding.weight"
EXPERT = "mlp.experts."
G = 128


@dataclass(frozen=True)
class Unit:
    source: str
    name: str
    shape: tuple
    dtype: torch.dtype
    quantize: bool
    expert: int | None = None

    def entries(self):
        if not self.quantize:
            return {self.name: (self.shape, self.dtype)}
        n, k = self.shape
        stem = self.name.removesuffix(".weight")
        return {
            stem + ".weight": ((n, k // 2), torch.uint8),
            stem + ".weight_scale": ((n, k // G), torch.float16),
            stem + ".weight_zero": ((n, k // G), torch.float16),
        }


def layer_units(source, idx, names):
    prefix = f"model.layers.{idx}."
    for name in sorted(names):
        leaf = name[len(prefix) :]
        if leaf == ENGRAM:
            continue
        spec = source.spec(name)
        if leaf in (EXPERT + "down_proj", EXPERT + "gate_up_proj"):
            experts, n, k = spec.shape
            projection = leaf.removeprefix(EXPERT)
            for expert in range(experts):
                target = f"{prefix}{EXPERT}{expert}.{projection}.weight"
                yield Unit(name, target, (n, k), spec.dtype, True, expert)
        else:
            yield Unit(name, name, spec.shape, spec.dtype, target_matrix(leaf))


def write_shards(
    output,
    label,
    source,
    units,
    moments,
    manifest,
    max_bytes,
    rows_per_tile,
    writer_type=DirectTensorWriter,
):
    def save(group, part):
        name = f"{label}-{part:04d}.safetensors"
        specs = {k: v for unit in group for k, v in unit.entries().items()}
        with writer_type(output / name, specs, source.io) as writer:
            for unit in group:
                origin = (
                    0 if unit.expert is None else unit.expert * math.prod(unit.shape)
                )
                if not unit.quantize:
                    # Safetensor tensor boundaries need not coincide with GPU
                    # staging-window or page boundaries, even for embeddings.
                    chunk = max(1, (32 << 20) // unit.dtype.itemsize)
                    count = math.prod(unit.shape)
                    for pos in range(0, count, chunk):
                        value = source.read(
                            unit.source,
                            start=origin + pos,
                            shape=(min(count - pos, chunk),),
                        )
                        writer.append(unit.name, value)
                    continue
                n, k = unit.shape
                diag = moments[unit.source]
                if unit.expert is not None:
                    diag = diag[unit.expert]
                codes = torch.empty(
                    (n, k // 2), device=source.device, dtype=torch.uint8
                )
                scales = torch.empty(
                    (n, k // G), device=source.device, dtype=torch.float16
                )
                zeros = torch.empty_like(scales)
                for pos in range(0, n, rows_per_tile):
                    take = min(n - pos, rows_per_tile)
                    weights = source.read(
                        unit.source, start=origin + pos * k, shape=(take, k)
                    )
                    group_fit = quantize_rows(weights, diag)
                    span = slice(pos, pos + take)
                    codes[span], scales[span], zeros[span] = (
                        group_fit.codes,
                        group_fit.scales,
                        group_fit.zeros,
                    )
                for key, data in zip(unit.entries(), (codes, scales, zeros)):
                    writer.append(key, data)
        manifest.update(dict.fromkeys(specs, name))

    part, group, size = 0, [], 0
    for unit in units:
        length = sum(math.prod(s) * d.itemsize for s, d in unit.entries().values())
        if group and size + length > max_bytes:
            save(group, part)
            part, group, size = part + 1, [], 0
        group.append(unit)
        size += length
    if group:
        save(group, part)


def module_name(unit):
    if unit.expert is not None:
        return unit.source.rsplit(".", 1)[0]
    name = unit.name.removesuffix(".weight").replace(
        ".compressed_indexer.", ".indexer."
    )
    for component in (".mlp.", ".shared_experts."):
        for projection in ("up_proj", "gate_proj"):
            if name.endswith(component + projection):
                return name.removesuffix(projection) + "gate_up_proj"
    return name


def calibration_tokens(tokenizer, corpus, samples, length, device):
    sequences = []
    with corpus.open() as f:
        for line in f:
            row = json.loads(line)
            ids = tokenizer.encode(row["text"], add_special_tokens=False)
            if len(ids) < 2:
                continue
            ids = ids[:length]
            sequences.append(
                [torch.tensor([ids], device=device, dtype=torch.long), None]
            )
            if len(sequences) == samples:
                return sequences
    raise ValueError(f"Need {samples} usable records in the calibration corpus")


@torch.inference_mode()
def convert(args):
    from transformers import AutoTokenizer
    from transformers.models.hy_v4_flash.configuration_hy_v4_flash import (
        HYV4FlashConfig,
    )
    from transformers.models.hy_v4_flash.modeling_hy_v4_flash import (
        HYV4FlashRotaryEmbedding,
    )

    cfg = HYV4FlashConfig.from_pretrained(args.source).text_config
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    rotary = HYV4FlashRotaryEmbedding(cfg, device=device).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.source)
    corpus = calibration_tokens(
        tokenizer, args.corpus, args.samples, args.length, device
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest, modules, audit = {}, {}, []
    from contextlib import ExitStack

    from scripts.hy4_gpu_checkpoint import GPUCheckpoint
    from scripts.hy4_hub_source import HubDirectSource

    with ExitStack() as stack:
        io = stack.enter_context(DirectStorage(device))
        bank = None
        if getattr(args, "hub_repo", None):
            from scripts.hy4_engram_source import corpus_identity

            source = stack.enter_context(
                HubDirectSource(
                    args.source,
                    io,
                    args.workspace,
                    args.hub_repo,
                    args.revision,
                    args.window_mib,
                )
            )
            source.calibration_dir = args.workspace
            source.corpus_identity = corpus_identity(
                args.corpus, args.samples, args.length
            )
        else:
            source = stack.enter_context(DirectCheckpoint(args.source, io))
        if getattr(args, "storage_gpus", None):
            bank = GPUCheckpoint(device, args.storage_gpus, args.workspace)
        writer = bank.writer if bank is not None else DirectTensorWriter
        examples = None
        if getattr(args, "draft_cache", None):
            from scripts.hy4_draft_examples import DFlyExamples

            examples = DFlyExamples(args.draft_cache, source, cfg)
        for idx in range(cfg.num_hidden_layers):
            if hasattr(source, "begin_layer"):
                source.begin_layer(idx)
            print(
                f"Loading language layer {idx + 1}/{cfg.num_hidden_layers}", flush=True
            )
            layer, names = load_teacher_layer(source, cfg, idx)
            print(
                f"Measuring language layer {idx + 1}/{cfg.num_hidden_layers}",
                flush=True,
            )
            finish = examples.layer_hook(layer, idx) if examples is not None else None
            diag = calibrate_layer(source, cfg, idx, layer, names, corpus, rotary)
            if finish is not None:
                finish()
                del finish
            del layer
            units = list(layer_units(source, idx, names.values()))
            pooled = {}
            for name, values in diag.items():
                if values.ndim == 2:
                    used = values.sum(1) > 0
                    pooled[name] = int((~used).sum())
                    if not bool(used.any()):
                        raise ValueError(f"Calibration never routed to {name}")
                    # Use the measured layer moment for an expert with zero
                    # observations. All observed experts retain their own moment.
                    values[~used] = values[used].sum(0)
                elif not bool(values.sum() > 0):
                    raise ValueError(f"Calibration did not exercise {name}")
            for unit in units:
                if unit.quantize:
                    modules[module_name(unit)] = 4
            write_shards(
                output,
                f"layer-{idx:02d}",
                source,
                units,
                diag,
                manifest,
                args.shard_mib << 20,
                args.row_tile,
                writer,
            )
            audit.append({"layer": idx, "unobserved_experts": pooled})
            print(
                f"Converted language layer {idx + 1}/{cfg.num_hidden_layers}",
                flush=True,
            )
        if hasattr(source, "begin_layer"):
            source.begin_layer(-1)
            source.embedding = None
        if examples is not None:
            examples.final(corpus)
        other = []
        for name in source.index:
            if name.startswith("model.layers.") or name.startswith(
                ("model.dfly_layers.", "vit.")
            ):
                continue
            spec = source.spec(name)
            other.append(Unit(name, name, spec.shape, spec.dtype, False))
        write_shards(
            output,
            "backbone",
            source,
            other,
            {},
            manifest,
            args.shard_mib << 20,
            args.row_tile,
            writer,
        )
        if bank is not None:
            print(
                "All quantized backbone tensors are on the GPUs. Starting export.",
                flush=True,
            )
            bank.export()
    # The assembler joins these with trained vision weights and the Q4 tables.
    for filename, payload in (
        ("backbone.index.json", {"weight_map": manifest}),
        (
            "backbone.quant.json",
            {
                "quant_method": "hy4_neuqi",
                "group_size": G,
                "quantized_modules": modules,
            },
        ),
        (
            "calibration.json",
            {
                "samples": len(corpus),
                "tokens": sum(x[0].numel() for x in corpus),
                "layers": audit,
            },
        ),
    ):
        temp = output / (filename + ".partial")
        temp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temp, output / filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--corpus", type=Path, required=True, help="JSONL with a text field per record"
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--row-tile", type=int, default=512)
    parser.add_argument("--shard-mib", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--draft-cache", type=Path, help="Strict-NVMe target states for DFly PTQR"
    )
    parser.add_argument("--hub-repo")
    parser.add_argument("--revision")
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Native P2P-capable NVMe workspace for bounded imports and exports",
    )
    parser.add_argument("--window-mib", type=int, default=64)
    parser.add_argument(
        "--storage-gpus",
        type=lambda value: [int(d) for d in value.split(",")],
        help="Keep all completed backbone shards on these peer GPUs until export",
    )
    args = parser.parse_args()
    if args.hub_repo and not (args.revision and args.workspace):
        parser.error("Pinned Hub acquisition requires --revision and --workspace")
    if args.storage_gpus and not args.workspace:
        parser.error("Deferred checkpoint export requires --workspace")
    if min(args.samples, args.length, args.row_tile, args.shard_mib) < 1:
        parser.error("Samples, length, tile and shard size must be positive")
    os.environ.setdefault("HIPFILE_ALLOW_COMPAT_MODE", "false")
    os.environ.setdefault("HIPFILE_FORCE_COMPAT_MODE", "false")
    convert(args)


if __name__ == "__main__":
    main()
