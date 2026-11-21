#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Train and export HY4's W8A8 vision encoder against the original GPU teacher."""

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from scripts.hy4_hub_source import HubDirectSource
from scripts.ptqr_train_target import PTQRLinear, _weight_integer, anneal_tau
from vllm.fs_gpu.checkpoint import DirectCheckpoint
from vllm.fs_gpu.direct import DirectStorage
from vllm.fs_gpu.safetensor_writer import DirectTensorWriter


class GroupedPTQR(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.padding = -original.in_features % 128
        w = (
            F.pad(original.weight.detach(), (0, self.padding))
            if self.padding
            else original.weight
        )
        self.linear = PTQRLinear(w, group=128, act_group=128)
        # Tiny and all-zero learned groups, including padded groups, need a
        # positive scale after it is rounded to the checkpoint's FP16 grid.
        with torch.no_grad():
            self.linear.scale.clamp_min_(2**-24)
        self.register_buffer(
            "bias", None if original.bias is None else original.bias.detach()
        )

    def forward(self, x):
        if self.padding:
            x = F.pad(x, (0, self.padding))
        result = self.linear(x)
        return result if self.bias is None else result + self.bias


def hf_name(name):
    return (
        name.removeprefix("vit.")
        .replace(".mlp.dense_h_to_4h.", ".mlp.fc1.")
        .replace(".mlp.dense_4h_to_h.", ".mlp.fc2.")
        .replace(".input_layernorm.", ".layer_norm1.")
        .replace(".post_attention_layernorm.", ".layer_norm2.")
        .replace("perceive.norm.", "patch_merger.before_rms.")
        .replace("perceive.proj.0.", "patch_merger.proj_conv.")
        .replace("perceive.proj.2.", "patch_merger.proj_out.")
        .replace("perceive.", "patch_merger.")
    )


def serving_module(name):
    name = name.removesuffix(".weight").replace("vit.", "visual.", 1)
    for suffix in ("q_proj", "k_proj", "v_proj"):
        if name.endswith(".self_attn." + suffix):
            return name.removesuffix(suffix) + "qkv"
    return name


def load_models(source, config, *, trainable=True):
    from transformers.models.hy_v4_vl.modeling_hy_v4_vl import HYV4VLVisionModel

    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        teacher = HYV4VLVisionModel(config)
    names = {hf_name(raw): raw for raw in source.index if raw.startswith("vit.")}
    params = dict(teacher.named_parameters())
    if names.keys() != params.keys():
        raise ValueError(
            f"Mismatched vision keys: {sorted(names.keys() ^ params.keys())}"
        )
    missing_buffers = [k for k, b in teacher.named_buffers() if b.device.type == "meta"]
    if missing_buffers:
        raise ValueError(f"Uninitialized vision buffers: {missing_buffers}")
    order = (
        sorted(params, key=lambda k: source.file_order(names[k]))
        if hasattr(source, "file_order")
        else params
    )
    for i, k in enumerate(order):
        p = params[k]
        mod, _, leaf = k.rpartition(".")
        value = source.read(names[k])
        if value.shape != p.shape:
            raise ValueError(
                f"Vision shape mismatch at {k}: {value.shape} vs {p.shape}"
            )
        teacher.get_submodule(mod)._parameters[leaf] = nn.Parameter(
            value.to(torch.float16), requires_grad=False
        )
        if (i + 1) % 80 == 0:
            print(f"Loaded {i + 1}/{len(params)} vision parameters", flush=True)
    if not trainable:
        return teacher.eval()
    student = copy.deepcopy(teacher).eval()
    replaced = {}
    for name, module in list(student.named_modules()):
        if isinstance(module, nn.Linear) and name.startswith("layers."):
            replacement = GroupedPTQR(module)
            parent, _, leaf = name.rpartition(".")
            setattr(student.get_submodule(parent), leaf, replacement)
            replaced[name] = replacement.linear
    return teacher.eval(), student, names, replaced


@torch.no_grad()
def get_examples(teacher, processor, images, count, pixels, device, student):
    examples = []
    for path in images[:count]:
        with Image.open(path) as image:
            inputs = processor(
                images=image.convert("RGB"),
                return_tensors="pt",
                size={"shortest_edge": 64 * 64, "longest_edge": pixels},
            )
        args = {
            "pixel_values": inputs.pixel_values.to(device=device, dtype=torch.float16),
            "grid_thw": inputs.image_grid_thw.to(device),
        }
        expect = teacher(**args).pooler_output.detach()
        if not bool(torch.isfinite(expect).all()):
            raise ValueError(f"Nonfinite teacher features for {path.name}")
        if not examples:
            handles = []

            def checked(name):
                def hook(mod, inputs, output):
                    if isinstance(output, torch.Tensor) and not bool(
                        output.isfinite().all()
                    ):
                        raise ValueError(
                            f"Nonfinite vision student activation at {name}"
                        )

                return hook

            try:
                for name, m in student.named_modules():
                    if isinstance(m, (GroupedPTQR, nn.LayerNorm, nn.Linear)):
                        handles.append(m.register_forward_hook(checked(name)))
                test = student(**args).pooler_output
                if not bool(test.isfinite().all()):
                    raise ValueError(
                        "Nonfinite exported vision features on first image"
                    )
            finally:
                for handle in handles:
                    handle.remove()
        examples.append((args, expect))
    return examples


def feature_loss(y, target):
    err = (y.float() - target.float()).square().mean()
    return err / target.float().square().mean().clamp_min(1e-8)


@torch.no_grad()
def evaluate(student, examples):
    errors, cos = [], []
    for args, target in examples:
        y = student(**args).pooler_output
        errors.append(feature_loss(y, target))
        cos.append(F.cosine_similarity(y.float(), target.float(), dim=-1).mean())
    return {
        "normalized_mse": float(torch.stack(errors).mean()),
        "cosine": float(torch.stack(cos).mean()),
    }


def train(student, replaced, train_set, val_set, args):
    base = evaluate(student, val_set)
    if not all(math.isfinite(v) for v in base.values()):
        raise ValueError("Nonfinite vision calibration baseline")
    history = [{"step": 0, **base}]
    print(f"Held-out before PTQR {base}", flush=True)
    scale_optimizer = torch.optim.AdamW(
        [q.scale for q in replaced.values()], lr=args.scale_lr, weight_decay=0
    )
    best = None
    for step in range(1, args.steps + 1):
        batch, target = train_set[(step - 1) % len(train_set)]
        tau = (
            args.dither
            if args.constant_dither or not args.dither
            else anneal_tau(step - 1, args.steps, args.dither, args.dither)
        )
        for q in replaced.values():
            q.tau = tau
        loss = feature_loss(student(**batch).pooler_output, target)
        if not bool(torch.isfinite(loss)):
            raise ValueError(f"Nonfinite vision feature loss at step {step}")
        loss.backward()
        with torch.no_grad():
            for q in replaced.values():
                g = q.weight.grad
                if g is None or q.scale.grad is None:
                    raise RuntimeError("PTQR gradient missing from a vision projection")
                q.weight.add_(
                    g * (1.0 / g.float().norm().clamp_min(1.0)), alpha=-args.lr
                )
                q.weight.grad = None
        scale_optimizer.step()
        scale_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for q in replaced.values():
                q.scale.clamp_min_(2**-24)
                q.tau = 0.0
        if step == args.steps or step % args.eval_every == 0:
            row = {"step": step, **evaluate(student, val_set)}
            history.append(row)
            print(f"Held-out vision {row}", flush=True)
            if best is None or row["normalized_mse"] < best[0]["normalized_mse"]:
                best = (
                    row,
                    {
                        name: (q.weight.detach().clone(), q.scale.detach().clone())
                        for name, q in replaced.items()
                    },
                )
    if best is None:
        raise ValueError("PTQR produced no trained vision state")
    with torch.no_grad():
        for name, (weight, scale) in best[1].items():
            replaced[name].weight.copy_(weight)
            replaced[name].scale.copy_(scale)
    return {"baseline": base, "selected": best[0], "history": history}


@torch.no_grad()
def export(directory, io, teacher, names, replaced, writer_type=DirectTensorWriter):
    entries, values, manifest = {}, {}, {}
    for local, p in teacher.named_parameters():
        name = names[local]
        method = (
            replaced.get(local.removesuffix(".weight"))
            if local.endswith(".weight")
            else None
        )
        if method is None:
            entries[name] = (p.shape, p.dtype)
            values[name] = p
        else:
            entries[name] = (method.weight.shape, torch.int8)
            entries[name.removesuffix(".weight") + ".weight_scale"] = (
                method.scale.shape,
                torch.float16,
            )
            values[name] = method
            manifest[serving_module(name)] = 8
    filename = "vision-ptqr.safetensors"
    with writer_type(directory / filename, entries, io) as writer:
        for name, value in values.items():
            if isinstance(value, PTQRLinear):
                codes = _weight_integer(value.weight, value.scale, 128).to(torch.int8)
                writer.append(name, codes)
                writer.append(
                    name.removesuffix(".weight") + ".weight_scale", value.scale.half()
                )
            else:
                writer.append(name, value)
    (directory / "vision.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(entries, filename)}, indent=2)
    )
    (directory / "vision.quant.json").write_text(
        json.dumps({"quantized_modules": manifest}, indent=2)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("--hub-repo")
    parser.add_argument("--revision")
    parser.add_argument("--download-workspace", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--train-images", type=int, default=60)
    parser.add_argument("--validation-images", type=int, default=12)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--scale-lr", type=float, default=1e-7)
    parser.add_argument("--dither", type=float, default=0.001)
    parser.add_argument(
        "--constant-dither",
        action="store_true",
        help="Keep the training dither through the final step (published ViT).",
    )
    parser.add_argument("--seed", type=int, default=4017)
    args = parser.parse_args()
    if min(args.steps, args.train_images, args.validation_images, args.eval_every) < 1:
        parser.error("Training steps and image counts must be positive")
    if args.hub_repo and not (args.download_workspace and args.revision):
        parser.error(
            "Hub acquisition needs a pinned revision and a native NVMe workspace"
        )
    os.environ.setdefault("HIPFILE_ALLOW_COMPAT_MODE", "false")
    os.environ.setdefault("HIPFILE_FORCE_COMPAT_MODE", "false")
    from transformers.models.hy_v4_flash.configuration_hy_v4_flash import (
        HYV4FlashConfig,
    )
    from transformers.models.hy_v4_vl.image_processing_hy_v4_vl import (
        HYV4VLImageProcessor,
    )

    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    paths = sorted(
        p
        for p in args.images.rglob("*")
        if p.suffix.lower() in (".jpg", ".png", ".webp", ".jpeg")
    )
    random.Random(args.seed).shuffle(paths)
    count = args.train_images + args.validation_images
    if len(paths) < count:
        raise ValueError(f"Need {count} distinct images; got {len(paths)}")
    config = HYV4FlashConfig.from_pretrained(args.source).vision_config
    pc = json.loads((args.source / "processor_config.json").read_text())[
        "image_processor"
    ]
    processor = HYV4VLImageProcessor(**pc)
    args.output.mkdir(parents=True, exist_ok=True)
    with DirectStorage(device) as io:
        src = (
            HubDirectSource(
                args.source, io, args.download_workspace, args.hub_repo, args.revision
            )
            if args.hub_repo
            else DirectCheckpoint(args.source, io)
        )
        with src:
            teacher, student, names, replaced = load_models(src, config)
        examples = get_examples(
            teacher, processor, paths, count, args.max_pixels, device, student
        )
        results = train(
            student,
            replaced,
            examples[: args.train_images],
            examples[args.train_images :],
            args,
        )
        results.update(
            seed=args.seed,
            input_images=[p.name for p in paths[:count]],
            train_images=args.train_images,
            validation_images=args.validation_images,
            max_pixels=args.max_pixels,
            steps=args.steps,
            lr=args.lr,
            scale_lr=args.scale_lr,
            dither=args.dither,
            zero_dither_final_fraction=0.0 if args.constant_dither else 0.1,
            source_revision=args.revision,
        )
        if args.download_workspace:
            from scripts.hy4_gpu_checkpoint import GPUCheckpoint

            bank = GPUCheckpoint(device, [device.index], args.download_workspace)
            export(args.output, io, teacher, names, replaced, bank.writer)
            bank.export()
        else:
            export(args.output, io, teacher, names, replaced)
        (args.output / "vision-training.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
