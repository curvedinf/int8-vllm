# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fit Inkling target projections on observed teacher inputs and export G128 W4A8."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from scripts.ptqr_train_target import PTQRLinear, _weight_integer, anneal_tau


@dataclass
class FitResult:
    codes: torch.Tensor
    scales: torch.Tensor
    initial_error: float
    trained_error: float


@torch.no_grad()
def initialize_group_scales(weights, activations, full_range, bounds=(-8, 7)):
    """Fit deployed FP16 scales to the training activation second moments.

    A diagonal approximation to input covariance regularizes rarely visited
    experts. Each candidate minimizes its own group's weighted weight error.
    """
    grouped = weights.view(*full_range.shape, 128)
    moment = activations.square().mean(0).view(1, -1, 128)
    # Shrink noisy channel estimates towards the per-group mean.
    moment = moment + moment.mean(-1, keepdim=True)
    best_loss = torch.full_like(full_range, torch.inf)
    best_scale = full_range
    for fraction in (1.0, 0.96, 0.92, 0.88, 0.84, 0.80, 0.74):
        step = (full_range * fraction).clamp_min(2**-24).half().float()
        codes = (grouped / step.unsqueeze(-1)).round().clamp_(*bounds)
        # Solve for the scale with fixed codes and reassign to the new grid.
        denom = (codes.square() * moment).sum(-1)
        fitted = (grouped * codes * moment).sum(-1) / denom.clamp_min(1e-30)
        step = torch.where(denom > 0, fitted, step).clamp_min(2**-24).half().float()
        codes = (grouped / step.unsqueeze(-1)).round().clamp_(*bounds)
        loss = ((grouped - codes * step.unsqueeze(-1)).square() * moment).sum(-1)
        best_scale = torch.where(loss < best_loss, step, best_scale)
        best_loss = torch.minimum(loss, best_loss)
    return best_scale


def fit_rows(weights, training, validation, *, steps, batch_size, generator, bits=4):
    """Learn weights and group scales against unquantized teacher projections.

    A row tile is independent of the other output rows. Keep masters and the
    optimizer's state in FP32. Students see the MI100 G128 INT8 activation grid.
    """
    if not len(training) or not len(validation) or steps < 1:
        raise ValueError(
            "PTQR requires training and validation rows and positive steps"
        )
    x = training.to(device=weights.device, dtype=torch.float32)
    valid = validation.to(device=weights.device, dtype=torch.float32)
    target = weights.float()
    student = PTQRLinear(target, group=128, act_group=128, weight_bits=bits)
    with torch.no_grad():
        y = F.linear(x, target)
        y_valid = F.linear(valid, target)
        normalizer = y_valid.square().mean().clamp_min(1e-10)

        # Scale the independent updates by the step of each G128 weight grid.
        full_range = student.scale.detach().clone()
        start_scale = initialize_group_scales(
            target, x, full_range, student.weight_bounds
        )
        train_normalizer = y.square().mean().clamp_min(1e-10)

    # Optimize in units of the original grid step. Adam's moments then see
    # comparable update sizes in a quiet group and a high-magnitude group.
    delta = torch.nn.Parameter(torch.zeros_like(target))
    ratio = torch.nn.Parameter(torch.ones_like(start_scale))
    optimizer = torch.optim.Adam(
        [
            {"params": [delta], "lr": 0.003},
            {"params": [ratio], "lr": 0.0004},
        ]
    )

    def project(rows, tau):
        from scripts.ptqr_train_target import _PTQRLinFn

        weight = target + (
            delta.view(*start_scale.shape, 128) * start_scale[..., None]
        ).reshape_as(target)
        scale = (ratio * start_scale).clamp_min(2**-24)
        return _PTQRLinFn.apply(
            rows.half(),
            weight.half(),
            scale,
            128,
            tau,
            True,
            128,
            student.weight_bounds,
        ).float()

    with torch.no_grad():
        # Report the ordinary full-range nearest-rounding baseline.
        from scripts.ptqr_train_target import _PTQRLinFn

        initial = _PTQRLinFn.apply(
            valid.half(),
            target.half(),
            full_range,
            128,
            0.0,
            True,
            128,
            student.weight_bounds,
        ).float()
        initial_error = (initial - y_valid).square().mean() / normalizer
    for step in range(steps):
        # Dither is applied to rounded codes. A width above one explores
        # adjacent codes; the final steps reproduce the deployed grid.
        student.tau = anneal_tau(step, steps, 1.45, 0.8)
        ids = torch.randint(
            len(x), (min(batch_size, len(x)),), generator=generator, device=x.device
        )
        prediction = project(x[ids], student.tau)
        loss = (prediction - y[ids]).square().mean() / train_normalizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        student.tau = 0.0
        student.weight.copy_(
            target
            + (delta.view(*start_scale.shape, 128) * start_scale[..., None]).reshape_as(
                target
            )
        )
        student.weight.copy_(student.weight.half())
        student.scale.copy_((ratio * start_scale).clamp_min(2**-24))
        error = (project(valid, 0.0) - y_valid).square().mean() / normalizer
        if not torch.isfinite(error):
            raise ValueError("Nonfinite held-out projection error")
        codes = _weight_integer(
            student.weight, student.scale, 128, student.weight_bounds
        )
        if bits == 4:
            codes = (codes + 8).to(torch.uint8)
            codes = codes[..., ::2] | (codes[..., 1::2] << 4)
        else:
            codes = codes.to(torch.int8)
        return FitResult(
            codes, student.scale.half(), initial_error.item(), error.item()
        )


def source_weight(name):
    """Map calibrated HF linear modules back to original checkpoint names."""
    if name == "lm_head":
        return "model.llm.unembed.weight"
    if name.startswith("model.vision_tower."):
        name = name.removeprefix("model.vision_tower.")
        if name.startswith("merger.linear_"):
            return (
                "model.visual."
                + {
                    "merger.linear_1": "proj_in.weight",
                    "merger.linear_2": "proj_out.weight",
                }[name]
            )
        return (
            "model.visual.backbone."
            + name.replace(".mlp.fc1", ".mlp.c_fc").replace(".mlp.fc2", ".mlp.c_proj")
            + ".weight"
        )
    prefix, projection = name.split(".self_attn.")
    raw_projection = {
        "q_proj.linear": "wq_du",
        "k_proj.linear": "wk_dv",
        "v_proj.linear": "wv_dv",
        "o_proj.linear": "wo_ud",
        "o_proj.gate_proj": "gate_attn",
    }[projection]
    return (
        prefix.replace("model.language_model.", "model.llm.")
        + f".attn.{raw_projection}.weight"
    )


def fit_projection(weights, training, validation, options, generator, *, bits=4):
    packed = []
    scales = []
    errors = []
    for tile in weights.split(options.row_tile):
        fit = fit_rows(
            tile,
            training,
            validation,
            steps=options.steps,
            batch_size=options.batch,
            generator=generator,
            bits=bits,
        )
        packed.append(fit.codes)
        scales.append(fit.scales)
        errors.append([fit.initial_error, fit.trained_error])
    scale = torch.cat(scales)
    fields = (torch.cat(packed), scale)
    if bits == 4:
        fields += (torch.full_like(scale, 8),)
    return fields, errors


def load_inputs(directory, split, name, device):
    path = directory / split / (name + ".pt")
    return [
        t.to(device) for t in torch.load(path, weights_only=True, map_location="cpu")
    ]


@torch.no_grad()
def route_margins(gate, inputs, routes):
    scores = gate.score_fn(F.linear(inputs.float(), gate.weight))
    scores += gate.e_score_correction_bias
    return scores - scores.gather(1, routes.long()).amin(1, keepdim=True)


def expert_inputs(rows, routes, margins, expert, minimum):
    routed = (routes == expert).any(dim=-1)
    selected = routed.nonzero().flatten()
    count = selected.numel()
    needed = min(minimum, len(rows)) - count
    if needed > 0:
        # For rare experts, supplement genuine routes with the teacher rows
        # closest to its learned top-k selection boundary.
        nearest = (
            margins[:, expert].masked_fill(routed, -torch.inf).topk(needed).indices
        )
        selected = torch.cat((selected, nearest))
    return rows[selected], count


def fit_experts(name, moe, options, generator):
    module = moe.experts
    device = module.gate_up_proj.device
    train, train_routes = load_inputs(options.samples, "train", name, device)
    valid, valid_routes = load_inputs(options.samples, "eval", name, device)
    train_margins = route_margins(moe.gate, train, train_routes)
    valid_margins = route_margins(moe.gate, valid, valid_routes)
    projections = ("w13_weight", "w2_weight")
    fields = ("", "_scale", "_zero")
    prefix = name.replace("model.language_model.", "model.llm.") + "."
    blocks = {}
    errors = []
    for expert in range(module.num_experts):
        x, train_routed = expert_inputs(
            train, train_routes, train_margins, expert, options.batch
        )
        # Evaluate genuinely routed held-out states whenever available. If
        # none were observed, report a boundary example explicitly.
        v, valid_routed = expert_inputs(valid, valid_routes, valid_margins, expert, 1)
        gate_up = module.gate_up_proj[expert].detach()
        weights = (gate_up, module.down_proj[expert].detach())
        down_x = module._apply_gate(F.linear(x.to(gate_up.dtype), gate_up))
        down_v = module._apply_gate(F.linear(v.to(gate_up.dtype), gate_up))
        for label, w, train_rows, valid_rows in zip(
            projections, weights, (x, down_x), (v, down_v)
        ):
            grid, scores = fit_projection(w, train_rows, valid_rows, options, generator)
            for field, tensor in zip(fields, grid):
                if label == "w13_weight":
                    # The release stores gate and up in alternating rows.
                    tensor = tensor.unflatten(0, (2, -1)).transpose(0, 1).flatten(0, 1)
                key = prefix + label + field
                if key not in blocks:
                    blocks[key] = tensor.new_empty((module.num_experts, *tensor.shape))
                blocks[key][expert].copy_(tensor)
            errors.append(
                {
                    "expert": expert,
                    "projection": label,
                    "train_rows": len(x),
                    "routed_train_rows": train_routed,
                    "validation_rows": len(v),
                    "routed_validation_rows": valid_routed,
                    "tile_errors": scores,
                }
            )
    return blocks, errors


def export_group(tag, raw, calibrated, options, device):
    import json

    from scripts.calibrate_inkling2 import quantized_projection
    from scripts.hy4_gpu_checkpoint import GPUCheckpoint

    generator = torch.Generator(device=device).manual_seed(options.seed)
    payload = dict(raw)
    report = {}
    for name, module in calibrated:
        if name.endswith(".mlp.experts"):
            values, report[name] = fit_experts(name, module, options, generator)
            payload.update(values)
        elif quantized_projection(name, module):
            raw_name = source_weight(name)
            w = raw[raw_name]
            (train,) = load_inputs(options.samples, "train", name, device)
            (valid,) = load_inputs(options.samples, "eval", name, device)
            grid, errors = fit_projection(w.detach(), train, valid, options, generator)
            for suffix, tensor in zip(("", "_scale", "_zero"), grid):
                payload[raw_name + suffix] = tensor
            report[name] = {
                "train_rows": len(train),
                "validation_rows": len(valid),
                "tile_errors": errors,
            }
    shard = options.output / f"{tag}.safetensors"
    specs = {name: (value.shape, value.dtype) for name, value in payload.items()}
    bank = GPUCheckpoint(device, [device.index], options.workspace)
    with bank.writer(shard, specs, None) as writer:
        for name, value in payload.items():
            writer.append(name, value)
    bank.export()
    (options.work / (tag + ".json")).write_text(json.dumps(report))
    print(f"Exported {tag} with {len(payload)} tensors", flush=True)
    return {name: shard.name for name in payload}


def main():
    import argparse
    import os
    import shutil
    from pathlib import Path
    from tempfile import mkdtemp

    from scripts.assemble_inkling2 import assemble
    from scripts.calibrate_inkling2 import (
        add_sampling_options,
        collect_splits,
    )
    from scripts.inkling2_teacher import load_teacher

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "samples", "output", "work", "workspace"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--row-tile", type=int, default=768)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument(
        "--corpus", type=Path, help="Collect fresh inputs from this JSONL corpus"
    )
    parser.add_argument(
        "--images", type=Path, help="Directory containing calibration images"
    )
    add_sampling_options(parser)
    options = parser.parse_args()
    if bool(options.corpus) != bool(options.images):
        parser.error("Pass --corpus and --images together")
    for path in (options.output, options.work):
        path.mkdir(parents=True, exist_ok=True)
    if any(options.output.iterdir()):
        parser.error("Use an empty output directory for checkpoint assembly")
    os.environ.setdefault("HIPFILE_ALLOW_COMPAT_MODE", "false")
    os.environ.setdefault("HIPFILE_FORCE_COMPAT_MODE", "false")
    model, raw = load_teacher(options.source, keep_dense_source=True)
    if options.corpus:
        collect_splits(model, options)
    destination = options.output
    options.output = Path(mkdtemp(prefix=".inkling2-pieces-", dir=destination.parent))
    print(f"Writing GPU-exported pieces to {options.output}", flush=True)
    train_loaded(model, raw, options)
    assemble(options.output, destination)
    shutil.rmtree(options.output)


def train_loaded(model, raw, options):
    import hashlib
    import json
    import re
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    from scripts.assemble_inkling2 import copy_assets
    from scripts.calibrate_inkling2 import quantized_projection
    from scripts.inkling2_teacher import SOURCE_REVISION

    groups = defaultdict(dict)
    modules = defaultdict(list)
    ranks = {}

    def group_name(name):
        layer = re.search(r"^model\.(llm|language_model)\.layers\.(\d+)\.", name)
        if layer:
            return "text-layer-" + layer[2].zfill(2)
        return (
            "vision"
            if name.startswith(("model.visual.", "model.vision_tower."))
            else "text"
        )

    for name, weight in raw.items():
        tag = group_name(name)
        groups[tag][name] = weight
        ranks[tag] = weight.device
    for name, module in model.named_modules():
        if name.endswith(".mlp.experts") or quantized_projection(name, module):
            tag = group_name(name)
            if name.endswith(".mlp.experts"):
                module = model.get_submodule(name.removesuffix(".experts"))
            modules[tag].append((name, module))

    def fit_rank(rank):
        result = {}
        with torch.cuda.device(rank):
            # PTQR dithering uses the current device's default generator.
            torch.cuda.manual_seed(options.seed + rank)
            for tag in sorted(groups):
                if ranks[tag].index == rank:
                    result.update(
                        export_group(
                            tag, groups[tag], modules[tag], options, ranks[tag]
                        )
                    )
        return result

    weight_map = {}
    workers = torch.cuda.device_count()
    with ThreadPoolExecutor(workers) as pool:
        for manifest in pool.map(fit_rank, range(workers)):
            weight_map.update(manifest)
    raw_names = set(raw)
    if not raw_names.issubset(weight_map):
        raise ValueError(
            f"Missing original non-expert tensors: {raw_names - weight_map.keys()}"
        )
    config = json.loads((options.source / "config.json").read_text())
    config.pop("mtp_config", None)
    config["dtype"] = config["text_config"]["torch_dtype"] = "float16"
    config["quantization_config"] = {
        "quant_method": "inkling2_ptqr",
        "bits": 4,
        "group_size": 128,
        "quantized_modules": sorted(
            key.removesuffix("_scale")
            for key in weight_map
            if key.endswith("weight_scale")
        ),
    }
    (options.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    copy_assets(options.source, options.output)
    (options.output / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}, indent=2) + "\n"
    )
    (options.work / "training.json").write_text(
        json.dumps(
            {
                "source_revision": SOURCE_REVISION,
                "seed": options.seed,
                "steps": options.steps,
                "batch": options.batch,
                "row_tile": options.row_tile,
                "dither_seeds": [options.seed + rank for rank in range(workers)],
                "dither_width": [1.45, 0.8],
                "train_texts": options.train_texts,
                "eval_texts": options.eval_texts,
                "train_images": options.train_images,
                "eval_images": options.eval_images,
                "corpus_sha256": (
                    hashlib.sha256(options.corpus.read_bytes()).hexdigest()
                    if options.corpus
                    else None
                ),
                "dtype": "W4A8-G128",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
