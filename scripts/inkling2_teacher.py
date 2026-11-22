# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the released Inkling-2-Mini target through AIS for PTQR calibration.

Use the Transformers Inkling2Loop implementation and its original-checkpoint
converters. The teacher uses the decoded NVFP4/MXFP8 weights in BF16.
"""

import os
import re
from contextlib import ExitStack
from pathlib import Path

import torch

from vllm.fs_gpu.checkpoint import DirectCheckpoint
from vllm.fs_gpu.direct import DirectStorage

SOURCE_REPO = "thinkingmachines/Inkling-2-Mini"
SOURCE_REVISION = "b85c3cfa615fc413cc9af3379f491b963f3b586a"


def place_teacher_inputs(model):
    """Run each physical layer on its weight device and return its residual to GPU 0.

    The recurrent loop keeps prelude and previous-pass states outside modules.
    Accelerate's nested offload hooks otherwise move some operands back to the
    root device inside a remote layer. All teacher parameters are GPU resident.
    """
    from accelerate.hooks import remove_hook_from_module
    from accelerate.utils import send_to_device

    remove_hook_from_module(model, recurse=True)
    root = model.get_input_embeddings().weight.device
    for layer in model.model.language_model.layers:
        device = next(layer.parameters()).device
        if device == root:
            continue

        def enter(module, args, kwargs, *, dest=device):
            return send_to_device(args, dest), send_to_device(kwargs, dest)

        def leave(module, args, output):
            return send_to_device(output, root)

        layer.register_forward_pre_hook(enter, with_kwargs=True)
        layer.register_forward_hook(leave)


def read_source(directory: Path):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    if config.model_type != "inkling2_loop":
        raise ValueError(f"Expected Inkling-2-Mini, got {config.model_type}")
    # The HF format converter dequantizes the original experts. Invoking
    # ModelOpt here would instead select the CUDA serving kernels.
    del config.quantization_config
    devices = torch.cuda.device_count()
    if devices < 2:
        raise ValueError("The decoded Inkling teacher requires multiple MI100s")
    layers = config.text_config.num_hidden_layers
    ranks = [i * devices // layers for i in range(layers)]
    device_map = {"model.vision_tower": 0, "lm_head": 0}
    for name in ("embed_tokens", "final_input", "norm", "message_input"):
        device_map["model.language_model." + name] = 0
    device_map.update(
        {f"model.language_model.layers.{i}": rank for i, rank in enumerate(ranks)}
    )
    with ExitStack() as stack:
        io = stack.enter_context(DirectStorage(torch.device("cuda", 0)))
        workspace = os.environ.get("INKLING_AIS_WORKSPACE")
        if workspace:
            from scripts.hy4_hub_source import HubDirectSource

            source = stack.enter_context(
                HubDirectSource(
                    directory,
                    io,
                    workspace,
                    SOURCE_REPO,
                    SOURCE_REVISION,
                    window_mib=128,
                )
            )
            order = sorted(source.index, key=source.file_order)
        else:
            source = stack.enter_context(DirectCheckpoint(directory, io))
            order = source.index
        weights = {}
        for name in order:
            if not name.startswith(("model.llm.", "model.visual.")):
                raise ValueError(f"Unexpected tensor in target checkpoint: {name}")
            match = re.match(r"model\.llm\.layers\.(\d+)\.", name)
            rank = ranks[int(match[1])] if match else 0
            weights[name] = source.read(name).to(torch.device("cuda", rank))
    return config, device_map, weights


def materialize_teacher(config, device_map, weights):
    from transformers import Inkling2LoopForConditionalGeneration

    teacher, info = Inkling2LoopForConditionalGeneration.from_pretrained(
        None,
        state_dict=weights,
        config=config,
        device_map=device_map,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        experts_implementation="eager",
        local_files_only=True,
        output_loading_info=True,
    )
    errors = {key: values for key, values in info.items() if values}
    if errors:
        raise ValueError(f"Incomplete Inkling teacher load: {errors}")
    place_teacher_inputs(teacher)
    teacher.eval().requires_grad_(False)
    return teacher


def load_teacher(directory: Path, *, keep_dense_source=False):
    config, device_map, weights = read_source(directory)
    teacher = materialize_teacher(config, device_map, weights)
    if keep_dense_source:
        return teacher, {
            name: value
            for name, value in weights.items()
            if ".mlp.experts." not in name
        }
    return teacher


def read_expert(source: DirectCheckpoint, prefix: str, expert: int) -> torch.Tensor:
    """Decode one original expert projection using the Transformers reference.

    Returned rows follow the release order (w13 alternates gate and up rows).
    Only the requested expert's tensors are read into GPU memory.
    """
    import math

    from transformers.core_model_loading import Inkling2LoopDequantizeExperts

    fields = [
        prefix + suffix
        for suffix in (
            "_weight",
            "_weight_scale",
            "_weight_amax",
            "_weight_outer_scale_denominator",
        )
    ]
    values = {}
    for name in fields:
        if name not in source.index:
            continue
        shape = source.spec(name).shape
        if name.endswith("_denominator"):
            values[name] = source.read(name)
        else:
            if not 0 <= expert < shape[0]:
                raise IndexError(f"Expert {expert} outside {name}")
            values[name] = source.read(
                name, start=expert * math.prod(shape[1:]), shape=(1, *shape[1:])
            )
    result = Inkling2LoopDequantizeExperts().convert(values, fields, ["weight"])
    return result["weight"][0]
