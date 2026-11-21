# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load packed HY4 tensor groups into GPU windows before TP sharding."""

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather
from vllm.fs_gpu.direct import DirectStorage
from vllm.fs_gpu.region import ALIGN, RegisteredFile
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

logger = init_logger(__name__)


def hy4_windowed_weights(source, files, *, require_fastpath, allowed=None):
    """A single sequential read covers hundreds of small expert scales and zeros.

    TP consumers take views of these original GPU tensors. The pack's per-expert
    down projections otherwise produce millions of noncontiguous host read-plan
    entries. Each GPU window remains valid until its final consumer releases it.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    config = get_current_vllm_config_or_none()
    group = None
    if (
        config is not None
        and config.additional_config.get("hy4_share_weight_reads", False)
        and config.parallel_config.tensor_parallel_size > 1
    ):
        group = get_tp_group()
    started = time.monotonic()
    read_bytes = 0
    # Stay below bcache's sequential bypass threshold on the compat path.
    transfer = (16 if require_fastpath else 2) << 20
    window = 32 << 20
    with DirectStorage(device, transfer, require_fastpath=require_fastpath) as io:
        for path in files:
            with RegisteredFile(str(path), direct=True) as file:
                header = io.safetensor_header(file)
                names = None if allowed is None else allowed[Path(path).name]
                if names is not None and names.difference(header.tensors):
                    raise ValueError(f"Missing indexed tensors in {Path(path).name}")
                specs = sorted(
                    (
                        spec
                        for spec in header
                        if (names is None or spec.name in names)
                        and not (
                            group is None
                            and source.is_unused_weight
                            and source.is_unused_weight(spec.name)
                        )
                    ),
                    key=lambda entry: entry.start,
                )
                i = 0
                while i < len(specs):
                    first = specs[i].start
                    j = i + 1
                    while j < len(specs) and specs[j].end - first <= window:
                        j += 1
                    length = specs[j - 1].end - first
                    offset = header.data_begin + first
                    if group is not None and length >= transfer:
                        # Adjacent rank slices are reassembled on the GPU. This
                        # reads the original window once across the TP group.
                        span = ALIGN * math.ceil(length / (group.world_size * ALIGN))
                        local = torch.empty(span, dtype=torch.uint8, device=device)
                        start = group.rank_in_group * span
                        take = max(0, min(span, length - start))
                        if take:
                            io.read(file, offset + start, local[:take])
                        local[take:].zero_()
                        read_bytes += take
                        data = tensor_model_parallel_all_gather(local, dim=0)[:length]
                    else:
                        data = torch.empty(length, dtype=torch.uint8, device=device)
                        io.read(file, offset, data)
                        read_bytes += length
                    for spec in specs[i:j]:
                        if (
                            group is not None
                            and source.is_unused_weight
                            and source.is_unused_weight(spec.name)
                        ):
                            continue
                        if spec.nbytes != math.prod(spec.shape) * spec.dtype.itemsize:
                            raise ValueError(f"HY4 tensor extent mismatch: {spec.name}")
                        raw = data[spec.start - first : spec.end - first]
                        if raw.storage_offset() % spec.dtype.itemsize:
                            raw = raw.clone()
                        yield (
                            source.prefix + spec.name,
                            raw.view(spec.dtype).view(spec.shape),
                        )
                    i = j
    logger.info(
        "HY4 rank read %.2f GiB in %.2f seconds (shared reads: %s)",
        read_bytes / (1 << 30),
        time.monotonic() - started,
        group is not None,
    )


class HY4DirectModelLoader(DefaultModelLoader):
    def _get_weights_iterator(self, source):
        root = Path(source.model_or_path)
        if source.subfolder:
            root /= source.subfolder
        index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        per_file = defaultdict(set)
        for name, filename in index.items():
            if Path(filename).name != filename:
                raise ValueError(f"Unexpected HY4 shard filename: {filename}")
            per_file[filename].add(name)
        self.counter_before_loading_weights = time.perf_counter()
        return hy4_windowed_weights(
            source,
            [root / name for name in per_file],
            require_fastpath=True,
            allowed=per_file,
        )

    def download_model(self, model_config):
        if not Path(model_config.model_weights or model_config.model).is_dir():
            raise ValueError(
                "Download the HY4 checkpoint to the direct-storage volume first"
            )
