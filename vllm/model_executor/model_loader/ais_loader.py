# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AIS (AMD Infinity Storage / hipFile) direct-to-GPU model loader.

Yields device-resident tensors filled by direct file->VRAM reads — no
safetensors mmap, no CPU heap copy, no pageable H2D. Everything downstream
(the weight_loader chain's TP narrowing, GPTQ->AITER GPU-side repack in
process_weights_after_loading) is unchanged; from its point of view the
checkpoint simply arrives already on the device.

Selected by ``--load-format ais`` or automatically for safetensors
checkpoints when ``VLLM_FS_GPU=weights``. Non-safetensors sources and
(non-fatal) AIS init failures fall back to the stock loader when
``VLLM_FS_GPU_ALLOW_COMPAT`` is set (default); failures mid-load are always
loud.
"""

from __future__ import annotations

import time
from collections.abc import Generator

import torch
from tqdm import tqdm
from torch import nn

from vllm.config import ModelConfig
from vllm.envs import VLLM_AIS_STAGING_MB, VLLM_FS_GPU_ALLOW_COMPAT
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    _BAR_FORMAT,
    _natural_sort_key,
    enable_tqdm,
    should_skip_weight,
)
from vllm.fs_gpu.ais import FsGpuError
from vllm.fs_gpu.io import read_into_tensor
from vllm.fs_gpu.region import GpuStagingPool, RegisteredFile
from vllm.fs_gpu.safetensors import SafetensorFile
from vllm.fs_gpu.stats import fs_gpu_stats
from vllm.model_executor.model_loader import register_model_loader

logger = init_logger(__name__)


def ais_safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    local_expert_ids: set[int] | None = None,
    device: str = "cuda",
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield (name, device tensor) for every tensor in the shard set."""
    pool = GpuStagingPool(VLLM_AIS_STAGING_MB << 20, device=device)
    try:
        for st_file in tqdm(
            sorted(hf_weights_files, key=_natural_sort_key),
            desc="Loading safetensors checkpoint shards (AIS direct-to-GPU)",
            disable=not enable_tqdm(use_tqdm_on_load),
            bar_format=_BAR_FORMAT,
        ):
            index = SafetensorFile(st_file)
            with RegisteredFile(st_file) as rf:
                for spec in index:
                    if should_skip_weight(spec.name, local_expert_ids):
                        continue
                    tensor = torch.empty(spec.shape, dtype=spec.dtype, device=device)
                    read_into_tensor(rf, pool, tensor, index.data_begin + spec.start)
                    yield spec.name, tensor
                    del tensor
    finally:
        pool.close()
        logger.info_once("%s", fs_gpu_stats.snapshot())


@register_model_loader("ais")
class AisModelLoader(DefaultModelLoader):
    """DefaultModelLoader with the safetensors read path routed via AIS."""

    def _get_weights_iterator(
        self, source: DefaultModelLoader.Source
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        try:
            hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
                source.model_or_path,
                source.subfolder,
                source.revision,
                source.fall_back_to_pt,
                source.allow_patterns_overrides,
            )
            if not use_safetensors:
                raise FsGpuError(
                    f"AIS loader requires safetensors checkpoints; "
                    f"{source.model_or_path} is not safetensors",
                    5000 + 18,
                )
            weights_iterator = ais_safetensors_weights_iterator(
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
                local_expert_ids=self.local_expert_ids,
            )
        except FsGpuError as e:
            if not VLLM_FS_GPU_ALLOW_COMPAT:
                raise
            logger.warning(
                "AIS direct-to-GPU loading unavailable (%s); falling back to "
                "the stock (host-staged) loader.",
                e,
            )
            weights_iterator = None

        if weights_iterator is None:
            return super()._get_weights_iterator(source)

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return (
            (source.prefix + name, tensor) for (name, tensor) in weights_iterator
        )
