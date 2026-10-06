# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AIS (AMD Infinity Storage / hipFile) direct-to-GPU model loader.

Yields device-resident tensors filled by direct file->VRAM reads — no
safetensors mmap, no CPU heap copy, no pageable H2D. Everything downstream
(the weight_loader chain's TP narrowing, GPTQ->AITER GPU-side repack in
process_weights_after_loading) is unchanged; from its point of view the
checkpoint simply arrives already on the device.

Per-rank reads (VLLM_AIS_PER_RANK, default on): before the real load, the
loader runs the same ``model.load_weights`` chain once against recording
tensors (vllm.fs_gpu.recorder) whose copies are no-op'd and whose consumed
views are logged. The real pass then fills only the byte ranges the
consumer actually copies — each TP rank reads its own shard slices instead
of every tensor in full (4x NVMe traffic reduction at TP4). Weights whose
consumers do anything the recorder cannot track fall back to full reads.

Selected by ``--load-format ais`` or automatically for safetensors
checkpoints when ``VLLM_FS_GPU=weights``. Non-safetensors sources and
(non-fatal) AIS init failures fall back to the stock loader when
``VLLM_FS_GPU_ALLOW_COMPAT`` is set (default); failures mid-load are always
loud.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm import tqdm
from torch import nn

from vllm.config import ModelConfig
from vllm.envs import (
    VLLM_AIS_IO_THREADS,
    VLLM_AIS_MERGE_GAP_KB,
    VLLM_AIS_MERGE_MAX_KB,
    VLLM_AIS_PER_RANK,
    VLLM_AIS_STAGING_MB,
    VLLM_FS_GPU_ALLOW_COMPAT,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    _BAR_FORMAT,
    _natural_sort_key,
    enable_tqdm,
    should_skip_weight,
)
from vllm.fs_gpu.ais import FsGpuError
from vllm.fs_gpu.io import fill_ranges, merge_ranges, read_into_tensor
from vllm.fs_gpu.recorder import ReadPlan, with_active_plan
from vllm.fs_gpu.region import GpuStagingPool, RegisteredFile
from vllm.fs_gpu.safetensors import SafetensorFile
from vllm.fs_gpu.stats import fs_gpu_stats
from vllm.model_executor.model_loader import register_model_loader

logger = init_logger(__name__)


def _iter_checkpoint_tensors(
    loader: DefaultModelLoader, source: DefaultModelLoader.Source
) -> Generator[tuple[str, str, SafetensorFile, object], None, None]:
    """Yield (prefixed name, file, index, spec) for one source."""
    hf_folder, hf_weights_files, use_safetensors = loader._prepare_weights(
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
    for st_file in sorted(hf_weights_files, key=_natural_sort_key):
        index = SafetensorFile(st_file)
        for spec in index:
            if should_skip_weight(spec.name, loader.local_expert_ids):
                continue
            yield source.prefix + spec.name, st_file, index, spec


def ais_safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    local_expert_ids: set[int] | None = None,
    device: str = "cuda",
    prefix: str = "",
    plan: ReadPlan | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield (name, device tensor) for every tensor in the shard set."""
    pool = None
    executor = (
        ThreadPoolExecutor(max_workers=VLLM_AIS_IO_THREADS) if plan is not None else None
    )
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
                    if spec.nbytes > 0:
                        entry = plan.get(prefix + spec.name) if plan else None
                        if entry is not None and not entry.full and entry.copies:
                            data_base = index.data_begin + spec.start
                            ranges = [
                                (bo, data_base + bo, n) for (bo, n) in entry.ranges
                            ]
                            ranges = merge_ranges(
                                ranges,
                                VLLM_AIS_MERGE_GAP_KB << 10,
                                VLLM_AIS_MERGE_MAX_KB << 10,
                            )
                            fill_ranges(rf, tensor, ranges, executor)
                        else:
                            if pool is None:
                                pool = GpuStagingPool(
                                    VLLM_AIS_STAGING_MB << 20, device=device
                                )
                            read_into_tensor(
                                rf, pool, tensor, index.data_begin + spec.start
                            )
                    yield spec.name, tensor
                    del tensor
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if pool is not None:
            pool.close()
        logger.info_once("%s", fs_gpu_stats.snapshot())


@register_model_loader("ais")
class AisModelLoader(DefaultModelLoader):
    """DefaultModelLoader with the safetensors read path routed via AIS."""

    def __init__(self, load_config):
        super().__init__(load_config)
        self._read_plan: ReadPlan | None = None

    def _build_read_plan(self, model: nn.Module, model_config: ModelConfig) -> ReadPlan:
        plan = ReadPlan()
        sources = [
            DefaultModelLoader.Source(
                model_config.model,
                model_config.revision,
                prefix="",
                fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
            )
        ]
        sources.extend(getattr(model, "secondary_weights", ()))

        def recorder_iter():
            for source in sources:
                for name, _f, _idx, spec in _iter_checkpoint_tensors(self, source):
                    yield name, plan.new_recorder(
                        name, spec.shape, spec.dtype, device="cuda"
                    )

        with with_active_plan(plan):
            model.load_weights(recorder_iter())
        logger.info("fs_gpu %s", plan.summary())
        return plan

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        self._read_plan = None
        if VLLM_AIS_PER_RANK:
            try:
                self._read_plan = self._build_read_plan(model, model_config)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "fs_gpu per-rank read plan unavailable (%s: %s); "
                    "falling back to full-tensor reads.",
                    type(e).__name__,
                    e,
                )
                self._read_plan = None
        return super().load_weights(model, model_config)

    def _get_weights_iterator(
        self, source: DefaultModelLoader.Source
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        try:
            # Validate + materialize the file list once (also raises for
            # non-safetensors checkpoints).
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
                prefix=source.prefix,
                plan=self._read_plan,
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
