# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AIS (AMD Infinity Storage / hipFile) direct-to-GPU model loader.

Yields device-resident tensors filled by direct file->VRAM reads — no
safetensors mmap, no CPU heap copy, no pageable H2D. Everything downstream
(the weight_loader chain's TP narrowing, GPTQ->AITER GPU-side repack in
process_weights_after_loading) is unchanged.

Three layers, fastest first:

1. **Repacked cache** (``VLLM_AIS_REPACK=1``, default): the first AIS boot
   writes each rank's planned byte runs into a compact per-rank file
   (vllm.fs_gpu.repack). Cache-hit boots do only large sequential reads and
   skip the plan pass — deterministic, NVMe-bandwidth-bound.
2. **Per-rank plan reads** (``VLLM_AIS_PER_RANK=1``, default): a recording
   pass over ``model.load_weights`` (vllm.fs_gpu.recorder) derives the byte
   ranges this rank's consumers actually copy; the fill reads only those
   ranges (gap-merged; hipFile 0.3.0 is IOPS-limited).
3. **Full reads**: every tensor read whole (v1 behavior; the fallback for
   unplannable weights).

Selected by ``--load-format ais`` or automatically (``VLLM_FS_GPU=auto``,
the default) whenever AIS is usable in this process. Non-safetensors
sources fall back to the stock loader under ``VLLM_FS_GPU_ALLOW_COMPAT``
(default); failures mid-load are always loud.
"""

from __future__ import annotations

import os
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
    VLLM_AIS_REPACK,
    VLLM_AIS_REPACK_WINDOW_MB,
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
from vllm.fs_gpu.repack import (
    RepackCache,
    RepackWriter,
    canonical_ranges,
    default_repack_root,
    run_groups,
)
from vllm.fs_gpu.recorder import ReadPlan, with_active_plan
from vllm.fs_gpu.region import GpuStagingPool, RegisteredFile
from vllm.fs_gpu.safetensors import SafetensorFile
from vllm.fs_gpu.stats import fs_gpu_stats
from vllm.model_executor.model_loader import register_model_loader

logger = init_logger(__name__)


def _sorted_files(files: list[str]) -> list[str]:
    return sorted(files, key=_natural_sort_key)


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
    for st_file in _sorted_files(hf_weights_files):
        index = SafetensorFile(st_file)
        for spec in index:
            if should_skip_weight(spec.name, loader.local_expert_ids):
                continue
            yield source.prefix + spec.name, st_file, index, spec


def _scatter_from_compact(
    tensor: torch.Tensor, compact: torch.Tensor, runs: list[tuple[int, int]]
) -> None:
    """Place a compact uint8 block into the full-shaped tensor (one copy per group)."""
    es = tensor.element_size()
    off = 0
    for first, length, period, count in run_groups(runs, es):
        n = length * count * es
        src = compact[off : off + n].view(tensor.dtype).view(count, length)
        tensor.as_strided((count, length), (period, 1), first).copy_(src)
        off += n


def _gather_into_compact(
    tensor: torch.Tensor, compact: torch.Tensor, runs: list[tuple[int, int]]
) -> None:
    """Reverse of _scatter_from_compact (repack write side)."""
    es = tensor.element_size()
    off = 0
    for first, length, period, count in run_groups(runs, es):
        n = length * count * es
        dst = compact[off : off + n].view(tensor.dtype).view(count, length)
        dst.copy_(tensor.as_strided((count, length), (period, 1), first))
        off += n


def ais_repacked_weights_iterator(
    cache: RepackCache,
    use_tqdm_on_load: bool,
    device: str = "cuda",
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Cache-hit path: windowed sequential reads + strided scatter per tensor.

    Blocks are laid out contiguously in iteration order, so consecutive
    tensors are coalesced into ~``VLLM_AIS_REPACK_WINDOW_MB`` reads. Four
    concurrent 1-block-at-a-time streams thrash this DRAM-less NVMe down to
    ~0.5 GiB/s aggregate; 32 MiB chunks sustained ~1.75 GiB/s in the v1
    boots (ledger AIS_W1/W2).
    """
    specs = cache.tensors
    assert specs is not None
    max_block = max(t.block_len for t in specs) if specs else 0
    window = VLLM_AIS_REPACK_WINDOW_MB << 20
    pool = GpuStagingPool(max(VLLM_AIS_STAGING_MB << 20, max_block, window), device=device)
    if max(max_block, window) > pool.capacity:
        pool.grow(max(max_block, window))
    bin_size = os.path.getsize(cache.bin_path)
    try:
        with RegisteredFile(cache.bin_path) as rf:
            i = 0
            pbar = tqdm(
                total=len(specs),
                desc="Loading repacked AIS rank cache",
                disable=not enable_tqdm(use_tqdm_on_load),
                bar_format=_BAR_FORMAT,
            )
            while i < len(specs):
                t = specs[i]
                if t.block_len >= window:
                    tensor = torch.empty(t.shape, dtype=t.dtype, device=device)
                    rf.ais.read(rf.fh, pool.base, t.block_len, t.block_off, 0)
                    fs_gpu_stats.reads += 1
                    fs_gpu_stats.bytes_read += t.block_len
                    if t.block_len:
                        _scatter_from_compact(tensor, pool.view(0, t.block_len), t.runs)
                    pbar.update(1)
                    yield t.name, tensor
                    del tensor
                    torch.cuda.current_stream().synchronize()
                    i += 1
                    continue
                # coalesce following blocks into one window read
                n = min(window, bin_size - t.block_off)
                j = i
                while j < len(specs) and specs[j].block_off + specs[j].block_len <= t.block_off + n:
                    j += 1
                rf.ais.read(rf.fh, pool.base, n, t.block_off, 0)
                fs_gpu_stats.reads += 1
                fs_gpu_stats.bytes_read += n
                for k in range(i, j):
                    s = specs[k]
                    rel = s.block_off - t.block_off
                    tensor = torch.empty(s.shape, dtype=s.dtype, device=device)
                    if s.block_len:
                        _scatter_from_compact(
                            tensor, pool.view(rel, s.block_len), s.runs
                        )
                    pbar.update(1)
                    yield s.name, tensor
                    del tensor
                    # Consumer copy_ kernels reading the yielded tensor may
                    # still be queued; raw DMA for the next allocation must
                    # not preempt them (caching-allocator reuse).
                    torch.cuda.current_stream().synchronize()
                i = j
            pbar.close()
    finally:
        pool.close()
        logger.info_once("%s", fs_gpu_stats.snapshot())


def ais_safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    local_expert_ids: set[int] | None = None,
    device: str = "cuda",
    prefix: str = "",
    plan: ReadPlan | None = None,
    writer: RepackWriter | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Plan-driven fill from the original checkpoint (+ optional repack write)."""
    pool = None
    executor = ThreadPoolExecutor(max_workers=VLLM_AIS_IO_THREADS) if plan else None
    try:
        for st_file in tqdm(
            _sorted_files(hf_weights_files),
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
                            runs = entry.ranges
                        else:
                            if pool is None:
                                pool = GpuStagingPool(
                                    VLLM_AIS_STAGING_MB << 20, device=device
                                )
                            read_into_tensor(
                                rf, pool, tensor, index.data_begin + spec.start
                            )
                            runs = [(0, spec.nbytes)]
                    else:
                        runs = []
                    if writer is not None and (runs or spec.nbytes == 0):
                        runs = canonical_ranges(runs)
                        _append_to_writer(
                            writer, spec.name, spec.dtype, spec.shape, tensor, runs, device
                        )
                    yield spec.name, tensor
                    del tensor
                    torch.cuda.current_stream().synchronize()
        if writer is not None:
            ok = writer.commit()
            logger.info("fs_gpu repack cache %s", "written" if ok else "FAILED")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if pool is not None:
            pool.close()
        logger.info_once("%s", fs_gpu_stats.snapshot())


def _append_to_writer(
    writer: RepackWriter,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    tensor: torch.Tensor,
    runs: list[tuple[int, int]],
    device: str,
) -> None:
    total = sum(n for _, n in runs)
    if total == 0:
        writer.append(name, dtype, shape, torch.empty(0, dtype=torch.uint8, device=device), runs)
        return
    # Double-buffered gather pools: hipFileWrite's DMA can outlive its
    # return (probed: 6/2405 blocks corrupted sharing one pool), so a
    # lingering write must never share a buffer with the next gather.
    pools = getattr(writer, "_gather_pools", None)
    if pools is None:
        pools = [
            GpuStagingPool(max(VLLM_AIS_STAGING_MB << 20, total), device=device)
            for _ in range(2)
        ]
        writer._gather_pools = pools  # type: ignore[attr-defined]
        writer._gather_idx = 0  # type: ignore[attr-defined]
    pool = pools[writer._gather_idx]  # type: ignore[attr-defined]
    writer._gather_idx = 1 - writer._gather_idx  # type: ignore[attr-defined]
    if total > pool.capacity:
        pool.grow(total)
    _gather_into_compact(tensor, pool.view(0, total), runs)
    # The gather is stream-enqueued; hipFileWrite reads the pool from the
    # host immediately and would race kernels still in flight.
    torch.cuda.current_stream().synchronize()
    writer.append(name, dtype, shape, pool.view(0, total), runs)


@register_model_loader("ais")
class AisModelLoader(DefaultModelLoader):
    """DefaultModelLoader with the safetensors read path routed via AIS."""

    def __init__(self, load_config):
        super().__init__(load_config)
        self._read_plan: ReadPlan | None = None
        self._repack_hit: RepackCache | None = None
        self._repack_write: bool = False
        self._repack_model: str | None = None

    # -- repack cache resolution ---------------------------------------------

    def _resolve_repack(self, model_config: ModelConfig) -> None:
        self._repack_hit = None
        self._repack_write = False
        if not (VLLM_AIS_PER_RANK and VLLM_AIS_REPACK):
            return
        try:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
        except Exception:
            return
        if tp_size <= 1:
            return
        model_dir = model_config.model
        if not model_dir or not model_dir.startswith("/"):
            return  # non-local (hub id) checkpoints are not repacked
        hf_folder, files, use_st = self._prepare_weights(
            model_dir, None, model_config.revision, True, None
        )
        if not use_st:
            return
        cache = RepackCache(default_repack_root(model_dir), tp_size, tp_rank)
        if cache.validate(files):
            self._repack_hit = cache
            self._repack_model = model_dir
            logger.info(
                "fs_gpu repack cache HIT: %s (%d tensors)",
                cache.bin_path,
                len(cache.tensors or []),
            )
        else:
            self._repack_write = True
            self._repack_model = model_dir
            logger.info_once(
                "fs_gpu repack cache MISS for %s (tp%d rank%d); writing after load",
                model_dir,
                tp_size,
                tp_rank,
            )
            self._pending_repack = (cache, files)

    # -- plan ------------------------------------------------------------------

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
        try:
            self._resolve_repack(model_config)
        except Exception as e:  # noqa: BLE001
            logger.warning("fs_gpu repack resolve failed (%s); ignoring cache", e)
            self._repack_hit = None
            self._repack_write = False
        if self._repack_hit is None and VLLM_AIS_PER_RANK:
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

    # -- iterator ----------------------------------------------------------------

    def _get_weights_iterator(
        self, source: DefaultModelLoader.Source
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        weights_iterator = None
        try:
            if (
                self._repack_hit is not None
                and not source.prefix
                and source.model_or_path == self._repack_model
            ):
                weights_iterator = ais_repacked_weights_iterator(
                    self._repack_hit, self.load_config.use_tqdm_on_load
                )
            if weights_iterator is None:
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
                writer = None
                if (
                    self._repack_write
                    and not source.prefix
                    and source.model_or_path == self._repack_model
                ):
                    cache, files = self._pending_repack
                    writer = cache.writer(files)
                weights_iterator = ais_safetensors_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    local_expert_ids=self.local_expert_ids,
                    prefix=source.prefix,
                    plan=self._read_plan,
                    writer=writer,
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
