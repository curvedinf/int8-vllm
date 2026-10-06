# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic FS->GPU direct-IO layer (AMD Infinity Storage / hipFile).

One interface for every "file bytes belong in device memory" task in the
fork: weight loading today; KV-tier store/load and a VRAM page tier later.

Backend selection is env-driven (VLLM_FS_GPU):

- ``auto`` (default): use AIS wherever it is available in this process —
  the library loads, resolves onto a single HIP runtime, and the cache
  filesystem accepts O_DIRECT. Otherwise silently fall back to stock.
- ``weights`` / ``all``: force on (fail loudly if unusable).
- ``off``: never use AIS.
"""

from __future__ import annotations

import os

from vllm.envs import VLLM_AIS_STAGING_MB, VLLM_FS_GPU
from vllm.fs_gpu.ais import AisLib, FsGpuError, get_ais, verify_single_hip_runtime
from vllm.fs_gpu.io import fill_ranges, merge_ranges, read_into_tensor, write_from_tensor
from vllm.fs_gpu.repack import RepackCache, RepackWriter, default_repack_root, run_groups
from vllm.fs_gpu.recorder import ReadPlan, WeightPlan, with_active_plan
from vllm.fs_gpu.region import GpuStagingPool, RegisteredFile
from vllm.fs_gpu.safetensors import SafetensorFile, TensorSpec
from vllm.fs_gpu.stats import fs_gpu_stats
from vllm.logger import init_logger

__all__ = [
    "AisLib",
    "FsGpuError",
    "get_ais",
    "verify_single_hip_runtime",
    "GpuStagingPool",
    "RegisteredFile",
    "read_into_tensor",
    "write_from_tensor",
    "fill_ranges",
    "merge_ranges",
    "SafetensorFile",
    "TensorSpec",
    "ReadPlan",
    "WeightPlan",
    "with_active_plan",
    "RepackCache",
    "RepackWriter",
    "default_repack_root",
    "run_groups",
    "fs_gpu_stats",
    "fs_gpu_enabled",
    "ais_available",
]

_logger = init_logger(__name__)


def ais_available() -> bool:
    """Cheap runtime probe: can AIS serve this process? (cached per process)."""
    global _available
    if _available is not None:
        return _available
    try:
        get_ais()
        maps = verify_single_hip_runtime()
        if len(maps) != 1:
            raise FsGpuError(f"multiple libamdhip64 mappings: {maps}", 5000 + 30)
        from vllm.envs import VLLM_CACHE_ROOT

        os.makedirs(VLLM_CACHE_ROOT, exist_ok=True)
        probe = os.path.join(VLLM_CACHE_ROOT, f".ais_avail_probe.{os.getpid()}")
        fd = os.open(probe, os.O_DIRECT | os.O_CREAT | os.O_RDWR, 0o644)
        os.close(fd)
        os.unlink(probe)
        _available = True
    except Exception as e:  # noqa: BLE001
        _logger.info_once("fs_gpu auto-mode: AIS unavailable (%s); using stock paths", e)
        _available = False
    return _available


_available: bool | None = None


def fs_gpu_enabled(component: str) -> bool:
    """Whether AIS direct IO is enabled for ``component`` (e.g. 'weights')."""
    mode = VLLM_FS_GPU
    if mode == "off" or not mode:
        return False
    if mode == "all":
        return True
    if mode == "weights":
        return component == "weights"
    if mode == "auto":
        return component == "weights" and ais_available()
    return False
