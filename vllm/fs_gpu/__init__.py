# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic FS->GPU direct-IO layer (AMD Infinity Storage / hipFile).

One interface for every "file bytes belong in device memory" task in the
fork: weight loading today; KV-tier store/load and a VRAM page tier later.
Backend selection is env-driven (:data:`VLLM_FS_GPU`) with a hard failure
mode (no silent host-staging fallback) unless compat is explicitly allowed.
"""

from __future__ import annotations

from vllm.envs import VLLM_AIS_STAGING_MB, VLLM_FS_GPU
from vllm.fs_gpu.ais import AisLib, FsGpuError, get_ais, verify_single_hip_runtime
from vllm.fs_gpu.io import read_into_tensor, write_from_tensor
from vllm.fs_gpu.region import GpuStagingPool, RegisteredFile
from vllm.fs_gpu.safetensors import SafetensorFile, TensorSpec
from vllm.fs_gpu.stats import fs_gpu_stats

__all__ = [
    "AisLib",
    "FsGpuError",
    "get_ais",
    "verify_single_hip_runtime",
    "GpuStagingPool",
    "RegisteredFile",
    "read_into_tensor",
    "write_from_tensor",
    "SafetensorFile",
    "TensorSpec",
    "fs_gpu_stats",
    "fs_gpu_enabled",
]


def fs_gpu_enabled(component: str) -> bool:
    """Whether AIS direct IO is enabled for ``component`` (e.g. 'weights')."""
    mode = VLLM_FS_GPU
    return mode == "all" or mode == component
