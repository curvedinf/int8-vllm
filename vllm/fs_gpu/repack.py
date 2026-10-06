# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank repacked checkpoint cache for AIS weight loading.

hipFile 0.3.0 executes ~one IO per process, so per-rank strided shard reads
are IOPS-bound no matter how they are merged (ledger AIS_W2). This cache
eliminates the geometry problem: each TP rank's planned byte runs — exactly
what the plan-pass recorder derives from the weight_loader chain — are
stored contiguously in one file per rank, so a cache-hit boot performs only
large sequential reads (deterministic NVMe-bandwidth-bound load) and skips
the plan pass entirely.

Layout (per rank, all under one directory keyed by model + TP size):

    rank{r}.bin        concatenated compact tensor blocks (8-byte aligned)
    rank{r}.meta.json  manifest: source fingerprint + per-tensor run maps

The manifest is written last; a rank is cache-valid only when its manifest
exists, the format version matches, and every source safetensors file still
matches the recorded (size, mtime) fingerprint. Writes go to
``.tmp-<pid>`` files renamed into place, so a crash mid-repack can never
produce a half-valid cache.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

import torch

from vllm.fs_gpu.ais import FsGpuError, get_ais
from vllm.fs_gpu.region import GpuStagingPool, RegisteredFile

REPACK_VERSION = 1


@dataclasses.dataclass
class RepackedTensor:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    block_off: int
    block_len: int
    runs: list[tuple[int, int]]  # (tensor-relative byte off, byte len)


class RepackCache:
    """Read/write access to one rank's repacked cache."""

    def __init__(self, root: str, tp_size: int, rank: int):
        self.root = root
        self.tp_size = tp_size
        self.rank = rank
        self.bin_path = os.path.join(root, f"rank{rank}.bin")
        self.meta_path = os.path.join(root, f"rank{rank}.meta.json")
        self.tensors: list[RepackedTensor] | None = None

    # -- fingerprinting -----------------------------------------------------

    @staticmethod
    def fingerprint(files: list[str]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for f in sorted(files):
            st = os.stat(f)
            out[os.path.basename(f)] = [st.st_size, st.st_mtime_ns]
        return out

    def validate(self, files: list[str]) -> bool:
        """Cache-hit test: manifest present, version/tp match, sources unchanged."""
        try:
            with open(self.meta_path) as f:
                meta = json.load(f)
            if meta.get("version") != REPACK_VERSION or meta.get("tp_size") != self.tp_size:
                return False
            if meta.get("rank") != self.rank:
                return False
            if meta.get("sources") != self.fingerprint(files):
                return False
            if not os.path.exists(self.bin_path):
                return False
            size_ok = all(
                t["block_off"] + t["block_len"] <= os.path.getsize(self.bin_path)
                for t in meta.get("tensors", [])
            )
            if not size_ok:
                return False
            self.tensors = [
                RepackedTensor(
                    name=t["name"],
                    dtype=_STR_TO_DTYPE[t["dtype"]],
                    shape=tuple(t["shape"]),
                    block_off=t["block_off"],
                    block_len=t["block_len"],
                    runs=[(r[0], r[1]) for r in t["runs"]],
                )
                for t in meta.get("tensors", [])
            ]
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False

    # -- write side ----------------------------------------------------------

    def writer(self, files: list[str]) -> RepackWriter:
        return RepackWriter(self, files)


class RepackWriter:
    """Appends compact tensor blocks to the rank bin; publishes meta last."""

    def __init__(self, cache: RepackCache, files: list[str]):
        os.makedirs(cache.root, exist_ok=True)
        self.cache = cache
        self.tmp_bin = f"{cache.bin_path}.tmp-{os.getpid()}"
        self.entries: list[dict[str, Any]] = []
        self.off = 0
        self.ais = get_ais()
        self.fd = os.open(self.tmp_bin, os.O_DIRECT | os.O_CREAT | os.O_RDWR, 0o644)
        self.fh = self.ais.handle_register(self.fd)
        self.sources = cache.fingerprint(files)

    def append(
        self,
        name: str,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        compact: torch.Tensor,
        runs: list[tuple[int, int]],
    ) -> None:
        """Write one tensor's compact block (a contiguous uint8 GPU buffer)."""
        n = compact.numel()
        self.ais.write(self.fh, compact.data_ptr(), n, self.off, 0)
        self.entries.append(
            {
                "name": name,
                "dtype": _DTYPE_TO_STR[dtype],
                "shape": list(shape),
                "block_off": self.off,
                "block_len": n,
                "runs": [[bo, ln] for (bo, ln) in runs],
            }
        )
        self.off += (n + 7) & ~7  # keep blocks 8-byte aligned

    def commit(self) -> bool:
        try:
            self.ais.handle_deregister(self.fh)
            os.close(self.fd)
            meta = {
                "version": REPACK_VERSION,
                "tp_size": self.cache.tp_size,
                "rank": self.cache.rank,
                "sources": self.sources,
                "tensors": self.entries,
            }
            tmp_meta = f"{self.cache.meta_path}.tmp-{os.getpid()}"
            with open(tmp_meta, "w") as f:
                json.dump(meta, f)
            os.replace(self.tmp_bin, self.cache.bin_path)
            os.replace(tmp_meta, self.cache.meta_path)
            return True
        except OSError:
            return False
        finally:
            self._cleanup()

    def abort(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        for p in (self.tmp_bin, f"{self.cache.meta_path}.tmp-{os.getpid()}"):
            try:
                os.unlink(p)
            except OSError:
                pass


# -- run helpers (element-units, for strided scatter/gather) -----------------


def canonical_ranges(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort by offset, drop duplicates, and fuse overlapping/abutting runs.

    The recorder appends ranges in copy order; for merged-column weights
    (q/k/v shards of one checkpoint tensor) that interleaves run families,
    which the repack gather/scatter cannot consume. Canonical order makes
    the compact layout deterministic and identical on both sides.
    """
    if not runs:
        return []
    out: list[tuple[int, int]] = []
    for bo, n in sorted(runs):
        if out:
            e = out[-1][0] + out[-1][1]
            if bo <= e:  # overlap or abut
                if bo + n > e:
                    out[-1] = (out[-1][0], bo + n - out[-1][0])
                continue
        out.append((bo, n))
    return out


def run_groups(runs: list[tuple[int, int]], es: int) -> list[tuple[int, int, int, int]]:
    """Group byte runs into maximal constant-stride groups.

    Returns (first_off_elem, run_len_elem, period_elem, count) tuples — each
    maps onto a single 2-D strided view, so scatter/gather is one copy
    kernel per group instead of one per run.
    """
    groups: list[tuple[int, int, int, int]] = []
    idx = 0
    while idx < len(runs):
        bo0, n0 = runs[idx]
        period = None
        count = 1
        j = idx + 1
        while j < len(runs):
            bo1, n1 = runs[j]
            if n1 != n0:
                break
            d = (bo1 - bo0) // es
            if period is None:
                period = d
            elif d != period * count:
                break
            count += 1
            j += 1
        groups.append((bo0 // es, n0 // es, period if period else n0 // es, count))
        idx = j
    return groups


_STR_TO_DTYPE = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}
_DTYPE_TO_STR = {v: k for k, v in _STR_TO_DTYPE.items()}


def default_repack_root(model_dir: str) -> str:
    from vllm.envs import VLLM_AIS_REPACK_DIR, VLLM_CACHE_ROOT

    base = VLLM_AIS_REPACK_DIR or os.path.join(VLLM_CACHE_ROOT, "ais_repack")
    tag = os.path.basename(os.path.normpath(model_dir))
    return os.path.join(base, tag)
