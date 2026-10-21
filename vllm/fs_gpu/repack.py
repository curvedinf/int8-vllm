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

import numpy as np
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
    # list[tuple] on the legacy JSON path, np.ndarray view on the fast path.
    runs: Any  # (tensor-relative byte off, byte len)


class RepackCache:
    """Read/write access to one rank's repacked cache."""

    def __init__(self, root: str, tp_size: int, rank: int):
        self.root = root
        self.tp_size = tp_size
        self.rank = rank
        self.bin_path = os.path.join(root, f"rank{rank}.bin")
        self.meta_path = os.path.join(root, f"rank{rank}.meta.json")
        self.fast_meta_path = os.path.join(root, f"rank{rank}.meta.fast.json")
        self.runs_path = os.path.join(root, f"rank{rank}.runs.npy")
        self.tensors: list[RepackedTensor] | None = None

    # -- fingerprinting -----------------------------------------------------

    @staticmethod
    def fingerprint(files: list[str]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for f in sorted(files):
            st = os.stat(f)
            out[os.path.basename(f)] = [st.st_size, st.st_mtime_ns]
        return out

    def _validate_fast(self, files: list[str]) -> bool:
        """O(1) manifest load: small JSON + mmap'd int64 runs sidecar."""
        if not (
            os.path.exists(self.fast_meta_path) and os.path.exists(self.runs_path)
        ):
            return False
        try:
            with open(self.fast_meta_path) as f:
                meta = json.load(f)
            if (
                meta.get("version") != REPACK_VERSION
                or meta.get("tp_size") != self.tp_size
                or meta.get("rank") != self.rank
                or meta.get("sources") != self.fingerprint(files)
                or not os.path.exists(self.bin_path)
            ):
                return False
            bin_size = os.path.getsize(self.bin_path)
            runs = np.load(self.runs_path, mmap_mode="r")
            if runs.ndim != 2 or runs.shape[1] != 2:
                return False
            tensors: list[RepackedTensor] = []
            for t in meta.get("tensors", []):
                start = int(t["runs_off"])
                count = int(t["runs_count"])
                if (
                    start < 0
                    or count < 0
                    or start + count > runs.shape[0]
                    or t["block_off"] + t["block_len"] > bin_size
                ):
                    return False
                tensors.append(
                    RepackedTensor(
                        name=t["name"],
                        dtype=_STR_TO_DTYPE[t["dtype"]],
                        shape=tuple(t["shape"]),
                        block_off=t["block_off"],
                        block_len=t["block_len"],
                        runs=runs[start : start + count],
                    )
                )
            self.tensors = tensors
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False

    def validate(self, files: list[str]) -> bool:
        """Cache-hit test: manifest present, version/tp match, sources unchanged."""
        if self._validate_fast(files):
            return True
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
            # One-time migration for caches built before the fast sidecar.
            try:
                self._publish_fast(self.tensors, self.fingerprint(files))
            except (OSError, ValueError):
                pass
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False

    def _publish_fast(
        self, tensors: list[RepackedTensor], sources: dict[str, list[int]]
    ) -> None:
        runs_flat: list[int] = []
        entries: list[dict[str, Any]] = []
        for t in tensors:
            start = len(runs_flat) // 2
            for bo, n in t.runs:
                runs_flat.extend((int(bo), int(n)))
            entries.append(
                {
                    "name": t.name,
                    "dtype": _DTYPE_TO_STR[t.dtype],
                    "shape": list(t.shape),
                    "block_off": t.block_off,
                    "block_len": t.block_len,
                    "runs_off": start,
                    "runs_count": len(t.runs),
                }
            )
        self._write_fast_files(entries, sources, runs_flat)

    def _write_fast_files(
        self,
        entries: list[dict[str, Any]],
        sources: dict[str, list[int]],
        runs_flat: list[int],
    ) -> None:
        runs = np.asarray(runs_flat, dtype=np.int64).reshape(-1, 2)
        runs_tmp = f"{self.runs_path}.tmp-{os.getpid()}.npy"
        np.save(runs_tmp, runs)
        os.replace(runs_tmp, self.runs_path)
        meta = {
            "version": REPACK_VERSION,
            "tp_size": self.tp_size,
            "rank": self.rank,
            "sources": sources,
            "tensors": entries,
        }
        meta_tmp = f"{self.fast_meta_path}.tmp-{os.getpid()}"
        with open(meta_tmp, "w") as f:
            json.dump(meta, f)
        os.replace(meta_tmp, self.fast_meta_path)

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
        self._runs_flat: list[int] = []
        self.off = 0
        # Thousands of small expert writes can serialize on filesystem journal
        # commits with hipFile. This opt-in changes cache construction only;
        # source reads and subsequent cache reads still use direct GPU IO.
        cpu_write = os.environ.get("VLLM_AIS_REPACK_CPU_WRITE") == "1"
        self.ais = None if cpu_write else get_ais()
        flags = os.O_CREAT | os.O_RDWR | (0 if cpu_write else os.O_DIRECT)
        self.fd = os.open(self.tmp_bin, flags, 0o644)
        self.fh = self.ais.handle_register(self.fd) if self.ais else None
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
        if self.ais is not None:
            self.ais.write(self.fh, compact.data_ptr(), n, self.off, 0)
        else:
            # Blocking D2H preserves stream ordering and keeps the host view
            # alive until all bytes have been accepted by the filesystem.
            data = memoryview(compact.cpu().numpy()).cast("B")
            done = 0
            while done < n:
                written = os.pwrite(self.fd, data[done:], self.off + done)
                if written == 0:
                    raise OSError("Repack cache write made no progress")
                done += written
        runs_off = len(self._runs_flat) // 2
        runs_count = len(runs)
        self._runs_flat.extend(int(v) for pair in runs for v in pair)
        self.entries.append(
            {
                "name": name,
                "dtype": _DTYPE_TO_STR[dtype],
                "shape": list(shape),
                "block_off": self.off,
                "block_len": n,
                "runs": [[bo, ln] for (bo, ln) in runs],
                "runs_off": runs_off,
                "runs_count": runs_count,
            }
        )
        self.off += (n + 7) & ~7  # keep blocks 8-byte aligned

    def commit(self) -> bool:
        try:
            if self.ais is not None:
                self.ais.handle_deregister(self.fh)
            else:
                os.fsync(self.fd)
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
            fast_entries = [
                {k: v for k, v in e.items() if k != "runs"} for e in self.entries
            ]
            self.cache._write_fast_files(
                fast_entries, self.sources, self._runs_flat
            )
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


def _run_groups_numpy(
    runs: np.ndarray, es: int
) -> list[tuple[int, int, int, int]]:
    """Vectorized grouping for the fast manifest's mmap'd runs array.

    The 1,024-run down-expert pattern must not cost 1,024 Python iterations
    per tensor: the equal-length segments and the constant-stride check are
    done in C, with the scalar walk only for genuinely irregular segments.
    """
    bo = runs[:, 0]
    n = runs[:, 1]
    total = len(runs)
    groups: list[tuple[int, int, int, int]] = []
    starts = np.r_[0, np.flatnonzero(n[1:] != n[:-1]) + 1]
    ends = np.r_[starts[1:], total]
    for a, b in zip(starts.tolist(), ends.tolist()):
        if b - a == 1:
            groups.append((int(bo[a]) // es, int(n[a]) // es, int(n[a]) // es, 1))
            continue
        d = np.diff(bo[a:b]) // es
        if d.size and d[0] > 0 and bool(np.all(d == d[0])):
            groups.append((int(bo[a]) // es, int(n[a]) // es, int(d[0]), b - a))
            continue
        idx = a
        while idx < b:
            bo0, n0 = int(bo[idx]), int(n[idx])
            period = None
            count = 1
            j = idx + 1
            while j < b:
                bo1, n1 = int(bo[j]), int(n[j])
                if n1 != n0:
                    break
                d2 = (bo1 - bo0) // es
                if period is None:
                    period = d2
                elif d2 != period * count:
                    break
                count += 1
                j += 1
            groups.append(
                (bo0 // es, n0 // es, period if period else n0 // es, count)
            )
            idx = j
    return groups


def run_groups(runs: Any, es: int) -> list[tuple[int, int, int, int]]:
    """Group byte runs into maximal constant-stride groups.

    Returns (first_off_elem, run_len_elem, period_elem, count) tuples — each
    maps onto a single 2-D strided view, so scatter/gather is one copy
    kernel per group instead of one per run. ``runs`` may be the legacy list
    or an mmap'd numpy slice from the fast manifest.
    """
    if isinstance(runs, np.ndarray):
        return _run_groups_numpy(runs, es)
    groups: list[tuple[int, int, int, int]] = []
    idx = 0
    while idx < len(runs):
        bo0, n0 = int(runs[idx][0]), int(runs[idx][1])
        period = None
        count = 1
        j = idx + 1
        while j < len(runs):
            bo1, n1 = int(runs[j][0]), int(runs[j][1])
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
