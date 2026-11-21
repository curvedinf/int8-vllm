# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Page-aligned GGML Q4_0 storage for HY4's 96-component Engram rows."""

from __future__ import annotations

import errno
import mmap
import os
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.fs_gpu.direct import DirectStorage
from vllm.fs_gpu.region import ALIGN, RegisteredFile
from vllm.models.hy_v4_flash.engram_kernel import decompress_q4_rows, lookup_q4_pages
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

WIDTH = 96
BLOCK = 32
BLOCK_BYTES = 18
ROW_BYTES = WIDTH // BLOCK * BLOCK_BYTES
ROWS_PER_PAGE = ALIGN // ROW_BYTES


def table_path(directory: Path, layer_id: int) -> Path:
    return directory / f"engram-{layer_id}.q4_0"


def n_pages(nrows: int) -> int:
    return (nrows + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE


def encode_rows(rows: torch.Tensor) -> torch.Tensor:
    """GGML Q4_0 uses the sign of the first largest-magnitude weight."""
    if not rows.is_cuda or rows.ndim != 2 or rows.shape[1] != WIDTH:
        raise ValueError("Expected GPU Engram rows shaped [N, 96]")
    groups = rows.reshape(-1, WIDTH // BLOCK, BLOCK).float()
    extreme = groups.gather(-1, groups.abs().argmax(-1, keepdim=True))
    scale = extreme * (-1.0 / 8.0)
    inv = torch.where(scale != 0, scale.reciprocal(), 0)
    q = (groups * inv + 8.5).floor().clamp_(0, 15).to(torch.uint8)
    packed = q[..., :16] | (q[..., 16:] << 4)
    return torch.cat((scale.half().view(torch.uint8), packed), -1).reshape(
        -1, ROW_BYTES
    )


def decode_rows(data: torch.Tensor) -> torch.Tensor:
    groups = data.reshape(*data.shape[:-1], WIDTH // BLOCK, BLOCK_BYTES)
    scale = groups[..., :2].contiguous().view(torch.float16)
    packed = groups[..., 2:]
    q = torch.cat((packed & 15, packed >> 4), -1).to(torch.float16)
    return ((q - 8) * scale).reshape(*data.shape[:-1], WIDTH)


def page_rows(pages: torch.Tensor) -> torch.Tensor:
    return pages[:, : ROWS_PER_PAGE * ROW_BYTES].view(-1, ROWS_PER_PAGE, ROW_BYTES)


class RowCachePlan(NamedTuple):
    indices: np.ndarray
    slots: np.ndarray
    rows: np.ndarray
    tags: np.ndarray


class PackedEngramTable:
    """One layer. Every TP rank owns its head range and a bounded GPU page cache."""

    def __init__(
        self,
        path: Path,
        total: int,
        start: int,
        stop: int,
        mode: str,
        cache_mib: int,
        device: torch.device,
        *,
        row_threshold=256,
        cache_rows=False,
        prefetch_pages=0,
    ):
        if not 0 <= start < stop <= total:
            raise ValueError("Invalid TP Engram range")
        if path.stat().st_size != n_pages(total) * ALIGN:
            raise ValueError(f"Incorrect packed Engram file length: {path}")
        self.start, self.stop = start, stop
        self.device = device
        self.mode = mode
        if row_threshold < 0:
            raise ValueError("The packed row threshold must be nonnegative")
        self.row_threshold = row_threshold
        if prefetch_pages < 0:
            raise ValueError("The minimum page prefetch count must be nonnegative")
        self.prefetch_pages = prefetch_pages
        self.hinted = None
        self.hint_visits = 0
        self.first_page = start // ROWS_PER_PAGE
        self.io = self.handle = self.file = self.mapping = None
        self.cache: OrderedDict[int, int] = OrderedDict()
        self.capacity = min(
            cache_mib * (1 << 20) // ALIGN, n_pages(stop) - self.first_page
        )
        self.pages_read = self.page_hits = 0
        self.row_tags = None
        self.registered_host_ptr = None
        if mode == "ram":
            length = n_pages(stop) - self.first_page
            # Each production slice is just over 2 GiB. The caching pinned
            # allocator rounds that to 4 GiB. Register the exact page span.
            self.mapping = mmap.mmap(
                -1,
                length * ALIGN,
                flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            self.host = torch.frombuffer(self.mapping, dtype=torch.uint8).view(
                length, ALIGN
            )
            try:
                runtime = torch.cuda.cudart()
                ptr = self.host.data_ptr()
                status = runtime.cudaHostRegister(ptr, self.host.nbytes, 2)
                if status.value:
                    raise RuntimeError(
                        f"Registering {self.host.nbytes} Engram bytes failed: {status}"
                    )
                self.registered_host_ptr = ptr
                if not self.host.is_pinned():
                    raise RuntimeError(
                        "HIP did not expose the registered Engram pages as pinned"
                    )
                with (
                    path.open("rb", buffering=0) as source,
                    memoryview(self.mapping) as data,
                ):
                    start_offset = self.first_page * ALIGN
                    source.seek(start_offset)
                    for pos in range(0, len(data), 8 << 20):
                        span = min(8 << 20, len(data) - pos)
                        dest = pos
                        while dest < pos + span:
                            count = source.readinto(data[dest : pos + span])
                            if not count:
                                raise OSError(f"Short Engram table: {path}")
                            dest += count
                        os.posix_fadvise(
                            source.fileno(),
                            start_offset + pos,
                            span,
                            os.POSIX_FADV_DONTNEED,
                        )
                self.pages = get_accelerator_view_from_cpu_tensor(self.host)
            except BaseException:
                self.close()
                raise
            return
        if self.capacity < 1:
            raise ValueError("The Engram GPU page cache must hold at least one page")
        self.pages = torch.empty(
            (self.capacity, ALIGN), device=device, dtype=torch.uint8
        )
        self.batch = min(self.capacity, 256)
        self.staging = self.pages.new_empty((self.batch, ALIGN))
        if mode == "direct":
            self.handle = RegisteredFile(str(path), direct=True)
            try:
                self.io = DirectStorage(device, self.batch * ALIGN)
            except BaseException:
                self.handle.close()
                raise
        elif mode == "bcache":
            self.file = path.open("rb", buffering=0)
            if self.prefetch_pages and hasattr(os, "posix_fadvise"):
                self.hinted = np.zeros(n_pages(stop) - self.first_page, dtype=bool)
            self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            if hasattr(self.mapping, "madvise"):
                self.mapping.madvise(mmap.MADV_RANDOM)
            self.disk_pages = np.ndarray(
                (n_pages(total), ALIGN), np.uint8, buffer=self.mapping
            )
            self.disk_rows = np.ndarray(
                (n_pages(total), ROWS_PER_PAGE, ROW_BYTES),
                np.uint8,
                buffer=self.mapping,
                strides=(ALIGN, ROW_BYTES, 1),
            )
            self.host = torch.empty_like(self.staging, device="cpu", pin_memory=True)
            if cache_rows:
                # Reserve a tail for simultaneous rows that hash to the same slot.
                available = self.pages.numel() // ROW_BYTES
                reserve = max(1, min(2048, available // 8))
                persistent = 1 << ((available - reserve).bit_length() - 1)
                self.row_tags = np.full(persistent, -1, dtype=np.int64)
                self.row_cache = self.pages.view(-1)[: available * ROW_BYTES].view(
                    available, ROW_BYTES
                )
        else:
            raise ValueError(f"Unknown packed Engram storage: {mode}")

    def close(self):
        self.disk_rows = self.disk_pages = None
        if self.mode == "ram":
            if self.registered_host_ptr is not None:
                torch.cuda.synchronize(self.device)
                status = torch.cuda.cudart().cudaHostUnregister(
                    self.registered_host_ptr
                )
                if status.value:
                    raise RuntimeError(
                        f"Releasing registered Engram pages failed: {status}"
                    )
                self.registered_host_ptr = None
            self.pages = self.host = None
        for item in (self.io, self.handle, self.mapping, self.file):
            if item is not None:
                item.close()
        self.io = self.handle = self.mapping = self.file = None

    def _page_slots(self, pages: np.ndarray) -> np.ndarray:
        hits = [int(page) for page in pages if int(page) in self.cache]
        for page in hits:
            self.cache.move_to_end(page)
        self.page_hits += len(hits)
        missing = [int(page) for page in pages if int(page) not in self.cache]
        slots = []
        for page in missing:
            if len(self.cache) == self.capacity:
                _, slot = self.cache.popitem(last=False)
            else:
                slot = len(self.cache)
            self.cache[page] = slot
            slots.append(slot)
        i = 0
        while i < len(missing):
            if self.mapping is not None:
                end = min(i + self.batch, len(missing))
                count = end - i
                # The virtual mapping is reclaimable. Batch scattered pages into
                # one pinned transfer instead of synchronizing the GPU per miss.
                np.take(
                    self.disk_pages,
                    missing[i:end],
                    axis=0,
                    mode="clip",
                    out=self.host[:count].numpy(),
                )
                self.staging[:count].copy_(self.host[:count])
            else:
                end = i + 1
                while (
                    end < len(missing)
                    and end - i < self.batch
                    and missing[end] == missing[end - 1] + 1
                ):
                    end += 1
                count = end - i
                self.io.read(self.handle, missing[i] * ALIGN, self.staging[:count])
            dest = torch.tensor(slots[i:end], dtype=torch.long, device=self.device)
            self.pages.index_copy_(0, dest, self.staging[:count])
            self.pages_read += count
            i = end
        return np.fromiter(
            (self.cache[int(p)] for p in pages), dtype=np.int64, count=len(pages)
        )

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        if self.mode == "ram":
            return lookup_q4_pages(
                self.pages, ids, self.first_page, self.start, self.stop
            )
        out = torch.empty((*ids.shape, WIDTH), dtype=torch.float16, device=ids.device)
        self._disk_lookup(ids, out)
        return out

    @eager_break_during_capture
    def _disk_lookup(self, ids: torch.Tensor, out: torch.Tensor):
        """Resolve row IDs and page offsets on the host for disk lookups."""
        self.lookup_host(ids.detach().reshape(-1).cpu().numpy(), out)

    def _advise_pages(self, pages):
        """Queue first visits together so synchronous row faults can overlap."""
        if self.hinted is None or len(pages) < self.prefetch_pages:
            return
        pages = np.unique(pages)
        local = pages - self.first_page
        self.hint_visits += len(pages)
        if self.hint_visits >= 2 * len(self.hinted):
            self.hinted.fill(False)
            self.hint_visits = 0
        new = ~self.hinted[local]
        self.hinted[local[new]] = True
        try:
            for page in pages[new]:
                os.posix_fadvise(
                    self.file.fileno(), int(page) * ALIGN, ALIGN, os.POSIX_FADV_WILLNEED
                )
        except OSError as err:
            if err.errno not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
                raise
            self.hinted = None

    def _plan_cached_rows(self, numbers, live):
        ids, inverse = np.unique(numbers[live], return_inverse=True)
        slots = (ids & (len(self.row_tags) - 1)).astype(np.int64)
        _, first = np.unique(slots, return_index=True)
        persistent = np.zeros(len(ids), dtype=bool)
        persistent[first] = True
        conflicts = len(ids) - len(first)
        if conflicts > len(self.row_cache) - len(self.row_tags):
            return None
        hit = persistent & (self.row_tags[slots] == ids)
        slots[~persistent] = len(self.row_tags) + np.arange(conflicts)
        missing = ids[~hit]
        pages, offsets = np.divmod(missing, ROWS_PER_PAGE)
        self._advise_pages(pages)
        packed = self.disk_rows[pages, offsets]
        index = np.full(len(numbers), -1, dtype=np.int32)
        index[live] = slots[inverse]
        return RowCachePlan(index, slots[~hit], packed, missing)

    def plan_rows(self, numbers: np.ndarray, *, readahead=False):
        numbers = numbers.reshape(-1)
        if self.row_tags is not None:
            live = (numbers >= self.start) & (numbers < self.stop)
            plan = self._plan_cached_rows(numbers, live)
            if plan is not None:
                return plan
        if self.row_tags is not None or (
            self.mapping is not None
            and self.row_threshold
            and len(numbers) >= self.row_threshold
        ):
            # Large batches rarely reuse whole 4 KiB pages. Gather the 54-byte
            # payloads in request order on the filesystem's page cache.
            live = (numbers >= self.start) & (numbers < self.stop)
            safe = np.where(live, numbers, self.start)
            pages, offsets = np.divmod(safe, ROWS_PER_PAGE)
            self._advise_pages(pages)
            rows = self.disk_rows[pages, offsets]
            rows[~live] = 0
            return rows
        valid = np.flatnonzero((numbers >= self.start) & (numbers < self.stop))
        order = valid[np.argsort(numbers[valid], kind="stable")]
        ordered = numbers[order]
        pages, starts = np.unique(ordered // ROWS_PER_PAGE, return_index=True)
        if readahead and self.mapping is not None:
            # Fault missing filesystem pages concurrently across Engram layers.
            # One byte per page suffices; the Q4 payload stays in the mapping.
            missing = [int(p) for p in pages if int(p) not in self.cache]
            if missing:
                self._advise_pages(missing)
                self.disk_pages[missing, 0]
        return order, ordered, pages, starts

    def lookup_host(self, numbers: np.ndarray, out: torch.Tensor):
        self.lookup_plan(self.plan_rows(numbers), out)

    def _row_staging(self):
        capacity = self.staging.numel() // ROW_BYTES
        host = self.host.view(-1)[: capacity * ROW_BYTES].view(capacity, ROW_BYTES)
        stage = self.staging.view(-1)[: capacity * ROW_BYTES].view_as(host)
        return host, stage

    def lookup_plan(self, plan, out: torch.Tensor):
        if isinstance(plan, RowCachePlan):
            host, stage = self._row_staging()
            try:
                for start in range(0, len(plan.rows), len(host)):
                    rows = plan.rows[start : start + len(host)]
                    n = len(rows)
                    host[:n].copy_(torch.from_numpy(rows))
                    stage[:n].copy_(host[:n])
                    slots = torch.as_tensor(
                        plan.slots[start : start + n], device=self.device
                    )
                    self.row_cache.index_copy_(0, slots, stage[:n])
                indices = torch.as_tensor(plan.indices, device=self.device)
                decompress_q4_rows(self.row_cache, out.view(-1, WIDTH), indices)
                keep = plan.slots < len(self.row_tags)
                self.row_tags[plan.slots[keep]] = plan.tags[keep]
            except Exception:
                self.row_tags.fill(-1)
                raise
            return
        if isinstance(plan, np.ndarray):
            host, stage = self._row_staging()
            capacity = len(host)
            output = out.view(-1, WIDTH)
            for start in range(0, len(plan), capacity):
                batch = plan[start : start + capacity]
                count = len(batch)
                np.copyto(host[:count].numpy(), batch)
                stage[:count].copy_(host[:count])
                decompress_q4_rows(stage[:count], output[start : start + count])
            return
        order, ordered, pages, starts = plan
        out.zero_()
        for i in range(0, len(pages), self.capacity):
            chunk = pages[i : i + self.capacity]
            end = i + len(chunk)
            rows = slice(starts[i], starts[end] if end < len(pages) else len(order))
            try:
                slots = self._page_slots(chunk)
            except Exception:
                self.cache.clear()
                raise
            indexes = ordered[rows]
            slot_ids = slots[np.searchsorted(chunk, indexes // ROWS_PER_PAGE)]

            def gpu_index(values):
                return torch.as_tensor(values, dtype=torch.long, device=out.device)

            physical_rows = gpu_index(
                slot_ids * ROWS_PER_PAGE + indexes % ROWS_PER_PAGE
            )
            lookup_q4_pages(
                self.pages,
                physical_rows,
                0,
                0,
                self.capacity * ROWS_PER_PAGE,
                out=out,
                destinations=gpu_index(order[rows]),
            )
