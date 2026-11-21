#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Populate bcache's SSD read cache for the packed HY4 Engram tables."""

import argparse
import math
import mmap
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

PAGE = 4096
CHUNK = 64 << 10
BAND = 2 << 20


@contextmanager
def direct_reader(path, size):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        with mmap.mmap(-1, size) as space, memoryview(space) as buffer:
            yield fd, buffer
    finally:
        os.close(fd)


def probe(path, seed=None):
    pages = path.stat().st_size // PAGE
    offsets = random.Random(seed).sample(range(pages), min(64, pages))
    times = []
    with direct_reader(path, PAGE) as (fd, buf):
        for index in offsets:
            start = time.monotonic_ns()
            if os.preadv(fd, [buf], index * PAGE) != PAGE:
                raise OSError(f"Short probe read from {path.name}")
            times.append((time.monotonic_ns() - start) / 1e6)
    return statistics.median(times), sorted(times)[int(0.9 * (len(times) - 1))]


def is_bcache(path):
    dev = path.stat().st_dev
    sysdev = Path("/sys/dev/block") / f"{os.major(dev)}:{os.minor(dev)}"
    return (sysdev / "bcache").is_dir()


def warm(path, force, probe_only, threshold):
    size = path.stat().st_size
    if not size or size % PAGE:
        raise ValueError(f"Expected a nonempty page-aligned table: {path.name}")
    median, p90 = probe(path)
    print(
        f"{path.name} initial random reads p50 {median:.2f} ms, p90 {p90:.2f} ms",
        flush=True,
    )
    if probe_only or (not force and p90 < threshold):
        return
    start = time.monotonic()
    # Reverse the order of short bands so a stream does not cross bcache's
    # sequential bypass threshold. Small independent reads can fill gaps
    # between hot pages in an extent that is only partly on the SSD.
    with direct_reader(path, CHUNK) as (fd, buffer):
        for band in reversed(range(0, size, BAND)):
            for offset in range(band, min(band + BAND, size), CHUNK):
                count = min(CHUNK, size - offset)
                if os.preadv(fd, [buffer[:count]], offset) != count:
                    raise OSError(f"Short table read from {path.name} at {offset}")
    duration = time.monotonic() - start
    median, p90 = probe(path)
    print(
        f"{path.name} read {size / 2**30:.2f} GiB in {duration:.1f} s, "
        f"independent probe p50 {median:.2f} ms, p90 {p90:.2f} ms",
        flush=True,
    )


def available_ram_bytes(proc=Path("/proc"), root=Path("/sys/fs/cgroup")):
    """Return reclaimable host memory, bounded by cgroup-v2 limits."""
    with (proc / "meminfo").open() as stream:
        available = next(
            int(line.split()[1]) * 1024
            for line in stream
            if line.startswith("MemAvailable:")
        )
    for line in (proc / "self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            group = root / line[3:].lstrip("/")
            break
    else:
        return available
    while group.is_relative_to(root):
        limit_path = group / "memory.max"
        if limit_path.is_file() and (limit := limit_path.read_text().strip()) != "max":
            current = int((group / "memory.current").read_text())
            stat = dict(
                line.split()
                for line in (group / "memory.stat").read_text().splitlines()
            )
            reclaimable = int(stat.get("inactive_file", 0))
            available = min(available, max(0, int(limit) - current + reclaimable))
        group = group.parent
    return available


def check_ram(directory, mode, reserve_gib, runtime_gib=0, kv_gib=0):
    available = available_ram_bytes()
    tables = []
    if mode == "ram":
        tables = list(directory.glob("engram-*.q4_0"))
        if not tables:
            raise ValueError(f"No packed Engram tables in {directory}")
    pinned = sum(p.stat().st_size for p in tables)
    budget = (available - pinned) / 2**30 - reserve_gib - runtime_gib
    kv_gib = max(0, math.floor(budget)) if kv_gib == "auto" else int(kv_gib)
    if kv_gib < 0 or kv_gib > budget:
        raise ValueError(
            f"HY4 has {available / 2**30:.2f} GiB RAM available. "
            f"Engram mode {mode} needs {pinned / 2**30:.2f} GiB of pinned tables "
            f"and the host reserve is {reserve_gib:g} GiB. "
            f"Runtime allowance is {runtime_gib} GiB and host KV is {kv_gib} GiB."
        )
    print(
        f"HY4 host RAM available {available / 2**30:.2f} GiB, "
        f"reserve {reserve_gib:g} GiB, pinned Engram {pinned / 2**30:.2f} GiB, "
        f"runtime allowance {runtime_gib} GiB, host KV {kv_gib} GiB",
        file=sys.stderr,
        flush=True,
    )
    return kv_gib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--auto", action="store_true", help="Skip filesystems outside bcache."
    )
    parser.add_argument(
        "--force", action="store_true", help="Scan already warm tables as well."
    )
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--threshold-ms", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--check-ram",
        choices=("bcache", "direct", "ram"),
        help="Check host headroom for the selected Engram mode and exit",
    )
    parser.add_argument("--ram-reserve-gib", type=float, default=8)
    parser.add_argument("--runtime-gib", type=int, default=0)
    parser.add_argument("--host-kv-gib", default="0")
    args = parser.parse_args()
    if not 0 <= args.ram_reserve_gib < float("inf"):
        parser.error("RAM reserve must be a finite, nonnegative number")
    if args.check_ram:
        try:
            if args.runtime_gib < 0:
                raise ValueError("Runtime allowance must be nonnegative")
            size = check_ram(
                args.directory, args.check_ram, args.ram_reserve_gib,
                args.runtime_gib, args.host_kv_gib,
            )
            print(size)
        except ValueError as exc:
            parser.error(str(exc))
        return
    if args.workers < 1 or args.threshold_ms <= 0:
        parser.error("Workers and the probe threshold must be positive")
    files = sorted(args.directory.glob("engram-*.q4_0"))
    if not files:
        parser.error(f"No packed Engram tables in {args.directory}")
    if args.auto and not is_bcache(files[0]):
        return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        calls = [
            pool.submit(warm, path, args.force, args.probe_only, args.threshold_ms)
            for path in files
        ]
        for call in calls:
            call.result()


if __name__ == "__main__":
    main()
