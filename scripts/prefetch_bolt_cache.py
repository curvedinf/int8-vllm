#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Warm the page cache with Bolt's per-rank weight caches before workers load.

bcache on this host only retains a fraction of the working set, so the workers
would otherwise stream their repack/fused files from the HDD at ~50 MiB/s.
This prefetcher reads the same byte ranges with buffered ``preadv``, which
lands them in the page cache; AIS then reads them at >1 GiB/s.

The repack tensors are read first, then the fused-expert files, matching the
workers' load order. The workers wait for the sentinel written at the end
(``--done-file``), so no byte is read twice. At the measured ~170-260 MiB/s
the full ~30 GiB working set needs 2-3 minutes; the budget only bounds a
partial read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm.fs_gpu.repack import default_repack_root  # noqa: E402

CHUNK = 2 << 20
SKIP = ".mixer.experts."
DEFAULT_DONE = "/dev/shm/vllm_bolt_prefetch.done"


def non_expert_ranges(meta_path: str) -> list[tuple[int, int]]:
    meta = json.load(open(meta_path))
    ranges = [
        (int(t["block_off"]), int(t["block_len"]))
        for t in meta["tensors"]
        if SKIP not in t["name"] and int(t["block_len"]) > 0
    ]
    merged: list[tuple[int, int]] = []
    for off, length in sorted(ranges):
        if merged and off <= merged[-1][0] + merged[-1][1]:
            prev_off, prev_len = merged[-1]
            merged[-1] = (prev_off, max(prev_len, off + length - prev_off))
        else:
            merged.append((off, length))
    return merged


def read_range(fd: int, off: int, length: int, deadline: float) -> int:
    buf = bytearray(min(CHUNK, max(length, 1)))
    mv = memoryview(buf)
    done = 0
    while done < length:
        if time.monotonic() > deadline:
            break
        n = min(len(buf), length - done)
        got = os.preadv(fd, [mv[:n]], off + done)
        if got != n:
            raise OSError(f"short prefetch read {got}/{n}")
        done += n
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chunk-mb", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--done-file", default=os.environ.get("VLLM_BOLT_PREFETCH_DONE", DEFAULT_DONE))
    args = parser.parse_args()
    global CHUNK
    CHUNK = args.chunk_mb << 20

    try:
        os.unlink(args.done_file)
    except FileNotFoundError:
        pass

    root = default_repack_root(args.model_dir)
    per_rank: dict[int, list[tuple[int, int]]] = {}
    jobs: list[tuple[str, int, int]] = []
    for rank in args.ranks:
        meta_path = os.path.join(root, f"rank{rank}.meta.fast.json")
        if not os.path.exists(meta_path):
            print(f"no fast manifest for rank {rank}: {meta_path}", file=sys.stderr)
            return 1
        per_rank[rank] = non_expert_ranges(meta_path)
    # Round-robin by range index so every rank's early tensors are warm first.
    width = max(len(r) for r in per_rank.values())
    bins = {rank: os.path.join(root, f"rank{rank}.bin") for rank in args.ranks}
    for i in range(width):
        for rank in args.ranks:
            if i < len(per_rank[rank]):
                off, length = per_rank[rank][i]
                jobs.append((bins[rank], off, length))
    for rank in args.ranks:
        fused = os.path.join(root, f"rank{rank}.experts.bin")
        if os.path.exists(fused):
            jobs.append((fused, 0, os.path.getsize(fused)))

    deadline = time.monotonic() + args.seconds

    def fetch(job: tuple[str, int, int]) -> int:
        path, off, length = job
        fd = os.open(path, os.O_RDONLY)
        try:
            return read_range(fd, off, length, deadline)
        finally:
            os.close(fd)

    started = time.monotonic()
    status = "complete"
    try:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            total = sum(pool.map(fetch, jobs))
    except BaseException:
        status = "failed"
        raise
    finally:
        dt = time.monotonic() - started
        planned = sum(length for _, _, length in jobs)
        if status == "complete" and total < planned:
            status = "partial"
        with open(args.done_file, "w") as fh:
            fh.write(f"{status} {total} {dt:.1f}\n")
        print(
            f"prefetched {total / (1 << 30):.2f}/{planned / (1 << 30):.2f} GiB "
            f"in {dt:.1f}s = {total / (1 << 20) / max(dt, 1e-6):.1f} MiB/s "
            f"[{status}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
