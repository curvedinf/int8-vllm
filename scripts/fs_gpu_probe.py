#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AIS (hipFile) hardware probe for the gfx908 serving stack.

Phase-0 gate for the FS->GPU layer: verifies in-process coexistence with
the (patched) torch HIP runtime, then exercises the exact vllm.fs_gpu read
and write paths at aligned and unaligned offsets against CPU references,
and reports streaming read bandwidth. Exits non-zero on any failure.

Usage (idle-ish GPU; needs <100 MiB VRAM):
  .venv/bin/python scripts/fs_gpu_probe.py [--device cuda:0] [--scratch DIR]
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import tempfile
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_REPO, "vllm")):
    sys.path.insert(0, _REPO)  # repo checkout must shadow the site-packages wheel

import torch

SCRATCH_SIZE = 256 << 20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--scratch", default="/home/curved/models")
    args = ap.parse_args()

    failures: list[str] = []

    torch.cuda.init()
    torch.zeros(1024, device=args.device)  # touch the context first
    print(f"[probe] torch {torch.__version__}, device {args.device}")

    from vllm.fs_gpu import get_ais, verify_single_hip_runtime
    from vllm.fs_gpu.io import read_into_tensor, write_from_tensor
    from vllm.fs_gpu.region import GpuStagingPool, RegisteredFile

    ais = get_ais()
    print(f"[probe] libhipfile {ais.path} version {'.'.join(map(str, ais.version))}")

    maps = verify_single_hip_runtime()
    print(f"[probe] libamdhip64 mappings in-process: {maps}")
    if len(maps) != 1:
        failures.append(f"expected exactly one libamdhip64 mapping, got {maps}")

    scratch = os.path.join(args.scratch, f".fs_gpu_probe_{os.getpid()}.bin")
    pattern = secrets.token_bytes(SCRATCH_SIZE)
    with open(scratch, "wb") as f:
        f.write(pattern)
        f.flush()
        os.fsync(f.fileno())

    pool = GpuStagingPool(64 << 20, device=args.device)
    print(f"[probe] staging pool: base=0x{pool.base:x} capacity={pool.capacity}")
    try:
        with RegisteredFile(scratch, write=True) as rf:
            # 1. aligned read
            t = torch.empty(32 << 20, dtype=torch.uint8, device=args.device)
            t0 = time.monotonic()
            read_into_tensor(rf, pool, t, 0)
            dt = time.monotonic() - t0
            ok = t.cpu().numpy().tobytes() == pattern[: 32 << 20]
            print(f"[probe] aligned read 32MiB: {'OK' if ok else 'MISMATCH'} ({dt:.3f}s)")
            if not ok:
                failures.append("aligned read mismatch")

            # 2. unaligned offsets (8-byte safetensors-style + odd offsets)
            for off in (8, 8100, 4096 + 137):
                n = 1 << 20
                read_into_tensor(rf, pool, t[:n], off)
                ok = t[:n].cpu().numpy().tobytes() == pattern[off : off + n]
                print(f"[probe] unaligned read @{off}: {'OK' if ok else 'MISMATCH'}")
                if not ok:
                    failures.append(f"unaligned read @{off} mismatch")

            # 3. write roundtrip (aligned extent)
            wpat = secrets.token_bytes(16 << 20)
            src = torch.frombuffer(bytearray(wpat), dtype=torch.uint8).to(
                args.device
            )
            write_from_tensor(rf, pool, src, SCRATCH_SIZE)  # past the pattern
            read_into_tensor(rf, pool, t[: 16 << 20], SCRATCH_SIZE)
            ok = t[: 16 << 20].cpu().numpy().tobytes() == wpat
            print(f"[probe] write roundtrip 16MiB: {'OK' if ok else 'MISMATCH'}")
            if not ok:
                failures.append("write roundtrip mismatch")

            # 4. streaming read bandwidth (4x256MiB)
            total = 0
            t0 = time.monotonic()
            for _ in range(4):
                off = 0
                while off < SCRATCH_SIZE:
                    n = min(32 << 20, SCRATCH_SIZE - off)
                    read_into_tensor(rf, pool, t[:n], off)
                    off += n
                    total += n
            dt = time.monotonic() - t0
            print(
                f"[probe] streaming read: {total / (1 << 30):.2f} GiB in {dt:.2f}s "
                f"= {total / (1 << 30) / dt:.2f} GiB/s"
            )
    finally:
        pool.close()
        os.unlink(scratch)

    if failures:
        print("[probe] FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[probe] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
