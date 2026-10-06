#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-identity gate for AIS direct-to-GPU weight loading.

Verifies the exact production read path (vllm.fs_gpu.io.read_into_tensor
over the parsed safetensors index) against the on-disk bytes of the real
checkpoints, tensor by tensor, in <=128 MiB slices so it can run beside a
live server. No engine boot, no model residency required.

Usage:
  .venv/bin/python scripts/fs_gpu_weightcheck.py [--device cuda:0] \
      [MODEL_DIR ...]
(defaults to the recipe target + draft checkpoints)
"""

from __future__ import annotations

import argparse
import glob
import mmap
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_REPO, "vllm")):
    sys.path.insert(0, _REPO)  # repo checkout must shadow the site-packages wheel

import torch

DEFAULT_MODELS = [
    "/home/curved/models/Qwen3.8-27B-PTQR-R10S60",
    "/home/curved/models/dflash2-ptqr-r1",
]

SLICE = 128 << 20


def natural_key(p: str):
    import re

    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p)]


def check_model(model_dir: str, device: str) -> tuple[bool, int, int, float]:
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")), key=natural_key)
    if not files:
        print(f"[weightcheck] {model_dir}: no safetensors found")
        return False, 0, 0, 0.0

    from vllm.fs_gpu.io import read_into_tensor
    from vllm.fs_gpu.region import GpuStagingPool, RegisteredFile
    from vllm.fs_gpu.safetensors import SafetensorFile

    pool = GpuStagingPool(64 << 20, device=device)
    buf = torch.empty(SLICE, dtype=torch.uint8, device=device)
    checked = mismatches = 0
    total_bytes = 0
    t0 = time.monotonic()
    ok_all = True
    try:
        for path in files:
            index = SafetensorFile(path)
            ref = open(path, "rb")
            ref_map = mmap.mmap(ref.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                with RegisteredFile(path) as rf:
                    for spec in index:
                        base = index.data_begin + spec.start
                        off = 0
                        while off < spec.nbytes:
                            n = min(SLICE, spec.nbytes - off)
                            read_into_tensor(rf, pool, buf[:n], base + off)
                            got = buf[:n].cpu().numpy().tobytes()
                            want = ref_map[base + off : base + off + n]
                            checked += 1
                            total_bytes += n
                            if got != want:
                                mismatches += 1
                                ok_all = False
                                if mismatches <= 5:
                                    print(
                                        f"[weightcheck] MISMATCH {path} "
                                        f"{spec.name} @{base + off}+{n}"
                                    )
                            off += n
            finally:
                ref_map.close()
                ref.close()
            print(
                f"[weightcheck] {os.path.basename(path)}: done "
                f"({len(index)} tensors)"
            )
    finally:
        pool.close()
    return ok_all, checked, total_bytes, time.monotonic() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    args = ap.parse_args()
    models = args.models or DEFAULT_MODELS

    torch.cuda.init()
    torch.zeros(1024, device=args.device)

    ok_all = True
    grand_bytes = 0
    for m in models:
        print(f"[weightcheck] model: {m}")
        ok, chunks, nbytes, dt = check_model(m, args.device)
        grand_bytes += nbytes
        print(
            f"[weightcheck] {m}: {'PASS' if ok else 'FAIL'} — {chunks} chunk(s), "
            f"{nbytes / (1 << 30):.2f} GiB in {dt:.1f}s "
            f"({nbytes / (1 << 30) / dt:.2f} GiB/s)"
        )
        ok_all = ok_all and ok

    print(
        f"[weightcheck] {'PASS' if ok_all else 'FAIL'} — total "
        f"{grand_bytes / (1 << 30):.2f} GiB verified byte-identical"
    )
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
