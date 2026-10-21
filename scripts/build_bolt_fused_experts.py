#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build fused per-layer routed-expert blocks from an AIS repack cache."""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm.fs_gpu.fused_experts import build_fused_cache  # noqa: E402
from vllm.fs_gpu.repack import RepackCache, default_repack_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.model_dir, "*.safetensors")))
    if not files:
        raise SystemExit(f"no safetensors found under {args.model_dir}")
    root = default_repack_root(args.model_dir)
    for rank in args.ranks:
        cache = RepackCache(root, args.tp_size, rank)
        if not cache.validate(files):
            raise SystemExit(f"repack cache invalid for {args.model_dir} rank{rank}")
        build_fused_cache(cache, files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
