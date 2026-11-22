#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Size a native host KV cache from available RAM and the runtime footprint."""

import argparse
import math
import sys

from scripts.warm_hy4_engram import available_ram_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-kv-gib", default="auto")
    parser.add_argument("--ram-reserve-gib", type=float, default=8)
    parser.add_argument("--runtime-gib", type=float, required=True)
    args = parser.parse_args()
    if not 0 <= args.ram_reserve_gib < math.inf:
        parser.error("RAM reserve must be finite and nonnegative")
    if not 0 <= args.runtime_gib < math.inf:
        parser.error("Runtime allowance must be finite and nonnegative")
    available_gib = available_ram_bytes() / 2**30
    budget = max(0, math.floor(available_gib - args.ram_reserve_gib - args.runtime_gib))
    if args.host_kv_gib == "auto":
        size = budget
    else:
        try:
            size = int(args.host_kv_gib)
        except ValueError:
            parser.error("Host KV size must be auto or a nonnegative integer in GiB")
        if not 0 <= size <= budget:
            parser.error(f"Host KV size must be between 0 and {budget} GiB")
    print(
        f"Host RAM available {available_gib:.2f} GiB, "
        f"runtime allowance {args.runtime_gib:g} GiB, "
        f"reserve {args.ram_reserve_gib:g} GiB, host KV {size} GiB",
        file=sys.stderr,
    )
    print(size)


if __name__ == "__main__":
    main()
