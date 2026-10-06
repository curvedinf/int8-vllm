#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit gate for the fs_gpu read-plan recorder.

Mimics the fork's weight_loader consumer patterns (parameter.py /
linear.py: narrow on dim 0, dim 1, merged shards, packed dims) against
recorder tensors and verifies the planned byte ranges exactly cover the
consumed elements — no more, no less. Also verifies the two taint paths
(non-view op, storage-breaking view op) degrade to full-tensor fallback.

Usage: .venv/bin/python scripts/fs_gpu_plan_test.py
"""

from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_REPO, "vllm")):
    sys.path.insert(0, _REPO)

import torch

from vllm.fs_gpu.recorder import ReadPlan, with_active_plan

FAILURES: list[str] = []


def run_case(name: str, shape, dtype, consume) -> None:
    plan = ReadPlan()
    rec = plan.new_recorder(name, tuple(shape), dtype, device="cpu")
    with with_active_plan(plan):
        consume(rec)
    entry = plan.get(name)
    assert entry is not None

    # ground truth: replay the consumption on a range tensor
    marker = torch.arange(rec.numel(), dtype=torch.int64).reshape(rec.shape)

    def replay(t):
        out = consume(t)
        if isinstance(out, (tuple, list)):
            return out
        return (out,)

    parts = replay(marker)

    if entry.full:
        FAILURES.append(f"{name}: unexpected full-fallback")
        print(f"[plan-test] {name}: UNEXPECTED FULL FALLBACK")
        return
    want_elems: set[int] = set()
    for p in parts:
        off = p.storage_offset()
        strides = list(p.stride())
        for multi in torch.cartesian_prod(
            *[torch.arange(s) for s in p.shape]
        ).tolist():
            el = off + sum(c * st for c, st in zip(multi, strides))
            want_elems.add(int(el))
    got_elems: set[int] = set()
    for bo, n in entry.ranges:
        es = rec.element_size()
        got_elems.update(range(bo // es, (bo + n) // es))
    if got_elems == want_elems:
        print(
            f"[plan-test] {name}: OK ({len(entry.ranges)} ranges, "
            f"{len(got_elems)} elements)"
        )
    else:
        FAILURES.append(
            f"{name}: range mismatch (missing {len(want_elems - got_elems)} "
            f"elements, extra {len(got_elems - want_elems)})"
        )
        print(
            f"[plan-test] {name}: MISMATCH missing="
            f"{len(want_elems - got_elems)} extra={len(got_elems - want_elems)}"
        )


def main() -> int:
    def copied(fn):
        """Wrap a slice fn: narrow views then copy into a real param."""

        def wrap(t):
            out = fn(t)
            if isinstance(out, (tuple, list)):
                for v in out:
                    torch.empty(v.shape, dtype=t.dtype).copy_(v)
            else:
                torch.empty(out.shape, dtype=t.dtype).copy_(out)
            return out

        return wrap

    # dim-0 shard (row-parallel: o_proj/down_proj qweight)
    run_case(
        "row_shard",
        (128, 64),
        torch.int32,
        copied(lambda t: t.narrow(0, 2 * 32, 32)),
    )
    # dim-1 shard (column-parallel qkv/gate_up) — strided
    run_case(
        "col_shard",
        (128, 64),
        torch.int32,
        copied(lambda t: t.narrow(1, 1 * 16, 16)),
    )
    # merged column shard with explicit shard offset (linear.py pattern)
    def merged(t):
        parts = [t.narrow(1, off, 16) for off in (16, 48)]
        return tuple(parts)

    run_case("merged_shards", (96, 64), torch.int32, copied(merged))
    # packed qweight (K//4, N) narrow on packed dim 0
    run_case(
        "packed_dim0",
        (256, 96),
        torch.int32,
        copied(lambda t: t.narrow(0, 64, 64)),
    )
    # 3-D: (R, H, D) narrow on dim 1 (heads shard)
    run_case(
        "head_shard",
        (16, 8, 128),
        torch.bfloat16,
        copied(lambda t: t.narrow(1, 4, 2)),
    )
    # full copy (replicated: norms)
    run_case("replicated", (33, 7), torch.float16, copied(lambda t: t))
    # transpose then narrow (view ops only)
    run_case(
        "transposed",
        (64, 32),
        torch.float16,
        copied(lambda t: t.t().narrow(0, 8, 8)),
    )

    # taint path: non-view op (arithmetic) -> full fallback
    plan = ReadPlan()
    rec = plan.new_recorder("tainted_math", (16, 16), torch.float32, device="cpu")
    with with_active_plan(plan):
        _ = rec * 2
    e = plan.get("tainted_math")
    ok = e.full
    print(f"[plan-test] tainted_math -> full fallback: {'OK' if ok else 'FAIL'}")
    if not ok:
        FAILURES.append("taint via non-view op not detected")

    # taint path: reshape-copy of a non-contiguous view (new storage)
    plan = ReadPlan()
    rec = plan.new_recorder("tainted_reshape", (16, 16), torch.float32, device="cpu")
    with with_active_plan(plan):
        _ = rec.narrow(1, 4, 8).reshape(-1)
    e = plan.get("tainted_reshape")
    ok = e.full
    print(f"[plan-test] tainted_reshape -> full fallback: {'OK' if ok else 'FAIL'}")
    if not ok:
        FAILURES.append("taint via storage-breaking view op not detected")

    # no-op copy_ leaves dst untouched (data never flows in plan pass)
    plan = ReadPlan()
    rec = plan.new_recorder("noop_copy", (8, 8), torch.float32, device="cpu")
    dst = torch.full((4, 8), 7.0)
    with with_active_plan(plan):
        dst.copy_(rec.narrow(0, 2, 4))
    ok = bool((dst == 7.0).all()) and plan.get("noop_copy").copies == 1
    print(f"[plan-test] noop_copy: {'OK' if ok else 'FAIL'}")
    if not ok:
        FAILURES.append("plan-pass copy_ was not a no-op")

    if FAILURES:
        print("[plan-test] FAIL:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("[plan-test] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
