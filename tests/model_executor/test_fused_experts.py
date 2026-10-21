# SPDX-License-Identifier: Apache-2.0
"""Fused expert cache must reproduce the per-expert TP-narrowed bytes."""

from types import SimpleNamespace

import numpy as np

from vllm.fs_gpu.fused_experts import _expert_destination


def _w2_case(tp_size: int, hidden: int, intermediate: int):
    blocks = intermediate // 256
    row = blocks * 66
    full = np.arange(hidden * row, dtype=np.uint32).astype(np.uint8)
    shape = [hidden, blocks, 66]
    local = intermediate // tp_size
    assert local % 256 == 0
    width = local // 256
    for rank in range(tp_size):
        begin = rank * local
        first = begin // 256
        runs = []
        compact = bytearray()
        for r in range(hidden):
            off = r * row + first * 66
            runs.append((off, width * 66))
            compact += full[off : off + width * 66].tobytes()
        spec = SimpleNamespace(shape=shape, runs=runs)
        dest, n = _expert_destination(spec, bytes(compact), "w2", tp_size, rank)
        assert n == len(compact)
        assert dest == bytes(compact), f"rank {rank} w2 slice misaligned"


def _w13_case(tp_size: int, hidden_blocks: int, intermediate: int):
    row = hidden_blocks * 66
    full = np.arange(intermediate * row, dtype=np.uint32).astype(np.uint8)
    shape = [intermediate, hidden_blocks, 66]
    local = intermediate // tp_size
    for rank in range(tp_size):
        begin = rank * local
        runs = [(r * row, row) for r in range(begin, begin + local)]
        compact = b"".join(full[off : off + n].tobytes() for off, n in runs)
        spec = SimpleNamespace(shape=shape, runs=runs)
        dest, _ = _expert_destination(spec, compact, "w13", tp_size, rank)
        assert dest == compact, f"rank {rank} w13 slice misaligned"


def test_w2_uses_absolute_block_offset():
    _w2_case(tp_size=4, hidden=8, intermediate=1024)


def test_w2_unaligned_begin():
    _w2_case(tp_size=4, hidden=4, intermediate=2048)


def test_w13_tp_slice():
    _w13_case(tp_size=4, hidden_blocks=4, intermediate=1024)
