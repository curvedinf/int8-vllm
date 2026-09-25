#!/usr/bin/env python3
"""Probe Gluon int8 MFMA on gfx908 (production-tile literals)."""
import torch
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def _probe(A, B, C):
    a_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[4, 16],
        warps_per_cta=[4, 1], order=[1, 0])
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=1, instr_shape=[16, 16, 32],
        transposed=True, warps_per_cta=[4, 1])
    am = gl.arange(0, 64, layout=gl.SliceLayout(1, a_layout))
    ak = gl.arange(0, 128, layout=gl.SliceLayout(0, a_layout))
    a = gl.load(A + am[:, None] * 128 + ak[None, :])
    a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, k_width=4))
    bn = gl.arange(0, 32, layout=gl.SliceLayout(1, a_layout))
    bk = gl.arange(0, 128, layout=gl.SliceLayout(0, a_layout))
    b = gl.load(B + bn[:, None] * 128 + bk[None, :])  # [32, 128] keys
    b = gl.convert_layout(gl.permute(b, (1, 0)), gl.DotOperandLayout(1, mma, k_width=4))
    acc = gl.zeros([64, 32], dtype=gl.int32, layout=mma)
    acc = gl.amd.cdna3.mfma(a, b, acc)
    out_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 2], threads_per_warp=[16, 4],
        warps_per_cta=[4, 1], order=[1, 0])
    acc32 = gl.convert_layout(acc, out_layout)
    om = gl.arange(0, 64, layout=gl.SliceLayout(1, out_layout))
    on = gl.arange(0, 32, layout=gl.SliceLayout(0, out_layout))
    gl.store(C + om[:, None] * 32 + on[None, :], acc32)


if __name__ == "__main__":
    a = torch.randint(-127, 127, (64, 128), device="cuda", dtype=torch.int8)
    b = torch.randint(-127, 127, (32, 128), device="cuda", dtype=torch.int8)
    c = torch.empty(64, 32, device="cuda", dtype=torch.int32)
    try:
        _probe[(1,)](a, b, c)
        torch.cuda.synchronize()
        ref = a.int() @ b.int().t()
        exact = torch.equal(c, ref)
        print("int8 MFMA OK; exact:", exact,
              "| got", c[0, :3].tolist(), "ref", ref[0, :3].tolist())
    except Exception as e:
        print("PROBE FAIL:", type(e).__name__, str(e)[-600:])
