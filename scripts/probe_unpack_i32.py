#!/usr/bin/env python3
"""Solve the int32->4xint8 register unpack for wide-KV loads (GOALOPT).

Kernel A loads a [32,256] int8 tile via 64 int32 lanes and unpacks each
i32 into 4 int8 through shift-and-mask arithmetic, producing a
[32,256] int8 tensor that is stored back so all lanes stay live.
Correctness is checked against the direct int8 load; the AMDGCN is
inspected for both the load and the unpack cost shape.
"""
import os
import re
import sys

import torch
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

sys.path.insert(0, "/home/curved/vllm-gfx908")
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_unpack"


@g.jit
def unpack_kernel(K32, OUT, OUT8):
    kv_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[4, 16],
        warps_per_cta=[4, 1], order=[1, 0])
    tn = gl.arange(0, 32, layout=gl.SliceLayout(1, kv_layout))
    kq = gl.arange(0, 64, layout=gl.SliceLayout(0, kv_layout))
    w = gl.load(K32 + tn[:, None] * 130 + kq[None, :])  # [32, 64] i32
    # Unpack: each i32 lane -> 4 int8 columns. Build the [32, 256] view
    # by re-mapping the column index: col c maps to lane c//4, byte c%4.
    cols = gl.arange(0, 256, layout=gl.SliceLayout(0, kv_layout))
    # gather the word for each column via modulo arithmetic on registers
    # (the [32,256] expansion of the [32,64] word tensor)
    word_idx = cols // 4
    byte_idx = cols % 4
    # word gather: since word_idx spans 0..63 and w holds 64 lanes, use
    # shift-free math: expand w to [32, 256] by multiplying with an
    # one-hot gather is not available; instead compute the byte directly
    # from w by re-loading per column group is what we are avoiding, so
    # use reshape trick: w [32,64] -> repeat each lane 4x along axis 1.
    # Gluon lacks register repeat; do the unpack on a [32,64,4] virtual
    # via strided math on the ORIGINAL load with per-byte shifts:
    b0 = (w & 0xFF).to(gl.int8)
    b1 = ((w >> 8) & 0xFF).to(gl.int16).to(gl.int8)
    b2 = ((w >> 16) & 0xFF).to(gl.int16).to(gl.int8)
    b3 = ((w >> 24) & 0xFF).to(gl.int16).to(gl.int8)
    # Store the four byte-planes so every lane is live and verifiable.
    st_n = gl.arange(0, 32, layout=gl.SliceLayout(1, kv_layout))
    st_q = gl.arange(0, 64, layout=gl.SliceLayout(0, kv_layout))
    gl.store(OUT8 + st_n[:, None] * 256 + st_q[None, :], b0)
    gl.store(OUT8 + st_n[:, None] * 256 + 64 + st_q[None, :], b1)
    gl.store(OUT8 + st_n[:, None] * 256 + 128 + st_q[None, :], b2)
    gl.store(OUT8 + st_n[:, None] * 256 + 192 + st_q[None, :], b3)
    gl.store(OUT + st_n[:, None] * 64 + st_q[None, :], w)


def main():
    dev = "cuda"
    torch.manual_seed(0)
    k = torch.randint(-128, 127, (32, 520), device=dev, dtype=torch.int8)
    k32 = k.view(torch.int32)
    out8 = torch.zeros(32, 256, device=dev, dtype=torch.int8)
    out32 = torch.zeros(32, 64, device=dev, dtype=torch.int32)
    unpack_kernel[(1,)](k32, out32, out8)
    torch.cuda.synchronize()
    ref = k[:, :256].contiguous()
    got = out8
    ok = torch.equal(ref, got)
    print("unpack byte-planes == reference:", ok)
    if not ok:
        d = (ref.int() - got.int()).abs()
        print("max_abs:", d.max().item(), "mismatches:",
              int((d > 0).sum()), "/", d.numel())
        print("words match:", torch.equal(out32, k32[:, :64]))
        for r, c in (d > 0).nonzero()[:4].tolist():
            print(r, c, "ref", int(ref[r, c]), "got", int(got[r, c]),
                  "word", hex(int(out32[r, c // 4]) & 0xFFFFFFFF))
    import glob
    for h in sorted(glob.glob("/tmp/triton_unpack/*/*.amdgcn"),
                    key=os.path.getmtime):
        src = open(h, errors="ignore").read()
        counts = {}
        for m in re.finditer(r"(global_load_[a-z0-9_]+)", src):
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
        print(os.path.basename(os.path.dirname(h))[:8],
              dict(sorted(counts.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
