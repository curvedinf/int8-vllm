#!/usr/bin/env python3
"""Verify int32-lane packed-KV loads emit wide AMDGCN loads (GOALOPT).

Two minimal kernels store a per-thread loaded value to global memory so
nothing is optimized away; the AMDGCN is then inspected for the global
load widths of the int8-lane vs int32-lane variants.
"""
import os
import re
import sys

import torch
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

sys.path.insert(0, "/home/curved/vllm-gfx908")
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_wide2"

LAYOUT: gl.constexpr = None  # placeholder to satisfy linters; set in jit


@g.jit
def load_i8_kernel(K, OUT):
    kv_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[4, 16],
        warps_per_cta=[4, 1], order=[1, 0])
    tn = gl.arange(0, 32, layout=gl.SliceLayout(1, kv_layout))
    kd = gl.arange(0, 256, layout=gl.SliceLayout(0, kv_layout))
    t = gl.load(K + tn[:, None] * 520 + kd[None, :]).to(gl.int32)
    # per-lane sum over kd so each thread's loads are all live
    s = gl.sum(t, axis=1)
    st = gl.arange(0, 32, layout=gl.SliceLayout(1, kv_layout))
    gl.store(OUT + st, s)


@g.jit
def load_i32_kernel(K32, OUT):
    kv_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[4, 16],
        warps_per_cta=[4, 1], order=[1, 0])
    tn = gl.arange(0, 32, layout=gl.SliceLayout(1, kv_layout))
    kd = gl.arange(0, 64, layout=gl.SliceLayout(0, kv_layout))
    t = gl.load(K32 + tn[:, None] * 130 + kd[None, :])
    s = gl.sum(t, axis=1)
    st = gl.arange(0, 32, layout=gl.SliceLayout(1, kv_layout))
    gl.store(OUT + st, s)


def main():
    dev = "cuda"
    k = torch.randint(-127, 127, (32, 520), device=dev, dtype=torch.int8)
    k32 = k.view(torch.int32)
    out = torch.zeros(32, device=dev, dtype=torch.int32)
    load_i8_kernel[(1,)](k, out)
    load_i32_kernel[(1,)](k32, out)
    torch.cuda.synchronize()

    import glob
    hits = sorted(glob.glob("/tmp/triton_wide2/*/*.amdgcn"),
                  key=os.path.getmtime)
    for h in hits:
        src = open(h, errors="ignore").read()
        counts = {}
        for m in re.finditer(r"(global_load_[a-z0-9_]+)", src):
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
        head = open(h.replace(".amdgcn", ".json"), errors="ignore").read(0) or ""
        print(os.path.basename(os.path.dirname(h))[:8],
              dict(sorted(counts.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
