#!/usr/bin/env python3
"""Probe: int32-wide packed-KV loads in Gluon vs int8 scalar loads.

Loads the K tile [32 keys, 256 dims] as int32 lanes [32, 64] through a
reinterpreted pointer and dumps the AMDGCN load widths. If int32 loads
vectorize to dwordx2/dwordx4 while int8 stays ubyte, the K/V path gets
a 4-16x load-instruction reduction.
"""
import os
import re
import sys

import torch
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

sys.path.insert(0, "/home/curved/vllm-gfx908")
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_wide"


@g.jit
def _load_i8(K, OUT):
    kv_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[4, 16],
        warps_per_cta=[4, 1], order=[1, 0])
    tn = gl.arange(0, 32, layout=gl.SliceLayout(1, kv_layout))
    kd = gl.arange(0, 256, layout=gl.SliceLayout(0, kv_layout))
    t = gl.load(K + tn[:, None] * 520 + kd[None, :])
    acc = gl.sum(t.to(gl.int32), axis=0, keep_dims=True)
    gl.store(OUT + (st := 0) + 0, gl.reshape(acc, (1,)))


@g.jit
def _load_i32(K32, OUT):
    kv_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[4, 16],
        warps_per_cta=[4, 1], order=[1, 0])
    tn = gl.arange(0, 32, layout=gl.SliceLayout(1, kv_layout))
    kd = gl.arange(0, 64, layout=gl.SliceLayout(0, kv_layout))
    t = gl.load(K32 + tn[:, None] * 130 + kd[None, :])
    s1 = gl.sum(t, axis=0)
    st = gl.arange(0, 1, layout=gl.SliceLayout(1, kv_layout))
    gl.store(OUT + st, s1)


def main():
    dev = "cuda"
    k = torch.randint(-127, 127, (32, 520), device=dev, dtype=torch.int8)
    k32 = k.view(torch.int32)  # [32, 130]
    out = torch.zeros(4, device=dev, dtype=torch.int32)
    _load_i8[(1,)](k, out)
    _load_i32[(1,)](k32, out)
    torch.cuda.synchronize()
    for name in ("_load_i8", "_load_i32"):
        fn = {"_load_i8": _load_i8, "_load_i32": _load_i32}[name]
        # find the newest cache dirs for each
        base = "/tmp/triton_wide"
        hits = []
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".amdgcn"):
                    hits.append(os.path.join(root, f))
        # map by kernel name
        for h in hits:
            src = open(h, errors="ignore").read()
            if name.strip("_") in h or name.strip("_") in src[:400]:
                counts = {}
                for m in re.finditer(
                    r"(global_load_[a-z0-9_]+)", src
                ):
                    counts[m.group(1)] = counts.get(m.group(1), 0) + 1
                if counts:
                    print(name, dict(sorted(counts.items(), key=lambda kv: -kv[1])))
                    break


if __name__ == "__main__":
    main()
