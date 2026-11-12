#!/usr/bin/env python3
"""Stage-2 gate: native small-M int8 GEMM vs the Triton GS128 blockscale op.

Builds the native kernel, preshuffles the weight into B-fragment order,
checks it against a float32 dequantized reference, then times it against the
production Triton call at the decode shapes (CUDA-graph replay, kernel time).

  HIP_VISIBLE_DEVICES=0 PYTHONPATH=$PWD:$PWD/../int8-aiter \
    .venv/bin/python scripts/bench_native_gemm.py
"""

import ctypes
import sys

import torch

sys.path.insert(0, ".")
from aiter.ops.int8_native import _build  # noqa: E402
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (  # noqa: E402
    gemm_a8w8_blockscale,
)
from vllm.model_executor.kernels.linear.mixed_precision.aiter_w8a16 import (  # noqa: E402
    _get_aiter_w8a8_config,
)

SITES = {"qkv": (1280, 2048), "o_proj": (2048, 1024),
         "shared gate_up": (2048, 2048), "draft qkv": (1536, 2048),
         "draft gate_up": (12288, 2048), "lm_head": (37536, 2048)}


def load_launcher():
    lib = ctypes.CDLL(str(_build("gemm_a8w8_m16_gfx908.hip")))
    fn = lib.launch_gemm_a8w8_m16
    fn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def preshuffle(w: torch.Tensor) -> torch.Tensor:
    N, K = w.shape
    w32 = w.view(torch.int32)  # [N, K/4]
    nt = torch.arange(N // 16, device=w.device)
    k16 = torch.arange(K // 16, device=w.device)
    lane = torch.arange(64, device=w.device)
    row, quad = lane % 16, lane // 16
    return w32[nt[:, None, None] * 16 + row[None, None, :],
               k16[None, :, None] * 4 + quad[None, None, :]].contiguous()


def dequant_ref(a, w, a_s, w_s):
    M, K = a.shape
    N = w.shape[0]
    G = K // 128
    ref = torch.zeros(M, N, dtype=torch.float32, device=a.device)
    for g in range(G):
        part = a[:, g * 128:(g + 1) * 128].float() @ \
            w[:, g * 128:(g + 1) * 128].float().T
        ref += part * a_s[:, g:g + 1] * w_s[:, g:g + 1].T
    return ref


def timed(fn, iters=30):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(2):
        g.replay()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def main():
    torch.manual_seed(0)
    fn = load_launcher()

    def call_native(a, wp, a_s, w_s, out, M, N, K, block_m=0, block_n=0,
                    ksplit=0, partial=None):
        return fn(a.data_ptr(), wp.data_ptr(), a_s.data_ptr(), w_s.data_ptr(),
                  out.data_ptr(),
                  partial.data_ptr() if partial is not None else 0,
                  M, N, K, block_m, block_n, ksplit,
                  torch.cuda.current_stream().cuda_stream)

    # correctness: several shapes including partial M/N tiles
    for M, N, K in ((8, 64, 256), (12, 80, 256), (20, 64, 384),
                    (32, 128, 256), (96, 1280, 2048)):
        a = torch.randint(-60, 61, (M, K), dtype=torch.int8, device="cuda")
        w = torch.randint(-60, 61, (N, K), dtype=torch.int8, device="cuda")
        a_s = torch.rand(M, K // 128, dtype=torch.float32, device="cuda") * 0.02
        w_s = torch.rand(N, K // 128, dtype=torch.float32, device="cuda") * 0.02
        wp = preshuffle(w)
        out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
        st = call_native(a, wp, a_s, w_s, out, M, N, K)
        torch.cuda.synchronize()
        ref = dequant_ref(a, w, a_s, w_s)
        err = (out.float() - ref).abs().max().item()
        rel = err / ref.abs().max().item()
        print(f"check M={M:3d} N={N:5d} K={K:4d} status={st} "
              f"max_abs={err:.3f} rel={rel:.2e}", flush=True)
        assert st == 0 and rel < 5e-3, "native GEMM mismatch"

    print(f"{'site':14s} {'M':>3s} {'native_us':>9s} {'triton_us':>9s} "
          f"{'gain':>7s}")
    for name, (N, K) in SITES.items():
        w = torch.randint(-60, 61, (N, K), dtype=torch.int8, device="cuda")
        w_s = torch.rand(N, K // 128, dtype=torch.float32, device="cuda") * 0.02
        wp = preshuffle(w)
        for M in (16, 32, 96):
            a = torch.randint(-60, 61, (M, K), dtype=torch.int8, device="cuda")
            a_s = torch.rand(M, K // 128, dtype=torch.float32, device="cuda") * 0.02
            out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
            cfg = _get_aiter_w8a8_config(M, N, K, 128)
            nat = timed(lambda: call_native(a, wp, a_s, w_s, out, M, N, K))
            tri = timed(lambda: gemm_a8w8_blockscale(
                a, w, a_s, w_s, dtype=torch.bfloat16, config=cfg))
            print(f"{name:14s} {M:3d} {nat:9.2f} {tri:9.2f} "
                  f"{(1 - nat / tri) * 100:6.1f}%", flush=True)


def tile_sweep(call_native):
    torch.manual_seed(1)
    print(f"{'site':14s} {'M':>3s} {'BMxBN':>7s} {'native_us':>9s} "
          f"{'triton_us':>9s} {'gain':>7s}")
    for name in ("qkv", "o_proj", "lm_head"):
        N, K = SITES[name]
        w = torch.randint(-60, 61, (N, K), dtype=torch.int8, device="cuda")
        w_s = torch.rand(N, K // 128, dtype=torch.float32, device="cuda") * 0.02
        wp = preshuffle(w)
        for M in (16, 96):
            a = torch.randint(-60, 61, (M, K), dtype=torch.int8, device="cuda")
            a_s = torch.rand(M, K // 128, dtype=torch.float32, device="cuda") * 0.02
            out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
            cfg = _get_aiter_w8a8_config(M, N, K, 128)
            tri = timed(lambda: gemm_a8w8_blockscale(
                a, w, a_s, w_s, dtype=torch.bfloat16, config=cfg))
            part = torch.empty(M * N, dtype=torch.float32, device="cuda")
            for bm, bn, ks in ((16, 32, 1), (32, 32, 1), (32, 64, 2),
                               (32, 64, 4), (32, 128, 4), (16, 64, 4)):
                nat = timed(lambda: call_native(a, wp, a_s, w_s, out, M, N, K,
                                                bm, bn, ks, part))
                print(f"{name:14s} {M:3d} {bm:3d}x{bn:<3d}k{ks} {nat:9.2f} "
                      f"{tri:9.2f} {(1 - nat / tri) * 100:6.1f}%", flush=True)


if __name__ == "__main__":
    main()
