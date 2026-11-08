#!/usr/bin/env python3
"""Tune the CK W8A8 kernel's splitK for Flint's per-rank GEMM shapes.

Flint TP4, NS=7: decode M=96 (12 seqs x 8 verify rows), prefill chunk
M=2048. Shapes are the per-rank partitions the model actually launches.
The aiter CK path reads splitK from aiter/configs/a8w8_tuned_gemm.csv and
falls back to splitK=0 when the shape is missing (the boot warning).
This script times splitK candidates, checks output equality against
splitK=0, and prints CSV rows for the winners.

Usage: HIP_VISIBLE_DEVICES=0 .venv/bin/python scripts/tune_flint_dense_ck.py
"""

import torch

from aiter.ops.gemm_op_a8w8 import gemm_a8w8_CK

# CK runs only on whole-K column-parallel linears; o_proj/down_proj are
# row-parallel (sharded K) and stay on the Triton blockscale path.
SHAPES = [
    ("qkv", 1280, 2048),
    ("dense_gate_up", 3072, 2048),
    ("shared_gate_up", 512, 2048),
]
MS = [96, 2048]
SPLITS = [0, 1, 2, 3, 4, 6, 8]


def _time(fn, warm=5, iters=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


def main():
    dev = "cuda"
    torch.manual_seed(0)
    rows = []
    for M in MS:
        for name, N, K in SHAPES:
            xq = torch.randint(-60, 61, (M, K), dtype=torch.int8, device=dev)
            xs = (torch.rand(M, 1, device=dev, dtype=torch.float32) * 0.02)
            wq = torch.randint(-60, 61, (N, K), dtype=torch.int8, device=dev)
            ws = (torch.rand(N, 1, device=dev) * 0.02).to(torch.float32)
            ref = gemm_a8w8_CK(xq, wq, xs, ws, None, torch.bfloat16, splitK=0)
            torch.cuda.synchronize()
            best = None
            for k in SPLITS:
                try:
                    out = gemm_a8w8_CK(xq, wq, xs, ws, None, torch.bfloat16,
                                       splitK=k)
                    torch.cuda.synchronize()
                    same = torch.equal(out, ref) or torch.allclose(
                        out, ref, atol=1e-3, rtol=1e-3)
                    us = _time(lambda k=k: gemm_a8w8_CK(
                        xq, wq, xs, ws, None, torch.bfloat16, splitK=k))
                except Exception as exc:  # noqa: BLE001
                    print(f"  {name} M={M} splitK={k} FAIL {exc}")
                    continue
                print(f"  {name} M={M} N={N} K={K} splitK={k} {us:.1f} us "
                      f"ok={same}", flush=True)
                if same and (best is None or us < best[0]):
                    best = (us, k)
            if best:
                rows.append(
                    f"gfx908,120,{M},{N},{K},torch.int8,0,{best[1]},"
                    f"{best[0]:.3f},ck_splitk_tuned,0.0,0.0,Null")
    print("CSV_ROWS")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
