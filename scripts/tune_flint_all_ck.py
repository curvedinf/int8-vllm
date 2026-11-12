#!/usr/bin/env python3
"""Tune every CK W8A8 shape used by Flint + the DFlash2 draft, all M buckets.

The CK lookup pads M to 16/32/64 buckets (get_padded_m), so tuning the bucket
values covers every production M. Shapes are the per-rank GEMMs seen in the
boot log: Flint qkv (1280x2048), Flint o_proj (2048x1024), draft qkv
(1536x2048). Winners are appended to aiter's a8w8_tuned_gemm.csv.

  HIP_VISIBLE_DEVICES=0 .venv/bin/python scripts/tune_flint_all_ck.py
"""

import argparse
from pathlib import Path

import torch

from aiter.ops.gemm_op_a8w8 import gemm_a8w8_CK

SHAPES = [("flint_qkv", 1280, 2048), ("flint_o_proj", 2048, 1024),
          ("draft_qkv", 1536, 2048)]


def m_buckets():
    ms = list(range(16, 257, 16))                    # <=256: /16
    ms += list(range(288, 1025, 32))                 # <=1024: /32
    ms += list(range(1088, 2049, 64))                # <=2048 (MNBT): /64
    return ms


def _time(fn, warm=5, iters=20):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(
        Path.home() / "int8-aiter/aiter/configs/a8w8_tuned_gemm.csv"))
    ap.add_argument("--splits", default="0,1,2,3,4,6,8")
    args = ap.parse_args()
    splits = [int(x) for x in args.splits.split(",")]
    dev = "cuda"
    torch.manual_seed(0)
    rows = []
    for name, N, K in SHAPES:
        wq = torch.randint(-60, 61, (N, K), dtype=torch.int8, device=dev)
        ws = (torch.rand(N, 1, device=dev) * 0.02).to(torch.float32)
        for M in m_buckets():
            xq = torch.randint(-60, 61, (M, K), dtype=torch.int8, device=dev)
            xs = (torch.rand(M, 1, device=dev, dtype=torch.float32) * 0.02)
            ref = gemm_a8w8_CK(xq, wq, xs, ws, None, torch.bfloat16, splitK=0)
            torch.cuda.synchronize()
            best = None
            for k in splits:
                try:
                    out = gemm_a8w8_CK(xq, wq, xs, ws, None, torch.bfloat16,
                                       splitK=k)
                    torch.cuda.synchronize()
                    same = torch.equal(out, ref) or torch.allclose(
                        out, ref, atol=1e-3, rtol=1e-3)
                    us = _time(lambda k=k: gemm_a8w8_CK(
                        xq, wq, xs, ws, None, torch.bfloat16, splitK=k))
                except Exception as exc:  # noqa: BLE001
                    print(f"  {name} M={M} splitK={k} FAIL {exc}", flush=True)
                    continue
                if same and (best is None or us < best[0]):
                    best = (us, k)
            if best:
                rows.append(f"gfx908,120,{M},{N},{K},torch.int8,0,{best[1]},"
                            f"{best[0]:.3f},ck_splitk_tuned,0.0,0.0,Null")
            print(f"{name} M={M} N={N} K={K} best={best}", flush=True)
    if rows:
        with open(args.out, "a") as f:
            for r in rows:
                f.write(r + "\n")
        print(f"appended {len(rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
