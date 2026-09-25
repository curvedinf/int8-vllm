#!/usr/bin/env python3
"""Single-process a8w8 CK tuning sweep for small-M decode shapes (GOALOPT).

Replaces the mp-based tuner (whose spawn workers thrash on the JIT lock):
iterates gemm_a8w8_tune over every (kernelId, splitK), times with CUDA
events, checks against a dequantized fp32 reference, and writes the best
row per shape in the tuned-CSV schema.
"""
import csv
import sys
import time

import torch

sys.path.insert(0, "/home/curved/aiter")
sys.path.insert(0, "/home/curved/aiter/csrc/ck_gemm_a8w8")
sys.path.insert(0, "/home/curved/vllm-gfx908")

from gemm_a8w8_common import kernels_list  # noqa: E402
import aiter  # noqa: E402
from aiter import gemm_a8w8_tune  # noqa: E402

dev = "cuda"
torch.manual_seed(0)
import os
SPLIT_CAP = int(os.environ.get("TUNE_SPLIT_CAP", "4"))


def bench_config(xq, w, xs, ws, out, kid, sk, iters=20, warmup=5):
    try:
        for _ in range(warmup):
            gemm_a8w8_tune(xq, w, xs, ws, out, kid, sk)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            gemm_a8w8_tune(xq, w, xs, ws, out, kid, sk)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters * 1000
    except RuntimeError:
        return None


def main() -> None:
    shapes_path = sys.argv[1] if len(sys.argv) > 1 else \
        "logs/goal_opt/gemm_tune/untuned_msmall.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else \
        "logs/goal_opt/gemm_tune/tuned_msmall.csv"
    rows = list(csv.DictReader(open(shapes_path)))
    results = []
    for r in rows:
        M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
        xq = torch.randint(-127, 127, (M, K), device=dev, dtype=torch.int8)
        xs = torch.rand(M, device=dev, dtype=torch.float32) * 0.02 + 0.005
        w = torch.randint(-127, 127, (N, K), device=dev, dtype=torch.int8)
        ws = torch.rand(N, device=dev, dtype=torch.float32) * 0.02 + 0.005
        out = torch.empty(M, N, device=dev, dtype=torch.float16)
        ref = (xq.float() * xs[:, None]) @ (w.float() * ws[:, None]).t()
        best = None
        t0 = time.time()
        for kid in sorted(kernels_list):
            kern = kernels_list[kid]
            try:
                maxsk = aiter.compute_gemm_SplitK(
                    M, N, K, kern.MPerBLOCK, kern.NPerBLOCK, kern.KPerBLOCK
                )
            except Exception:
                maxsk = 0
            for sk in range(min(maxsk, SPLIT_CAP) + 1):
                out.zero_()
                try:
                    gemm_a8w8_tune(xq, w, xs, ws, out, kid, sk)
                    torch.cuda.synchronize()
                except RuntimeError:
                    continue
                err = (out.float() - ref).abs().max().item()
                if err > 0.05 * max(1.0, ref.abs().max().item()):
                    continue
                us = bench_config(xq, w, xs, ws, out, kid, sk)
                if us is None:
                    continue
                if best is None or us < best["us"]:
                    best = {"us": us, "kid": kid, "sk": sk,
                            "name": kern.name}
        if best:
            tflops = 2 * M * N * K / (best["us"] * 1e-6) / 1e12
            bw = (M * K + N * K) / (best["us"] * 1e-6) / 1e9
            results.append({
                "gfx": "gfx908", "cu_num": 120, "M": M, "N": N, "K": K,
                "q_dtype_w": "torch.int8", "kernelId": best["kid"],
                "splitK": best["sk"], "us": round(best["us"], 3),
                "kernelName": best["name"], "tflops": round(tflops, 2),
                "bw": round(bw, 1), "errRatio": "Null",
            })
            print(f"M={M} N={N} K={K}: best {best['us']:.1f}us "
                  f"kid={best['kid']} sk={best['sk']} {best['name'][:40]} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        else:
            print(f"M={M} N={N} K={K}: NO VALID CONFIG", flush=True)
        with open(out_path, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=[
                "gfx", "cu_num", "M", "N", "K", "q_dtype_w", "kernelId",
                "splitK", "us", "kernelName", "tflops", "bw", "errRatio"])
            wtr.writeheader()
            wtr.writerows(results)
    print(f"wrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
