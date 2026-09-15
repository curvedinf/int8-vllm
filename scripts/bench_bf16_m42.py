#!/usr/bin/env python3
"""bf16 M=42 GEMM bench: rocBLAS (F.linear) vs AITER gemm_a16w16.

The decode-only profile shows hipBLASLt bf16 at M=42 running 380us/call
(MT128x128x64 -> ~40 CTAs on 120 CUs, ~137GB/s). These are the bf16 GDN
projection surfaces. If AITER a16w16 picks better tiles at skinny M, the
dispatch whitelist should cover these shapes.
"""
import sys, torch

sys.path.insert(0, "/home/curved/aiter")

dev = "cuda"
torch.manual_seed(0)
H, I = 5120, 4352
shapes = [
    ("qkvz_proj", H, H * 3 // 2 + I),
    ("o_proj", H, H),
    ("ba_proj", H, 2 * H // 5),
    ("gate_up", H, 2 * I),
    ("down_proj", I, H),
]


def bench(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1000  # us


for M in (42, 6):
    print(f"--- M={M} ---")
    for name, K, N in shapes:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        t_blas = bench(lambda: torch.nn.functional.linear(x, w))
        # N-chunked linear: raise CTA count at constant weight bytes
        best_ch, best_t = 1, t_blas
        for ch in (2, 4, 8):
            ws = torch.chunk(w, ch, dim=0)
            def run_ch(x=x, ws=ws, ch=ch):
                return torch.cat([torch.nn.functional.linear(x, wi) for wi in ws], dim=1)
            t_ch = bench(run_ch, iters=100, warmup=10)
            if t_ch < best_t:
                best_t, best_ch = t_ch, ch
        floor = (N * K * 2 + M * K * 2) / 1.2e6  # us at 1.2TB/s
        print(f"{name:10s} K={K} N={N}: blas={t_blas:7.1f}us "
              f"floor={floor:6.1f}us  "
              f"{f'CHUNK{best_ch} x{t_blas/best_t:.2f}' if best_ch > 1 else 'plain wins'}")
