#!/usr/bin/env python3
"""Fused RMSNorm + per-token int8 quant correctness/perf probe (gfx908).

Compares the fused Triton kernel against both unfused chains it replaces:

- aiter chain (RMSNorm.forward_hip large-M path): aiter rms_norm ->
  pertoken_quant — the fused kernel is bit-exact against this one;
- ir-native chain (compiled vllm_ir baseline): native rms_norm ->
  pertoken_quant — compared against the registered native-numerics op
  (what the graph pass inserts), which matches up to tl.sum reduction
  order (last-ulp noise); the fp32 residual add itself is bit-exact.

Reports payload mismatch counts, max scale rel err, and kernel timings.
"""

import torch
import triton
from aiter import pertoken_quant
from aiter.ops.triton.normalization.rmsnorm import (
    rms_norm as aiter_rms_norm,
    rmsnorm2d_fwd_with_add as aiter_rmsnorm_add,
)

from vllm.model_executor.layers.rms_norm_int8_quant import (
    rms_norm_int8_quant,
)

M_LIST = [1, 8, 64, 88, 96, 2048]
K = 5120
EPS = 1e-6


def native_rms_norm(x, w, eps):
    x32 = x.float()
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    return ((x32 * torch.rsqrt(var + eps)).to(w.dtype) * w).to(x.dtype)


def native_fused_add(x, res, w, eps):
    x32 = x.float() + res.float()
    res_out = x32.to(x.dtype)
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    return ((x32 * torch.rsqrt(var + eps)).to(w.dtype) * w).to(x.dtype), res_out


def stats(q_fused, s_fused, q_ref, s_ref):
    payload = (q_fused != q_ref).sum().item()
    rows = (q_fused != q_ref).any(dim=-1).sum().item()
    sre = ((s_fused - s_ref).abs() / s_ref.clamp(min=1e-30)).max().item()
    return payload, rows, sre


def main():
    torch.manual_seed(0)
    dev = "cuda"
    print(f"{'dtype':8} {'M':>5} {'var':4} "
          f"{'aiter:payload':>14} {'rows':>7} {'scale_relerr':>12} "
          f"{'native:payload':>15} {'rows':>7} {'scale_relerr':>12}")

    for dtype in (torch.float16, torch.bfloat16):
        for M in M_LIST:
            x = (torch.randn(M, K, device=dev) * 3).to(dtype)
            w = (torch.randn(K, device=dev) * 0.5 + 1).to(dtype)
            res = torch.randn(M, K, device=dev).to(dtype)

            # --- plain ---
            normed_a = aiter_rms_norm(x, w, EPS)
            q_a, s_a = pertoken_quant(
                normed_a.to(torch.float16), quant_dtype=torch.int8
            )
            normed_n = native_rms_norm(x, w, EPS)
            q_n, s_n = pertoken_quant(
                normed_n.to(torch.float16), quant_dtype=torch.int8
            )

            out_f, q_f, s_f = rms_norm_int8_quant(x, w, EPS)
            # native-numerics registered op = the compiled-graph replacement
            q_g, s_g = torch.ops.vllm.rocm_rms_norm_int8_quant(x, w, EPS)
            pa = stats(q_f, s_f, q_a, s_a)
            pn = stats(q_g, s_g, q_n, s_n)
            print(f"{str(dtype):8} {M:>5} {'norm':4} "
                  f"{pa[0]:>14} {pa[1]:>4}/{M:<2} {pa[2]:>12.3e} "
                  f"{pn[0]:>15} {pn[1]:>4}/{M:<2} {pn[2]:>12.3e}")

            # --- fused add ---
            out_t = torch.empty_like(x)
            res_t = torch.empty_like(x)
            aiter_rmsnorm_add(out_t, x, res, res_t, w, EPS)
            q_ar, s_ar = pertoken_quant(out_t.to(torch.float16), quant_dtype=torch.int8)
            normed_n2, res_n = native_fused_add(x, res, w, EPS)
            q_nr, s_nr = pertoken_quant(
                normed_n2.to(torch.float16), quant_dtype=torch.int8
            )

            out_f2, res_f2, q_f2, s_f2 = rms_norm_int8_quant(x, w, EPS, residual=res)
            q_g2, s_g2, res_g2 = torch.ops.vllm.rocm_rms_norm_add_int8_quant(
                x, res, w, EPS
            )
            pa2 = stats(q_f2, s_f2, q_ar, s_ar)
            pn2 = stats(q_g2, s_g2, q_nr, s_nr)
            assert (out_f2 == out_t).all().item() and (res_f2 == res_t).all().item()
            assert (res_g2 == res_n).all().item(), "native residual must be bit-exact"
            print(f"{'':8} {M:>5} {'add':4} "
                  f"{pa2[0]:>14} {pa2[1]:>4}/{M:<2} {pa2[2]:>12.3e} "
                  f"{pn2[0]:>15} {pn2[1]:>4}/{M:<2} {pn2[2]:>12.3e}")

    bench()


def bench():
    dev = "cuda"
    print("\nkernel timing (median of 50, CUDA events):")
    print(f"{'dtype':8} {'M':>5} {'unfused ms':>11} {'fused ms':>10} {'speedup':>8}")
    for dtype in (torch.float16,):
        for M in (8, 96, 2048):
            x = (torch.randn(M, K, device=dev) * 3).to(dtype)
            w = (torch.randn(K, device=dev) * 0.5 + 1).to(dtype)
            res = torch.randn(M, K, device=dev).to(dtype)

            def unfused():
                out = torch.empty_like(x)
                res_out = torch.empty_like(x)
                aiter_rmsnorm_add(out, x, res, res_out, w, EPS)
                q, s = pertoken_quant(out.to(torch.float16), quant_dtype=torch.int8)
                return q, s

            def fused():
                return rms_norm_int8_quant(x, w, EPS, residual=res)[2:]

            for fn in (unfused, fused):
                fn()
            times = {}
            for name, fn in (("u", unfused), ("f", fused)):
                evs = [
                    (torch.cuda.Event(True), torch.cuda.Event(True))
                    for _ in range(50)
                ]
                for s0, s1 in evs:
                    s0.record()
                    fn()
                    s1.record()
                torch.cuda.synchronize()
                times[name] = sorted(
                    s0.elapsed_time(s1) for s0, s1 in evs
                )[25]
            print(f"{str(dtype):8} {M:>5} {times['u']:>11.4f} {times['f']:>10.4f} "
                  f"{times['u'] / times['f']:>7.2f}x")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
