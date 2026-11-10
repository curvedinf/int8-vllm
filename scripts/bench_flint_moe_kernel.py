#!/usr/bin/env python3
"""Graph-timed bench of the routed-expert GEMMs at Flint's production shapes.

Shapes (TP4, per rank): gate_up K=2048 N=256 (2*I_p), down K=I_p=128 N=2048,
E=256 experts, top-6 routing, decode verify rows M=96 (C12 x NS7+1).
Both calls use the production kwargs (fused SwiGLU/GS128 quant epilogue for
gate_up, down-quant-out for down) and are timed inside a CUDA graph, because
eager per-call timings are dispatch-bound.

Reported GB/s uses the weight + weight-scale bytes that must stream from HBM
(read once per step no matter how the tiles are cut).

Usage:
  drivers:  HIP_VISIBLE_DEVICES=0 .venv/bin/python scripts/bench_flint_moe_kernel.py
  one cfg:  ... scripts/bench_flint_moe_kernel.py one <kernel> <M> <bn> <warps> <stages>
            (single config per process: a bad tile size faults the device and
            would poison every later candidate in the same process)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from aiter.ops.triton.moe import moe_op_gemm_a8w8_blockscale as moe_mod  # noqa: E402
from vllm.model_executor.layers.quantization.gptq_moe_a8w8 import (  # noqa: E402
    _quant_act,
    _routing_from_topk,
)

E, K, NGU, I, ND, GS, TOPK = 256, 2048, 256, 128, 2048, 128, 6
_ORIG = moe_mod.get_kernel_config


def _set_cfg(cfg):
    def get_kernel_config(m, n, k, routing_data, blockscale_n=128):
        base = dict(_ORIG(m, n, k, routing_data, blockscale_n))
        base.update(cfg)
        return base

    moe_mod.get_kernel_config = get_kernel_config


def _time_graph(fn, warmup=3, iters=30, reps=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    best = float("inf")
    for _ in range(reps):
        g.replay()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters * 1000)
    return best


def _bench(kernel, M, bn, nw, ns):
    dev = "cuda"
    torch.manual_seed(0)
    w13 = torch.randint(-60, 61, (E, K, NGU), dtype=torch.int8, device=dev)
    s13 = torch.rand(E, K // GS, NGU, dtype=torch.float32, device=dev) * 0.02
    w2 = torch.randint(-60, 61, (E, I, ND), dtype=torch.int8, device=dev)
    s2 = torch.rand(E, I // GS, ND, dtype=torch.float32, device=dev) * 0.02
    nbytes = (w13.numel() + s13.numel() * 4 if kernel == "gate_up"
              else w2.numel() + s2.numel() * 4)

    ids = torch.randint(0, E, (M, TOPK), device=dev)
    w = torch.rand(M, TOPK, device=dev)
    w = w / w.sum(-1, keepdim=True)
    rd, gi, si = _routing_from_topk(ids, w, E)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev) / 10
    xq, xs = _quant_act(x)
    # down consumes the GATHERED rows (gate_up's quant-out output), so its
    # activation is [M*top_k, I] -- indexing gathered rows against a
    # token-sized tensor reads out of bounds.
    y = torch.randn(M * TOPK, I, dtype=torch.bfloat16, device=dev) / 10
    yq, ys = _quant_act(y)
    kw = dict(out_dtype=torch.bfloat16, per_row_x_scale=True, w_blockscale_n=1)

    if kernel == "gate_up":
        fn = lambda: moe_mod.moe_gemm_a8w8_blockscale(  # noqa: E731
            xq, w13, xs, s13, routing_data=rd, gather_indx=gi,
            apply_swiglu=True, swiglu_add_residual=False, quant_out=True, **kw)
    else:
        fn = lambda: moe_mod.moe_gemm_a8w8_blockscale(  # noqa: E731
            yq, w2, ys, s2, routing_data=rd, scatter_indx=si,
            gammas=rd.gate_scal, apply_swiglu=False, down_quant_out=True, **kw)

    _set_cfg(dict(block_m=rd.block_m, block_k=128, block_n=bn,
                  num_warps=nw, num_stages=ns, group_m=1,
                  xcd_swizzle=8, blockscale_m=128, blockscale_n=1,
                  blockscale_k=128, split_k=1, waves_per_eu=1,
                  matrix_instr_nonkdim=16, kpack=1,
                  w_cache_modifier=os.environ.get("BENCH_WCACHE") or None))
    us = _time_graph(fn)
    return us, nbytes


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        kernel, M, bn, nw, ns = sys.argv[2], int(sys.argv[3]), int(
            sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
        try:
            us, nbytes = _bench(kernel, M, bn, nw, ns)
        except Exception as exc:  # noqa: BLE001
            print(f"RESULT {kernel} M={M} bn={bn} w={nw} s={ns} FAIL "
                  f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}",
                  flush=True)
            os._exit(0)
        print(f"RESULT {kernel} M={M} bn={bn} w={nw} s={ns} {us:.1f}us "
              f"{nbytes/us/1e3:.0f}GB/s", flush=True)
        return

    py = sys.executable
    for M in (96, 192):
        for kernel, bns in (("gate_up", [256]), ("down", [128, 256, 512, 1024])):
            rows = []
            for bn in bns:
                for nw in (4, 8):
                    for ns in (2, 3):
                        r = subprocess.run(
                            [py, __file__, "one", kernel, str(M), str(bn),
                             str(nw), str(ns)],
                            capture_output=True, text=True,
                            env=dict(os.environ), timeout=600)
                        line = [ln for ln in r.stdout.splitlines()
                                if ln.startswith("RESULT")]
                        if not line:
                            print(f"  {kernel} M={M} bn={bn} w={nw} s={ns} "
                                  f"CRASH {r.stderr.splitlines()[-1][:80] if r.stderr else ''}",
                                  flush=True)
                            continue
                        print("  " + line[0], flush=True)
                        if "FAIL" not in line[0]:
                            rows.append((float(line[0].split()[-2][:-2]),
                                         (bn, nw, ns)))
            if rows:
                rows.sort()
                print(f"BEST {kernel} M={M} {rows[0]}", flush=True)


if __name__ == "__main__":
    main()
