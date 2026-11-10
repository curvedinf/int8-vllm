#!/usr/bin/env python3
"""Retune grouped a8w8 MoE GEMM configs for the REAL Flint TP4 shapes.

Production routing: E_local = 256/4 = 64 experts/rank? No: w13_local is
[E=256/K=2048/N=2*I_p=256] with I_p = 128 per rank (TP4 shards the
intermediate); down is [E=256, K=128, N=2048]. Decode verify rows:
M_gather = 72 (C12, 12 tokens x top-6); prefill chunk 2048 tokens.

For each (M, kernel) pair, patch get_kernel_config with ONE candidate,
time gate_up+down through the production _run path is NOT used (that
chunk-splits); instead this sweeps the two GEMMs with production routing
structures (via _routing_from_topk at the real E/topk) and reports us/call.

Usage: HIP_VISIBLE_DEVICES=0 .venv/bin/python scripts/tune_flint_moe_tp4.py
"""
import itertools, json, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aiter.ops.triton.moe import moe_op_gemm_a8w8_blockscale as moe_mod
from vllm.model_executor.layers.quantization.gptq_moe_a8w8 import (
    _quant_act, _routing_from_topk,
)
_ORIG = moe_mod.get_kernel_config
E, K, NGU, I, ND, GS, TOPK = 256, 2048, 256, 128, 2048, 128, 6

GRID = list(itertools.product(
    [64, 128, 256], [1, 2], [4, 8], [2, 3], [8],
))

def _set_cand(cand, block_m):
    def get_kernel_config(m, n, k, routing_data, blockscale_n=128):
        base = _ORIG(m, n, k, routing_data, blockscale_n)
        cfg = dict(base); cfg.update({
            "block_m": block_m, "block_n": cand[0], "block_k": 128,
            "group_m": cand[4], "xcd_swizzle": 1, "num_warps": cand[2],
            "num_stages": cand[3], "w_cache_modifier": None,
            "split_k": cand[1]})
        return cfg
    moe_mod.get_kernel_config = get_kernel_config

def _time_capped(fn, warmup=10, iters=50, cap_us=250):
    """Time one candidate; skip-and-abort the sweep if it is slow.

    Returns None for kernels slower than cap_us (their result would not
    be production-relevant) instead of raising; the sweep then stays
    bounded instead of probing pathological configs forever.
    """
    times = []
    for _ in range(warmup + iters):
        fn()

    t0 = torch.cuda.Event(True)
    t1 = torch.cuda.Event(True)
    t0.record()
    fn()
    t1.record()
    torch.cuda.synchronize()
    one_us = t0.elapsed_time(t1) * 1000
    if one_us > cap_us:
        return None
    return _time(fn, warmup, iters)


def _time(fn, warmup=10, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters*1000

def main():
    dev = "cuda"; torch.manual_seed(0)
    for M in (12, 512):
        w13 = torch.randint(-60, 61, (E, K, NGU), dtype=torch.int8, device=dev)
        s13 = torch.rand(E, K//GS, NGU, dtype=torch.float32, device=dev)*0.02
        w2 = torch.randint(-60, 61, (E, I, ND), dtype=torch.int8, device=dev)
        s2 = torch.rand(E, I//GS, ND, dtype=torch.float32, device=dev)*0.02
        ids = torch.randint(0, E, (M, TOPK), device=dev)
        w = torch.rand(M, TOPK, device=dev); w = w/w.sum(-1, keepdim=True)
        rd, gi, si = _routing_from_topk(ids, w, E)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)/10
        xq, xs = _quant_act(x)
        y = torch.randn(M, I, dtype=torch.bfloat16, device=dev)/10
        yq, ys = _quant_act(y)
        kw = dict(routing_data=rd, out_dtype=torch.bfloat16, per_row_x_scale=True, w_blockscale_n=1)
        def gu(): return moe_mod.moe_gemm_a8w8_blockscale(xq, w13, xs, s13, gather_indx=gi, apply_swiglu=False, **kw)
        def dn(): return moe_mod.moe_gemm_a8w8_blockscale(yq, w2, ys, s2, scatter_indx=si, gammas=rd.gate_scal, apply_swiglu=False, **kw)
        for name, fn in (("gate_up", gu), ("down", dn)):
            rows = []
            for cand in GRID:
                c = _time_capped(fn)
                if c is not None:
                    rows.append((c, cand))
            rows.sort()
            print(f"{name} M={M} block_m={rd.block_m} best5:")
            for us, c in rows[:5]:
                print(f"   {us:8.1f}us block_n={c[0]} split_k={c[1]} warps={c[2]} stages={c[3]} group_m={c[4]}", flush=True)
            print("WORST:", f"{rows[-1][0]:.1f}us {rows[-1][1]}", flush=True)
    print("DONE")

if __name__ == "__main__":
    main()
