#!/usr/bin/env python3
"""Sweep grouped a8w8-blockscale MoE kernel configs for Flint shapes.

Flint's routed experts: local E=64/rank (TP4), top-6, gate_up N=1024
K=2048, down N=2048 K=512, GS128. Decode verify rows M=96 (C12, NS7),
prefill chunk M=2048. The aiter kernel picks its config from a hardcoded
heuristic; this script times candidate configs with the exact routing
structures the model produces and prints the best per (kernel, M).

Usage:
  HIP_VISIBLE_DEVICES=0 .venv/bin/python scripts/tune_flint_moe_gemm.py
"""

import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aiter.ops.triton.moe import moe_op_gemm_a8w8_blockscale as moe_mod  # noqa: E402
from vllm.model_executor.layers.quantization.gptq_moe_a8w8 import (  # noqa: E402
    _quant_act,
    _routing_from_topk,
)

_ORIG = moe_mod.get_kernel_config
E, K, N_GU, I, N_D = 64, 2048, 1024, 512, 2048
GS = 128

GRID = list(itertools.product(
    [128, 256],            # block_n
    [1, 2],                # split_k
    [4, 8],                # num_warps
    [2, 3],                # num_stages
    [1, 8],                # group_m
    [1],                   # xcd_swizzle (8 == 1 on this mesh)
    [None, ".cg"],         # w_cache_modifier
))


def _make_override(cand, block_m):
    cfg = {
        "block_m": block_m, "block_n": cand[0], "block_k": 128,
        "group_m": cand[4], "xcd_swizzle": cand[5],
        "num_warps": cand[2], "num_stages": cand[3],
        "blockscale_m": 128, "blockscale_n": 1, "blockscale_k": 128,
        "w_cache_modifier": cand[6], "split_k": cand[1],
        "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 1,
    }
    return cfg


def _set_cand(cand):
    def get_kernel_config(m, n, k, routing_data, blockscale_n=128):
        base = _ORIG(m, n, k, routing_data, blockscale_n)
        return _make_override(cand, base["block_m"])
    moe_mod.get_kernel_config = get_kernel_config


def _time(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us


def main():
    dev = "cuda"
    torch.manual_seed(0)
    results = {}
    for M in (96, 2048):
        # synthetic expert weights in the kernel's [E,K,N] int8 layout
        w13 = torch.randint(-60, 61, (E, K, N_GU), dtype=torch.int8, device=dev)
        s13 = torch.rand(E, K // GS, N_GU, dtype=torch.float32, device=dev) * 0.02
        w2 = torch.randint(-60, 61, (E, I, N_D), dtype=torch.int8, device=dev)
        s2 = torch.rand(E, I // GS, N_D, dtype=torch.float32, device=dev) * 0.02
        # routing structures exactly as the model builds them
        ids = torch.randint(0, E, (M, 6), device=dev)
        w = torch.rand(M, 6, device=dev)
        w = w / w.sum(-1, keepdim=True)
        rd, gather_idx, scatter_idx = _routing_from_topk(ids, w, E)
        block_m = rd.block_m
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev) / 10
        xq, xs = _quant_act(x)
        y = torch.randn(M, I, dtype=torch.bfloat16, device=dev) / 10
        yq, ys = _quant_act(y)
        kw = dict(routing_data=rd, out_dtype=torch.bfloat16, per_row_x_scale=True,
                  w_blockscale_n=1)

        def gu():
            return moe_mod.moe_gemm_a8w8_blockscale(
                xq, w13, xs, s13, gather_indx=gather_idx,
                apply_swiglu=False, **kw)

        def dn():
            return moe_mod.moe_gemm_a8w8_blockscale(
                yq, w2, ys, s2, scatter_indx=scatter_idx,
                gammas=rd.gate_scal, apply_swiglu=False, **kw)

        for name, fn in (("gate_up", gu), ("down", dn)):
            _set_cand((256, 1, 8 if block_m > 16 else 4, 2, 4, 8, None))
            base_us = _time(fn)
            rows = []
            for ci, cand in enumerate(GRID):
                print(f"  [{name} M={M}] {ci+1}/{len(GRID)} {cand}", flush=True)
                if cand[1] > 1 and name == "down" and False:
                    continue
                _set_cand(cand)
                try:
                    us = _time(fn)
                except Exception as exc:  # noqa: BLE001
                    continue
                rows.append((us, cand))
            rows.sort()
            best = rows[:5]
            key = f"{name}_M{M}"
            results[key] = {
                "block_m": block_m,
                "heuristic_like_us": round(base_us, 2),
                "best_us": round(best[0][0], 2),
                "best": [
                    {"us": round(us, 2),
                     "block_n": c[0], "split_k": c[1], "num_warps": c[2],
                     "num_stages": c[3], "group_m": c[4],
                     "xcd_swizzle": c[5], "w_cache_modifier": c[6]}
                    for us, c in best
                ],
            }
            print(key, json.dumps(results[key]), flush=True)
    print("BEST_JSON", json.dumps(results))


if __name__ == "__main__":
    main()
