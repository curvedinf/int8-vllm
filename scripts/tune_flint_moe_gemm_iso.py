#!/usr/bin/env python3
"""Crash-isolated sweep of grouped a8w8-blockscale MoE configs for Flint.

Same shapes/grid as scripts/tune_flint_moe_gemm.py, but candidates run in
child processes in batches: one illegal-access config cannot poison the
sweep. A batch that dies is retried candidate-by-candidate. Each batch
brackets the measurements with the production heuristic (first and last) so
a clock ramp shows up as a reference drift instead of being read as a
config win.

Usage:
  HIP_VISIBLE_DEVICES=0 .venv/bin/python scripts/tune_flint_moe_gemm_iso.py
"""

import itertools
import json
import os
import subprocess
import sys

E, K, N_GU, I, N_D = 64, 2048, 1024, 512, 2048
GS = 128
BATCH = 8

GRID = list(itertools.product(
    [128, 256],            # block_n
    [1, 2],                # split_k
    [4, 8],                # num_warps
    [2, 3],                # num_stages
    [1, 8],                # group_m
    [1],                   # xcd_swizzle
    [None, ".cg"],         # w_cache_modifier
))

CHILD = r"""
import json, sys
import torch
from aiter.ops.triton.moe import moe_op_gemm_a8w8_blockscale as moe_mod
from vllm.model_executor.layers.quantization.gptq_moe_a8w8 import (
    _quant_act, _routing_from_topk)

spec = json.loads(sys.argv[1])
M, name, cands = spec["m"], spec["kernel"], spec["cands"]
E, K, N_GU, I, N_D, GS = 64, 2048, 1024, 512, 2048, 128
_ORIG = moe_mod.get_kernel_config

def make_override(cand, block_m):
    cfg = {"block_m": block_m, "block_n": cand[0], "block_k": 128,
           "group_m": cand[4], "xcd_swizzle": cand[5],
           "num_warps": cand[2], "num_stages": cand[3],
           "blockscale_m": 128, "blockscale_n": 1, "blockscale_k": 128,
           "w_cache_modifier": cand[6], "split_k": cand[1],
           "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 1}
    return cfg

def set_cand(cand):
    def g(m, n, k, routing_data, blockscale_n=128):
        base = _ORIG(m, n, k, routing_data, blockscale_n)
        return make_override(cand, base["block_m"])
    moe_mod.get_kernel_config = g

def time_fn(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0

torch.manual_seed(0)
w13 = torch.randint(-60, 61, (E, K, N_GU), dtype=torch.int8, device="cuda")
s13 = torch.rand(E, K // GS, N_GU, dtype=torch.float32, device="cuda") * 0.02
w2 = torch.randint(-60, 61, (E, I, N_D), dtype=torch.int8, device="cuda")
s2 = torch.rand(E, I // GS, N_D, dtype=torch.float32, device="cuda") * 0.02
ids = torch.randint(0, E, (M, 6), device="cuda")
w = torch.rand(M, 6, device="cuda")
w = w / w.sum(-1, keepdim=True)
rd, gather_idx, scatter_idx = _routing_from_topk(ids, w, E)
x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / 10
xq, xs = _quant_act(x)
y = torch.randn(M, I, dtype=torch.bfloat16, device="cuda") / 10
yq, ys = _quant_act(y)
kw = dict(routing_data=rd, out_dtype=torch.bfloat16, per_row_x_scale=True,
          w_blockscale_n=1)

def fn():
    if name == "gate_up":
        return moe_mod.moe_gemm_a8w8_blockscale(
            xq, w13, xs, s13, gather_indx=gather_idx,
            apply_swiglu=False, **kw)
    return moe_mod.moe_gemm_a8w8_blockscale(
        yq, w2, ys, s2, scatter_indx=scatter_idx,
        gammas=rd.gate_scal, apply_swiglu=False, **kw)

out = []
for c in cands:
    if c is None:
        moe_mod.get_kernel_config = _ORIG
    else:
        set_cand(tuple(c))
    try:
        out.append([c, round(time_fn(fn), 3)])
    except Exception as exc:
        out.append([c, "err:" + type(exc).__name__])
print("RESULT", json.dumps(out))
"""


def run_batch(m, kernel, cands):
    spec = {"m": m, "kernel": kernel, "cands": cands}
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, json.dumps(spec)],
        capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return None


def main():
    results = {}
    for M in (96, 2048):
        for kernel in ("gate_up", "down"):
            key = f"{kernel}_M{M}"
            rows = []
            i = 0
            while i < len(GRID):
                batch = GRID[i:i + BATCH]
                # bracket the batch with the production heuristic
                payload = [None] + [list(c) for c in batch] + [None]
                got = run_batch(M, kernel, payload)
                if got is not None:
                    refs = [r[1] for r in got
                            if r[0] is None and isinstance(r[1], (int, float))]
                    if len(refs) == 2 and max(refs) > min(refs) * 1.10:
                        print(f"  [{key}] reference drift {refs}", flush=True)
                    rows.extend(r for r in got if r[0] is not None)
                else:
                    print(f"  [{key}] batch crashed, retrying singly "
                          f"{i}-{i+len(batch)-1}", flush=True)
                    for c in batch:
                        one = run_batch(M, kernel, [list(c)])
                        if one is None:
                            rows.append([list(c), "crash"])
                        else:
                            rows.extend(one)
                i += BATCH
                print(f"  [{key}] {i}/{len(GRID)} done", flush=True)
            ok = [(u, c) for c, u in rows if isinstance(u, (int, float))]
            ok.sort()
            refs = [u for c, u in rows if c is None and isinstance(u, (int, float))]
            results[key] = {
                "heuristic_us": refs,
                "best_us": ok[0][0] if ok else None,
                "best": ok[:5],
                "crashes": [c for c, u in rows if u == "crash"],
            }
            print(key, json.dumps(results[key]), flush=True)
    print("BEST_JSON", json.dumps(results))


if __name__ == "__main__":
    main()
