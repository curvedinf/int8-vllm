#!/usr/bin/env python3
"""Isolated down-config probe through the production _run path.

Usage: probe_flint_moe_down.py BN WARPS STAGES GROUP SPLITK
Prints: CONFIG ... ok=<bool> us=<float>  (one line per process)
"""
import sys
import types

import torch

from aiter.ops.triton.moe import moe_op_gemm_a8w8_blockscale as moe_mod
from vllm.model_executor.layers.quantization.gptq_moe_a8w8 import (
    A8W8BlockscaleMoEMethod,
)

E, K, NGU, I, ND, G = 64, 2048, 1024, 512, 2048, 128
M = 96
bn, warps, stages, group, sk = (int(x) for x in sys.argv[1:6])
dev = "cuda"
torch.manual_seed(0)
layer = types.SimpleNamespace(
    w13_qweight=torch.randint(-60, 61, (E, K, NGU), dtype=torch.int8, device=dev),
    w13_scales=torch.rand(E, K // G, NGU, dtype=torch.float32, device=dev) * 0.02,
    w2_qweight=torch.randint(-60, 61, (E, I, ND), dtype=torch.int8, device=dev),
    w2_scales=torch.rand(E, I // G, ND, dtype=torch.float32, device=dev) * 0.02,
    layer_name="probe",
)
m = object.__new__(A8W8BlockscaleMoEMethod)
m.quant_config = types.SimpleNamespace(group_size=128)
m.input_dtype = None
m._use_a8w8 = True
x = torch.randn(M, K, dtype=torch.bfloat16, device=dev) / 10
w = torch.rand(M, 6, device=dev)
w = w / w.sum(-1, keepdim=True)
ids = torch.argsort(torch.rand(M, E, device=dev), dim=1)[:, :6]

orig = moe_mod.get_kernel_config


def gkc(mm, nn, kk, routing_data, blockscale_n=128):
    base = orig(mm, nn, kk, routing_data, blockscale_n)
    if kk == I:  # down only
        base.update({"block_n": bn, "num_warps": warps, "num_stages": stages,
                     "group_m": group, "split_k": sk,
                     "w_cache_modifier": None})
    return base


moe_mod.get_kernel_config = orig
ref = m._run(layer, x, w, ids, E)
torch.cuda.synchronize()
if bn == 0:
    base_us = None
    for _ in range(5):
        m._run(layer, x, w, ids, E)
    torch.cuda.synchronize()
    s0, e0 = torch.cuda.Event(True), torch.cuda.Event(True)
    s0.record()
    for _ in range(20):
        m._run(layer, x, w, ids, E)
    e0.record()
    torch.cuda.synchronize()
    print(f"CONFIG heuristic ok=True us={s0.elapsed_time(e0)/20*1000:.1f}")
    raise SystemExit
moe_mod.get_kernel_config = gkc
out = m._run(layer, x, w, ids, E)
torch.cuda.synchronize()
ok = torch.allclose(out, ref, atol=2e-2, rtol=2e-2)
for _ in range(5):
    m._run(layer, x, w, ids, E)
torch.cuda.synchronize()
s, e = torch.cuda.Event(True), torch.cuda.Event(True)
s.record()
for _ in range(20):
    m._run(layer, x, w, ids, E)
e.record()
torch.cuda.synchronize()
print(f"CONFIG bn={bn} w={warps} s={stages} g={group} sk={sk} "
      f"ok={ok} us={s.elapsed_time(e)/20*1000:.1f}")
