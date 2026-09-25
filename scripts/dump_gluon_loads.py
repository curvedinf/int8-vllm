#!/usr/bin/env python3
"""Dump compiled PTX/SASS global-load widths for the Gluon cores (GOALOPT).

Compiles the g128 prefill core at production geometry and counts global
load instructions by vector width in the generated assembly, answering
whether the packed-KV loads already issue wide (128-bit) transactions
(the 1Cat PR #268 lever) or are narrow enough to justify a
load-vectorization kernel.
"""
import os
import re
import sys

import torch

sys.path.insert(0, "/home/curved/vllm-gfx908")
os.environ["VLLM_G128_PREFILL_GLUON"] = "1"

from triton.experimental import gluon as g  # noqa: E402
from triton.experimental.gluon import language as gl  # noqa: E402

import vllm.v1.attention.ops.gfx908_g128_gluon_prefill as pf  # noqa: E402

dev = "cuda"
G, BLOCK, NQ, NKV, D = 128, 1664, 6, 1, 256
SEQS, QTOK, CTX = 1, 2048, 8192
PAD = D + 2 * (D // G)
CONTENT = 2 * PAD
NB = SEQS * ((CTX + BLOCK - 1) // BLOCK) + 8
packed = torch.randint(-127, 127, (NB, NKV, BLOCK, CONTENT),
                       device=dev, dtype=torch.int8)
k_data = packed.transpose(1, 2)[..., :D]
v_data = packed.transpose(1, 2)[..., PAD:PAD + D]
raw = packed.untyped_storage()
base_f16 = torch.tensor([], dtype=torch.float16, device=dev).set_(raw)


def f16u(n):
    return n // 2


g8_k = torch.as_strided(
    base_f16, (NB, BLOCK, NKV, D // G),
    (f16u(packed.stride(0)), f16u(packed.stride(2)), f16u(packed.stride(1)), 1),
    storage_offset=f16u(D),
)
g8_v = torch.as_strided(
    base_f16, (NB, BLOCK, NKV, D // G),
    (f16u(packed.stride(0)), f16u(packed.stride(2)), f16u(packed.stride(1)), 1),
    storage_offset=f16u(PAD + D),
)
q = torch.randn(QTOK, NQ, D, device=dev, dtype=torch.bfloat16)
out = torch.empty_like(q)
cu_q = torch.tensor([0, QTOK], device=dev, dtype=torch.int32)
seqused = torch.full((SEQS,), CTX, device=dev, dtype=torch.int32)
bt = torch.arange(
    (CTX + BLOCK - 1) // BLOCK, device=dev, dtype=torch.int32
).view(1, -1).contiguous()

pf.g128_prefill_core[(QTOK // 64 + SEQS, NQ)](
    q, k_data, v_data, g8_k, g8_v, bt, seqused, cu_q, out,
    SCALE=D ** -0.5, NUM_SEQS=SEQS, NUM_QHEADS=NQ, NQ_PER_KV=NQ,
    BLOCK_SIZE=BLOCK,
    BT_STRIDE=bt.stride(0), Q_STRIDE0=q.stride(0), Q_STRIDE1=q.stride(1),
    K_STRIDE0=k_data.stride(0), K_STRIDE1=k_data.stride(1),
    K_STRIDE2=k_data.stride(2),
    V_STRIDE0=v_data.stride(0), V_STRIDE1=v_data.stride(1),
    V_STRIDE2=v_data.stride(2),
    S_STRIDE0=g8_k.stride(0), S_STRIDE1=g8_k.stride(1), S_STRIDE2=g8_k.stride(2),
    OUT_STRIDE0=out.stride(0), OUT_STRIDE1=out.stride(1),
    MMA_DT=gl.float16, MMA_FP16=True,
    num_warps=4,
)
torch.cuda.synchronize()

kern = pf.g128_prefill_core
for key, compiled in kern.device_caches[0][0].items() if hasattr(
    kern, "device_caches"
) else []:
    pass

# Triton keeps compiled kernels on the JITFunction; find any CompiledKernel
found = None
for cache in getattr(kern, "device_caches", {}).values():
    d = cache[0] if isinstance(cache, tuple) else cache
    if hasattr(d, "values"):
        for ck in d.values():
            found = ck
            break
if found is None and hasattr(kern, "cache"):
    for ckdict in kern.cache.values():
        for ck in ckdict.values():
            found = ck
if found is None:
    print("could not locate compiled kernel object; attrs:",
          [a for a in dir(kern) if "cache" in a.lower()])
    sys.exit(1)

asm = getattr(found, "asm", None)
if asm is None:
    print("compiled kernel has no asm attr:", type(found))
    sys.exit(1)
ptx = asm.get("ptx", "")
open("/tmp/prefill_core.ptx", "w").write(ptx)
counts = {}
for m in re.finditer(r"ld\.global(\.[a-z0-9]+)?(?:\.v(\d))?\.([a-z0-9]+)", ptx):
    vec = m.group(2) or "1"
    typ = m.group(3)
    width = {"u8": 1, "s8": 1, "b8": 1, "u16": 2, "b16": 2, "f16": 2,
             "b32": 4, "u32": 4, "f32": 4, "b64": 8, "f64": 8,
             "b128": 16}.get(typ, 0)
    key = f"v{vec}x{typ}"
    counts[key] = counts.get(key, 0) + 1
total_bytes = sum(int(k.split("x")[0][1:]) * {"u8": 1, "s8": 1, "b8": 1,
                                              "u16": 2, "b16": 2, "f16": 2,
                                              "b32": 4, "u32": 4, "f32": 4,
                                              "b64": 8, "f64": 8,
                                              "b128": 16}.get(
    k.split("x")[1], 0) * v for k, v in counts.items())
print("global load widths:", dict(sorted(counts.items(),
                                         key=lambda kv: -kv[1])))
print(f"total global-load bytes/instruction estimate: {total_bytes}")
print("PTX at /tmp/prefill_core.ptx")
