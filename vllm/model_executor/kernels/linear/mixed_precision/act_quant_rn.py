"""Fused per-token int8 activation quant with round-to-nearest for gfx908.

Replaces the aiter ``pertoken_quant`` eager chain (fp32 cast -> abs -> max ->
div -> trunc-cast, ~4 passes over [M,K]) with a one-kernel two-sweep Triton
implementation that rounds to nearest-even instead of truncating toward zero.
Phase-1 replay measured the truncation leg at 10-15% mean rel-L2 per GEMM
output; rounding alone recovers roughly half of it offline.

Selected per process via ``VLLM_GFX908_ACT_QUANT=round`` in
``aiter_w8a16.apply_weights``; the default remains the aiter trunc path.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _pertoken_quant_rn_kernel(
    X,  # [M, K] fp16, row-major
    Q,  # [M, K] int8 out
    S,  # [M, 1] fp32 out
    K,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    x_ptr = X + row * K
    q_ptr = Q + row * K

    amax = -1.0
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        a = tl.abs(x)
        amax = tl.maximum(amax, tl.max(a, axis=0))

    s = tl.maximum(amax, 0.0) / 127.0
    s = tl.where(s == 0.0, 1.0, s)
    tl.store(S + row, s)

    rcp = 1.0 / s
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        q = libdevice.rint(x * rcp)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(q_ptr + offs, q.to(tl.int8), mask=mask)


def pertoken_quant_rn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token dynamic symmetric int8 quant, round-to-nearest-even.

    Returns (q, s): q int8 [M,K], s fp32 [M,1] — the same contract as
    aiter.pertoken_quant(x, quant_dtype=torch.int8).
    """
    assert x.dim() == 2 and x.dtype in (torch.float16, torch.bfloat16)
    x = x.contiguous()
    M, K = x.shape
    q = torch.empty((M, K), dtype=torch.int8, device=x.device)
    s = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    BLOCK = 4096 if K <= 8192 else 8192
    _pertoken_quant_rn_kernel[(M,)](x, q, s, K, BLOCK=BLOCK, num_warps=8)
    return q, s
