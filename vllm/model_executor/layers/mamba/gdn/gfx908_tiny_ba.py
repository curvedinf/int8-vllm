# SPDX-License-Identifier: Apache-2.0
"""Small BF16 GDN gate projection for gfx908 decode batches."""

import torch
import triton
import triton.language as tl


@triton.jit
def _tiny_ba_kernel(
    X,
    W,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    red = tl.arange(0, BK)
    accum = tl.full((BN, BK), 0, tl.float32)
    for tile in range(tl.cdiv(K, BK)):
        k = tile * BK + red
        x = tl.load(X + row * K + k, mask=k < K, other=0).to(tl.float32)
        w = tl.load(
            W + col[:, None] * K + k[None, :],
            mask=(col[:, None] < N) & (k[None, :] < K),
            other=0,
        ).to(tl.float32)
        accum += x[None, :] * w
    value = tl.sum(accum, axis=1)
    tl.store(Y + row * N + col, value, mask=col < N)


def tiny_ba_projection(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Project a small token batch through contiguous BF16 gate weights."""
    rows, inner = x.shape
    cols = weight.shape[0]
    out = torch.empty((rows, cols), dtype=x.dtype, device=x.device)
    _tiny_ba_kernel[(rows, triton.cdiv(cols, 4))](
        x, weight, out, N=cols, K=inner, BN=4, BK=128, num_warps=4
    )
    return out
