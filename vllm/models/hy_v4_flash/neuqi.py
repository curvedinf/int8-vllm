# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Training-free grouped affine INT4 using NeUQI (arXiv:2505.17595).

Stored weights reconstruct as scale * (unsigned_nibble - zero). Both
scale and zero are floating point, and activations use signed INT8.
"""

from dataclasses import dataclass

import torch

GROUP_SIZE = 128
LEVELS = 15


def _best_offset(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Exact shift for a fixed step using the paper's two-unit search interval.

    The clipping surrogate is convex. Locate its minimizer by bisection, then
    enumerate the rounding discontinuities within one step on either side.
    On each interval the weighted error is a quadratic with unit curvature.
    """
    lo = x.amin(-1, keepdim=True) - LEVELS - 1
    hi = x.amax(-1, keepdim=True) + 1
    for _ in range(23):
        center = (lo + hi) * 0.5
        delta = center - x
        slope = torch.where(
            delta < -LEVELS - 0.5,
            delta + LEVELS,
            torch.where(delta > 0.5, delta, 0),
        )
        right = (h * slope).sum(-1, keepdim=True) < 0
        lo, hi = torch.where(right, center, lo), torch.where(right, hi, center)
    left = (lo + hi) * 0.5 - 1
    right = left + 2
    level = (x + 0.5 - left).floor()
    y = x - level.clamp(0, LEVELS)
    start1 = (h * y).sum(-1, keepdim=True)
    start2 = (h * y.square()).sum(-1, keepdim=True)
    first = x + 0.5 - level
    valid0 = (level >= 1) & (level <= LEVELS) & (first <= right)
    valid1 = (level >= 2) & (level <= LEVELS + 1) & (first + 1 <= right)
    times = torch.cat(
        (torch.where(valid0, first, right), torch.where(valid1, first + 1, right)), -1
    )
    mass = torch.cat((h * valid0, h * valid1), -1)
    times, order = times.sort(dim=-1)
    mass = mass.gather(-1, order)
    s1 = torch.cat((start1, start1 + mass.cumsum(-1)), -1)
    s2 = torch.cat((start2, start2 + (2 * times * mass).cumsum(-1)), -1)
    bounds = torch.cat((left, times, right), -1)
    optimum = s1.clamp(min=bounds[..., :-1], max=bounds[..., 1:])
    score = s2 + optimum * (optimum - 2 * s1)
    return optimum.gather(-1, score.argmin(-1, keepdim=True))


@dataclass
class AffineInt4:
    codes: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor


@torch.no_grad()
def quantize_rows(
    weights: torch.Tensor,
    h_diag: torch.Tensor,
    coarse_steps: int = 8,
    fine_steps: int = 32,
    *,
    fused: bool = True,
) -> AffineInt4:
    """Fit independent G128 quantizers to bounded GPU row tiles.

    h_diag is the measured diagonal input second moment, broadcastable to the
    row tile, including a separate diagonal per routed expert where available.
    The scale search minimizes error after rounding both metadata fields to
    their deployed FP16 representation.
    """
    if not weights.is_cuda or h_diag.device != weights.device:
        raise ValueError("NeUQI weights and calibration must reside on the same GPU")
    if weights.shape[-1] % GROUP_SIZE or weights.ndim < 2:
        raise ValueError("NeUQI requires complete groups of 128 input features")
    if coarse_steps < 2 or fine_steps < coarse_steps or fine_steps % coarse_steps:
        raise ValueError("fine_steps must be a multiple of coarse_steps")
    if h_diag.shape[-1] != weights.shape[-1]:
        raise ValueError("Calibration input width differs from the weight tile")
    if (
        fused
        and weights.ndim == 2
        and weights.is_contiguous()
        and h_diag.stride(-1) == 1
    ):
        from vllm.models.hy_v4_flash.neuqi_kernel import fit_gpu

        return AffineInt4(*fit_gpu(weights, h_diag, coarse_steps, fine_steps))
    width = weights.shape[-1] // GROUP_SIZE
    shape = (*weights.shape[:-1], width, GROUP_SIZE)
    w = weights.float().reshape(shape)
    h = h_diag.float().broadcast_to(weights.shape).reshape(shape)
    h = h.clamp_min(0)
    norm = h.sum(-1, keepdim=True)
    h = torch.where(norm > 0, h / norm.clamp_min(1e-30), 1.0 / GROUP_SIZE)
    # Constant groups have a floating translation and reconstruct exactly.
    lower, upper = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
    constant = upper == lower
    full = torch.where(constant, torch.ones_like(upper), (upper - lower) / LEVELS)
    best_score = torch.full_like(full, torch.inf)
    best_scale = torch.ones_like(full)
    best_zero = torch.zeros_like(full)

    def candidate(step):
        nonlocal best_score, best_scale, best_zero
        s = step.clamp_min(2**-24).half().float()
        x = w / s
        z = (-_best_offset(x, h)).half().float()
        code = (x + z).round().clamp_(0, LEVELS)
        error = (h * (s * (code - z) - w).square()).sum(-1, keepdim=True)
        better = error < best_score
        best_scale = torch.where(better, s, best_scale)
        best_zero = torch.where(better, z, best_zero)
        best_score = torch.minimum(error, best_score)

    for i in range(max(1, coarse_steps // 2), coarse_steps + 1):
        candidate(full * (i / coarse_steps))
    center = best_scale
    for i in range(1, fine_steps // coarse_steps // 2 + 1):
        candidate(center - full * (i / fine_steps))
        candidate(center + full * (i / fine_steps))
    scale = torch.where(constant, full, best_scale)
    zero = torch.where(constant, -lower, best_zero)
    q = (w / scale + zero).round().clamp_(0, LEVELS).to(torch.uint8)
    nibbles = q[..., 0::2] | (q[..., 1::2] << 4)
    return AffineInt4(
        nibbles.reshape(*weights.shape[:-1], -1),
        scale.squeeze(-1).half(),
        zero.squeeze(-1).half(),
    )
