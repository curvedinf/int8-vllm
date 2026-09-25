# SPDX-License-Identifier: Apache-2.0
"""gfx908 Gluon split-KV reduce for the g128 decode path.

One program per (token, head): loads the per-segment LSE partials
(PMAX/PSUM/PARTIAL in the layout g128_core stores), merges them with a
log-sum-exp reduction, and writes the normalized output row. Mirrors the
Triton ``reduce_segments`` math op-for-op so results are bit-compatible.

The merge is a statically-unrolled per-segment loop over [D]-shaped rows with
scalar per-segment maxima - no expand_dims and no cross-axis reductions,
which Gluon's layout verifier rejects for this shape on gfx908.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def reduce_segments_gfx908(
    OUT,
    PARTIAL,
    PMAX,
    PSUM,
    SEQLENS,
    OUT_STRIDE0: gl.constexpr,
    OUT_STRIDE1: gl.constexpr,
    H: gl.constexpr,
    SPLITS: gl.constexpr,
    TILE: gl.constexpr,
    D: gl.constexpr,
):
    token = gl.program_id(0)
    head = gl.program_id(1)

    seq_len = gl.load(SEQLENS + token)
    tiles_per_seg = (seq_len + SPLITS * TILE - 1) // (SPLITS * TILE)
    denom = tiles_per_seg * TILE
    act_num_segments = (seq_len + denom - 1) // denom

    row: gl.constexpr = gl.SliceLayout(1, gl.BlockedLayout(
        size_per_thread=[1, D // 128],
        threads_per_warp=[1, 64],
        warps_per_cta=[1, 4],
        order=[1, 0],
    ))
    doff = gl.arange(0, D, layout=row)

    base_m = token * (H * SPLITS) + head * SPLITS
    base_p = token * (H * SPLITS * D) + head * (SPLITS * D)

    # seed from segment 0, then fold the rest (avoids gl.full so every value
    # carries the row layout from a real load)
    sm0 = gl.load(PMAX + base_m)
    sm0 = gl.where(0 < act_num_segments, sm0, float("-inf"))
    se0 = gl.load(PSUM + base_m)
    se0 = gl.where(0 < act_num_segments, se0, 0.0)
    p0 = gl.load(PARTIAL + base_p + doff)
    p0 = gl.where(0 < act_num_segments, p0, 0.0)

    m = sm0
    for s in gl.static_range(1, SPLITS):
        sm = gl.load(PMAX + base_m + s)
        sm = gl.where(s < act_num_segments, sm, float("-inf"))
        m = gl.maximum(m, sm)

    r0 = gl.exp(sm0 - m)
    l = se0 * r0
    acc = p0 * r0
    for s in gl.static_range(1, SPLITS):
        sm = gl.load(PMAX + base_m + s)
        sm = gl.where(s < act_num_segments, sm, float("-inf"))
        se = gl.load(PSUM + base_m + s)
        se = gl.where(s < act_num_segments, se, 0.0)
        p = gl.load(PARTIAL + base_p + s * D + doff)
        p = gl.where(s < act_num_segments, p, 0.0)
        r = gl.exp(sm - m)
        l += se * r
        acc += p * r

    out = acc / l
    ooff = token * OUT_STRIDE0 + head * OUT_STRIDE1 + doff
    gl.store(OUT + ooff, out)
