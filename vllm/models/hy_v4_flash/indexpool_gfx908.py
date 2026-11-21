# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MI100 preparation of grouped INT8 keys, queries and logical attention indices."""

import torch

from vllm.models.hy_v4_flash.mla_gfx908 import _quantize_int8
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import _hadamard_stage


@triton.jit
def _index_quant(
    V,
    Q,
    S,
    M,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    GROUP: tl.constexpr,
    ROWS: tl.constexpr,
):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.arange(0, 128)
    v = tl.load(
        V + r[:, None] * ROW_STRIDE + c[None, :] * COL_STRIDE, r[:, None] < M, 0
    ).to(tl.float32)
    for stage in tl.static_range(7):
        v = _hadamard_stage(v, 1 << stage, ROWS * 128, 128)
    codes, scales = _quantize_int8(v * 0.08838834764831845, ROWS * 128, GROUP)
    tl.store(
        Q + r[:, None] * 128 + c[None, :], codes.reshape((ROWS, 128)), r[:, None] < M
    )
    g = tl.arange(0, 128 // GROUP)
    tl.store(
        S + r[:, None] * (128 // GROUP) + g[None, :],
        scales.reshape((ROWS, 128 // GROUP)),
        r[:, None] < M,
    )


def quantize_index_vectors(vectors, group, *, rows=None, warps=4):
    if (
        vectors.ndim != 2
        or vectors.shape[1] != 128
        or group not in (4, 8, 16, 32, 64, 128)
    ):
        raise ValueError(
            "Expected 128-wide IndexPool rows and a valid signed INT8 group"
        )
    m = vectors.shape[0]
    if rows is None:
        if m <= 128:
            rows = 1
        elif m <= 16384:
            rows = 8
        else:
            rows = 4 if group == 128 else 16
    codes = torch.empty((m, 128), device=vectors.device, dtype=torch.int8)
    scales = torch.empty((m, 128 // group), device=vectors.device, dtype=torch.float16)
    if m:
        _index_quant[(triton.cdiv(m, rows),)](
            vectors,
            codes,
            scales,
            m,
            *vectors.stride(),
            group,
            rows,
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return codes, scales


@triton.jit
def _expand_sorted(
    GROUPS,
    SCORES,
    POS,
    LIVE,
    OUT,
    N: tl.constexpr,
    SELECTED: tl.constexpr,
    R: tl.constexpr,
    SORT: tl.constexpr,
    GROUP_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    GB: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    pos = tl.load(POS + row)
    upper = pos // R
    live = tl.load(LIVE + row) & (pos >= 0)
    i = tl.arange(0, GB)
    ids = tl.load(GROUPS + row * GROUP_STRIDE + i, i < SELECTED, -1).to(tl.int32)
    if SORT:
        empty = (
            -9223372036854775808
            if SCORES.dtype.element_ty == tl.int64
            else -float("inf")
        )
        score = tl.load(SCORES + row * SELECTED + i, i < SELECTED, empty)
        ids = tl.sort(tl.where(score != empty, ids, 2147483647))
    # The selected IDs are now chronological, with missing groups at the end.
    count = tl.sum(((ids >= 0) & (ids < upper)).to(tl.int32), 0)
    col = tl.arange(0, B)
    if SORT:
        group = tl.gather(ids, tl.minimum(col // R, GB - 1), 0)
    else:
        group = tl.load(GROUPS + row * GROUP_STRIDE + col // R, col < count * R, 0)
    expanded = group * R + col % R
    tail = upper * R + col - count * R
    value = tl.where(col < count * R, expanded, tl.where(tail <= pos, tail, -1))
    tl.store(OUT + row * OUT_STRIDE + col, tl.where(live, value, -1), col < (N + 1) * R)


def expand_sorted_groups(
    groups, positions, ratio, live, out, *, warps=None, scores=None
):
    """Restore chronological order when consuming raw top-k IDs and values."""
    n = out.shape[1] // ratio - 1
    selected = groups.shape[1]
    if selected > n or (scores is None and selected != n):
        raise ValueError("Invalid number of selected IndexPool groups")
    width = (n + 1) * ratio
    if warps is None:
        warps = 8 if positions.numel() >= 1024 and width >= 2048 else 4
    if out.shape != (positions.numel(), width) or groups.shape[0] != positions.numel():
        raise ValueError(
            "The IndexPool output must cover all selected groups and the tail"
        )
    if positions.numel():
        _expand_sorted[(positions.numel(),)](
            groups,
            scores if scores is not None else groups,
            positions,
            live,
            out,
            n,
            selected,
            ratio,
            scores is not None,
            groups.stride(0),
            out.stride(0),
            triton.next_power_of_2(n),
            triton.next_power_of_2(width),
            num_warps=warps,
        )
    return out


@triton.jit
def _radix_digit(ordered, above, equal, slots, shift: tl.constexpr):
    bins = tl.arange(0, 256)
    digit = ((ordered >> shift) & 255).to(tl.int32)
    hist = tl.histogram(digit, 256, mask=equal)
    higher = tl.sum(hist, 0) - tl.cumsum(hist, 0)
    chosen = tl.max(tl.where((higher < slots) & (slots <= higher + hist), bins, 0), 0)
    slots -= tl.sum(tl.where(bins == chosen, higher, 0), 0)
    return above | (equal & (digit > chosen)), equal & (digit == chosen), slots


@triton.jit
def _radix_membership(scores, present, budget: tl.constexpr, block: tl.constexpr):
    """Find the FP32 cutoff by byte, keeping the earliest ties."""
    bits = scores.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    ordered = tl.where(scores == 0.0, 0x80000000, ordered)
    above = tl.full((block,), False, tl.int1)
    above, equal, slots = _radix_digit(ordered, above, present, budget, 24)
    above, equal, slots = _radix_digit(ordered, above, equal, slots, 16)
    # Skip lower bytes once the complete bin fits or its maximum is sufficient.
    if slots == 1:
        equal &= ordered == tl.max(tl.where(equal, ordered, 0), 0)
    elif tl.sum(equal.to(tl.int32), 0) > slots:
        above, equal, slots = _radix_digit(ordered, above, equal, slots, 8)
        if slots == 1:
            equal &= ordered == tl.max(tl.where(equal, ordered, 0), 0)
        elif tl.sum(equal.to(tl.int32), 0) > slots:
            above, equal, slots = _radix_digit(ordered, above, equal, slots, 0)
    return above | (equal & (tl.cumsum(equal.to(tl.int32), 0) <= slots))


@triton.jit
def _expand_index_row(
    logits,
    external_cutoff,
    pos,
    live,
    out,
    upper,
    width: tl.constexpr,
    n: tl.constexpr,
    ratio: tl.constexpr,
    block: tl.constexpr,
    out_block: tl.constexpr,
    external: tl.constexpr,
):
    col = tl.arange(0, block)
    scores = tl.load(logits + col, col < tl.minimum(upper, width), -float("inf"))
    present = scores != -float("inf")
    count = tl.sum(present.to(tl.int32), 0)
    if count > n:
        if external or block <= 2048:
            if external:
                cutoff = external_cutoff
            else:
                k: tl.constexpr = min(block, triton.next_power_of_2(n))
                best = tl.topk(scores, k)
                cutoff = tl.max(
                    tl.where(tl.arange(0, k) == n - 1, best, -float("inf")), 0
                )
            above = scores > cutoff
            equal = scores == cutoff
            slots = n - tl.sum(above.to(tl.int32), 0)
            take = above | (equal & (tl.cumsum(equal.to(tl.int32), 0) <= slots))
        else:
            take = _radix_membership(scores, present, n, block)
        count = n
    else:
        take = present
    dest = tl.cumsum(take.to(tl.int32), 0) - 1
    for offset in tl.static_range(ratio):
        tl.store(
            out + tl.maximum(dest, 0) * ratio + offset,
            col * ratio + offset,
            live & take,
        )
    col_out = tl.arange(0, out_block)
    tail = pos // ratio * ratio + col_out - count * ratio
    value = tl.where(live & (tail <= pos), tail, -1)
    tl.store(
        out + col_out,
        value,
        (col_out < (n + 1) * ratio) & ((col_out >= count * ratio) | ~live),
    )


@triton.jit
def _expand_threshold(
    LOGITS,
    TOP,
    POS,
    LIVE,
    OUT,
    WIDTH: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
    BO: tl.constexpr,
    LOCAL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    pos = tl.load(POS + row)
    live = tl.load(LIVE + row) & (pos >= 0)
    upper = pos // R
    base = LOGITS + row * WIDTH
    out = OUT + row * ((N + 1) * R)
    if not LOCAL:
        j = tl.arange(0, triton.next_power_of_2(K))
        cutoff = tl.min(tl.load(TOP + row * K + j, j < K, float("inf")), 0)
        _expand_index_row(base, cutoff, pos, live, out, upper, WIDTH, N, R, B, BO, True)
    else:
        # A captured decode graph reserves the maximum context. Read and scan
        # only the history used by this query, without a host length sync.
        if upper <= N:
            _expand_index_row(
                base,
                0.0,
                pos,
                live,
                out,
                upper,
                WIDTH,
                N,
                R,
                min(B, triton.next_power_of_2(N)),
                BO,
                False,
            )
        elif WIDTH > 2048 and upper <= 2048:
            _expand_index_row(
                base, 0.0, pos, live, out, upper, WIDTH, N, R, 2048, BO, False
            )
        elif WIDTH > 4096 and upper <= 4096:
            _expand_index_row(
                base, 0.0, pos, live, out, upper, WIDTH, N, R, 4096, BO, False
            )
        elif WIDTH > 8192 and upper <= 8192:
            _expand_index_row(
                base, 0.0, pos, live, out, upper, WIDTH, N, R, 8192, BO, False
            )
        else:
            _expand_index_row(
                base, 0.0, pos, live, out, upper, WIDTH, N, R, B, BO, False
            )


def local_group_selection(rows, width):
    return width <= 2048 or (rows <= 128 and width <= 16384)


def select_and_expand(logits, positions, ratio, live, out):
    """Select historical tokens deterministically and append the causal tail."""
    rows, width = logits.shape
    budget = out.shape[1] // ratio - 1
    k = min(budget, width)
    if not logits.is_floating_point():
        top, indices = logits.topk(k, dim=-1, sorted=False)
        return expand_sorted_groups(indices, positions, ratio, live, out, scores=top)
    local = local_group_selection(rows, width)
    top = logits if local else logits.topk(k, dim=-1, sorted=False).values
    _expand_threshold[(rows,)](
        logits,
        top,
        positions,
        live,
        out,
        width,
        budget,
        k,
        ratio,
        triton.next_power_of_2(width),
        triton.next_power_of_2(out.shape[1]),
        local,
        num_warps=8,
    )
    return out


@triton.jit
def _pack_verify_groups(
    SELECTED,
    PACKED,
    LOCAL_ROWS: tl.constexpr,
    FIRST_ROW: tl.constexpr,
    GROUPS: tl.constexpr,
    RATIO: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.arange(0, GROUPS)
    local = row - FIRST_ROW
    first = tl.load(
        SELECTED + (local * GROUPS + group) * RATIO,
        (local >= 0) & (local < LOCAL_ROWS),
        -1,
    )
    # One rank owns each row. Adding one represents -1 padding as zero.
    tl.store(PACKED + row * GROUPS + group, first.to(tl.float32) + 1.0)


@triton.jit
def _unpack_verify_groups(
    PACKED,
    POS,
    OUT,
    GROUPS: tl.constexpr,
    RATIO: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, GROUPS * RATIO)
    first = tl.load(PACKED + row * GROUPS + col // RATIO).to(tl.int32) - 1
    token = first + col % RATIO
    position = tl.load(POS + row)
    tl.store(
        OUT + row * GROUPS * RATIO + col,
        tl.where((first >= 0) & (token <= position), token, -1),
    )


def gather_verify_indices(tp, selected, positions, out, ratio):
    """Exchange group starts and reconstruct the causally masked tail on each rank."""
    rows, width = out.shape
    groups = width // ratio
    local_rows = selected.shape[0]
    staging = torch.empty((rows, groups), dtype=torch.float32, device=out.device)
    _pack_verify_groups[(rows,)](
        selected, staging, local_rows, local_rows * tp.rank_in_group, groups, ratio
    )
    packed = tp.device_communicator.ca_comm.custom_all_reduce(staging)
    if packed is None:
        torch.distributed.all_gather_into_tensor(out, selected, group=tp.device_group)
        return
    _unpack_verify_groups[(rows,)](packed, positions, out, groups, ratio)
