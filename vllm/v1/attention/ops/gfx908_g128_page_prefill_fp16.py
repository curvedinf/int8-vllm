# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Causal G128 page prefill with selectable FP16/BF16 MMA.

The original gfx908_g128_page_prefill module is root-owned in this checkout.
This copy keeps the page-outer lookup and 4256-token page contract, and adds
the MMA dtype knob measured at ~2.1-2.3x on MI100 (FP16 vs BF16).
"""

import os

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import (
    cdiv_fn,
    compute_tile_loop_bounds,
    resolve_seq_and_query_len,
    softmax_step,
)


@triton.jit
def page_prefill_kernel(
    output_ptr,
    query_ptr,
    key_ptr,
    value_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    query_start_ptr,
    scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    table_stride: tl.int64,
    query_stride0: tl.int64,
    query_stride1: tl.int64,
    output_stride0: tl.int64,
    output_stride1: tl.int64,
    k_stride0: tl.int64,
    k_stride1: tl.int64,
    k_stride2: tl.int64,
    k_stride3: tl.constexpr,
    v_stride0: tl.int64,
    v_stride1: tl.int64,
    v_stride2: tl.int64,
    v_stride3: tl.constexpr,
    scale_stride0: tl.int64,
    scale_stride1: tl.int64,
    scale_stride2: tl.int64,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 4256,
    TILE_SIZE: tl.constexpr = 32,
    HEAD_SIZE: tl.constexpr = 128,
    MMA_DT: tl.constexpr = tl.float16,
):
    tl.static_assert(BLOCK_SIZE == 4256 and TILE_SIZE == 32)
    tl.static_assert(HEAD_SIZE == 128 and num_queries_per_kv == 8)
    tl.static_assert(BLOCK_M == 16 or BLOCK_M == 64)
    BLOCK_Q: tl.constexpr = BLOCK_M // num_queries_per_kv
    TILES_PER_PAGE: tl.constexpr = BLOCK_SIZE // TILE_SIZE
    q_block = tl.program_id(0)
    kv_head = tl.program_id(1)
    seq_idx, local_block, first_query, query_len, seq_len = resolve_seq_and_query_len(
        query_start_ptr, seq_lens_ptr, q_block, num_seqs, BLOCK_Q
    )
    if local_block * BLOCK_Q >= query_len:
        return
    rows = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_SIZE)
    keys = tl.arange(0, TILE_SIZE)
    query_pos = local_block * BLOCK_Q + rows // num_queries_per_kv
    query_row = first_query + query_pos
    query_head = kv_head * num_queries_per_kv + rows % num_queries_per_kv
    row_valid = query_pos < query_len
    head_valid = query_head < num_query_heads
    query_offset = (
        query_row[:, None] * query_stride0
        + query_head[:, None] * query_stride1
        + dims[None, :]
    )
    Q = tl.load(
        query_ptr + query_offset,
        mask=row_valid[:, None] & head_valid[:, None],
        other=0.0,
    ).to(MMA_DT)
    M = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    L = tl.full((BLOCK_M,), 1.0, tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_SIZE), tl.float32)
    context_len = seq_len - query_len
    loop_lo, loop_hi, max_prefix = compute_tile_loop_bounds(
        context_len,
        seq_len,
        query_len,
        local_block,
        0,
        0,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        0,
        False,
        False,
        True,
        False,
        -1,
        -1,
    )
    # Preserve the original key-tile recurrence, including page boundaries.
    for page in range(loop_lo // TILES_PER_PAGE, cdiv_fn(loop_hi, TILES_PER_PAGE)):
        physical = tl.load(block_tables_ptr + seq_idx * table_stride + page).to(
            tl.int64
        )
        first_tile = tl.maximum(loop_lo, page * TILES_PER_PAGE)
        last_tile = tl.minimum(loop_hi, (page + 1) * TILES_PER_PAGE)
        for j in range(first_tile, last_tile):
            seq_offset = j * TILE_SIZE + keys
            slots = seq_offset - page * BLOCK_SIZE
            tile_valid = seq_offset < max_prefix
            k_offset = (
                physical * k_stride0
                + kv_head * k_stride2
                + dims[:, None] * k_stride3
                + slots[None, :] * k_stride1
            )
            v_offset = (
                physical * v_stride0
                + kv_head * v_stride2
                + dims[None, :] * v_stride3
                + slots[:, None] * v_stride1
            )
            K = tl.load(key_ptr + k_offset, mask=tile_valid[None, :], other=0).to(
                tl.float32
            )
            V = tl.load(value_ptr + v_offset, mask=tile_valid[:, None], other=0).to(
                tl.float32
            )
            scale_offset = (
                physical * scale_stride0
                + slots * scale_stride1
                + kv_head * scale_stride2
            )
            sk = tl.load(k_scale_ptr + scale_offset, mask=tile_valid, other=1.0).to(
                tl.float32
            )
            sv = tl.load(v_scale_ptr + scale_offset, mask=tile_valid, other=1.0).to(
                tl.float32
            )
            K = (K * sk[None, :]).to(MMA_DT)
            V = (V * sv[:, None]).to(MMA_DT)
            seq_mask = seq_offset[None, :] <= context_len + query_pos[:, None]
            S = tl.zeros((BLOCK_M, TILE_SIZE), tl.float32)
            S += scale * tl.dot(Q, K)
            S = tl.where(
                row_valid[:, None] & head_valid[:, None] & seq_mask, S, float("-inf")
            )
            M, L, P, alpha = softmax_step(S, M, L)
            acc = acc * alpha[:, None]
            acc += tl.dot(P.to(MMA_DT), V)
    acc = acc / L[:, None]
    output_offset = (
        query_row[:, None] * output_stride0
        + query_head[:, None] * output_stride1
        + dims[None, :]
    )
    tl.store(
        output_ptr + output_offset, acc, mask=row_valid[:, None] & head_valid[:, None]
    )


def page_prefill(q, k, v, sk, sv, out, starts, lens, table, scale, *, block_m=64):
    assert k.shape[1] == v.shape[1] == 4256
    assert sk.stride() == sv.stride()
    heads = q.shape[1]
    per_kv = heads // k.shape[2]
    block_q = block_m // per_kv
    mode = os.environ.get("VLLM_G128_PAGE_MMA", "fp16")
    mma_dt = tl.float16 if mode == "fp16" else tl.bfloat16
    return page_prefill_kernel[
        (q.shape[0] // block_q + lens.numel(), k.shape[2])
    ](
        out,
        q,
        k,
        v,
        sk,
        sv,
        table,
        lens,
        starts,
        scale,
        heads,
        per_kv,
        table.stride(0),
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        *k.stride(),
        *v.stride(),
        sk.stride(0),
        sk.stride(1),
        sk.stride(2),
        lens.numel(),
        block_m,
        MMA_DT=mma_dt,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
