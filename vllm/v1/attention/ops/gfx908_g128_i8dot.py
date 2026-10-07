# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908 int8-MFMA decode/verify attention core for G128 KV caches.

MI100's int8 matrix rate is 2x every other dtype — the fork's whole
doctrine — but the existing G128 decode paths dequantize KV to fp16/bf16
before the QK dot (generic Triton path) or pay a register-permute + LDS
round-trip per tile (gluon m64 core). Both stream KV at ~6-11 GB/s vs
~980 GB/s of HBM (ledger G128_DECODE_ATTN_9GBPS).

This core keeps K int8 through the dot: Q is quantized per (row,
128-dim group) with round-to-nearest (the same doctrine as
VLLM_GFX908_ACT_QUANT=round), the QK product runs as two int8 MFMA dots
(one per group), and the group scales fold into the score sum
afterwards. V is dequantized in registers (it must meet fp32
probabilities anyway). One program: one sequence, 64 rows = up to 10
verify tokens x 6 query heads, KV streamed in 64-token tiles.

Partials are written in the same (token, head, split, 256) layout the
m64 core uses, so the existing reduce_segments finalizes them.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _g128_i8dot_kernel(
    Q, K, V, SK, SV, BT, SEQLENS, CUQ,
    PARTIAL, PMAX, PSUM,
    scale,
    num_query_heads: tl.constexpr,
    nq_per_kv: tl.constexpr,
    num_seqs: tl.constexpr,
    block_size: tl.constexpr,
    splits: tl.constexpr,
    bt_stride,
    q_stride0,
    q_stride1,
    k_stride0,
    k_stride1,
    k_stride2,
    v_stride0,
    v_stride1,
    v_stride2,
    s_stride0,
    s_stride1,
    s_stride2,
    ROWS: tl.constexpr,   # 64
    TILE: tl.constexpr,   # 64 KV tokens per iteration
    D: tl.constexpr,      # 256
    DG: tl.constexpr,     # 128
):
    pid = tl.program_id(0)
    seg = tl.program_id(1)

    # map pid -> (seq, local q block of 10 tokens); block ids are
    # per-sequence prefixed (start // 10 + seq) like the m64 core.
    seq = 0
    first_block = 0
    for s in tl.static_range(num_seqs):
        start = tl.load(CUQ + s)
        fb = start // 10 + s
        seq = tl.where(pid >= fb, s, seq)
        first_block = tl.where(pid >= fb, fb, first_block)
    q_start = tl.load(CUQ + seq)
    q_end = tl.load(CUQ + seq + 1)
    q_len = q_end - q_start
    local_block = pid - first_block
    if local_block * 10 >= q_len:
        return

    seq_len = tl.load(SEQLENS + seq)
    context = seq_len - q_len
    max_prefix = tl.minimum(context + local_block * 10 + 11, seq_len)

    # ---- Q: ROWS rows = token(10) x head(6); RN-quantize per group ----
    offs_m = tl.arange(0, ROWS)
    qpos = local_block * 10 + offs_m // nq_per_kv
    qhead = offs_m % nq_per_kv
    qvalid = (qpos < q_len) & (qhead < num_query_heads)
    offs_d = tl.arange(0, DG)

    q8_lo = tl.zeros((ROWS, DG), dtype=tl.int8)
    q8_hi = tl.zeros((ROWS, DG), dtype=tl.int8)
    sq_lo = tl.zeros((ROWS,), dtype=tl.float32)
    sq_hi = tl.zeros((ROWS,), dtype=tl.float32)
    for gi in tl.static_range(2):
        ptr = (
            Q
            + (q_start + qpos[:, None]) * q_stride0
            + qhead[:, None] * q_stride1
            + (offs_d[None, :] + gi * DG)
        )
        x = tl.load(ptr, mask=qvalid[:, None], other=0.0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-6)
        s = amax / 127.0
        qi = x / s[:, None]
        qi = qi + tl.where(qi >= 0, 0.5, -0.5)
        qi = tl.floor(qi)
        qi = tl.minimum(tl.maximum(qi, -128.0), 127.0).to(tl.int8)
        if gi == 0:
            q8_lo = qi
            sq_lo = s
        else:
            q8_hi = qi
            sq_hi = s

    m_i = tl.full((ROWS,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((ROWS,), dtype=tl.float32)
    acc = tl.zeros((ROWS, D), dtype=tl.float32)

    tiles_per_seg = tl.cdiv(seq_len, splits * TILE)
    lo = seg * tiles_per_seg
    if lo * TILE >= seq_len:
        return
    hi = tl.minimum((seg + 1) * tiles_per_seg, tl.cdiv(max_prefix, TILE))

    offs_n = tl.arange(0, TILE)
    offs_dv = tl.arange(0, D)
    for j in range(lo, hi):
        slot = (j * TILE) % block_size
        physical = tl.load(
            BT + seq * bt_stride + (j * TILE) // block_size
        ).to(tl.int64)
        kv_valid = tl.minimum(TILE, max_prefix - j * TILE)
        valid_n = offs_n < kv_valid

        kb = K + physical * k_stride0 + slot * k_stride1
        k_off = offs_n[:, None] * k_stride1 + offs_dv[None, :]
        k8 = tl.load(kb + k_off, mask=valid_n[:, None], other=0)  # (TILE, D)
        # split halves via separate masked loads (triton can't slice)
        k_off_lo = offs_n[:, None] * k_stride1 + offs_d[None, :]
        k_off_hi = offs_n[:, None] * k_stride1 + (offs_d[None, :] + DG)
        k_lo_t = tl.load(kb + k_off_lo, mask=valid_n[:, None], other=0)
        k_hi_t = tl.load(kb + k_off_hi, mask=valid_n[:, None], other=0)
        k_lo = tl.trans(k_lo_t)   # (DG, TILE) int8
        k_hi = tl.trans(k_hi_t)   # (DG, TILE) int8

        s_row = SK + physical * s_stride0 + (slot + offs_n) * s_stride1
        sk_lo = tl.load(s_row, mask=valid_n, other=1.0).to(tl.float32)
        sk_hi = tl.load(s_row + 1, mask=valid_n, other=1.0).to(tl.float32)

        p_lo = tl.dot(q8_lo, k_lo, out_dtype=tl.int32).to(tl.float32)
        p_hi = tl.dot(q8_hi, k_hi, out_dtype=tl.int32).to(tl.float32)
        scores = (
            p_lo * (sq_lo[:, None] * sk_lo[None, :])
            + p_hi * (sq_hi[:, None] * sk_hi[None, :])
        ) * scale

        kvpos = j * TILE + offs_n
        mask = qvalid[:, None] & (kvpos[None, :] < max_prefix) & (
            kvpos[None, :] <= context + qpos[:, None]
        )
        scores = tl.where(mask, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        m_new = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
        acc = acc * alpha[:, None]

        vb = V + physical * v_stride0 + slot * v_stride1
        v_off = offs_n[:, None] * v_stride1 + offs_dv[None, :]
        v8 = tl.load(vb + v_off, mask=valid_n[:, None], other=0).to(tl.float32)
        sv_row = SV + physical * s_stride0 + (slot + offs_n) * s_stride1
        sv_lo = tl.load(sv_row, mask=valid_n, other=1.0).to(tl.float32)
        sv_hi = tl.load(sv_row + 1, mask=valid_n, other=1.0).to(tl.float32)
        sv = tl.where(offs_dv[None, :] < DG, sv_lo[:, None], sv_hi[:, None])
        vf = (v8 * sv).to(tl.bfloat16)

        acc = tl.dot(p.to(tl.bfloat16), vf, acc)

    # ---- store per-split partials (m64 layout); reduce finalizes ----
    # NOTE: raw (un-normalized) partial sums; reduce_segments applies
    # exp(max - overall_max) rescaling and the expsum division.
    pbase = (
        PARTIAL
        + (q_start + qpos[:, None]) * (num_query_heads * splits * D)
        + qhead[:, None] * (splits * D)
        + seg * D
        + offs_dv[None, :]
    )
    tl.store(pbase, acc, mask=qvalid[:, None])
    moff = (q_start + qpos) * (num_query_heads * splits) + qhead * splits + seg
    tl.store(PMAX + moff, m_i, mask=qvalid)
    tl.store(PSUM + moff, l_i, mask=qvalid)
