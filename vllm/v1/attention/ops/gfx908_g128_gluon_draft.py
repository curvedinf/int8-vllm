# SPDX-License-Identifier: Apache-2.0
"""Gfx908 grouped-int8 DFlash2 sliding-window verify attention."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def draft_g128_core(
    Q,
    K,
    V,
    SK,
    SV,
    BT,
    SEQLENS,
    CUQ,
    PARTIAL,
    PMAX,
    PSUM,
    SCALE: gl.constexpr,
    WINDOW: gl.constexpr,
    NUM_SEQS: gl.constexpr,
    NUM_QHEADS: gl.constexpr,
    NQ_PER_KV: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    SPLITS: gl.constexpr,
    BT_STRIDE: gl.constexpr,
    Q_STRIDE0: gl.constexpr,
    Q_STRIDE1: gl.constexpr,
    K_STRIDE0: gl.constexpr,
    K_STRIDE1: gl.constexpr,
    K_STRIDE2: gl.constexpr,
    V_STRIDE0: gl.constexpr,
    V_STRIDE1: gl.constexpr,
    V_STRIDE2: gl.constexpr,
    S_STRIDE0: gl.constexpr,
    S_STRIDE1: gl.constexpr,
    S_STRIDE2: gl.constexpr,
):
    block_id = gl.program_id(0)
    kv_head = gl.program_id(1)
    seg = gl.program_id(2)
    seq = 0
    for s in gl.static_range(NUM_SEQS):
        start = gl.load(CUQ + s)
        first_block = start // 8 + s
        seq = gl.where(block_id >= first_block, s, seq)
    q_start = gl.load(CUQ + seq)
    q_end = gl.load(CUQ + seq + 1)
    q_len = q_end - q_start
    local_block = block_id - (q_start // 8 + seq)
    if local_block * 8 >= q_len:
        return
    seq_len = gl.load(SEQLENS + seq)
    context = seq_len - q_len

    # Partition the active sliding window, including draft tokens to the
    # right of an earlier noncausal query, across all split-K segments.
    first_key = gl.maximum(context + local_block * 8 - WINDOW + 1, 0)
    first_tile = first_key // 32
    last_tile = (seq_len + 31) // 32
    tiles_per_seg = (last_tile - first_tile + SPLITS - 1) // SPLITS
    lo = first_tile + seg * tiles_per_seg
    hi = gl.minimum(lo + tiles_per_seg, last_tile)

    q_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[4, 16],
        warps_per_cta=[4, 1], order=[1, 0]
    )
    kv_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[4, 16],
        warps_per_cta=[4, 1], order=[1, 0]
    )
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=1, instr_shape=[16, 16, 8],
        transposed=True, warps_per_cta=[4, 1]
    )
    out_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 2], threads_per_warp=[16, 4],
        warps_per_cta=[4, 1], order=[1, 0]
    )
    qm = gl.arange(0, 32, layout=gl.SliceLayout(1, q_layout))
    qd = gl.arange(0, 128, layout=gl.SliceLayout(0, q_layout))
    qpos = local_block * 8 + qm // NQ_PER_KV
    qhead = kv_head * NQ_PER_KV + qm % NQ_PER_KV
    qvalid = (qpos < q_len) & (qhead < NUM_QHEADS)
    qptr = (
        Q + (q_start + qpos[:, None]) * Q_STRIDE0
        + qhead[:, None] * Q_STRIDE1 + qd[None, :]
    )
    q = gl.load(qptr, mask=qvalid[:, None], other=0.0)
    q = gl.convert_layout(q, gl.DotOperandLayout(0, mma, k_width=2))

    score_rows = gl.arange(0, 32, layout=gl.SliceLayout(1, mma))
    score_cols = gl.arange(0, 32, layout=gl.SliceLayout(0, mma))
    score_qpos = local_block * 8 + score_rows // NQ_PER_KV
    score_qhead = kv_head * NQ_PER_KV + score_rows % NQ_PER_KV
    score_qvalid = (score_qpos < q_len) & (score_qhead < NUM_QHEADS)
    m = gl.full((32,), float('-inf'), gl.float32, layout=gl.SliceLayout(1, mma))
    l = gl.full((32,), 1.0, gl.float32, layout=gl.SliceLayout(1, mma))
    acc = gl.full((32, 128), 0.0, gl.float32, layout=mma)

    tn = gl.arange(0, 32, layout=gl.SliceLayout(1, kv_layout))
    kd = gl.arange(0, 128, layout=gl.SliceLayout(0, kv_layout))
    for j in range(lo, hi):
        slot = (j * 32) % BLOCK_SIZE
        physical = gl.load(BT + seq * BT_STRIDE + (j * 32) // BLOCK_SIZE)
        kv_valid = gl.minimum(32, seq_len - j * 32)
        valid_t = tn < kv_valid
        kb = K + physical * K_STRIDE0 + kv_head * K_STRIDE2 + slot * K_STRIDE1
        vb = V + physical * V_STRIDE0 + kv_head * V_STRIDE2 + slot * V_STRIDE1
        off_k = tn[:, None] * K_STRIDE1 + kd[None, :]
        off_v = tn[:, None] * V_STRIDE1 + kd[None, :]
        sptr_k = (
            SK + physical * S_STRIDE0 + kv_head * S_STRIDE2
            + (slot + tn) * S_STRIDE1
        )
        sptr_v = (
            SV + physical * S_STRIDE0 + kv_head * S_STRIDE2
            + (slot + tn) * S_STRIDE1
        )
        if kv_valid == 32:
            k = gl.load(kb + off_k)
            v = gl.load(vb + off_v)
            sk = gl.load(sptr_k).to(gl.float32)
            sv = gl.load(sptr_v).to(gl.float32)
        else:
            k = gl.load(kb + off_k, mask=valid_t[:, None], other=0)
            v = gl.load(vb + off_v, mask=valid_t[:, None], other=0)
            sk = gl.load(sptr_k, mask=valid_t, other=1.0).to(gl.float32)
            sv = gl.load(sptr_v, mask=valid_t, other=1.0).to(gl.float32)
        # Dequantize before MFMA to match the generic draft path's BF16
        # rounding at the K and V interfaces.
        k = (k.to(gl.float32) * sk[:, None]).to(gl.bfloat16)
        v = (v.to(gl.float32) * sv[:, None]).to(gl.bfloat16)
        k = gl.convert_layout(
            gl.permute(k, (1, 0)), gl.DotOperandLayout(1, mma, k_width=2)
        )
        v = gl.convert_layout(v, gl.DotOperandLayout(1, mma, k_width=4))
        scores = gl.full((32, 32), 0.0, gl.float32, layout=mma)
        scores = gl.amd.cdna3.mfma(q, k, scores) * SCALE
        kvpos = j * 32 + score_cols
        dist = context + score_qpos[:, None] - kvpos[None, :]
        mask = (
            score_qvalid[:, None]
            & (kvpos[None, :] < seq_len)
            & (dist < WINDOW)
            & (dist > -WINDOW)
        )
        scores = gl.where(mask, scores, float('-inf'))
        m_new = gl.maximum(m, gl.max(scores, axis=1))
        m_new = gl.where(m_new > float('-inf'), m_new, 0.0)
        p = gl.exp(scores - m_new[:, None])
        alpha = gl.exp(m - m_new)
        l = l * alpha + gl.sum(p, axis=1)
        m = m_new
        acc = acc * alpha[:, None]
        p = gl.convert_layout(p.to(gl.bfloat16), gl.DotOperandLayout(0, mma, k_width=4))
        acc = gl.amd.cdna3.mfma(p, v, acc)

    acc = gl.convert_layout(acc, out_layout)
    om = gl.arange(0, 32, layout=gl.SliceLayout(1, out_layout))
    od = gl.arange(0, 128, layout=gl.SliceLayout(0, out_layout))
    opos = local_block * 8 + om // NQ_PER_KV
    ohead = kv_head * NQ_PER_KV + om % NQ_PER_KV
    ovalid = (opos < q_len) & (ohead < NUM_QHEADS)
    base = (
        PARTIAL + (q_start + opos[:, None]) * (NUM_QHEADS * SPLITS * 128)
        + ohead[:, None] * (SPLITS * 128) + seg * 128 + od[None, :]
    )
    gl.store(base, acc, mask=ovalid[:, None])
    moff = (q_start + score_qpos) * (NUM_QHEADS * SPLITS) + score_qhead * SPLITS + seg
    gl.store(PMAX + moff, m, mask=score_qvalid)
    gl.store(PSUM + moff, l, mask=score_qvalid)
