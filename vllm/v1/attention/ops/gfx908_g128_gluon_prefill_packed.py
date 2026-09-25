# SPDX-License-Identifier: Apache-2.0
"""gfx908 grouped-int8 G128 MFMA prefill core, GQA-packed (GOALOPT iter 9).

Packs all 6 GQA query heads into each CTA's row block (rows are
(token, head) pairs, 10 tokens x 6 heads in a 64-row block - the m64
decode-core pattern) so every K/V tile load and group-scale load feeds
all six heads' QK/PV dots, and Q rows become contiguous in memory
instead of 1536B-strided segments. Same packed-KV format, scale math
and flash update as the one-head-per-CTA core.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def g128_prefill_core_packed(
    Q,
    K,
    V,
    SK,
    SV,
    BT,
    SEQLENS,
    CUQ,
    OUT,
    SCALE: gl.constexpr,
    NUM_SEQS: gl.constexpr,
    NUM_QHEADS: gl.constexpr,
    NQ_PER_KV: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    QTOKENS_PER_CTA: gl.constexpr,
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
    OUT_STRIDE0: gl.constexpr,
    OUT_STRIDE1: gl.constexpr,
    MMA_DT: gl.constexpr,
    MMA_FP16: gl.constexpr,
):
    block_id = gl.program_id(0)
    kv_head = gl.program_id(1)
    # Map block_id -> (seq, local QTOKENS_PER_CTA-token block).
    seq = 0
    my_first = 0
    acc = 0
    for s in gl.static_range(NUM_SEQS):
        s_start = gl.load(CUQ + s)
        s_end = gl.load(CUQ + s + 1)
        s_blks = (s_end - s_start + QTOKENS_PER_CTA - 1) // QTOKENS_PER_CTA
        sel = block_id >= acc + s_blks
        seq = gl.where(sel, s + 1, seq)
        acc = acc + s_blks
    seq = gl.minimum(seq, NUM_SEQS - 1)
    my_first = 0
    for s in gl.static_range(NUM_SEQS - 1):
        s_start = gl.load(CUQ + s)
        s_end = gl.load(CUQ + s + 1)
        my_first = gl.where(
            s < seq, my_first + (s_end - s_start + QTOKENS_PER_CTA - 1)
            // QTOKENS_PER_CTA, my_first
        )
    q_start = gl.load(CUQ + seq)
    q_end = gl.load(CUQ + seq + 1)
    q_len = q_end - q_start
    local_block = block_id - my_first
    if local_block * QTOKENS_PER_CTA >= q_len:
        return
    seq_len = gl.load(SEQLENS + seq)
    context = seq_len - q_len
    max_prefix = gl.minimum(
        context + local_block * QTOKENS_PER_CTA + QTOKENS_PER_CTA, seq_len
    )
    hi = (max_prefix + 31) // 32

    q_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[4, 16],
        warps_per_cta=[4, 1], order=[1, 0]
    )
    if MMA_FP16:
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=1, instr_shape=[16, 16, 16],
            transposed=True, warps_per_cta=[4, 1]
        )
    else:
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=1, instr_shape=[16, 16, 8],
            transposed=True, warps_per_cta=[4, 1]
        )
    out_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 2], threads_per_warp=[16, 4],
        warps_per_cta=[4, 1], order=[1, 0]
    )
    qm = gl.arange(0, 64, layout=gl.SliceLayout(1, q_layout))
    qd = gl.arange(0, 128, layout=gl.SliceLayout(0, q_layout))
    qpos = local_block * QTOKENS_PER_CTA + qm // NQ_PER_KV
    qhead = qm % NQ_PER_KV
    qvalid = qpos < q_len
    # Rows are (token-major, head-minor): consecutive 256B Q rows.
    qptr = (
        Q
        + (q_start + qpos[:, None]) * Q_STRIDE0
        + qhead[:, None] * Q_STRIDE1
        + qd[None, :]
    )
    q0 = gl.load(qptr, mask=qvalid[:, None], other=0.0).to(MMA_DT)
    q1 = gl.load(qptr + 128, mask=qvalid[:, None], other=0.0).to(MMA_DT)
    q0 = gl.convert_layout(q0, gl.DotOperandLayout(0, mma, k_width=2))
    q1 = gl.convert_layout(q1, gl.DotOperandLayout(0, mma, k_width=2))

    score_rows = gl.arange(0, 64, layout=gl.SliceLayout(1, mma))
    score_cols = gl.arange(0, 32, layout=gl.SliceLayout(0, mma))
    score_qpos = local_block * QTOKENS_PER_CTA + score_rows // NQ_PER_KV
    score_qvalid = score_qpos < q_len
    m = gl.full((64,), float('-inf'), gl.float32, layout=gl.SliceLayout(1, mma))
    l = gl.full((64,), 1.0, gl.float32, layout=gl.SliceLayout(1, mma))
    acc0 = gl.full((64, 128), 0.0, gl.float32, layout=mma)
    acc1 = gl.full((64, 128), 0.0, gl.float32, layout=mma)

    tn = gl.arange(0, 32, layout=gl.SliceLayout(1, q_layout))
    kd = gl.arange(0, 128, layout=gl.SliceLayout(0, q_layout))
    for j in range(0, hi):
        slot = (j * 32) % BLOCK_SIZE
        physical = gl.load(BT + seq * BT_STRIDE + (j * 32) // BLOCK_SIZE)
        kv_valid = gl.minimum(32, max_prefix - j * 32)
        valid_t = tn < kv_valid
        kb = K + physical * K_STRIDE0 + kv_head * K_STRIDE2 + slot * K_STRIDE1
        off_k = tn[:, None] * K_STRIDE1 + kd[None, :]
        if kv_valid == 32:
            k0 = gl.load(kb + off_k).to(MMA_DT)
            k1 = gl.load(kb + 128 + off_k).to(MMA_DT)
        else:
            k0 = gl.load(kb + off_k, mask=valid_t[:, None], other=0).to(MMA_DT)
            k1 = gl.load(kb + 128 + off_k, mask=valid_t[:, None], other=0).to(MMA_DT)
        k0 = gl.convert_layout(gl.permute(k0, (1, 0)), gl.DotOperandLayout(1, mma, k_width=2))
        k1 = gl.convert_layout(gl.permute(k1, (1, 0)), gl.DotOperandLayout(1, mma, k_width=2))
        sptr_k = SK + physical * S_STRIDE0 + kv_head * S_STRIDE2 + (slot + tn) * S_STRIDE1
        if kv_valid == 32:
            sk0 = gl.load(sptr_k).to(gl.float32)
            sk1 = gl.load(sptr_k + 1).to(gl.float32)
        else:
            sk0 = gl.load(sptr_k, mask=valid_t, other=1.0).to(gl.float32)
            sk1 = gl.load(sptr_k + 1, mask=valid_t, other=1.0).to(gl.float32)
        sk0 = gl.convert_layout(sk0, gl.SliceLayout(0, mma))
        sk1 = gl.convert_layout(sk1, gl.SliceLayout(0, mma))
        score0 = gl.full((64, 32), 0.0, gl.float32, layout=mma)
        score1 = gl.full((64, 32), 0.0, gl.float32, layout=mma)
        score0 = gl.amd.cdna3.mfma(q0, k0, score0)
        score1 = gl.amd.cdna3.mfma(q1, k1, score1)
        scores = score0 * (SCALE * sk0[None, :]) + score1 * (SCALE * sk1[None, :])
        kvpos = j * 32 + score_cols
        mask = score_qvalid[:, None] & (kvpos[None, :] <= context + score_qpos[:, None])
        scores = gl.where(mask, scores, float('-inf'))
        m_new = gl.maximum(m, gl.max(scores, axis=1))
        m_new = gl.where(m_new > float('-inf'), m_new, 0.0)
        p = gl.exp(scores - m_new[:, None])
        alpha = gl.exp(m - m_new)
        l = l * alpha + gl.sum(p, axis=1)
        m = m_new
        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]
        vb = V + physical * V_STRIDE0 + kv_head * V_STRIDE2 + slot * V_STRIDE1
        off_v = tn[:, None] * V_STRIDE1 + kd[None, :]
        if kv_valid == 32:
            v0 = gl.load(vb + off_v).to(MMA_DT)
            v1 = gl.load(vb + 128 + off_v).to(MMA_DT)
        else:
            v0 = gl.load(vb + off_v, mask=valid_t[:, None], other=0).to(MMA_DT)
            v1 = gl.load(vb + 128 + off_v, mask=valid_t[:, None], other=0).to(MMA_DT)
        v0 = gl.convert_layout(v0, gl.DotOperandLayout(1, mma, k_width=4))
        v1 = gl.convert_layout(v1, gl.DotOperandLayout(1, mma, k_width=4))
        sptr_v = SV + physical * S_STRIDE0 + kv_head * S_STRIDE2 + (slot + tn) * S_STRIDE1
        if kv_valid == 32:
            sv0 = gl.load(sptr_v).to(gl.float32)
            sv1 = gl.load(sptr_v + 1).to(gl.float32)
        else:
            sv0 = gl.load(sptr_v, mask=valid_t, other=1.0).to(gl.float32)
            sv1 = gl.load(sptr_v + 1, mask=valid_t, other=1.0).to(gl.float32)
        sv0 = gl.convert_layout(sv0, gl.SliceLayout(0, mma))
        sv1 = gl.convert_layout(sv1, gl.SliceLayout(0, mma))
        p0 = gl.convert_layout((p * sv0[None, :]).to(MMA_DT), gl.DotOperandLayout(0, mma, k_width=4))
        p1 = gl.convert_layout((p * sv1[None, :]).to(MMA_DT), gl.DotOperandLayout(0, mma, k_width=4))
        acc0 = gl.amd.cdna3.mfma(p0, v0, acc0)
        acc1 = gl.amd.cdna3.mfma(p1, v1, acc1)

    acc0 = gl.convert_layout(acc0, out_layout)
    acc1 = gl.convert_layout(acc1, out_layout)
    l_out = gl.convert_layout(l, gl.SliceLayout(1, out_layout))
    om = gl.arange(0, 64, layout=gl.SliceLayout(1, out_layout))
    od = gl.arange(0, 128, layout=gl.SliceLayout(0, out_layout))
    opos = local_block * QTOKENS_PER_CTA + om // NQ_PER_KV
    ohead = om % NQ_PER_KV
    ovalid = opos < q_len
    base = (
        OUT
        + (q_start + opos[:, None]) * OUT_STRIDE0
        + ohead[:, None] * OUT_STRIDE1
        + od[None, :]
    )
    inv_l = 1.0 / l_out
    gl.store(base, acc0 * inv_l[:, None], mask=ovalid[:, None])
    gl.store(base + 128, acc1 * inv_l[:, None], mask=ovalid[:, None])
