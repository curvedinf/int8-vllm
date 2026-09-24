# SPDX-License-Identifier: Apache-2.0
"""64-row gfx908 grouped-int8 split-K decode core for verify batches."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def g128_core(
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
        first_block = start // 10 + s
        seq = gl.where(block_id >= first_block, s, seq)
    q_start = gl.load(CUQ + seq)
    q_end = gl.load(CUQ + seq + 1)
    q_len = q_end - q_start
    local_block = block_id - (q_start // 10 + seq)
    if local_block * 10 >= q_len:
        return
    seq_len = gl.load(SEQLENS + seq)
    context = seq_len - q_len
    tiles_per_seg = (seq_len + SPLITS * 32 - 1) // (SPLITS * 32)
    lo = seg * tiles_per_seg
    if lo * 32 >= seq_len:
        return
    # The causal mask below excludes the conservative extra key in this bound.
    max_prefix = gl.minimum(context + local_block * 10 + 11, seq_len)
    hi = gl.minimum((seg + 1) * tiles_per_seg, (max_prefix + 31) // 32)

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
    qm = gl.arange(0, 64, layout=gl.SliceLayout(1, q_layout))
    qd = gl.arange(0, 128, layout=gl.SliceLayout(0, q_layout))
    qpos = local_block * 10 + qm // NQ_PER_KV
    qhead = kv_head * NQ_PER_KV + qm % NQ_PER_KV
    qvalid = (qpos < q_len) & (qhead < NUM_QHEADS)
    qptr = Q + (q_start + qpos[:, None]) * Q_STRIDE0 + qhead[:, None] * Q_STRIDE1 + qd[None, :]
    q0 = gl.load(qptr, mask=qvalid[:, None], other=0.0)
    q1 = gl.load(qptr + 128, mask=qvalid[:, None], other=0.0)
    q0 = gl.convert_layout(q0, gl.DotOperandLayout(0, mma, k_width=2))
    q1 = gl.convert_layout(q1, gl.DotOperandLayout(0, mma, k_width=2))

    score_rows = gl.arange(0, 64, layout=gl.SliceLayout(1, mma))
    score_cols = gl.arange(0, 32, layout=gl.SliceLayout(0, mma))
    score_qpos = local_block * 10 + score_rows // NQ_PER_KV
    score_qhead = kv_head * NQ_PER_KV + score_rows % NQ_PER_KV
    score_qvalid = (score_qpos < q_len) & (score_qhead < NUM_QHEADS)
    m = gl.full((64,), float('-inf'), gl.float32, layout=gl.SliceLayout(1, mma))
    l = gl.full((64,), 1.0, gl.float32, layout=gl.SliceLayout(1, mma))
    acc0 = gl.full((64, 128), 0.0, gl.float32, layout=mma)
    acc1 = gl.full((64, 128), 0.0, gl.float32, layout=mma)

    tn = gl.arange(0, 32, layout=gl.SliceLayout(1, kv_layout))
    kd = gl.arange(0, 128, layout=gl.SliceLayout(0, kv_layout))
    for j in range(lo, hi):
        slot = (j * 32) % BLOCK_SIZE
        physical = gl.load(BT + seq * BT_STRIDE + (j * 32) // BLOCK_SIZE)
        kv_valid = gl.minimum(32, max_prefix - j * 32)
        valid_t = tn < kv_valid
        kb = K + physical * K_STRIDE0 + kv_head * K_STRIDE2 + slot * K_STRIDE1
        off_k = tn[:, None] * K_STRIDE1 + kd[None, :]
        if kv_valid == 32:
            k0 = gl.load(kb + off_k).to(gl.bfloat16)
            k1 = gl.load(kb + 128 + off_k).to(gl.bfloat16)
        else:
            k0 = gl.load(kb + off_k, mask=valid_t[:, None], other=0).to(gl.bfloat16)
            k1 = gl.load(kb + 128 + off_k, mask=valid_t[:, None], other=0).to(gl.bfloat16)
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
        mask = score_qvalid[:, None] & (kvpos[None, :] < max_prefix) & (kvpos[None, :] <= context + score_qpos[:, None])
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
            v0 = gl.load(vb + off_v).to(gl.bfloat16)
            v1 = gl.load(vb + 128 + off_v).to(gl.bfloat16)
        else:
            v0 = gl.load(vb + off_v, mask=valid_t[:, None], other=0).to(gl.bfloat16)
            v1 = gl.load(vb + 128 + off_v, mask=valid_t[:, None], other=0).to(gl.bfloat16)
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
        p0 = gl.convert_layout((p * sv0[None, :]).to(gl.bfloat16), gl.DotOperandLayout(0, mma, k_width=4))
        p1 = gl.convert_layout((p * sv1[None, :]).to(gl.bfloat16), gl.DotOperandLayout(0, mma, k_width=4))
        acc0 = gl.amd.cdna3.mfma(p0, v0, acc0)
        acc1 = gl.amd.cdna3.mfma(p1, v1, acc1)

    acc0 = gl.convert_layout(acc0, out_layout)
    acc1 = gl.convert_layout(acc1, out_layout)
    om = gl.arange(0, 64, layout=gl.SliceLayout(1, out_layout))
    od = gl.arange(0, 128, layout=gl.SliceLayout(0, out_layout))
    opos = local_block * 10 + om // NQ_PER_KV
    ohead = kv_head * NQ_PER_KV + om % NQ_PER_KV
    ovalid = (opos < q_len) & (ohead < NUM_QHEADS)
    base = PARTIAL + (q_start + opos[:, None]) * (NUM_QHEADS * SPLITS * 256) + ohead[:, None] * (SPLITS * 256) + seg * 256 + od[None, :]
    gl.store(base, acc0, mask=ovalid[:, None])
    gl.store(base + 128, acc1, mask=ovalid[:, None])
    moff = (q_start + score_qpos) * (NUM_QHEADS * SPLITS) + score_qhead * SPLITS + seg
    gl.store(PMAX + moff, m, mask=score_qvalid)
    gl.store(PSUM + moff, l, mask=score_qvalid)

