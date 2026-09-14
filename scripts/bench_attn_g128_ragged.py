#!/usr/bin/env python3
"""Ragged-tail numerics gate for the g128 attention kernel.

The 2026-09-14 block-ptr gate failure slipped through the production-shape
bench because every sequence was exactly 20k (all tiles fully valid). This
gate runs mixed/ragged sequence lengths and requires the 3D path to match
the 2D reference within the established residual class (~1e-3 mean_rel).
Run after ANY change to the g8 tile loads.
"""
import os, sys, torch

sys.path.insert(0, "/home/curved/aiter")
sys.path.insert(0, "/home/curved/vllm-gfx908")

from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

dev = "cuda"
torch.manual_seed(0)

G = 128
BLOCK = 64
NQ, NKV, D = 6, 1, 256
QTOK = 7
# Ragged: lengths chosen to put partial tiles at every boundary class
LENS = [20000, 20001, 19998, 20047, 20033, 20064]
SEQS = len(LENS)

PAD = D + 2 * (D // G)
CONTENT = 2 * PAD
NB = sum((l + BLOCK - 1) // BLOCK for l in LENS) + 64

packed = torch.randint(
    -127, 127, (NB, NKV, BLOCK, CONTENT), device=dev, dtype=torch.int8
)
k_data = packed.transpose(1, 2)[..., :D]
v_data = packed.transpose(1, 2)[..., PAD:PAD + D]

raw = packed.untyped_storage()
base_f16 = torch.tensor([], dtype=torch.float16, device=dev).set_(raw)


def f16u(n):
    return n // 2


block_f16 = f16u(packed.stride(0))
head_f16 = f16u(packed.stride(1))
slot_f16 = f16u(packed.stride(2))
g8_k = torch.as_strided(
    base_f16, (NB, BLOCK, NKV, D // G),
    (block_f16, slot_f16, head_f16, 1), storage_offset=f16u(D),
)
g8_v = torch.as_strided(
    base_f16, (NB, BLOCK, NKV, D // G),
    (block_f16, slot_f16, head_f16, 1), storage_offset=f16u(PAD + D),
)
g8_k.copy_(torch.rand_like(g8_k) * 0.02 + 0.99)
g8_v.copy_(torch.rand_like(g8_v) * 0.02 + 0.99)

q = torch.randn(SEQS * QTOK, NQ, D, device=dev, dtype=torch.bfloat16)
out = torch.empty_like(q)
cu_q = torch.arange(0, SEQS * QTOK + 1, QTOK, device=dev, dtype=torch.int32)
seqused = torch.tensor(LENS, device=dev, dtype=torch.int32)
maxlen = max(LENS)
pages_per = [(l + BLOCK - 1) // BLOCK for l in LENS]
bt = torch.zeros((SEQS, max(pages_per)), device=dev, dtype=torch.int32)
cursor = 0
for i, p in enumerate(pages_per):
    bt[i, :p] = torch.arange(cursor, cursor + p, device=dev, dtype=torch.int32)
    cursor += p
scale = D ** -0.5

SPLITS = 64
segm_out = torch.empty((512, NQ, SPLITS, D), dtype=torch.float32, device=dev)
segm_max = torch.empty((512, NQ, SPLITS), dtype=torch.float32, device=dev)
segm_sum = torch.empty((512, NQ, SPLITS), dtype=torch.float32, device=dev)


def run(**extra):
    unified_attention(
        q, k_data, v_data, out,
        cu_seqlens_q=cu_q, max_seqlen_q=QTOK,
        seqused_k=seqused, max_seqlen_k=maxlen,
        softmax_scale=scale, causal=True,
        alibi_slopes=None, window_size=(-1, -1),
        block_table=bt, softcap=0.0,
        kv_quant_mode=KVQuantMode.INT8_BLOCK_G128,
        q_descale=None, k_descale=None, v_descale=None,
        sinks=None, output_scale=None,
        k_scale_cache=None, v_scale_cache=None,
        g8_k_scale=g8_k, g8_v_scale=g8_v,
        **extra,
    )


if __name__ == "__main__":
    run()
    torch.cuda.synchronize()
    ref = out.clone()
    os.environ["VLLM_UA_3D_MAXQ"] = "8"
    try:
        run(
            num_par_softmax_segments=SPLITS,
            softmax_segm_output=segm_out,
            softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_sum,
            seq_threshold_3D=SEQS,
            max_flash_decoding_splits=SPLITS,
        )
        torch.cuda.synchronize()
        d = (out.float() - ref.float()).abs()
        rel = d / ref.float().abs().clamp_min(1e-3)
        per_seq = [float(d[i * QTOK:(i + 1) * QTOK].max()) for i in range(SEQS)]
        mr = float(rel.mean())
        print(f"ragged 3D vs 2D: max_abs={float(d.max()):.4e} "
              f"mean_rel={mr:.4e} per-seq max={per_seq}")
        bad = mr > 5e-3 or float(d.max()) > 2.0
        print("RAGGED GATE:", "FAIL" if bad else "PASS")
        sys.exit(1 if bad else 0)
    finally:
        os.environ.pop("VLLM_UA_3D_MAXQ", None)
