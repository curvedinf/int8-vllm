#!/usr/bin/env python3
"""DFlash2-draft-geometry bench for the g8 2grp attention path.

Replicates the draft's KV layout from VLLM_UA_GEODUMP: D=128, G=64 (2
groups), 260B packed rows (K[0:128], k-scales[128:130], V[130:258],
v-scales[258:260]) with 2 kv-heads per row -> V base lands at 2 mod 4.
This layout must take the v2 int8 fallback (USE_I32_2GRP False): correct
results and sane speed. 3D vs 2D must match within the residual class.
"""
import os, sys, torch

sys.path.insert(0, "/home/curved/aiter")
sys.path.insert(0, "/home/curved/vllm-gfx908")

from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

dev = "cuda"
torch.manual_seed(0)

G = 64
BLOCK = 1664
NQ, NKV, D = 8, 2, 128
SEQS, CTX, QTOK = 6, 20000, 7
CONTENT = 2 * D + 2 * (D // G) * 2   # 260 bytes: K, ksc(2xf16), V, vsc
NB = (SEQS * (-(-CTX // BLOCK))) + 64

packed = torch.randint(
    -127, 127, (NB, NKV, BLOCK, CONTENT), device=dev, dtype=torch.int8
)
# engine layout: [block][head][token][row]; k at [0:128], v at [130:258]
# (2 mod 4), k-scales fp16 at bytes [128:130], v-scales at [258:260].
k_data = packed[..., 0:D].permute(0, 2, 1, 3)
v_data = packed[..., 2 + D:2 + 2 * D].permute(0, 2, 1, 3)

raw = packed.untyped_storage()
base_f16 = torch.tensor([], dtype=torch.float16, device=dev).set_(raw)


def f16u(n):
    return n // 2


g8_k = torch.as_strided(
    base_f16, (NB, BLOCK, NKV, D // G),
    (f16u(packed.stride(0)), f16u(packed.stride(2)),
     f16u(packed.stride(1)), 1),
    storage_offset=f16u(D),
)
g8_v = torch.as_strided(
    base_f16, (NB, BLOCK, NKV, D // G),
    (f16u(packed.stride(0)), f16u(packed.stride(2)),
     f16u(packed.stride(1)), 1),
    storage_offset=f16u(2 * D + 2 + 0 - 2),  # bytes 258..260
)
g8_k.copy_(torch.rand_like(g8_k.float()).half() * 0.02 + 0.99)
g8_v.copy_(torch.rand_like(g8_v.float()).half() * 0.02 + 0.99)

q = torch.randn(SEQS * QTOK, NQ, D, device=dev, dtype=torch.bfloat16)
out = torch.empty_like(q)
cu_q = torch.arange(0, SEQS * QTOK + 1, QTOK, device=dev, dtype=torch.int32)
seqused = torch.full((SEQS,), CTX, device=dev, dtype=torch.int32)
perm = torch.randperm(NB - 8, device=dev)[: (SEQS * (-(-CTX // BLOCK)))].to(torch.int32)
bt = perm.view(SEQS, -1).contiguous()
scale = D ** -0.5
SPLITS = 64
segm_out = torch.empty((512, NQ, SPLITS, D), dtype=torch.float32, device=dev)
segm_max = torch.empty((512, NQ, SPLITS), dtype=torch.float32, device=dev)
segm_sum = torch.empty((512, NQ, SPLITS), dtype=torch.float32, device=dev)


def run(**extra):
    unified_attention(
        q, k_data, v_data, out,
        cu_seqlens_q=cu_q, max_seqlen_q=QTOK,
        seqused_k=seqused, max_seqlen_k=CTX,
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


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


if __name__ == "__main__":
    print(f"layout: kptr%4={k_data.data_ptr()%4} vptr%4={v_data.data_ptr()%4} "
          f"kstride={k_data.stride()} vstride={v_data.stride()}")
    run()
    torch.cuda.synchronize()
    ref = out.clone()
    t2d = bench(run)
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
        t3d = bench(lambda: run(
            num_par_softmax_segments=SPLITS,
            softmax_segm_output=segm_out,
            softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_sum,
            seq_threshold_3D=SEQS,
            max_flash_decoding_splits=SPLITS,
        ))
        print(f"draft-geometry 2D: {t2d:9.1f} us/call")
        print(f"draft-geometry 3D: {t3d:9.1f} us/call  "
              f"max_abs={float(d.max()):.4e} mean_rel={float(rel.mean()):.4e}")
    finally:
        os.environ.pop("VLLM_UA_3D_MAXQ", None)
