#!/usr/bin/env python3
"""CUDA-graph capture repro for the i32 g8 attention kernel.

The i32 variant passes every eager bench yet failed twice in-engine
(drift + steady collapse). Standing suspect: graph-capture-time
compilation. This bench captures unified_attention inside a CUDAGraph
(like the engine), replays with mutated inputs, and compares replay vs
eager for the v2 (fallback) and i32 paths.
"""
import os, sys, torch

sys.path.insert(0, "/home/curved/aiter")
sys.path.insert(0, "/home/curved/vllm-gfx908")

from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

dev = "cuda"
torch.manual_seed(0)

G = 128
BLOCK = 1664
NQ, NKV, D = 6, 1, 256
SEQS, CTX, QTOK = 6, 20000, 7
PAD = D + 2 * (D // G)
CONTENT = 2 * PAD
NB = (SEQS * (-(-CTX // BLOCK))) + 8

packed = torch.randint(
    -127, 127, (NB, NKV, BLOCK, CONTENT), device=dev, dtype=torch.int8
)
k_data = packed.transpose(1, 2)[..., :D]
v_data = packed.transpose(1, 2)[..., PAD:PAD + D]
raw = packed.untyped_storage()
base_f16 = torch.tensor([], dtype=torch.float16, device=dev).set_(raw)


def f16u(n):
    return n // 2


g8_k = torch.as_strided(
    base_f16, (NB, BLOCK, NKV, D // G),
    (f16u(packed.stride(0)), f16u(packed.stride(2)), f16u(packed.stride(1)), 1),
    storage_offset=f16u(D),
)
g8_v = torch.as_strided(
    base_f16, (NB, BLOCK, NKV, D // G),
    (f16u(packed.stride(0)), f16u(packed.stride(2)), f16u(packed.stride(1)), 1),
    storage_offset=f16u(PAD + D),
)
g8_k.copy_(torch.rand_like(g8_k) * 0.02 + 0.99)
g8_v.copy_(torch.rand_like(g8_v) * 0.02 + 0.99)

# persistent graph-visible buffers
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

KW = dict(
    num_par_softmax_segments=SPLITS,
    softmax_segm_output=segm_out,
    softmax_segm_max=segm_max,
    softmax_segm_expsum=segm_sum,
    seq_threshold_3D=SEQS,
    max_flash_decoding_splits=SPLITS,
)


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


if __name__ == "__main__":
    os.environ["VLLM_UA_3D_MAXQ"] = "8"
    try:
        # eager reference
        run(**KW)
        torch.cuda.synchronize()
        eager = out.clone()

        # capture the SAME call in a graph (engine-like)
        for _ in range(3):
            run(**KW)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run(**KW)
        # replay with MUTATED inputs: change q and some KV bytes
        q.copy_(torch.randn_like(q) * 1.5)
        with torch.no_grad():
            # mutate ONLY the K/V data regions - never the inline fp16
            # scale bytes (random scales make both sides NaN and the
            # comparison meaningless - the first version of this bench
            # made exactly that mistake).
            sl = slice(0, 4)
            packed[sl, ..., :D] = torch.randint(
                -127, 127, (4, NKV, BLOCK, D), device=dev, dtype=torch.int8)
            packed[sl, ..., PAD:PAD + D] = torch.randint(
                -127, 127, (4, NKV, BLOCK, D), device=dev, dtype=torch.int8)
        run(**KW)  # eager on mutated inputs
        torch.cuda.synchronize()
        eager2 = out.clone()

        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        replay2 = out.clone()

        d_init = (replay2.float() - eager2.float()).abs()
        rel = d_init / eager2.float().abs().clamp_min(1e-3)
        print(f"graph replay vs eager (mutated inputs): "
              f"max_abs={float(d_init.max()):.4e} "
              f"mean_rel={float(rel.mean()):.4e}")
        ok = float(rel.mean()) < 5e-3 and float(d_init.max()) < 2.0
        print("CAPTURE REPRO:", "PASS (graph is faithful)" if ok
              else "FAIL - i32 breaks under capture")
    finally:
        os.environ.pop("VLLM_UA_3D_MAXQ", None)
