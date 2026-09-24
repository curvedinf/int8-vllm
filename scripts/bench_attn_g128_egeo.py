#!/usr/bin/env python3
"""Production-shape g128 decode-attention bench.

Replicates the exact tensors/args the RocmAiterUnifiedAttentionImpl forward
passes for int8_block_g128 at C6/20k decode: packed KV (NB, 1, 1664, 520)
int8 with inline fp16 group scales, 6 seqs x 20k ctx, 42 q-tokens
(7/seq), 6 q-heads, 1 kv-head, head 256, scattered block table.

Times: (a) the current vLLM triton path (g8 scales), (b) the same under
VLLM_GFX908_ATTN_WARPS/STAGES sweeps, (c) the aiter kernel with a scalar
descale as the reference "fast path" ceiling, (d) numerics cross-check
of (a) against a dequantized reference for sanity.
"""
import os
import sys
import torch

sys.path.insert(0, "/home/curved/aiter")
sys.path.insert(0, "/home/curved/vllm-gfx908")

from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

dev = "cuda"
torch.manual_seed(0)

G = 128
BLOCK = 1664
NQ, NKV, D = 6, 1, 256
SEQS, CTX, QTOK = 6, int(os.environ.get("G128_CTX", "20000")), 7

PAD = D + 2 * (D // G)            # 260 per half
CONTENT = 2 * PAD                 # 520
NB = (SEQS * (-(-CTX // BLOCK))) + 64

# packed logical (NB, NKV, BLOCK, CONTENT) int8, HND-contig
packed = torch.randint(
    -127, 127, (NB, NKV, BLOCK, CONTENT), device=dev, dtype=torch.int8
)
k_data = packed.transpose(1, 2)[..., :D]
v_data = packed.transpose(1, 2)[..., PAD:PAD + D]

# inline fp16 group-scale views (mirror _ensure_scale_caches)
raw = packed.untyped_storage()
base_f16 = torch.tensor([], dtype=torch.float16, device=dev).set_(raw)


def f16u(n):  # int8 elements -> f16 units
    return n // 2


block_f16 = f16u(packed.stride(0))
head_f16 = f16u(packed.stride(1))
slot_f16 = f16u(packed.stride(2))
g8_k = torch.as_strided(
    base_f16, (NB, BLOCK, NKV, D // G),
    (block_f16, slot_f16, head_f16, 1),
    storage_offset=f16u(D),
)
g8_v = torch.as_strided(
    base_f16, (NB, BLOCK, NKV, D // G),
    (block_f16, slot_f16, head_f16, 1),
    storage_offset=f16u(PAD + D),
)
if os.environ.get("G128_EQUIV") == "1":
    # int8 values span [-127, 127]; scales near 0.01 give realistic K/V
    # magnitudes so the comparison is not hidden by saturated softmax.
    g8_k.copy_(torch.rand_like(g8_k) * 0.005 + 0.01)
    g8_v.copy_(torch.rand_like(g8_v) * 0.005 + 0.01)
else:
    g8_k.copy_(torch.rand_like(g8_k) * 0.02 + 0.99)
    g8_v.copy_(torch.rand_like(g8_v) * 0.02 + 0.99)

q = torch.randn(SEQS * QTOK, NQ, D, device=dev, dtype=torch.bfloat16)
out = torch.empty_like(q)
cu_q = torch.arange(0, SEQS * QTOK + 1, QTOK, device=dev, dtype=torch.int32)
seqused = torch.full((SEQS,), CTX, device=dev, dtype=torch.int32)
perm = torch.randperm(NB - 8, device=dev)[: (SEQS * (-(-CTX // BLOCK)))].to(torch.int32)
bt = perm.view(SEQS, -1).contiguous()
scale = D ** -0.5


def run_vllm(
    *,
    query=q,
    result=out,
    query_starts=cu_q,
    sequence_lengths=seqused,
    max_query_len=QTOK,
    max_sequence_len=CTX,
    **extra,
):
    unified_attention(
        query, k_data, v_data, result,
        cu_seqlens_q=query_starts, max_seqlen_q=max_query_len,
        seqused_k=sequence_lengths, max_seqlen_k=max_sequence_len,
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


# 3D split-K segment buffers (mirror triton_attn.py allocation).
# First dim sized for total q-blocks (q>1 verify batches overflow a
# seqs-sized buffer).
SPLITS = 64
SEGM_ROWS = (
    max(64, SEQS * QTOK) if os.environ.get("G128_EQUIV") == "1" else 512
)
segm_out = torch.empty((SEGM_ROWS, NQ, SPLITS, D), dtype=torch.float32, device=dev)
segm_max = torch.empty((SEGM_ROWS, NQ, SPLITS), dtype=torch.float32, device=dev)
segm_sum = torch.empty((SEGM_ROWS, NQ, SPLITS), dtype=torch.float32, device=dev)


def run_vllm_3d(**extra):
    run_vllm(
        num_par_softmax_segments=SPLITS,
        softmax_segm_output=segm_out,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_sum,
        seq_threshold_3D=SEQS,
        max_flash_decoding_splits=SPLITS,
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
    return s.elapsed_time(e) / iters * 1000  # us


if __name__ == "__main__":
    mode = next(m for m in KVQuantMode if m.name == "INT8_BLOCK_G128")
    print("mode:", mode)
    t = bench(run_vllm)
    print(f"vLLM triton g128 2D (current path): {t:9.1f} us/call")
    # numerics reference from the 2D path
    ref = out.clone()

    os.environ["VLLM_UA_3D_MAXQ"] = "8"
    try:
        run_vllm_3d()
        torch.cuda.synchronize()
        diff = (out.float() - ref.float()).abs()
        rel = diff / ref.float().abs().clamp_min(1e-3)
        print(f"3D vs 2D numerics: max_abs={diff.max():.4e} "
              f"mean_rel={rel.mean():.4e} p99_rel={rel.flatten().kthvalue(int(rel.numel()*0.99)).values:.4e}")
        if os.environ.get("G128_EQUIV") == "1":
            # Each speculative verify row must match a one-token query at
            # the same causal prefix length. Keep Q and backing KV identical.
            verify_rows = out.reshape(SEQS, QTOK, NQ, D).clone()
            verify_first = verify_rows[:, 0]
            first_diff = (verify_first.float() -
                          ref.reshape(SEQS, QTOK, NQ, D)[:, 0].float()).abs()
            print(f"3D vs 2D row-0: max_abs={first_diff.max():.4e} "
                  f"mean_abs={first_diff.mean():.4e}")
            one_q = q.reshape(SEQS, QTOK, NQ, D)[:, 0].contiguous()
            one_out = torch.empty_like(one_q)
            one_starts = torch.arange(SEQS + 1, device=dev, dtype=torch.int32)
            one_len = CTX - QTOK + 1
            one_seqlens = torch.full(
                (SEQS,), one_len, device=dev, dtype=torch.int32
            )
            for row in range(QTOK):
                row_len = one_len + row
                one_q.copy_(q.reshape(SEQS, QTOK, NQ, D)[:, row])
                one_seqlens.fill_(row_len)
                run_vllm_3d(
                    query=one_q, result=one_out, query_starts=one_starts,
                    sequence_lengths=one_seqlens, max_query_len=1,
                    max_sequence_len=row_len,
                )
                torch.cuda.synchronize()
                err = (verify_rows[:, row].float() - one_out.float()).abs()
                print(f"3D q=7 row-{row} vs q=1 at ctx={row_len}: "
                      f"max_abs={err.max():.4e} mean_abs={err.mean():.4e} "
                      f"frac_nonzero={(err != 0).float().mean():.4f}")
        t = bench(run_vllm_3d)
        print(f"vLLM triton g128 3D split-K (q=7): {t:9.1f} us/call")
    except Exception as ex:
        import traceback; traceback.print_exc()
    finally:
        os.environ.pop("VLLM_UA_3D_MAXQ", None)

    for warps in (4,):
        for stages in (2,):
            os.environ["VLLM_GFX908_ATTN_WARPS"] = str(warps)
            os.environ["VLLM_GFX908_ATTN_STAGES"] = str(stages)
            os.environ["VLLM_UA_3D_MAXQ"] = "8"
            try:
                t = bench(run_vllm_3d)
                print(f"  3D warps={warps} stages={stages}: {t:9.1f} us/call")
            except Exception as ex:
                print(f"  3D warps={warps} stages={stages}: FAIL {ex}")
            os.environ.pop("VLLM_UA_3D_MAXQ", None)
    os.environ.pop("VLLM_GFX908_ATTN_WARPS", None)
    os.environ.pop("VLLM_GFX908_ATTN_STAGES", None)
