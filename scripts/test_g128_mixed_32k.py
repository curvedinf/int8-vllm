#!/usr/bin/env python3
"""Compare 32k speculative attention alone and beside long prefills.

Run on an idle MI100. Decoder queries and their packed g128 KV bytes are
identical between calls; only three independent prefill rows are added.
"""

import os

os.environ.setdefault("G128_EQUIV", "1")
os.environ.setdefault("G128_CTX", "32000")
os.environ.setdefault("VLLM_UA_3D_MAXQ", "8")

import torch

import bench_attn_g128_egeo as bench
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode


def attend(query, output, starts, lengths, table, max_q, **extra):
    unified_attention(
        query, bench.k_data, bench.v_data, output,
        cu_seqlens_q=starts,
        max_seqlen_q=max_q,
        seqused_k=lengths,
        max_seqlen_k=bench.CTX,
        softmax_scale=bench.scale,
        causal=True,
        alibi_slopes=None,
        window_size=(-1, -1),
        block_table=table,
        softcap=0.0,
        kv_quant_mode=KVQuantMode.INT8_BLOCK_G128,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        sinks=None,
        output_scale=None,
        k_scale_cache=None,
        v_scale_cache=None,
        g8_k_scale=bench.g8_k,
        g8_v_scale=bench.g8_v,
        **extra,
    )


def main():
    assert bench.SEQS == 6 and bench.QTOK == 7
    prefill_q = int(os.environ.get("G128_PREFILL_Q", "2048"))
    assert 1 <= prefill_q <= bench.CTX

    dec_q = bench.q[:3 * bench.QTOK].contiguous()
    dec_out = torch.empty_like(dec_q)
    dec_starts = torch.arange(0, dec_q.shape[0] + 1, bench.QTOK,
                              dtype=torch.int32, device=bench.dev)
    attend(dec_q, dec_out, dec_starts, bench.seqused[:3],
           bench.bt[:3], bench.QTOK)
    torch.cuda.synchronize()

    pre_q = torch.randn(3 * prefill_q, bench.NQ, bench.D,
                        dtype=dec_q.dtype, device=bench.dev)
    mixed_q = torch.cat((dec_q, pre_q))
    mixed_out = torch.empty_like(mixed_q)
    starts = torch.tensor(
        [0, 7, 14, 21, 21 + prefill_q, 21 + 2 * prefill_q,
         21 + 3 * prefill_q], device=bench.dev, dtype=torch.int32,
    )
    lengths = torch.full((6,), bench.CTX, device=bench.dev,
                         dtype=torch.int32)
    attend(mixed_q, mixed_out, starts, lengths, bench.bt, prefill_q)
    torch.cuda.synchronize()

    diff = (dec_out.float() - mixed_out[:dec_q.shape[0]].float()).abs()
    scale = dec_out.float().abs().mean().clamp_min(1e-6)
    print(
        f"ctx={bench.CTX} prefill_q={prefill_q} "
        f"decoder_rows={dec_q.shape[0]} mean_abs={diff.mean().item():.6g} "
        f"max_abs={diff.max().item():.6g} "
        f"mean_rel_scale={(diff.mean() / scale).item():.6g}",
        flush=True,
    )

    segm_kw = dict(
        num_par_softmax_segments=bench.SPLITS,
        softmax_segm_output=bench.segm_out,
        softmax_segm_max=bench.segm_max,
        softmax_segm_expsum=bench.segm_sum,
        seq_threshold_3D=bench.SEQS,
        max_flash_decoding_splits=bench.SPLITS,
    )
    dec_3d = torch.empty_like(dec_q)
    attend(dec_q, dec_3d, dec_starts, lengths[:3], bench.bt[:3],
           bench.QTOK, **segm_kw)
    split_out = torch.empty_like(mixed_q)
    attend(dec_q, split_out[:dec_q.shape[0]], dec_starts, lengths[:3],
           bench.bt[:3], bench.QTOK, **segm_kw)
    attend(mixed_q[dec_q.shape[0]:], split_out[dec_q.shape[0]:],
           starts[3:] - dec_q.shape[0], lengths[3:], bench.bt[3:],
           prefill_q)
    torch.cuda.synchronize()
    dec_delta = (dec_3d.float() - split_out[:dec_q.shape[0]].float()).abs()
    pre_delta = (mixed_out[dec_q.shape[0]:].float()
                 - split_out[dec_q.shape[0]:].float()).abs()
    print(f"split decoder_vs_standalone_3d max_abs={dec_delta.max().item():.6g} "
          f"prefill_vs_mixed_2d max_abs={pre_delta.max().item():.6g}",
          flush=True)
    assert dec_delta.max() == 0 and pre_delta.max() == 0


if __name__ == "__main__":
    main()
