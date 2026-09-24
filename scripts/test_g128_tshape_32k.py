#!/usr/bin/env python3
"""Compare production g128 attention at T=7 and seven matched T=1 reads.

Run only with an idle MI100. G128_CTX controls the context length.
"""

import os

os.environ.setdefault("G128_EQUIV", "1")
os.environ.setdefault("G128_CTX", "32000")

import torch

import bench_attn_g128_egeo as bench


def main() -> None:
    assert bench.QTOK == 7
    bench.run_vllm_3d()
    torch.cuda.synchronize()
    verify = bench.out.float().clone()
    cu_one = torch.arange(bench.SEQS + 1, device=bench.dev, dtype=torch.int32)

    for t in range(bench.QTOK):
        query = bench.q[t::bench.QTOK].contiguous()
        result = torch.empty_like(query)
        seq_len = bench.CTX - bench.QTOK + 1 + t
        seq_lens = torch.full(
            (bench.SEQS,), seq_len, device=bench.dev, dtype=torch.int32
        )
        bench.run_vllm_3d(
            query=query,
            result=result,
            query_starts=cu_one,
            sequence_lengths=seq_lens,
            max_query_len=1,
            max_sequence_len=seq_len,
        )
        torch.cuda.synchronize()
        delta = (verify[t::bench.QTOK] - result.float()).abs()
        scale = result.float().abs().mean().clamp_min(1e-6)
        print(
            f"row={t} seq_len={seq_len} "
            f"mean_abs={delta.mean().item():.6g} "
            f"max_abs={delta.max().item():.6g} "
            f"mean_rel_scale={(delta.mean() / scale).item():.6g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
