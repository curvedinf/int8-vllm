# SPDX-License-Identifier: Apache-2.0
"""Screen decode launch geometry with unchanged G128 arithmetic."""

import os
import sys
from pathlib import Path

os.environ.setdefault("G128_CTX", "40334")
os.environ.setdefault("G128_EQUIV", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import torch
import triton.testing

import bench_attn_g128_egeo as b
from vllm.v1.attention.ops.gfx908_g128_i8dot import _g128_i8dot_kernel


def main():
    for count in (1, 6):
        splits = 64 if count == 1 else 32
        q = b.q[:count * 7]
        cu = b.cu_q[:count + 1]
        lengths = b.seqused[:count]
        table = b.bt[:count]
        for instruction in (0, 16, 32):
            for pack in (1, 2):
                rows, warps = 64, 4
                stages = 1
                def run():
                    return _g128_i8dot_kernel[(q.shape[0] // 10 + count, splits)](
                        q, b.k_data, b.v_data, b.g8_k, b.g8_v, table, lengths, cu,
                        b.segm_out, b.segm_max, b.segm_sum, scale=b.scale,
                        num_query_heads=6, nq_per_kv=6, num_seqs=count,
                        block_size=b.BLOCK, splits=splits, bt_stride=table.stride(0),
                        q_stride0=q.stride(0), q_stride1=q.stride(1),
                        k_stride0=b.k_data.stride(0), k_stride1=b.k_data.stride(1),
                        k_stride2=b.k_data.stride(2),
                        v_stride0=b.v_data.stride(0), v_stride1=b.v_data.stride(1),
                        v_stride2=b.v_data.stride(2),
                        s_stride0=b.g8_k.stride(0), s_stride1=b.g8_k.stride(1),
                        s_stride2=b.g8_k.stride(2), ROWS=rows, TILE=64, D=256,
                        DG=128, num_warps=warps, num_stages=stages,
                        matrix_instr_nonkdim=instruction, kpack=pack,
                    )
                try:
                    kernel = run()
                    used = count * 7 * 6 * splits
                    actual = b.segm_out.flatten()[:used * 256].clone()
                    maxima = b.segm_max.flatten()[:used].clone()
                    sums = b.segm_sum.flatten()[:used].clone()
                    if instruction == 0 and pack == 1:
                        reference, refmax, refsum = actual, maxima, sums
                    # Compare raw FP32 partials, not just rounded final output.
                    error = (actual - reference).abs()
                    relative = (error.mean() / reference.abs().mean()).item()
                    max_error = error.max().item()
                    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
                    torch.testing.assert_close(maxima, refmax, rtol=0, atol=0)
                    torch.testing.assert_close(sums, refsum, rtol=1e-6, atol=1e-5)
                    us = triton.testing.do_bench_cudagraph(run, rep=100) * 1000
                    print(f"C={count} instruction={instruction} pack={pack} us={us:.2f} "
                          f"regs={kernel.n_regs} spills={kernel.n_spills} "
                          f"relative={relative:.8f} max_abs={max_error:.8f}", flush=True)
                except Exception as exc:
                    print(f"C={count} instruction={instruction} pack={pack} FAILED "
                          f"{type(exc).__name__}: {str(exc)[:250]}", flush=True)


if __name__ == "__main__":
    main()
