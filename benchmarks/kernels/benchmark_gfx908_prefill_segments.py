# SPDX-License-Identifier: Apache-2.0
"""Numerical and timing screen for segmented G128 prefill on idle gfx908.

Uses the existing packed-cache/scattered-block production-shape fixture.
Reports isolated kernel time, never a serving-speed claim.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("G128_CTX", "42000")
os.environ.setdefault("VLLM_G128_PREFILL_GLUON", "1")
os.environ.setdefault("VLLM_G128_GLUON", "1")
os.environ.setdefault("VLLM_G128_PREFILL_SPLITK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import torch
import triton.testing

from test_g128_mixed_32k import attend, bench


def main():
    workspace = dict(
        num_par_softmax_segments=64,
        softmax_segm_output=torch.empty((256, 6, 64, 256), device="cuda"),
        softmax_segm_max=torch.empty((256, 6, 64), device="cuda"),
        softmax_segm_expsum=torch.empty((256, 6, 64), device="cuda"),
        seq_threshold_3D=256, max_flash_decoding_splits=64,
    )
    cases = [([n], [ctx]) for ctx in (8192, 40334)
             for n in (512, 1664, 2048)]
    # Empty causal segments, ragged queries and a shifted output subview.
    cases += [([7, 1664, 128], [64, 40334, 255]), ([2048], [2048])]
    for counts, contexts in cases:
        torch.manual_seed(sum(counts))
        starts = torch.tensor([0] + counts, device="cuda", dtype=torch.int32).cumsum(0)
        starts = starts.to(torch.int32)
        lengths = torch.tensor(contexts, device="cuda", dtype=torch.int32)
        q = torch.randn((sum(counts), 6, 256), device="cuda", dtype=torch.bfloat16)
        guard = torch.full((sum(counts) + 2, 6, 256), 123., device="cuda",
                           dtype=torch.bfloat16)
        out = guard[1:-1]

        def run():
            attend(q, out, starts, lengths, bench.bt[:len(counts)], max(counts),
                   **workspace)

        os.environ["VLLM_G128_PREFILL_SEGMENTS"] = "1"
        run()
        reference = out.clone()
        baseline = triton.testing.do_bench(run, warmup=50, rep=100)
        for splits in (2, 4, 8):
            os.environ["VLLM_G128_PREFILL_SEGMENTS"] = str(splits)
            run()
            assert torch.isfinite(out).all()
            error = (out.float() - reference.float()).abs()
            relative = (error.mean() / reference.float().abs().mean()).item()
            assert relative < 0.005, (counts, contexts, splits, relative)
            saved = out.clone()
            run()
            assert torch.equal(out, saved), "non-deterministic or overlapping stores"
            assert (guard[0] == 123).all() and (guard[-1] == 123).all()
            elapsed = triton.testing.do_bench(run, warmup=50, rep=100)
            print(f"q={counts} ctx={contexts} splits={splits} "
                  f"relative={relative:.6f} max_abs={error.max().item():.6f} "
                  f"baseline_ms={baseline:.3f} candidate_ms={elapsed:.3f} "
                  f"speedup={baseline / elapsed:.2f}", flush=True)


if __name__ == "__main__":
    main()
