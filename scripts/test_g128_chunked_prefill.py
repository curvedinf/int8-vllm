#!/usr/bin/env python3
"""Check and time bounded split-K G128 prefill against the prefill kernel."""
import os

os.environ.setdefault("VLLM_G128_PREFILL_GLUON", "1")
os.environ.setdefault("VLLM_G128_GLUON", "1")
os.environ.setdefault("VLLM_G128_PREFILL_SPLITK", "1")

import torch
import triton.testing

from test_g128_mixed_32k import attend, bench


def main():
    workspace = dict(
        num_par_softmax_segments=bench.SPLITS,
        softmax_segm_output=torch.empty(
            (256, bench.NQ, bench.SPLITS, bench.D), device=bench.dev,
            dtype=torch.float32,
        ),
        softmax_segm_max=torch.empty(
            (256, bench.NQ, bench.SPLITS), device=bench.dev,
            dtype=torch.float32,
        ),
        softmax_segm_expsum=torch.empty(
            (256, bench.NQ, bench.SPLITS), device=bench.dev,
            dtype=torch.float32,
        ),
        seq_threshold_3D=bench.SEQS,
        max_flash_decoding_splits=bench.SPLITS,
    )
    for context in (64, 255, 2047, 32000):
        for count in (16, 32, 64, 128, 192, 256):
            if count > context:
                continue
            torch.manual_seed(count)
            q = torch.randn(count, bench.NQ, bench.D, device=bench.dev,
                            dtype=torch.bfloat16)
            starts = torch.tensor([0, count], device=bench.dev, dtype=torch.int32)
            lengths = torch.tensor([context], device=bench.dev, dtype=torch.int32)
            reference = torch.empty_like(q)
            actual = torch.empty_like(q)
            def run(out, **kw):
                attend(q, out, starts, lengths, bench.bt[:1], count, **kw)
            run(reference)
            run(actual, **workspace)
            assert torch.isfinite(actual).all()
            error = (actual.float() - reference.float()).abs()
            relative = (error.mean() / reference.float().abs().mean()).item()
            assert relative < 0.01, (context, count, relative)
            saved = actual.clone()
            run(actual, **workspace)
            assert torch.equal(saved, actual), "overlapping stores"
            baseline_ms = triton.testing.do_bench(lambda: run(reference))
            split_ms = triton.testing.do_bench(lambda: run(actual, **workspace))
            print(f"ctx={context} q={count} rel={relative:.6f} "
                  f"prefill_ms={baseline_ms:.3f} split_ms={split_ms:.3f} "
                  f"speedup={baseline_ms / split_ms:.2f}", flush=True)


if __name__ == "__main__":
    main()
