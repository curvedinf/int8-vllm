#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the serving GS32 transport, including interleaved native CAR.

Run with four torchrun ranks, ingress blocked and all GPUs otherwise idle.
VLLM_MLP_CAR_INT8_LIBRARY names the separately built HIP shared library.
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from benchmark_gfx908_car_quant import (
    error_metrics,
    make_partials,
    measure,
    reference_quant,
)
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.device_communicators.int8_gs32_all_reduce import Int8Gs32Allreduce


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    ap.add_argument("--trace-dir")
    args = ap.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    native = CustomAllreduce(dist.group.WORLD, rank, max_size=40 * 1024**2)
    quant = Int8Gs32Allreduce(native)
    results = {"accuracy": [], "latency": [], "stress_replays_per_rank": 0}
    for dtype in (torch.bfloat16, torch.float16):
        cases = ["normal", "outlier", "log_range", "cancellation", "zero"]
        if dtype == torch.bfloat16:
            cases += ["large_1e10", "large_1e30", "tiny_1e-30",
                      "bf16_extremes", "bf16_sweep"]
        for case in cases:
            partials = make_partials(case, 8 * 5120, dtype)
            x = partials[rank].cuda()
            actual = quant.all_reduce(x).cpu()
            decoded = [reference_quant(p, 0, 32) for p in partials]
            expected = decoded[0].clone()
            for p in decoded[1:]:
                expected += p
            expected = expected.to(dtype)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            reference = partials[0].float()
            for p in partials[1:]:
                reference += p.float()
            row = {"dtype": str(dtype), "case": case,
                   **error_metrics(actual, reference.to(dtype))}
            results["accuracy"].append(row)
    for rows in (1, 7, 42, 48, 52, 56, 128, 1664, 2048):
        x = torch.randn((rows, 5120), device="cuda", dtype=torch.bfloat16)
        y = torch.empty_like(x)
        results["latency"].append({"rows": rows,
            "native": measure(lambda: native.all_reduce(x, out=y, registered=False),
                              native, 10, 5, 16),
            "gs32": measure(lambda: quant.all_reduce(x), native, 10, 5, 16)})
        # Mixed shapes/native interleaving and changing inputs on graph replay.
        graph = torch.cuda.CUDAGraph()
        with native.capture(), torch.cuda.graph(graph):
            q1 = quant.all_reduce(x)
            plain = native.all_reduce(x, registered=False)
            q2 = quant.all_reduce(x)
        for iteration in range(30):
            x.fill_(float((iteration % 7) - 3) * (rank + 1))
            graph.replay()
            expected = float((iteration % 7) - 3) * 10
            assert bool((q1 == expected).all()) and bool((q2 == expected).all())
            assert bool((plain == expected).all())
            results["stress_replays_per_rank"] += 1
    if args.trace_dir:
        x = torch.randn((1664, 5120), device="cuda", dtype=torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with native.capture(), torch.cuda.graph(graph):
            for _ in range(4):
                native.all_reduce(x, registered=False)
                quant.all_reduce(x)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
        path = Path(args.trace_dir)
        path.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(path / f"rank{rank}.json.gz"))
    gathered = [None] * 4
    dist.all_gather_object(gathered, results)
    if rank == 0:
        Path(args.output).write_text(json.dumps({"ranks": gathered}, indent=2))
        print(json.dumps(results), flush=True)
    quant.close()
    native.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
