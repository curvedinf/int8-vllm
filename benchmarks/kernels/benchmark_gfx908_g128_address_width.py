# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interleave old/new G128 decode cores with identical, low-offset KV inputs."""

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if min(args.pairs, args.iters) <= 0:
        parser.error("pair and iteration counts must be positive")
    root = Path(__file__).resolve().parents[2]
    os.environ.update(
        G128_CTX="32000",
        G128_EQUIV="1",
        VLLM_UA_3D_MAXQ="8",
        VLLM_G128_GLUON="1",
        VLLM_GFX908_ATTN_WARPS="2",
        VLLM_G128_GLUON_MMA="fp16",
        VLLM_G128_REDUCE_GLUON="0",
    )
    sys.path.insert(0, str(root / "scripts"))
    import bench_attn_g128_egeo as bench
    import torch

    from vllm.v1.attention.ops import gfx908_g128_gluon_m64 as current

    # The old core is only safe to time below its address-overflow boundary.
    assert int(bench.bt.max()) * bench.packed.stride(0) < 1 << 31
    fixed_core = current.g128_core
    with tempfile.TemporaryDirectory(prefix="g128-address-width-") as temp_dir:
        old_path = Path(temp_dir) / "baseline_m64.py"
        old_path.write_bytes(
            subprocess.check_output(
                [
                    "git",
                    "show",
                    args.baseline_ref
                    + ":vllm/v1/attention/ops/gfx908_g128_gluon_m64.py",
                ],
                cwd=root,
            )
        )
        spec = importlib.util.spec_from_file_location("g128_baseline_m64", old_path)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        cores = {"before": baseline.g128_core, "after": fixed_core}
        outputs = {}
        try:
            for name, core in cores.items():
                current.g128_core = core
                bench.run_vllm_3d()
                torch.cuda.synchronize()
                outputs[name] = bench.out.clone()
            torch.testing.assert_close(
                outputs["before"], outputs["after"], rtol=0, atol=0
            )
            print(
                json.dumps(
                    {
                        "phase": "correctness",
                        "ctx": bench.CTX,
                        "seqs": bench.SEQS,
                        "query_tokens": bench.QTOK,
                        "max_block_id": int(bench.bt.max()),
                        "max_abs": 0.0,
                    }
                ),
                flush=True,
            )
            results = {"before": [], "after": []}
            for pair in range(args.pairs):
                order = ["before", "after"] if pair % 2 == 0 else ["after", "before"]
                for name in order:
                    current.g128_core = cores[name]
                    latency = bench.bench(bench.run_vllm_3d, args.iters, warmup=10)
                    results[name].append(latency)
                    print(
                        json.dumps(
                            {
                                "phase": "timing",
                                "pair": pair,
                                "arm": name,
                                "us": latency,
                            }
                        ),
                        flush=True,
                    )
            means = {name: statistics.mean(values) for name, values in results.items()}
            print(
                json.dumps(
                    {
                        "phase": "summary",
                        "means_us": means,
                        "delta_pct": 100 * (means["after"] / means["before"] - 1),
                        "raw_us": results,
                    }
                ),
                flush=True,
            )
        finally:
            current.g128_core = fixed_core


if __name__ == "__main__":
    main()
