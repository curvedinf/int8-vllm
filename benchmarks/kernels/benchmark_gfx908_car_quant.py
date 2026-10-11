#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Model-free TP4 CAR quantization study; no model/checkpoint imports.

Compile the adjacent HIP prototype separately, then run with torchrun:
  hipcc -O3 -std=c++17 --offload-arch=gfx908 -shared -fPIC \
    benchmarks/kernels/car_quant_study.cpp -o /tmp/car_quant_study.so
  PYTHONPATH="$PWD:$PWD/../aiter" .venv/bin/python -m torch.distributed.run \
    --nproc_per_node=4 benchmarks/kernels/benchmark_gfx908_car_quant.py \
    --library /tmp/car_quant_study.so --output /tmp/car_quant_results.json

Only run with all four GPUs idle and ingress blocked. Production CAR is
unmodified. Experimental transport uses its uncached IPC buffers/signals,
FP32 rank-ordered accumulation, and a separate quantization launch. Native
CAR comparisons include the staging copy, exactly as on gfx908 graph replay.
"""

import argparse
import ctypes
import datetime
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist

# name, kernel mode, group, metadata bytes/group
FORMATS = [
    ("linear_g32_fp32", 0, 32, 4),
    ("linear_g16_fp32", 0, 16, 4),
    ("linear_g32_fp16", 1, 32, 2),
    ("linear_g16_fp16", 1, 16, 2),
    ("linear_g16_bf16", 2, 16, 2),
    ("sqrt_g16_fp16", 3, 16, 2),
    ("cbrt_g16_fp16", 4, 16, 2),
    ("log_g16_fp16", 5, 16, 4),
    ("log_g16_fp32", 6, 16, 8),
    ("log_g16_fp16_bounded", 7, 16, 4),
]


def reference_quant(x, mode, group):
    """Independent CPU oracle, using the transmitted (rounded) parameters."""
    x = x.float().reshape(-1, group)
    maximum = x.abs().amax(-1, keepdim=True)
    if mode >= 5:
        minimum = x.abs().masked_fill(x == 0, float("inf")).amin(-1, keepdim=True)
        offset = torch.floor(torch.log2(minimum))
        offset = torch.where(maximum == 0, 0.0, offset)
        step = ((torch.log2(maximum) - offset) / 126).clamp_min(1e-4)
        step = torch.where(maximum == 0, 1.0, step)
        if mode in (5, 7):
            offset, step = offset.half().float(), step.half().float()
        k = ((torch.log2(x.abs()) - offset) / step).round().clamp(0, 126)
        y = x.sign() * torch.exp2(offset + k * step)
        if mode == 7:
            limit = torch.finfo(torch.bfloat16).max
            y = y.clamp(-limit, limit)
        return torch.where(x == 0, 0.0, y).flatten()
    scale = torch.where(maximum == 0, 1.0, maximum / 127)
    if mode == 1:
        scale = scale.half().float()
    elif mode == 2:
        scale = scale.bfloat16().float()
    elif mode == 3:
        scale = scale.sqrt().half().float().square()
    elif mode == 4:
        root = scale.pow(1 / 3).half().float()
        scale = root.square() * root
    good = (scale > 0) & torch.isfinite(scale)
    q = (x / scale).round().clamp(-127, 127)
    return torch.where(good, q * scale, float("nan")).flatten()


def make_partials(case, n, dtype, seed=1091):
    rng = torch.Generator().manual_seed(seed)
    if case == "bf16_sweep":
        bits = torch.arange(0x7f80, dtype=torch.int16)
        values = bits.view(torch.bfloat16)
        values = torch.cat((values, -values))
        x = torch.zeros(4, values.numel(), dtype=torch.bfloat16)
        x[0] = values
        return x
    x = torch.randn(4, n, generator=rng)
    if case == "outlier":
        x[:, ::53] *= 300
    elif case == "log_range":
        exponents = torch.rand(4, n, generator=rng) * 24 - 12
        x = x.sign() * torch.exp2(exponents)
    elif case == "cancellation":
        base = torch.randn(2, n, generator=rng) * 128
        # Different rank-local maxima give different quantization errors.
        x[0], x[1] = base[0], -base[0] + x[1] * 0.7
        x[2], x[3] = base[1], -base[1] + x[3] * 0.7
    elif case == "zero":
        x.zero_()
    elif case == "large_1e10":
        x *= 1e10
    elif case == "large_1e30":
        x *= 1e30
    elif case == "tiny_1e-30":
        x *= 1e-30
    elif case == "bf16_extremes":
        # Include largest finite value and smallest subnormal; no cancellation.
        x.zero_()
        x[:, ::32] = torch.finfo(torch.bfloat16).max
        x[:, 1::32] = 2.0 ** -133  # Same group as the maximum.
        x[:, 16::32] = 2.0 ** -133  # Also test a tiny-only GS16 group.
        x[1:] /= -8  # Rank sum finite; rank 0 retains both BF16 extremes.
    return x.to(dtype)


def error_metrics(actual, expected):
    actual, expected = actual.double(), expected.double()
    finite = torch.isfinite(actual)
    finite_expected = torch.isfinite(expected)
    valid = finite & finite_expected
    delta = actual[valid] - expected[valid]
    norm = torch.linalg.vector_norm(expected[valid])
    err = torch.linalg.vector_norm(delta)
    def rms(y):
        return float(torch.sqrt(torch.mean(y.square()))) if y.numel() else 0.0

    nz = valid & (expected != 0)
    relative = (actual[nz] - expected[nz]).abs() / expected[nz].abs()
    return {
        "nonfinite": int((~finite).sum()),
        "reference_nonfinite": int((~finite_expected).sum()),
        "nrmse": (
            None if not bool(finite.all())
            else float(err / norm) if norm > 0
            else (0.0 if err == 0 else None)
        ),
        "max_abs_error": float(delta.abs().max()) if delta.numel() else None,
        "rms_error": rms(delta),
        "reference_rms": rms(expected[valid]),
        "nonzeros_dropped": int(((actual == 0) & (expected != 0)).sum()),
        "relative_error_p95": (
            float(torch.quantile(relative, 0.95)) if relative.numel() else 0.0
        ),
    }


class Prototype:
    def __init__(self, path, ar):
        self.lib = ctypes.CDLL(str(Path(path).resolve()))
        self.fn = self.lib.car_quant
        self.fn.restype = ctypes.c_int
        self.fn.argtypes = (
            [ctypes.c_int] * 4 + [ctypes.c_void_p] * 2
            + [ctypes.c_int] * 2 + [ctypes.POINTER(ctypes.c_void_p)] * 2
            + [ctypes.c_void_p]
        )
        self.buffers = (ctypes.c_void_p * 4)(*ar.buffer_ptrs)
        self.signals = (ctypes.c_void_p * 4)(*ar.meta_ptrs)
        self.rank = ar.rank

    def call(self, x, out, fmt, phase=0):
        _, mode, group, _ = fmt
        if fmt[0].endswith("_2stage") and phase in (0, 2):
            phase = 4 if phase == 0 else 5
        rc = self.fn(
            phase, mode, group, int(x.dtype == torch.float16),
            x.data_ptr(), out.data_ptr(), x.numel(), self.rank,
            self.buffers, self.signals, torch.cuda.current_stream().cuda_stream,
        )
        if rc:
            raise RuntimeError(f"HIP launch error {rc}, phase={phase}, format={fmt}")


def measure(fn, ar, iterations, trials, burst):
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    # Native C++ CAR allocates new rank-pointer slots during capture even for
    # its pre-registered staging buffer; populate those slots before replay.
    with ar.capture(), torch.cuda.graph(graph):
        for _ in range(burst):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    timings, wall = [], []
    for _ in range(trials):
        dist.barrier()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        a.record()
        for _ in range(iterations):
            graph.replay()
        b.record()
        b.synchronize()
        wall.append((time.perf_counter() - start) * 1e6 / iterations / burst)
        timings.append(a.elapsed_time(b) * 1000 / iterations / burst)
    all_times = [None] * 4
    dist.all_gather_object(all_times, {"gpu": timings, "wall": wall})
    maxima = [max(t["gpu"][i] for t in all_times) for i in range(trials)]
    return {
        "gpu_us": statistics.median(maxima),
        "min_us": min(maxima),
        "max_us": max(maxima),
        "wall_us": max(statistics.median(t["wall"]) for t in all_times),
        "rank_trials_us": [t["gpu"] for t in all_times],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", default="1,7,42,128,512,1664,2048")
    parser.add_argument("--dtypes", default="bf16,fp16")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--burst", type=int, default=16)
    parser.add_argument("--stress", type=int, default=80)
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument(
        "--formats", default="all", help="Comma-separated compressed format names"
    )
    parser.add_argument(
        "--trace-dir", help="Optional bounded GPU-only Kineto trace per rank"
    )
    parser.add_argument("--algorithms", default="1stage,2stage")
    args = parser.parse_args()
    rows_list = [int(row) for row in args.rows.split(",")]
    if not rows_list or any(row < 1 or row > 4096 for row in rows_list):
        raise ValueError("Rows must be 1..4096 (40 MiB native CAR allocation)")
    if min(args.iterations, args.trials, args.burst) < 1 or args.stress < 0:
        raise ValueError("Timing counts must be positive; stress cannot be negative")
    formats = [
        f for f in FORMATS
        if args.formats == "all" or f[0] in args.formats.split(",")
    ]
    if not formats:
        raise ValueError("No requested formats recognized")
    bases = formats
    formats = []
    for algorithm in args.algorithms.split(","):
        if algorithm not in ("1stage", "2stage"):
            raise ValueError(f"Unknown algorithm {algorithm}")
        formats += [
            (f[0] + ("_2stage" if algorithm == "2stage" else ""), *f[1:])
            for f in bases
        ]
    torch.set_num_threads(2)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=180))
    if dist.get_world_size() != 4:
        raise RuntimeError("This study requires exactly four local ranks")
    from vllm import _custom_ops as ops
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    if ops.meta_size() != 4864:
        raise RuntimeError("Signal ABI changed; update the standalone prototype")
    ar = CustomAllreduce(group=dist.group.WORLD, device=rank, max_size=40 * 1024**2)
    if ar.disabled:
        raise RuntimeError("Native CAR unavailable on this topology")
    proto = Prototype(args.library, ar)
    result = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(rank),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__), Path(__file__).with_name("car_quant_study.cpp")
            )
        },
        "library_sha256": hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
        "config": vars(args),
        "method": "TP4 uncached IPC; FP32 sum; captured burst; maximum rank latency",
        "accuracy": [], "latency": [], "stress": [],
    }
    profiler = None
    if args.trace_dir:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        )
        profiler.start()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    for dtype_name in args.dtypes.split(","):
        dtype = dtype_map[dtype_name]
        cases = ["normal", "outlier", "log_range", "cancellation", "zero"]
        if dtype == torch.bfloat16:
            cases += [
                "large_1e10", "large_1e30", "tiny_1e-30", "bf16_extremes", "bf16_sweep"
            ]
        if args.skip_accuracy:
            cases = []
        for case in cases:
            partials = make_partials(case, 8 * 5120, dtype)
            x = partials[rank].cuda()
            out = torch.empty_like(x)
            local = torch.empty(x.numel(), dtype=torch.float32, device=x.device)
            ref = partials[0].float().clone()
            for r in range(1, 4):
                ref += partials[r].float()
            ref = ref.to(dtype)
            if dtype == torch.bfloat16:
                transported = partials.half().float()
                fp16_sum = transported[0].clone()
                for r in range(1, 4):
                    fp16_sum += transported[r]
                if rank == 0:
                    result["accuracy"].append({
                        "dtype": dtype_name, "case": case,
                        "format": "native_fp16_wire",
                        "reduction": error_metrics(fp16_sum.half().to(dtype), ref),
                    })
            for fmt in formats:
                proto.call(x, out, fmt)
                torch.cuda.synchronize()
                actual = out.cpu()
                proto.call(x, local, fmt, phase=3)
                torch.cuda.synchronize()
                local_actual = local.cpu()
                local_ref = reference_quant(partials[rank], fmt[1], fmt[2])
                decoded = [None] * 4
                dist.all_gather_object(decoded, local_actual)
                quant_sum = decoded[0].clone()
                for r in range(1, 4):
                    quant_sum += decoded[r]
                # Reduction correctness is independent of quantization error.
                reduction_error = error_metrics(actual, quant_sum.to(dtype))
                if reduction_error["max_abs_error"] not in (None, 0.0):
                    raise RuntimeError(
                        f"Incorrect peer sum: {case} {fmt}: {reduction_error}"
                    )
                row = {
                    "dtype": dtype_name, "case": case, "format": fmt[0],
                    "reduction": error_metrics(actual, ref),
                    "reconstruction": error_metrics(local_actual, partials[rank]),
                    "oracle": error_metrics(local_actual, local_ref),
                    "peer_sum_exact": True,
                }
                if rank == 0:
                    result["accuracy"].append(row)
            if rank == 0:
                print(f"accuracy {dtype_name} {case}: complete", flush=True)

        if not args.accuracy_only:
            for rows in rows_list:
                x = make_partials("normal", rows * 5120, dtype)[rank].cuda()
                out = torch.empty_like(x)
                variants = [("native_auto", None), ("native_1stage", None)]
                if dtype == torch.bfloat16:
                    variants += [("native_fp16_wire", None)]
                variants += [(f[0], f) for f in formats]
                random.Random(rows).shuffle(variants)
                for name, fmt in variants:
                    if fmt is None:
                        os.environ["VLLM_CUSTOM_ALLREDUCE_ALGO"] = (
                            "" if name == "native_auto" else "1stage"
                        )
                        if name == "native_fp16_wire":
                            x16 = torch.empty_like(x, dtype=torch.float16)
                            out16 = torch.empty_like(x, dtype=torch.float16)

                            def fn():
                                x16.copy_(x)
                                ar.all_reduce(x16, out=out16, registered=False)
                                out.copy_(out16)
                        else:
                            fn = lambda: ar.all_reduce(x, out=out, registered=False)
                    else:
                        fn = lambda: proto.call(x, out, fmt)
                    timing = measure(fn, ar, args.iterations, args.trials, args.burst)
                    row = {
                        "dtype": dtype_name, "rows": rows, "elements": x.numel(),
                        "format": name, "total": timing,
                        "bytes_per_value": 2 if fmt is None else 1 + fmt[3] / fmt[2],
                    }
                    if fmt is not None:
                        row["pack"] = measure(
                            lambda: proto.call(x, out, fmt, phase=1),
                            ar, args.iterations, args.trials, args.burst)
                        row["reduce"] = measure(
                            lambda: proto.call(x, out, fmt, phase=2),
                            ar, args.iterations, args.trials, args.burst)
                    if rank == 0:
                        result["latency"].append(row)
                        print(
                            f"{dtype_name} M={rows} {name}: {timing['gpu_us']:.2f} us",
                            flush=True,
                        )

        # Change buffers between graph replays and impose rank-dependent arrival
        # skew. Prevent a static-input benchmark from hiding stale peer reads.
        x = torch.empty(7 * 5120, dtype=dtype, device=f"cuda:{rank}")
        out = torch.empty_like(x)
        for fmt in formats if args.stress else []:
            x.fill_(rank + 1)
            proto.call(x, out, fmt)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                proto.call(x, out, fmt)
            for iteration in range(args.stress):
                value = (iteration % 9 + 1) * (rank + 1) * (-1 if iteration % 2 else 1)
                x.fill_(value)
                # Unequal compute on local default streams, no extra collective.
                if rank == iteration % 4:
                    torch.cuda._sleep(150000)
                graph.replay()
                torch.cuda.synchronize()
                # Exact expected sum of individually encoded scalar groups.
                expected = sum(float(reference_quant(
                    torch.full((fmt[2],), value / (rank + 1) * (r + 1), dtype=dtype),
                    fmt[1], fmt[2])[0]) for r in range(4))
                expected = float(torch.tensor(expected, dtype=dtype))
                if not bool((out == expected).all()):
                    raise RuntimeError(
                        f"Stale peer data: {dtype_name} {fmt}, iter={iteration}"
                    )
            if rank == 0:
                result["stress"].append({"dtype": dtype_name, "format": fmt[0],
                                         "replays": args.stress, "pass": True})
        if rank == 0 and args.stress:
            print(f"changing-input stress {dtype_name}: PASS", flush=True)
    torch.cuda.synchronize()
    dist.barrier()
    if profiler is not None:
        profiler.stop()
        trace_path = Path(args.trace_dir)
        trace_path.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path / f"rank{rank}.json.gz"))
    dist.barrier()
    ar.close()
    dist.barrier()
    if rank == 0:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"results: {path}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
