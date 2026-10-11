#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Four-rank numerical/graph/timing gate; requires an isolated idle host."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from benchmark_gfx908_car_quant import measure
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.device_communicators.decode_int8_all_reduce import (
    DecodeInt8Allreduce,
)
from vllm.model_executor.kernels.linear.mixed_precision.act_quant_rn import (
    pertoken_quant_rn,
)
from vllm.model_executor.layers.decode_int8 import _gemm, _prefill_consume
from vllm.model_executor.layers.layernorm import _gemma_fused_add_rmsnorm
from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv


def quantize(x):
    blocks = x.float().reshape(x.shape[0], -1, 128)
    scale = blocks.abs().amax(-1) / 127.0
    scale = torch.where(scale == 0, 1.0, scale)
    q = torch.floor(blocks / scale[..., None] + .5).clamp(-127, 127)
    return q.to(torch.int8).reshape(x.shape), scale


def gemm_reference(x, w, ws):
    q, s = quantize(x)
    result = torch.zeros((x.shape[0], w.shape[0]))
    for group in range(x.shape[1] // 128):
        a = q[:, group * 128:(group + 1) * 128].float()
        b = w[:, group * 128:(group + 1) * 128].float()
        result += ((a @ b.T) * s[:, group, None]) * ws[None, :, group]
    return quantize(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace-dir")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    native = CustomAllreduce(dist.group.WORLD, rank, max_size=40 * 1024**2)
    comm = DecodeInt8Allreduce(native)
    result = {"numeric": [], "latency": [], "graph_replays": 0}
    result["prefill_norm"] = []
    for rows in (128, 1664):
        torch.manual_seed(120)
        dense = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(dense)
        weight = torch.randn(5120, device="cuda", dtype=torch.bfloat16)
        for gemma in (False, True):
            out, res = _prefill_consume(dense, residual, weight, 1e-6,
                                        gemma, None, None, 0, 0)
            total = dense.cpu().float() + residual.cpu().float()
            norm = total * torch.rsqrt(total.square().mean(-1, keepdim=True) + 1e-6)
            ew = weight.cpu().float() + 1 if gemma else weight.cpu()
            expected = (norm.to(ew.dtype) * ew).bfloat16()
            torch.testing.assert_close(res.cpu(), total.bfloat16(), rtol=0, atol=0)
            torch.testing.assert_close(out.cpu(), expected, rtol=.016, atol=.016)
            result["prefill_norm"].append({"rows": rows, "gemma": gemma,
                "max_error": (out.cpu().float() - expected.float()).abs().max().item()})
    for m, k in ((1, 256), (7, 256), (1, 4352), (7, 4352),
                 (42, 4352), (48, 4352), (64, 4352)):
        torch.manual_seed(100 + rank)
        x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        w = torch.randint(-127, 128, (5120, k), dtype=torch.int8, device="cuda")
        ws = torch.rand((5120, k // 128), device="cuda") * .003 + .0001
        q, s = _gemm(x, w, ws)
        # Complete output groups: scale is not computable from a subset
        # of the group's columns. Compare the first 128 as an exact group.
        rq, rs = gemm_reference(x.cpu(), w[:128].cpu(), ws[:128].cpu())
        torch.testing.assert_close(s[:, :1].cpu(), rs, rtol=3e-6, atol=1e-7)
        delta_q = (q[:, :128].cpu().int() - rq.int()).abs().max().item()
        assert delta_q <= 1, delta_q  # threshold ties may differ by FP32 reduction
        gathered = [None] * 4
        dist.all_gather_object(gathered, (q.cpu(), s.cpu()))
        decoded = torch.zeros((m, 5120))
        for cq, cs in gathered:
            decoded += (cq.float().reshape(m, 40, 128) * cs[..., None]).reshape(m, 5120)
        decoded = decoded.bfloat16()
        torch.manual_seed(999)
        residual = torch.randn((m, 5120), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(5120, device="cuda", dtype=torch.bfloat16) * .1
        for conv in (False, True):
            cd = (torch.randn((m, 2, 320), device="cuda",
                              dtype=torch.bfloat16) * .1 if conv else None)
            cb = (torch.randn((2, 5120), device="cuda",
                              dtype=torch.bfloat16) * .1 if conv else None)
            values = (_grouped_conv(decoded, cd.cpu(), cb.cpu(), 7, 320, 16, 2)
                      if conv else decoded)
            total = values.float() + residual.cpu().float()
            inv = torch.rsqrt((total * total).mean(-1, keepdim=True) + 1e-6)
            for gemma in (False, True):
                ref = (total * inv * (1 + weight.cpu().float())).bfloat16() if gemma else (
                    (total * inv).bfloat16() * weight.cpu())
                out, res = comm.norm(q, s, residual, weight, 1e-6, gemma,
                                     cd, cb, 7, 16)
                torch.testing.assert_close(res.cpu(), total.bfloat16(), rtol=0, atol=0)
                torch.testing.assert_close(out.cpu(), ref, rtol=.016, atol=.016)
                result["numeric"].append({"m": m, "k": k, "conv": conv,
                    "gemma": gemma, "max_code_error": delta_q,
                    "max_norm_error": (out.cpu().float() - ref.float()).abs().max().item()})

        def candidate():
            iq, scale = _gemm(x, w, ws)
            return comm.norm(iq, scale, residual, weight, 1e-6, True,
                             None, None, 0, 0)

        # Same original checkpoint-equivalent weights, converted exactly
        # as the current CK loader does for the latency control.
        from aiter import gemm_a8w8_CK
        dense = (w.float().reshape(5120, k // 128, 128)
                 * ws[..., None]).reshape(5120, k).bfloat16().float()
        cs = dense.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127
        cw = (dense / cs).round().clamp(-127, 127).to(torch.int8)
        norm_weight = 1 + weight.float()

        def baseline():
            iq, scale = pertoken_quant_rn(x)
            partial = gemm_a8w8_CK(iq, cw, scale, cs, None, torch.bfloat16)
            reduced = native.all_reduce(partial, registered=False)
            return _gemma_fused_add_rmsnorm(reduced, residual,
                                            norm_weight, 1e-6)

        candidate()
        baseline()
        result["latency"].append({"m": m, "k": k,
            "native": measure(baseline, native, 10, 5, 8),
            "candidate": measure(candidate, native, 10, 5, 8)})
        graph = torch.cuda.CUDAGraph()
        with native.capture(), torch.cuda.graph(graph):
            output, resout = candidate()
            native.all_reduce(residual, registered=False)
        for iteration in range(12):
            x.fill_((iteration - 6) * .125 * (rank + 1))
            graph.replay()
            eager, eres = candidate()
            torch.testing.assert_close(output, eager, rtol=0, atol=0)
            torch.testing.assert_close(resout, eres, rtol=0, atol=0)
            result["graph_replays"] += 1
        if args.trace_dir and m == 42:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                for _ in range(5):
                    graph.replay()
                torch.cuda.synchronize()
            target = Path(args.trace_dir)
            target.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(target / f"rank{rank}.json.gz"))
    gathered = [None] * 4
    dist.all_gather_object(gathered, result)
    if rank == 0:
        Path(args.output).write_text(json.dumps({"ranks": gathered}, indent=2))
        print(json.dumps(result), flush=True)
    comm.close()
    native.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
