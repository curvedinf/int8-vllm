# SPDX-License-Identifier: Apache-2.0
"""TP4 MI100 regression for back-to-back GS128 collectives on changing inputs.

Run with torchrun --standalone --nproc-per-node=4 scripts/test_g128_exchange.py.
Set VLLM_RESIDUAL_G128_LIBRARY to the built transport shared object.
"""

import os
import time

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.device_communicators.residual_g128_all_reduce import (
    ResidualG128Allreduce,
)
from vllm.model_executor.layers.residual_int8 import ResidualInt8


def make_input(rows, rank, seed):
    rng = torch.Generator().manual_seed(seed)
    shape = (rows, 16, 128)
    total = torch.zeros(shape)
    for peer in range(4):
        q = torch.randint(-30, 31, shape, generator=rng, dtype=torch.int8)
        # Anchor each group's peak at 127 * scale. The other lanes include
        # half-integers and their neighbors, with an exact float32 oracle.
        q[..., 0] = 127 if peer == 0 else 0
        scale = (torch.arange(rows * 16).view(rows, 16) % 7 - 3).exp2()
        if peer:
            scale = scale * (0.5**peer)
        total += q.float() * scale.unsqueeze(-1)
        if peer == rank:
            local_q, local_s = q, scale
    # The final group also checks nonfinite canonicalization and a partial CTA.
    if seed % 2:
        total[-1, -1, :] = float("nan")
        if rank == 2:
            local_s[-1, -1] = float("nan")
    peak = total.abs().amax(dim=-1)
    expected_s = (peak / 127).clamp_min(1e-30)
    expected_q = torch.nan_to_num((total / expected_s.unsqueeze(-1)).round())
    offset = rank + 1
    storage = torch.empty(local_q.numel() + offset, device="cuda", dtype=torch.int8)
    codes = storage[offset:].view(rows, 2048)
    codes.copy_(local_q.view_as(codes))
    value = ResidualInt8(codes, local_s.cuda(), torch.bfloat16)
    return value, expected_q.to(torch.int8).view(rows, 2048), expected_s


def main():
    os.environ.setdefault("VLLM_GFX908_G128_REDUCE_SCATTER", "1")
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    dist.init_process_group("gloo")
    native = CustomAllreduce(dist.group.WORLD, device=torch.device("cuda", device))
    collective = ResidualG128Allreduce(native)
    try:
        rows = (8, 96, 1, 513, 48, 17, 2048, 3)
        inputs = [make_input(m, dist.get_rank(), i) for i, m in enumerate(rows)]
        # Set the local IPC signal page to UINT32_MAX. The first rendezvous
        # must wait for all four ranks to publish zero, even when one is late.
        CudaRTLibrary().cudaMemset(collective.signals[dist.get_rank()], 0xFF, 16384)
        torch.cuda.synchronize()
        dist.barrier()
        if dist.get_rank() == 3:
            time.sleep(0.1)
        # No sync between launches. This exercises reuse of the peer buffer
        # across different layouts, chunk boundaries, and signal grids.
        eager = [collective.all_reduce(value) for value, _, _ in inputs]
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = [collective.all_reduce(value) for value, _, _ in inputs]
        graph.replay()
        for outputs in (eager, captured):
            for output, (_, codes, scales) in zip(outputs, inputs):
                torch.testing.assert_close(output.codes.cpu(), codes, rtol=0, atol=0)
                torch.testing.assert_close(
                    output.scales.cpu(), scales, rtol=0, atol=0, equal_nan=True
                )
        dist.barrier()
        if dist.get_rank() == 0:
            print("GS128 TP4 codes and scales match on all four ranks.")
    finally:
        collective.close()
        native.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
