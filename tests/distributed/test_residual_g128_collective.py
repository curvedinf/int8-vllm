# SPDX-License-Identifier: Apache-2.0
"""Run directly with .venv/bin/python -m torch.distributed.run --nproc-per-node=4.

Real TP4 IPC reduction, changing shapes/chunks, interleaved native collectives,
and graph replay with changed inputs. Requires idle GPUs and the built library.
"""

import torch
import torch.distributed as dist

from vllm.model_executor.layers.residual_int8 import ResidualInt8


def main():
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    native = CustomAllreduce(dist.group.WORLD, torch.device("cuda", rank))
    assert not native.disabled and native.residual_g128 is not None
    collective = native.residual_g128
    # Small, >one chunk, tail chunk, then small again exercise epoch reuse.
    for rows in (1, 64, 257, 513, 8):
        base_q = ((torch.arange(rows * 4096, device="cuda") % 15) - 7).view(rows, 4096)
        base_s = ((torch.arange(rows * 32, device="cuda") % 17) + 1).view(
            rows, 32
        ).float() * 0.001
        q = (base_q * (rank + 1)).to(torch.int8)
        s = base_s * (rank + 1)
        # sum((rank+1)^2) = 30; vary every group and every lane.
        expected = base_q.float() * base_s.repeat_interleave(128, -1) * 30
        value = ResidualInt8(q, s, torch.bfloat16)
        for _ in range(3):
            result = collective.all_reduce(value)
        decoded = result.codes.float() * result.scales.repeat_interleave(128, -1)
        bound = result.scales.repeat_interleave(128, -1) * 0.501 + 1e-5
        assert torch.all((decoded - expected).abs() <= bound)
        dense = torch.full((4096,), rank + 1, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(
            native.all_reduce(dense, registered=False), torch.full_like(dense, 10)
        )
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        dist.barrier()
        with torch.cuda.graph(graph):
            result = collective.all_reduce(value)
        for step in (2, 3, 1):
            q.copy_(base_q * (rank + 1) * step)
            graph.replay()
            decoded = result.codes.float() * result.scales.repeat_interleave(128, -1)
            bound = result.scales.repeat_interleave(128, -1) * 0.501 + 1e-5
            assert torch.all((decoded - expected * step).abs() <= bound)
        torch.cuda.synchronize()
        dist.barrier()
    if rank == 0:
        print(
            "TP4 GS128 collective: eager, chunks, native interleave, graph replay PASS",
            flush=True,
        )
    native.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
