# SPDX-License-Identifier: Apache-2.0
"""Run directly with .venv/bin/python -m torch.distributed.run --nproc-per-node=4.

Real TP4 IPC reduction, changing shapes/chunks, interleaved native collectives,
and graph replay with changed inputs. Requires idle GPUs and the built library.
Pass --rccl with --nproc-per-node=2 to check the PCIe transport.
"""

import argparse
import os

import torch
import torch.distributed as dist

from vllm.model_executor.layers.residual_int8 import ResidualInt8


def main():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )
    from vllm.distributed.device_communicators import residual_g128_all_reduce
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    parser = argparse.ArgumentParser()
    parser.add_argument("--rccl", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if args.rccl:
        set_custom_all_reduce(False)
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(local_rank=rank)
            initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())
        collective = residual_g128_all_reduce
        dense_sum = get_tp_group().all_reduce
    else:
        dist.init_process_group("gloo")
        native = CustomAllreduce(dist.group.WORLD, torch.device("cuda", rank))
        assert not native.disabled and native.residual_g128 is not None
        collective = native.residual_g128

        def dense_sum(tensor):
            return native.all_reduce(tensor, registered=False)

    ranks = range(1, dist.get_world_size() + 1)
    sum_squares = sum(i * i for i in ranks)
    # Small, >one chunk, tail chunk, then small again exercise epoch reuse.
    for rows in (1, 64, 257, 513, 8):
        base_q = ((torch.arange(rows * 4096, device="cuda") % 15) - 7).view(rows, 4096)
        base_s = ((torch.arange(rows * 32, device="cuda") % 17) + 1).view(
            rows, 32
        ).float() * 0.001
        q = (base_q * (rank + 1)).to(torch.int8)
        s = base_s * (rank + 1)
        expected = base_q.float() * base_s.repeat_interleave(128, -1) * sum_squares
        value = ResidualInt8(q, s, torch.bfloat16)
        for _ in range(3):
            result = collective.all_reduce(value)
        decoded = result.codes.float() * result.scales.repeat_interleave(128, -1)
        bound = result.scales.repeat_interleave(128, -1) * 0.501 + 1e-5
        assert torch.all((decoded - expected).abs() <= bound)
        dense = torch.full((4096,), rank + 1, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(
            dense_sum(dense), torch.full_like(dense, sum(ranks))
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
            f"TP{dist.get_world_size()} GS128 collective: "
            "eager, chunks, interleave, graph replay PASS",
            flush=True,
        )
    if args.rccl:
        destroy_model_parallel()
    else:
        native.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
