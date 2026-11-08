#!/usr/bin/env python3
"""TP4 parity test for the fused AR+RMSNorm+GS128-quant collective.

Run: torchrun --nproc_per_node=4 scripts/test_fused_ar_rms.py
Compares against (all-gather sum -> bf16) + residual_norm_group_quant.
"""
import os
import sys

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    from vllm.distributed.device_communicators.ar_rms_g128 import ArRmsG128
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    cpu_group = dist.new_group(backend="gloo")
    ca = CustomAllreduce(cpu_group, torch.device("cuda", local),
                         symm_mem_enabled=False)
    if rank == 0:
        print(f"custom AR disabled={ca.disabled} world={ca.world_size}",
              file=sys.stderr)
    ar = ArRmsG128(ca, 2048, 2048)

    from vllm.model_executor.layers.quantization.gptq_moe_a8w8 import (
        residual_norm_group_quant,
    )

    dev = torch.device("cuda", local)
    for M in (96, 2048):
        torch.manual_seed(rank * 1000 + M)
        partials = (torch.randn(M, 2048, device=dev) / 10).to(torch.bfloat16)
        residual = torch.randn(M, 2048, device=dev).to(torch.bfloat16)
        w = torch.randn(2048, device=dev).to(torch.bfloat16)
        gathered = [torch.empty_like(partials) for _ in range(4)]
        dist.all_gather(gathered, partials)
        s_sum = torch.stack([g.float() for g in gathered]).sum(0).to(torch.bfloat16)
        res_ref = (s_sum.float() + residual.float()).to(torch.bfloat16)
        ref_normed, ref_q, ref_s = residual_norm_group_quant(
            s_sum, residual.clone(), w, 1e-6)
        normed, res, q, s = ar.fused(partials, residual, w, 1e-6)
        torch.cuda.synchronize()
        if rank == 0:
            if M == 96:
                print("fused_s", s[0, :4].tolist(),
                      "ref_s", ref_s[0, :4].tolist(), file=sys.stderr)
                print("fused_n", normed[0, :4].tolist(),
                      "ref_n", ref_normed[0, :4].tolist(), file=sys.stderr)
            print(
                f"M={M} res_exact={torch.equal(res, res_ref)} "
                f"normed_neq={(normed != ref_normed).sum().item()}/{normed.numel()} "
                f"q_neq={(q != ref_q).sum().item()}/{q.numel()} "
                f"s_neq={(s != ref_s).sum().item()}/{s.numel()}")

        def t(fn, n=20, warm=5):
            for _ in range(warm):
                fn()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            for _ in range(n):
                fn()
            b.record()
            torch.cuda.synchronize()
            return a.elapsed_time(b) / n * 1000

        def unfused():
            summed = ca.custom_all_reduce(partials)
            if summed is None:
                summed = partials.clone()
            return residual_norm_group_quant(summed, residual.clone(), w, 1e-6)

        if rank == 0:
            print(f"M={M} fused us={t(lambda: ar.fused(partials, residual, w, 1e-6)):.1f} "
                  f"unfused(AR+prologue) us={t(unfused):.1f}")
    dist.barrier()
    ar.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
