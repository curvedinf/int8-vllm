"""4-rank CPU equivalence check for FlintRef TP sharding (gloo).

torchrun --nproc_per_node=4 scripts/ptqr_flint_tp_check.py

Each rank shards the same seed-initialized tiny FlintRef (world=4); rank 0
also builds the unsharded reference; logits must match within fp32 tolerance
(tp_sum/all_gather are exact reductions; only op-order noise remains).
"""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptqr_flint_model import FlintRef, flint_tiny_config, shard_flint_tp  # noqa: E402


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()

    torch.manual_seed(0)  # identical init on every rank
    cfg = flint_tiny_config()
    cfg.attention_bias = False
    ref = FlintRef(cfg)  # unsharded, identical on all ranks
    with torch.no_grad():
        for p in ref.parameters():
            torch.nn.init.normal_(p, std=0.05)
        # distinct sinks so the head-slice path is exercised
        for layer in ref.layers:
            torch.nn.init.normal_(layer.attn.sink, std=0.3)

    sharded = FlintRef(cfg)
    sharded.load_state_dict(ref.state_dict())
    shard_flint_tp(sharded, world, rank, group=None)
    sharded.eval()
    ref.eval()

    torch.manual_seed(1234)
    ids = torch.randint(0, cfg.vocab_size, (3, 40))

    import ptqr_flint_model as M
    orig = M.tp_sum
    def traced(x):
        if dist.get_rank() == 0:
            print("tp_sum shape", tuple(x.shape), flush=True)
        return orig(x)
    M.tp_sum = traced
    with torch.no_grad():
        out_s = sharded(ids)
        out_r = ref(ids)

    if rank == 0:
        diff = (out_s - out_r).abs().max().item()
        denom = out_r.abs().max().item() or 1.0
        print(f"sharded-vs-ref max abs diff {diff:.3e} (scale {denom:.2f})")
        assert diff <= 1e-3 * max(denom, 1.0), (diff, denom)
        print("TP4 sharding equivalence PASS")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
