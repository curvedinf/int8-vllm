import sys
import torch
sys.path.insert(0, ".")
from vllm.triton_utils import triton, tl


@triton.jit
def _winners_kernel(out_ptr, seed_t, pos_ptr, V, PPOS: tl.constexpr,
                    BLOCK: tl.constexpr):
    # one program per position; gumbel-argmax over flat logits (0.0) with
    # lattice noise keyed (randint(seed,pos), token)
    p = tl.program_id(0)
    seed = tl.load(seed_t)
    pos = tl.load(pos_ptr + p)
    gumbel_seed = tl.randint(seed, pos)
    best = -1e30
    best_x = -1
    for v0 in range(0, V, BLOCK):
        offs = v0 + tl.arange(0, BLOCK)
        m = offs < V
        u = tl.rand(gumbel_seed, offs)
        u = tl.maximum(u, 1e-10)
        g = -tl.log(-tl.log(u))
        g = tl.where(m, g, -1e30)
        val = tl.max(g, axis=0)
        idx = tl.argmax(g, axis=0)
        if val > best:
            best = val
            best_x = v0 + idx
    tl.store(out_ptr + p, best_x)


V, P = 8192, 3000
seed_t = torch.tensor([7], device="cuda", dtype=torch.int32)
pos = torch.arange(1000000, 1000000 + P, device="cuda", dtype=torch.int32)
out = torch.empty(P, device="cuda", dtype=torch.int64)
_winners_kernel[(P,)](out, seed_t, pos, V, PPOS=1, BLOCK=1024)
w = out.cpu()
rep = sum(1 for i in range(1, P) if w[i] == w[i - 1])
print(f"lattice: same-token consecutive wins {rep}/{P-1} "
      f"(rate {rep/(P-1):.5f}, chance ~{1/V:.5f})")

# control: torch fresh RNG per position
u = torch.rand((P, V), generator=torch.Generator().manual_seed(7))
u = u.clamp_min(1e-10)
tw = (-torch.log(-torch.log(u))).argmax(-1)
rep2 = sum(1 for i in range(1, P) if tw[i] == tw[i - 1])
print(f"torch control: {rep2}/{P-1} (rate {rep2/(P-1):.5f})")
