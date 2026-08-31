import sys
import torch
sys.path.insert(0, ".")
from vllm.triton_utils import triton, tl


@triton.jit
def _pair_kernel(u_ptr, g_ptr, seed_t, pos_t, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    seed = tl.load(seed_t + offs)
    pos = tl.load(pos_t + offs)
    gumbel_seed = tl.randint(seed, pos)
    u = tl.rand(seed, pos)
    tl.store(u_ptr + offs, u, mask=m)
    tl.store(g_ptr + offs, gumbel_seed.to(tl.float32) / 2147483648.0, mask=m)


N = 1 << 22
seeds = torch.randint(1, 2**31 - 1, (N,), device="cuda", dtype=torch.int32)
poss = torch.randint(0, 100000, (N,), device="cuda", dtype=torch.int32)
u = torch.empty(N, device="cuda", dtype=torch.float32)
g = torch.empty(N, device="cuda", dtype=torch.float32)
_pair_kernel[(triton.cdiv(N, 1024),)](u, g, seeds, poss, N, BLOCK=1024)
u, g = u.cpu(), g.cpu()
corr = torch.corrcoef(torch.stack([u, g]))[0, 1].item()
print(f"corr(rand(seed,pos), randint(seed,pos)) = {corr:+.4f} over {N} pairs")
# also: mean u conditioned on gumbel_seed quartile
qs = torch.quantile(g, torch.tensor([0.25, 0.5, 0.75]))
for name, mask in [("g<Q1", g < qs[0]), ("Q1..Q2", (g >= qs[0]) & (g < qs[1])),
                   ("Q2..Q3", (g >= qs[1]) & (g < qs[2])), ("g>Q3", g >= qs[2])]:
    print(f"  {name}: mean_u={u[mask].mean().item():.4f}")
