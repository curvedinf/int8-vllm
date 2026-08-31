import sys
import torch
sys.path.insert(0, ".")
from vllm.triton_utils import triton, tl


@triton.jit
def _u_kernel(out_ptr, seed, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    u = tl.rand(seed, offs)
    tl.store(out_ptr + offs, u, mask=m)


@triton.jit
def _gumbel_kernel(out_ptr, seed_ptr, pos_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    seed = tl.load(seed_ptr)
    pos = tl.load(pos_ptr + offs)
    gumbel_seed = tl.randint(seed, pos)
    u = tl.rand(gumbel_seed, offs)
    tl.store(out_ptr + offs, u, mask=m)


N = 1 << 22
for seed in (7, 123456789):
    out = torch.empty(N, device="cuda", dtype=torch.float32)
    _u_kernel[(triton.cdiv(N, 1024),)](out, seed, N, BLOCK=1024)
    u = out.cpu()
    dec = torch.histc(u, bins=10, min=0.0, max=1.0) / N * 100
    print(f"tl.rand(seed={seed}): mean={u.mean().item():.6f} (exp 0.5) "
          f"deciles={[round(x, 1) for x in dec.tolist()]}", flush=True)

pos = torch.arange(N, device="cuda", dtype=torch.int32)
seed_t = torch.tensor([7], device="cuda", dtype=torch.int32)
out = torch.empty(N, device="cuda", dtype=torch.float32)
_gumbel_kernel[(triton.cdiv(N, 1024),)](out, seed_t, pos, N, BLOCK=1024)
u = out.cpu()
dec = torch.histc(u, bins=10, min=0.0, max=1.0) / N * 100
print(f"gumbel-path u: mean={u.mean().item():.6f} "
      f"deciles={[round(x, 1) for x in dec.tolist()]}", flush=True)
