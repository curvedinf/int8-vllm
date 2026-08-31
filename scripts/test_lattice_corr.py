import sys
import torch
sys.path.insert(0, ".")
from vllm.triton_utils import triton, tl


@triton.jit
def _lattice_kernel(out_ptr, seed_t, pos_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    seed = tl.load(seed_t)
    pos = tl.load(pos_ptr + offs)
    gumbel_seed = tl.randint(seed, pos)
    u = tl.rand(gumbel_seed, offs)
    tl.store(out_ptr + offs, u, mask=m)


N = 1 << 20
# For ONE fixed token key (offs == key), vary position: u(P) across P.
pos = torch.arange(N, device="cuda", dtype=torch.int32)
seed_t = torch.tensor([7], device="cuda", dtype=torch.int32)
out = torch.empty(N, device="cuda", dtype=torch.float32)
_lattice_kernel[(triton.cdiv(N, 1024),)](out, seed_t, pos, N, BLOCK=1024)
u = out.cpu()
# lag-k correlation of u across positions
for lag in (1, 2, 13):
    a, b = u[:-lag], u[lag:]
    va, vb = a - a.mean(), b - b.mean()
    r = (va * vb).sum() / (va.norm() * vb.norm())
    print(f"lag-{lag} position correlation (same key): {r.item():+.4f}")

# Also: same position, adjacent keys — correlation across token ids.
@triton.jit
def _keys_kernel(out_ptr, seed_t, P, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    seed = tl.load(seed_t)
    gumbel_seed = tl.randint(seed, P)
    u = tl.rand(gumbel_seed, offs)
    tl.store(out_ptr + offs, u, mask=m)

out2 = torch.empty(N, device="cuda", dtype=torch.float32)
_keys_kernel[(triton.cdiv(N, 1024),)](out2, seed_t, 12345, N, BLOCK=1024)
u2 = out2.cpu()
for lag in (1, 2):
    a, b = u2[:-lag], u2[lag:]
    va, vb = a - a.mean(), b - b.mean()
    r = (va * vb).sum() / (va.norm() * vb.norm())
    print(f"lag-{lag} key correlation (same position): {r.item():+.4f}")
