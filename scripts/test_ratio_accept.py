import sys, math, torch
sys.path.insert(0, ".")
sys.path.insert(0, "../aiter")
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

DEV = "cuda"; V = 1024; R = 777; N = 40000; NS = 1
torch.cuda.init()
# wall config: target p(R)=0.90, drafter q(R)=0.95 (sharper on R)
target = torch.full((V,), -30.0, device=DEV)
rest = torch.tensor([1.0 / (i + 1) for i in range(18)], device=DEV)
pR = 0.90
rest_n = rest / rest.sum() * (1 - pR)
target[R] = math.log(pR)
target[100:118] = torch.log(rest_n)
tl_batch = target.unsqueeze(0).expand(N * (NS + 1), V).contiguous()
qR = 0.95
draft = torch.full((N, NS, V), -float("inf"), device=DEV)
drest = rest / rest.sum() * (1 - qR)
draft[:, 0, R] = math.log(qR)
draft[:, 0, 100:116] = torch.log(drest[:16])
dsm = torch.zeros(N * (NS + 1), dtype=torch.int64, device=DEV)
dsm[1::2] = R
cu = torch.arange(0, N * (NS + 1) + 1, NS + 1, device=DEV, dtype=torch.int32)
pos = torch.arange(1000, 1000 + N * (NS + 1), device=DEV)
idx = torch.arange(N, device=DEV, dtype=torch.int32)
e_idx = idx.repeat_interleave(NS + 1)
e_pos = torch.arange(NS + 1, device=DEV, dtype=torch.int32).repeat(N)
temp = torch.ones(N, device=DEV)
seeds = torch.randint(1, 2**31 - 1, (N,), device=DEV, dtype=torch.int64)
s, n = rejection_sample(tl_batch, draft, dsm, cu, pos, idx, e_idx, e_pos,
                        temp, seeds, NS, None, use_fp64=True,
                        use_block_verification=False)
torch.cuda.synchronize()
rows = s.view(N, NS + 1)
accepted = (rows[:, 0] == R)
emp = accepted.float().mean().item()
ana = min(1.0, pR / qR)
print(f"analytic P(accept R) = {ana:.4f}   empirical = {emp:.4f}   ratio = {emp/ana:.4f}")
