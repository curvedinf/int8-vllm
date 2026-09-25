import torch, time, sys
x = torch.empty(256<<20, dtype=torch.uint8, device="cuda")   # 256 MiB
y = torch.empty_like(x)
for _ in range(3): y.copy_(x)
torch.cuda.synchronize()
ts=[]
for _ in range(10):
    t0=time.perf_counter(); y.copy_(x); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
ts.sort()
bw = 2*(256<<20)/ts[5]/1e12
print(f"copy bw (read+write): {bw:.2f} TB/s")
r = x.view(-1,1024).sum(dtype=torch.int64)  # read-mostly kernel
torch.cuda.synchronize()
ts=[]
for _ in range(10):
    t0=time.perf_counter(); r = x.view(-1,1024).sum(dtype=torch.int64); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
ts.sort()
bwr = (256<<20)/ts[5]/1e12
print(f"read-only sum bw: {bwr:.2f} TB/s")
