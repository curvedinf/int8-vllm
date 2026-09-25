import torch, triton, triton.language as tl, time

@triton.jit
def smallm_gemm(A, B, XS, WS, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(0, K, BK):
        a = tl.load(A + offs_m[:, None] * K + k0 + tl.arange(0, BK)[None, :],
                    mask=(offs_m[:, None] < M), other=0)
        b = tl.load(B + offs_n[:, None] * K + k0 + tl.arange(0, BK)[None, :],
                    mask=(offs_n[:, None] < N), other=0)
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.int32)
    xs = tl.load(XS + offs_m, mask=offs_m < M, other=0.0)
    ws = tl.load(WS + offs_n, mask=offs_n < N, other=0.0)
    out = acc.to(tl.float32) * xs[:, None] * ws[None, :]
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], out.to(C.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

M, N, K = 24, 5120, 4352
A = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8)
B = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8)
XS = torch.rand(M, device="cuda") * 0.02 + 0.005
WS = torch.rand(N, device="cuda") * 0.02 + 0.005
C = torch.empty(M, N, device="cuda", dtype=torch.float16)
ref = (A.float() * XS[:, None]) @ (B.float() * WS[:, None]).t()
best = None
for (bn, bk, nw) in [(128, 128, 4), (256, 64, 8), (128, 64, 4), (64, 128, 4), (256, 128, 8)]:
    bm = 32
    try:
        fn = lambda: smallm_gemm[(triton.cdiv(N, bn),)](A, B, XS, WS, C, M, N, K, bm, bn, bk, num_warps=nw)
        fn(); torch.cuda.synchronize()
        err = (C.float() - ref).abs().max().item()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(200): fn()
        e.record(); torch.cuda.synchronize()
        us = s.elapsed_time(e) / 200 * 1000
        wbytes = N * K / 1e6
        print(f"BN={bn} BK={bk} warps={nw}: {us:.1f}us ({wbytes/us*1e3:.0f} GB/s weights) err={err:.3f}")
        best = min(best or us, us)
    except Exception as ex:
        print(f"BN={bn} BK={bk} warps={nw}: FAIL {str(ex)[:60]}")
print(f"best triton: {best:.1f}us vs CK tuned 49.9us; floor ~23.4us")
