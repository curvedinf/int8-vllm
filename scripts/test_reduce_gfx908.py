import sys, torch
sys.path.insert(0, "/home/curved/vllm-gfx908")
from vllm.v1.attention.ops.gfx908_g128_gluon_reduce import reduce_segments_gfx908
from vllm.v1.attention.ops.triton_unified_attention import reduce_segments

torch.manual_seed(3)
T, H, S, D, TILE = 42, 6, 2, 256, 64
seqlens = torch.randint(2000, 40000, (T,), device="cuda", dtype=torch.int32)
pm = (torch.randn(T, H, S, device="cuda") * 3).float()
pl = torch.rand(T, H, S, device="cuda").float() * 400 + 1
po = (torch.randn(T, H, S, D, device="cuda") * 4).float()
out_tri = torch.empty(T, H, D, device="cuda", dtype=torch.float32)
out_glu = torch.empty_like(out_tri)
reduce_segments[(T, H)](
    output_ptr=out_tri, segm_output_ptr=po, segm_max_ptr=pm, segm_expsum_ptr=pl,
    seq_lens_ptr=seqlens, num_seqs=T, num_query_heads=H,
    out_scale_inv=1.0, output_stride_0=out_tri.stride(0), output_stride_1=out_tri.stride(1),
    block_table_stride=1, TILE_SIZE=TILE, HEAD_SIZE=D, HEAD_SIZE_PADDED=D,
    query_start_len_ptr=torch.zeros(T+1, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32),
    BLOCK_Q=1, NUM_SEGMENTS_PER_SEQ=S, USE_FP8=False)
cuq = torch.arange(0, T + 1, T // 6, device="cuda", dtype=torch.int32)
if cuq.numel() < T + 1:
    cuq = torch.cat([cuq, torch.full((T + 1 - cuq.numel(),), T, device="cuda", dtype=torch.int32)])
num_seqs = 6
reduce_segments_gfx908[(T, H)](
    out_glu, po, pm, pl, seqlens, cuq, num_seqs,
    OUT_STRIDE0=out_glu.stride(0), OUT_STRIDE1=out_glu.stride(1),
    H=H, SPLITS=S, TILE=TILE, D=D, num_warps=4)
d = (out_tri - out_glu).abs()
print(f"maxdiff={d.max().item():.3e} meandiff={d.mean().item():.3e}")

# perf: production decode shape T=42 H=6 S=2 D=256
import torch
def bench(fn, iters=2000):
    for _ in range(50): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us

t_tri = bench(lambda: reduce_segments[(T, H)](
    output_ptr=out_tri, segm_output_ptr=po, segm_max_ptr=pm, segm_expsum_ptr=pl,
    seq_lens_ptr=seqlens, num_seqs=T, num_query_heads=H,
    out_scale_inv=1.0, output_stride_0=out_tri.stride(0), output_stride_1=out_tri.stride(1),
    block_table_stride=1, TILE_SIZE=TILE, HEAD_SIZE=D, HEAD_SIZE_PADDED=D,
    query_start_len_ptr=torch.zeros(T+1, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32),
    BLOCK_Q=1, NUM_SEGMENTS_PER_SEQ=S, USE_FP8=False))
t_glu = bench(lambda: reduce_segments_gfx908[(T, H)](
    out_glu, po, pm, pl, seqlens, cuq, num_seqs,
    OUT_STRIDE0=out_glu.stride(0), OUT_STRIDE1=out_glu.stride(1),
    H=H, SPLITS=S, TILE=TILE, D=D, num_warps=4))
print(f"triton={t_tri:.2f}us gluon={t_glu:.2f}us ratio={t_tri/t_glu:.2f}x")
