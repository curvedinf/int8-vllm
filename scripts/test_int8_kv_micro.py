#!/usr/bin/env python3
"""Micro-correctness test for AITER int8 per-token-head KV attention."""

import torch
from aiter.ops.triton.attention.unified_attention import unified_attention


def quantize_per_token_head(x):
    """x: [..., head_size] -> (int8 x_q, float32 scale)."""
    absmax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
    x_q = (x / scale).clamp(-128, 127).round().to(torch.int8)
    return x_q, scale.squeeze(-1)


def build_kv_cache(seq_lens, num_kv_heads, head_size, block_size, num_blocks):
    k_cache = torch.zeros(
        (num_blocks, block_size, num_kv_heads, head_size), dtype=torch.int8, device="cuda"
    )
    v_cache = torch.zeros_like(k_cache)
    k_scale = torch.ones((num_blocks, block_size, num_kv_heads), dtype=torch.float32, device="cuda")
    v_scale = torch.ones_like(k_scale)
    return k_cache, v_cache, k_scale, v_scale


def fill_kv_cache(k_cache, v_cache, k_scale, v_scale, block_table, seq_lens, key, value):
    """key/value: [num_tokens, nkv, hs] bf16/fp16."""
    pos = 0
    num_seqs = len(seq_lens)
    for s in range(num_seqs):
        for i in range(seq_lens[s].item()):
            slot = block_table[s, i // k_cache.shape[1]].item() * k_cache.shape[1] + (
                i % k_cache.shape[1]
            )
            blk = slot // k_cache.shape[1]
            off = slot % k_cache.shape[1]
            k_q, k_sc = quantize_per_token_head(key[pos : pos + 1])
            v_q, v_sc = quantize_per_token_head(value[pos : pos + 1])
            k_cache[blk, off] = k_q.squeeze(0)
            v_cache[blk, off] = v_q.squeeze(0)
            k_scale[blk, off] = k_sc.squeeze(0)
            v_scale[blk, off] = v_sc.squeeze(0)
            pos += 1


def ref_attention(q, k, v, causal=True, scale=None):
    """Naive reference attention. q,k,v: [num_tokens, num_heads, hs]."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q_h = q.transpose(0, 1)  # [num_heads, seq_len, hs]
    k_h = k.transpose(0, 1)
    v_h = v.transpose(0, 1)
    scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale  # [H, Q, K]
    if causal:
        seq_len = q.shape[0]
        mask = torch.arange(seq_len, device=q.device).unsqueeze(1) >= torch.arange(
            seq_len, device=q.device
        ).unsqueeze(0)
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    p = torch.softmax(scores, dim=-1)
    out_h = torch.matmul(p, v_h)  # [H, Q, hs]
    return out_h.transpose(0, 1)


def run_one(num_seqs, seq_lens, num_heads, num_kv_heads, head_size, block_size, name):
    device = "cuda"
    dtype = torch.float16
    num_queries_per_kv = num_heads // num_kv_heads
    total_tokens = sum(seq_lens)

    q = torch.randn((total_tokens, num_heads, head_size), dtype=dtype, device=device)
    key = torch.randn((total_tokens, num_kv_heads, head_size), dtype=dtype, device=device)
    value = torch.randn((total_tokens, num_kv_heads, head_size), dtype=dtype, device=device)

    max_blocks = (max(seq_lens) + block_size - 1) // block_size + 1
    num_blocks = num_seqs * max_blocks
    k_cache, v_cache, k_scale, v_scale = build_kv_cache(
        seq_lens, num_kv_heads, head_size, block_size, num_blocks
    )
    block_table = torch.zeros((num_seqs, max_blocks), dtype=torch.int32, device=device)
    for s in range(num_seqs):
        for b in range((seq_lens[s] + block_size - 1) // block_size):
            block_table[s, b] = s * max_blocks + b
    fill_kv_cache(k_cache, v_cache, k_scale, v_scale, block_table, seq_lens, key, value)

    out = torch.zeros_like(q)
    cu_seqlens = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(seq_lens, dim=0)]
    ).to(torch.int32)

    unified_attention(
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=seq_lens.max().item(),
        seqused_k=seq_lens,
        max_seqlen_k=seq_lens.max().item(),
        softmax_scale=head_size ** -0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        k_scale_cache=k_scale,
        v_scale_cache=v_scale,
    )

    # Reference: un-quantize cache and run naive attention per sequence.
    k_full = torch.empty_like(key)
    v_full = torch.empty_like(value)
    pos = 0
    for s in range(num_seqs):
        for i in range(seq_lens[s].item()):
            slot = block_table[s, i // block_size].item() * block_size + (i % block_size)
            blk = slot // block_size
            off = slot % block_size
            k_full[pos] = k_cache[blk, off].to(dtype) * k_scale[blk, off].unsqueeze(-1)
            v_full[pos] = v_cache[blk, off].to(dtype) * v_scale[blk, off].unsqueeze(-1)
            pos += 1

    ref_out = torch.empty_like(q)
    start = 0
    for s in range(num_seqs):
        end = start + seq_lens[s].item()
        ref_out[start:end] = ref_attention(
            q[start:end],
            k_full[start:end].repeat_interleave(num_queries_per_kv, dim=1),
            v_full[start:end].repeat_interleave(num_queries_per_kv, dim=1),
            scale=head_size ** -0.5,
        )
        start = end

    max_diff = (out - ref_out).abs().max().item()
    mean_diff = (out - ref_out).abs().mean().item()
    ok = max_diff < 1e-1
    print(
        f"{name}: max_diff={max_diff:.4e} mean_diff={mean_diff:.4e} "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def run_decode(num_seqs, seq_lens, num_heads, num_kv_heads, head_size, block_size, name):
    device = "cuda"
    dtype = torch.float16
    num_queries_per_kv = num_heads // num_kv_heads
    total_tokens = sum(seq_lens)

    q_last = torch.randn((num_seqs, num_heads, head_size), dtype=dtype, device=device)
    key = torch.randn((total_tokens, num_kv_heads, head_size), dtype=dtype, device=device)
    value = torch.randn((total_tokens, num_kv_heads, head_size), dtype=dtype, device=device)

    max_blocks = (max(seq_lens) + block_size - 1) // block_size + 1
    num_blocks = num_seqs * max_blocks
    k_cache, v_cache, k_scale, v_scale = build_kv_cache(
        seq_lens, num_kv_heads, head_size, block_size, num_blocks
    )
    block_table = torch.zeros((num_seqs, max_blocks), dtype=torch.int32, device=device)
    for s in range(num_seqs):
        for b in range((seq_lens[s] + block_size - 1) // block_size):
            block_table[s, b] = s * max_blocks + b
    fill_kv_cache(k_cache, v_cache, k_scale, v_scale, block_table, seq_lens, key, value)

    out = torch.zeros_like(q_last)
    cu_seqlens = torch.arange(num_seqs + 1, dtype=torch.int32, device=device)

    unified_attention(
        q=q_last,
        k=k_cache,
        v=v_cache,
        out=out,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=1,
        seqused_k=seq_lens,
        max_seqlen_k=seq_lens.max().item(),
        softmax_scale=head_size ** -0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        k_scale_cache=k_scale,
        v_scale_cache=v_scale,
    )

    # Reference
    k_full = torch.empty((total_tokens, num_kv_heads, head_size), dtype=dtype, device=device)
    v_full = torch.empty_like(k_full)
    pos = 0
    for s in range(num_seqs):
        for i in range(seq_lens[s].item()):
            slot = block_table[s, i // block_size].item() * block_size + (i % block_size)
            blk = slot // block_size
            off = slot % block_size
            k_full[pos] = k_cache[blk, off].to(dtype) * k_scale[blk, off].unsqueeze(-1)
            v_full[pos] = v_cache[blk, off].to(dtype) * v_scale[blk, off].unsqueeze(-1)
            pos += 1

    ref_out = torch.empty_like(q_last)
    start = 0
    for s in range(num_seqs):
        end = start + seq_lens[s].item()
        ref_out[s] = ref_attention(
            q_last[s : s + 1],
            k_full[start:end].repeat_interleave(num_queries_per_kv, dim=1),
            v_full[start:end].repeat_interleave(num_queries_per_kv, dim=1),
            causal=True,
            scale=head_size ** -0.5,
        ).squeeze(0)
        start = end

    max_diff = (out - ref_out).abs().max().item()
    mean_diff = (out - ref_out).abs().mean().item()
    ok = max_diff < 1e-1
    print(
        f"{name}: max_diff={max_diff:.4e} mean_diff={mean_diff:.4e} "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def run_noncausal():
    """Non-causal (DFlash-draft-style) int8 PTH read/write path check.

    The unified kernel treats causality as a runtime flag; INT8 PTH scales
    are applied identically. This guards that combination.
    """
    num_heads, num_kv_heads, head_size, block_size = 6, 1, 256, 32
    seq_len = 512
    # unified_attention non-causal: query attends to all KV including future
    q = torch.randn(seq_len, num_heads, head_size, dtype=torch.float16, device="cuda")
    k = torch.randn(seq_len, num_kv_heads, head_size, dtype=torch.float16, device="cuda")
    v = torch.randn(seq_len, num_kv_heads, head_size, dtype=torch.float16, device="cuda")

    k_q, k_sc = quantize_per_token_head(k)
    v_q, v_sc = quantize_per_token_head(v)

    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode

    num_blocks = seq_len // block_size
    # kernel reads block_size=v.shape[1], nkv=k.shape[2] -> expects
    # [num_blocks, block_size, num_kv_heads, head_size] (NHD)
    kc = k_q.reshape(num_blocks, block_size, num_kv_heads, head_size).contiguous()
    vc = v_q.reshape(num_blocks, block_size, num_kv_heads, head_size).contiguous()
    ksc = k_sc.reshape(num_blocks, block_size, num_kv_heads).contiguous()
    vsc = v_sc.reshape(num_blocks, block_size, num_kv_heads).contiguous()

    # logical packed cache: (B, H, N, 2*hs) -> the impl passes split views
    out = torch.empty_like(q)
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    seqused = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    bt = torch.arange(num_blocks, dtype=torch.int32, device="cuda")[None]

    unified_attention(
        q=q, k=kc, v=vc, out=out,
        cu_seqlens_q=cu, max_seqlen_q=seq_len,
        seqused_k=seqused, max_seqlen_k=seq_len,
        causal=False,
        block_table=bt,
        softmax_scale=head_size ** -0.5,
        window_size=(-1, -1), softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        k_scale_cache=ksc.contiguous(), v_scale_cache=vsc.contiguous(),
    )
    # reference: dequantized non-causal attention
    k_full = (k_q.float() * k_sc.unsqueeze(-1))
    v_full = (v_q.float() * v_sc.unsqueeze(-1))
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(1),  # [1, H?] no — build [1, num_heads, L, D]
        q=None, key=None, attn_mask=None,
    ) if False else torch.zeros(1)  # placeholder replaced below
    # proper reference
    qs = q.permute(1, 0, 2).float()  # [H, L, D]
    ks = k_full.permute(1, 0, 2)
    vs = v_full.permute(1, 0, 2).repeat_interleave(num_heads, dim=0)
    ks_r = ks.repeat_interleave(num_heads, dim=0)
    ref = torch.nn.functional.scaled_dot_product_attention(qs, ks_r, vs)
    ref = ref.permute(1, 0, 2).to(torch.float16)
    max_diff = (out - ref).abs().max().item()
    print(f"noncausal_int8_pth: max_diff={max_diff:.4e}")
    return max_diff < 5e-2


def main():
    torch.manual_seed(0)
    all_ok = True
    # Prefill (2D kernel path)
    all_ok &= run_one(
        num_seqs=2,
        seq_lens=torch.tensor([127, 129], dtype=torch.int32, device="cuda"),
        num_heads=16,
        num_kv_heads=4,
        head_size=64,
        block_size=32,
        name="prefill_2d",
    )
    # Decode (3D kernel path)
    all_ok &= run_decode(
        num_seqs=4,
        seq_lens=torch.tensor([1024, 2048, 4096, 8192], dtype=torch.int32, device="cuda"),
        num_heads=16,
        num_kv_heads=4,
        head_size=64,
        block_size=32,
        name="decode_3d",
    )
    # Mixed prefill+decode
    all_ok &= run_one(
        num_seqs=2,
        seq_lens=torch.tensor([1, 64], dtype=torch.int32, device="cuda"),
        num_heads=16,
        num_kv_heads=4,
        head_size=64,
        block_size=32,
        name="mixed",
    )
    all_ok &= run_noncausal()
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
