# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise Inkling's G128 KV scale placement and per-request relative bias."""

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm backend")
@pytest.mark.parametrize("grouped", [True, False])
def test_binding_draft_cache_preserves_other_groups_live_scales(grouped):
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionImpl,
    )

    # Two groups share physical storage but own different block IDs. The
    # second layer may first bind after prefill has written the first group.
    width = 144 if grouped else 132
    pool = torch.zeros(4, 2, 16, 2 * width, dtype=torch.int8)

    def bind():
        impl = object.__new__(RocmAiterUnifiedAttentionImpl)
        impl._is_int8_block = grouped
        impl._block_g = 128 if grouped else None
        impl._shuffle_kv_layout = False
        impl.head_size = 128
        impl._ensure_scale_caches(pool)
        return (
            (impl._g8_k, impl._g8_v)
            if grouped
            else (impl._k_scale_cache, impl._v_scale_cache)
        )

    k, v = bind()
    k[2].fill_(0.125)
    v[2].fill_(0.0625)
    bind()
    assert (k[2] == 0.125).all() and (v[2] == 0.0625).all()
    assert (k[0] == 1).all() and (v[0] == 1).all()


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")
@pytest.mark.parametrize(
    "is_local,skip_past_bias", [(False, False), (False, True), (True, False)]
)
def test_paged_decode_with_empty_splits_and_padding(is_local, skip_past_bias):
    from vllm.models.inkling2.amd.ops.attention import (
        gated_rel_attention,
        rel_decode_attention,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(94)
    lengths = [1489, 35]
    queries = [3, 1]
    block_size = 16
    # Each request has its own nonmonotonic block table.
    table = torch.randperm(192, device=device, generator=generator).view(2, 96).int()
    cache = torch.zeros((192, 2, block_size, 288), device=device, dtype=torch.int8)
    key, value = cache.transpose(1, 2).chunk(2, -1)
    decoded = []
    for tensor in (key, value):
        codes = torch.randint(
            -127,
            128,
            (*tensor.shape[:-1], 128),
            device=device,
            dtype=torch.int8,
            generator=generator,
        )
        scales = (
            torch.rand((*tensor.shape[:-1], 1), device=device, generator=generator)
            * 0.025
            + 0.001
        ).half()
        tensor[..., :128].copy_(codes)
        tensor[..., 128:130].copy_(scales.view(torch.int8))
        decoded.append((codes.float() * scales).half().float())

    def random_fp16(*shape):
        return torch.randn(
            shape, device=device, generator=generator, dtype=torch.float16
        )

    q = random_fp16(6, 6, 128)
    bias = random_fp16(len(q) + 128, 6, 1280)
    gate = random_fp16(6, 768)
    sinks = random_fp16(6, 6)
    output = torch.empty_like(gate)
    cu = torch.tensor([0, 3, 4], device=device, dtype=torch.int32)
    lens = torch.tensor(lengths, device=device, dtype=torch.int32)
    ref = torch.zeros_like(q)
    start = 0
    for b, (query_len, length) in enumerate(zip(queries, lengths)):
        stop = start + query_len
        k, v = (
            t[table[b].long()].reshape(-1, 2, 128)[:length].repeat_interleave(3, 1)
            for t in decoded
        )
        logits = torch.einsum("thd,lhd->thl", q[start:stop].float(), k) * 128**-0.5
        key0 = ((length + 127) // 128) * 128 - bias.shape[-1]
        first_bias_key = max(0, key0)
        logits[..., first_bias_key:] += bias[
            start:stop, :, first_bias_key - key0 : length - key0
        ].float()
        keys = torch.arange(length, device=device)
        distance = keys[-query_len:, None] - keys[None, :]
        invalid = distance < 0
        if is_local:
            invalid |= distance > 1023
        logits.masked_fill_(invalid[:, None, :], -torch.inf)
        attention = torch.einsum("thl,lhd->thd", logits.softmax(-1), v)
        ref[start:stop] = (
            attention
            * gate[start:stop].reshape_as(attention).float().sigmoid()
            * (logits.logsumexp(-1) - sinks[start:stop]).sigmoid()[..., None]
        )
        start = stop

    for splits in (1, 16):
        rel_decode_attention(
            q,
            key[..., :128],
            value[..., :128],
            bias,
            gate,
            sinks,
            output,
            block_table=table,
            cache_seqlens=lens,
            cu_seqlens_q=cu,
            max_query_len=3,
            num_actual_tokens=sum(queries),
            softmax_scale=128**-0.5,
            is_local=is_local,
            num_splits=splits,
            skip_past_bias=skip_past_bias,
        )
        torch.testing.assert_close(output.view_as(ref), ref, atol=0.003, rtol=0.005)
    for fused in (False, True):
        gated_rel_attention(
            q,
            key[..., :128],
            value[..., :128],
            bias,
            gate,
            sinks,
            output,
            block_table=table,
            cache_seqlens=lens,
            cu_seqlens_q=cu,
            max_seqlen_q=3,
            num_actual_tokens=sum(queries),
            softmax_scale=128**-0.5,
            window=1023 if is_local else -1,
            fused_epilogue=fused,
            skip_past_bias=skip_past_bias,
        )
        torch.testing.assert_close(output.view_as(ref), ref, atol=0.003, rtol=0.005)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")
@pytest.mark.parametrize("window", [0, 40])
def test_dspark_block_shares_window_across_all_queries(window):
    from vllm.models.inkling2.amd.ops.attention import block_decode_attention

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(105)
    table = torch.randperm(10, device=device, generator=gen).reshape(2, 5).int()
    lengths = torch.tensor([79, 17], device=device, dtype=torch.int32)
    cu = torch.arange(0, 17, 8, device=device, dtype=torch.int32)
    q = torch.randn(16, 3, 128, device=device, generator=gen, dtype=torch.bfloat16)
    buffers, decoded = [], []
    for _ in range(2):
        buf = torch.zeros(10, 16, 1, 144, device=device, dtype=torch.int8)
        codes = torch.randint(
            -127, 128, (10, 16, 1, 128), device=device, generator=gen, dtype=torch.int8
        )
        scales = (
            torch.rand(10, 16, 1, 1, device=device, generator=gen) * 0.02 + 0.005
        ).half()
        buf[..., :128] = codes
        buf[..., 128:130] = scales.view(torch.int8)
        buffers.append(buf)
        decoded.append((codes.float() * scales).bfloat16())
    out = torch.empty_like(q).view(16, -1)
    block_decode_attention(
        q,
        *buffers,
        out,
        block_table=table,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
        block=8,
        window=window,
        num_splits=5,
    )
    for b, n in enumerate((79, 17)):
        k, v = [
            t[table[b].long()].reshape(-1, 128)[max(0, n - window) if window else 0 : n]
            for t in decoded
        ]
        segment = slice(b * 8, (b + 1) * 8)
        probs = (q[segment].float() @ k.float().T * 128**-0.5).softmax(-1)
        expected = probs @ v.float()
        torch.testing.assert_close(
            out[segment].view_as(expected).float(), expected, rtol=0.035, atol=0.018
        )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")
@pytest.mark.parametrize("window", [-1, 1023])
def test_mixed_mi100_tiles_preserve_gated_attention(window):
    from vllm.models.inkling2.amd.ops.attention import gated_rel_attention
    from vllm.platforms.rocm import on_gfx908

    if not on_gfx908():
        pytest.skip("MI100 tile specialization")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(9417)
    tokens = 73  # one 8-token verify and one 65-token prefill chunk
    physical = 80
    pages = 146
    table = torch.randperm(pages * 2, device=device, generator=generator)
    table = table.view(2, pages).int()
    cache = torch.randint(
        -95,
        96,
        (pages * 2, 2, 16, 288),
        device=device,
        dtype=torch.int8,
        generator=generator,
    )
    key, value = cache.transpose(1, 2).chunk(2, dim=-1)
    for kv in (key, value):
        scale = torch.rand((*kv.shape[:-1], 1), device=device, generator=generator)
        kv[..., 128:130].copy_((scale * 0.012 + 0.001).half().view(torch.int8))

    def random(*shape):
        return torch.randn(
            *shape, device=device, generator=generator, dtype=torch.float16
        )

    q = random(physical, 6, 128)
    bias = random(physical + 128, 6, 1280)
    gate = random(physical, 768)
    sinks = random(physical, 6)
    cu = torch.tensor([0, 8, tokens], device=device, dtype=torch.int32)
    lengths = torch.tensor([2113, 2305], device=device, dtype=torch.int32)

    def run(optimized):
        result = torch.empty_like(gate)
        gated_rel_attention(
            q,
            key[..., :128],
            value[..., :128],
            bias,
            gate,
            sinks,
            result,
            block_table=table,
            cache_seqlens=lengths,
            cu_seqlens_q=cu,
            max_seqlen_q=65,
            num_actual_tokens=tokens,
            softmax_scale=128**-0.5,
            window=window,
            skip_past_bias=optimized,
            mi100_tiles=optimized,
        )
        return result

    assert torch.equal(run(False).view(torch.int16), run(True).view(torch.int16))
