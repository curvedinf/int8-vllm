# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise Inkling's G128 KV scale placement and per-request relative bias."""

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")
@pytest.mark.parametrize("is_local", [False, True])
def test_paged_decode_with_empty_splits_and_padding(is_local):
    from vllm.models.inkling2.amd.ops.attention import (
        gated_rel_attention,
        rel_decode_attention,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(94)
    lengths = [1105, 35]
    queries = [3, 1]
    block_size = 16
    # Each request has its own nonmonotonic block table.
    table = torch.randperm(140, device=device, generator=generator).view(2, 70).int()
    cache = torch.zeros((140, 2, block_size, 288), device=device, dtype=torch.int8)
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
        logits += bias[start:stop, :, -key0 : length - key0].float()
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
        )
        torch.testing.assert_close(output.view_as(ref), ref, atol=0.003, rtol=0.005)
