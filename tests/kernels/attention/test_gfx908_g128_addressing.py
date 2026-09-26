# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""G128 KV reads must preserve offsets beyond the int32 address range."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

pytestmark = [
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(not current_platform.is_rocm(), reason="gfx908 kernels"),
]

BLOCK_SIZE = 1664
PAGE_BYTES = BLOCK_SIZE * 520
NUM_BLOCKS = 5001
GUARD_BYTES = 1 << 32


@pytest.fixture(scope="module")
def guarded_arena():
    from vllm.platforms.rocm import on_mi100

    if not on_mi100():
        pytest.skip("gfx908 kernels")
    required = GUARD_BYTES + NUM_BLOCKS * PAGE_BYTES
    free, _ = torch.cuda.mem_get_info()
    if free < required + (1 << 30):
        pytest.skip("large-offset regression requires an idle GPU with 10 GiB free")
    # The leading guard makes a regressed, wrapped read remain within this
    # allocation. It returns known foreign content instead of faulting the GPU.
    return torch.full((required,), 5, dtype=torch.int8, device="cuda")


@pytest.mark.parametrize("block_id", [1, 2481, 2482, 4964, 5000])
@pytest.mark.parametrize("core", ["decode32", "decode64", "draft", "prefill", "packed"])
@torch.inference_mode()
def test_g128_large_kv_offsets(guarded_arena, monkeypatch, block_id, core):
    monkeypatch.setenv("VLLM_UA_3D_MAXQ", "8")
    monkeypatch.setenv("VLLM_G128_GLUON", "32" if core == "decode32" else "1")
    monkeypatch.setenv("VLLM_G128_DRAFT_GLUON", "1")
    monkeypatch.setenv("VLLM_G128_PREFILL_GLUON", "1")
    monkeypatch.setenv("VLLM_G128_PREFILL_PACKED", "1" if core == "packed" else "0")
    monkeypatch.setenv("VLLM_GFX908_ATTN_WARPS", "2")
    monkeypatch.setenv("VLLM_G128_REDUCE_GLUON", "0")

    draft = core == "draft"
    prefill = core in ("prefill", "packed")
    head_size, num_heads, num_kv_heads = (128, 8, 2) if draft else (256, 6, 1)
    query_len = 256 if prefill else 7
    groups = head_size // 128
    pad = head_size + 2 * groups
    head_stride = BLOCK_SIZE * 2 * pad
    packed = torch.as_strided(
        guarded_arena,
        (NUM_BLOCKS, num_kv_heads, BLOCK_SIZE, 2 * pad),
        (PAGE_BYTES, head_stride, 2 * pad, 1),
        storage_offset=GUARD_BYTES,
    )
    base_f16 = torch.tensor([], dtype=torch.float16, device="cuda").set_(
        guarded_arena.untyped_storage()
    )
    k_scale = torch.as_strided(
        base_f16,
        (NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, groups),
        (PAGE_BYTES // 2, pad, head_stride // 2, 1),
        storage_offset=(GUARD_BYTES + head_size) // 2,
    )
    v_scale = torch.as_strided(
        base_f16,
        k_scale.shape,
        k_scale.stride(),
        storage_offset=(GUARD_BYTES + pad + head_size) // 2,
    )
    key = packed.transpose(1, 2)[..., :head_size]
    value = packed.transpose(1, 2)[..., pad : pad + head_size]
    torch.manual_seed(42)
    key[block_id, :query_len].random_(-80, 80)
    value[block_id, :query_len].random_(-80, 80)
    k_scale[block_id, :query_len].fill_(0.01)
    v_scale[block_id, :query_len].fill_(0.02)
    query = torch.randn(
        query_len, num_heads, head_size, dtype=torch.bfloat16, device="cuda"
    )
    output = torch.empty_like(query)
    cu_query = torch.tensor([0, query_len], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([query_len], dtype=torch.int32, device="cuda")
    table = torch.tensor([[block_id]], dtype=torch.int32, device="cuda")
    segments = {}
    if not prefill:
        segments = dict(
            num_par_softmax_segments=64,
            softmax_segm_output=torch.empty(
                (query_len, num_heads, 64, head_size), device="cuda"
            ),
            softmax_segm_max=torch.empty((query_len, num_heads, 64), device="cuda"),
            softmax_segm_expsum=torch.empty((query_len, num_heads, 64), device="cuda"),
            seq_threshold_3D=256,
            max_flash_decoding_splits=64,
        )
    unified_attention(
        query,
        key,
        value,
        output,
        cu_query,
        query_len,
        seq_lens,
        query_len,
        head_size**-0.5,
        not draft,
        (2047, 2047) if draft else (-1, -1),
        table,
        0.0,
        None,
        None,
        None,
        kv_quant_mode=KVQuantMode.INT8_BLOCK_G128,
        g8_k_scale=k_scale,
        g8_v_scale=v_scale,
        **segments,
    )
    k = key[block_id, :query_len].float()
    v = value[block_id, :query_len].float()
    k *= k_scale[block_id, :query_len].float().repeat_interleave(128, dim=-1)
    v *= v_scale[block_id, :query_len].float().repeat_interleave(128, dim=-1)
    repeats = num_heads // num_kv_heads
    k = k.repeat_interleave(repeats, dim=1).transpose(0, 1)
    v = v.repeat_interleave(repeats, dim=1).transpose(0, 1)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.float().transpose(0, 1), k, v, is_causal=not draft
    ).transpose(0, 1)
    torch.testing.assert_close(output.float(), expected, atol=0.015, rtol=0.015)
