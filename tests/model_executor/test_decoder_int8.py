# SPDX-License-Identifier: Apache-2.0
"""Direct producer epilogues and GS128 handoff gates on idle gfx908 GPUs."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.decoder_int8 import (
    gated_norm,
    grouped_conv,
    projection,
)
from vllm.model_executor.layers.residual_int8 import ResidualInt8, materialize

pytestmark = pytest.mark.skipif(
    os.environ.get("RESIDUAL_TEST_DEVICE") != "cuda", reason="explicit idle GPU gate"
)


def _pack(x):
    from aiter.ops.triton.gemm.basic.gemm_packed_g128 import quantize

    q, s = quantize(x.contiguous())
    return ResidualInt8(q, s, x.dtype)


def _decoded(q, s):
    return q.float() * s.repeat_interleave(128, -1)[..., : q.shape[-1]]


def _check(value, expected, tolerance=1e-4):
    actual = _decoded(value.codes, value.scales)
    bounds = value.scales.repeat_interleave(128, -1)[..., : actual.shape[-1]] * 0.501
    assert torch.isfinite(value.scales).all()
    assert torch.all(
        (actual - expected).abs() <= bounds + tolerance * expected.abs().clamp_min(1)
    )


@pytest.mark.parametrize(
    "m,n,k", [(1, 128, 128), (8, 1344, 4096), (64, 4096, 1344), (256, 256, 1024)]
)
@pytest.mark.parametrize("relu2", [False, True])
def test_direct_gemm(m, n, k, relu2):
    torch.manual_seed(11)
    x = _pack(torch.randn(m, k, device="cuda", dtype=torch.bfloat16))
    w = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    ws = torch.rand((1, n, (k + 127) // 128), device="cuda") * 0.001 + 0.0001
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    layer = SimpleNamespace(weight=w, weight_gs128=ws, bias=bias, tp_size=1)
    actual = projection(layer, x, relu2=relu2)
    expected = torch.zeros((m, n), device="cuda")
    for group in range((k + 127) // 128):
        sl = slice(group * 128, min(k, (group + 1) * 128))
        expected += (
            (x.codes[:, sl].float() @ w[:, sl].float().T)
            * x.scales[:, group, None]
            * ws[0, None, :, group]
        )
    expected += bias.float()
    if relu2:
        expected = expected.relu().square()
    _check(actual, expected)
    assert actual.codes.dtype == torch.int8


def test_reference_attention_projection():
    torch.manual_seed(17)
    x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, 1024, device="cuda", dtype=x.dtype) * 0.01
    layer = SimpleNamespace(weight=w, bias=None, tp_size=1)
    actual = projection(layer, x, attention=True)
    expected = F.linear(x, w).float()
    _check(actual, expected, tolerance=0.008)
    with pytest.raises(ValueError, match="resident GS128"):
        projection(layer, _pack(x))


def test_conv_and_gated_norm():
    from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv

    x = _pack(torch.randn(16, 4096, device="cuda", dtype=torch.bfloat16))
    base = torch.randn(2, 4096, device="cuda", dtype=torch.bfloat16) * 0.1
    # Real prepare() selects one of two sides, leaving a strided coefficient view.
    delta = torch.randn(16, 2, 2, 256, device="cuda", dtype=torch.bfloat16)[:, 0]
    expected = _grouped_conv(materialize(x), delta, base, 8, 256, 16, 2)
    _check(grouped_conv(x, delta, base, 8, 16), expected.float())
    x = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(8, 6144, device="cuda", dtype=torch.bfloat16)[:, :2048]
    norm = SimpleNamespace(
        use_rms_norm=True,
        n_groups=8,
        tp_size=4,
        group_size=1024,
        variance_epsilon=1e-6,
        weight=torch.randn(2048, device="cuda", dtype=x.dtype),
    )
    v = (x.float() * F.silu(gate.float())).view(8, 2, 1024)
    v *= torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6)
    expected = v.view(8, 2048).to(x.dtype) * norm.weight
    _check(gated_norm(norm, x, gate), expected.float(), tolerance=0.008)


@pytest.mark.parametrize("m,width", [(8, 672), (256, 128)])
def test_expert_direct_up_and_down(m, width):
    from aiter.ops.triton.gemm.basic.gemm_packed_g128 import make_grid, packed_experts

    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    torch.manual_seed(29)
    experts, hidden, topk = 4, 1024, 2
    up = torch.randint(0, 256, (experts, width, hidden // 256, 66), dtype=torch.uint8)
    down = torch.randint(
        0, 256, (experts, hidden, (width + 255) // 256, 66), dtype=torch.uint8
    )
    for weight in (up, down):
        # Valid nonzero GGML block scales, independent random payload bits.
        bits = torch.tensor([0.01], dtype=torch.float16).view(torch.uint8)
        weight[..., :2] = bits
    up, down = up.cuda(), down.cuda()
    value = _pack(torch.randn(m, hidden, device="cuda", dtype=torch.bfloat16))
    ids = torch.randint(0, experts, (m, topk), device="cuda", dtype=torch.int32)
    plan = moe_align_block_size(ids, 16, experts, pad_sorted_ids=True)
    grid = make_grid(value.device)
    reference_q, reference_s = packed_experts(
        value.codes, value.scales, up, grid, ids, plan=plan
    )
    q, s = packed_experts(
        value.codes, value.scales, up, grid, ids, plan=plan, direct_int8=True
    )
    torch.testing.assert_close(s, reference_s, atol=1e-5, rtol=1e-4)
    assert (q.int() - reference_q.int()).abs().max() <= 1
    expected = packed_experts(q, s, down, grid, ids, plan=plan, down=True)
    result_q, result_s = packed_experts(
        q, s, down, grid, ids, plan=plan, down=True, direct_int8=True
    )
    _check(ResidualInt8(result_q, result_s, value.compute_dtype), expected)


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("m,k,n", [(8, 1024, 512), (64, 4096, 3072)])
def test_fused_silu_projection(gated, m, k, n):
    torch.manual_seed(41)
    x = _pack(torch.randn(m, k, device="cuda", dtype=torch.bfloat16))
    width = n * (2 if gated else 1)
    w = torch.randint(-127, 128, (width, k), device="cuda", dtype=torch.int8)
    ws = torch.rand((1, width, k // 128), device="cuda") * 0.001 + 0.0001
    bias = torch.randn(width, device="cuda", dtype=torch.bfloat16)
    layer = SimpleNamespace(weight=w, weight_gs128=ws, bias=bias, tp_size=1)
    actual = projection(layer, x, activation="silu_mul" if gated else "silu")
    expected = torch.zeros((m, width), device="cuda")
    for group in range(k // 128):
        sl = slice(group * 128, (group + 1) * 128)
        expected += (
            (x.codes[:, sl].float() @ w[:, sl].float().T)
            * x.scales[:, group, None]
            * ws[0, None, :, group]
        )
    expected = (expected + bias.float()).bfloat16().float()
    expected = F.silu(expected[:, :n]) * expected[:, n:] if gated else F.silu(expected)
    _check(actual, expected.bfloat16().float(), tolerance=0.008)
