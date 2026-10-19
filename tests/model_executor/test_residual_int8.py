# SPDX-License-Identifier: Apache-2.0
"""Residual storage, arithmetic, model boundaries and graph replay gates."""

import os
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.residual_int8 import (
    ResidualInt8,
    add_residual,
    final_norm,
    materialize,
    residual_norm,
)

DEVICE = os.environ.get("RESIDUAL_TEST_DEVICE", "cpu")


@pytest.fixture(autouse=True)
def enable(monkeypatch):
    monkeypatch.setenv("VLLM_GFX908_RESIDUAL_INT8", "1")


def _decode_fp32(residual):
    return (
        residual.codes.float().unflatten(-1, (-1, 128)) * residual.scales.unsqueeze(-1)
    ).flatten(-2)


@pytest.mark.parametrize("shape", [(1, 4096), (64, 4096), (256, 4096), (2, 56, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_residual_sequence(shape, dtype):
    torch.manual_seed(47)
    norm = torch.nn.RMSNorm(shape[-1], eps=1e-6, device=DEVICE, dtype=dtype)
    residual = None
    for step in range(3):
        branch = torch.randn(shape, dtype=dtype, device=DEVICE) * (step + 1)
        expected_sum = branch.float().clone()
        if residual is not None:
            expected_sum += _decode_fp32(residual)
        expected_sum = expected_sum.to(dtype).float()
        expected_norm = expected_sum * torch.rsqrt(
            expected_sum.square().mean(-1, keepdim=True) + 1e-6
        )
        normal, residual, nq, ns = residual_norm(
            norm, branch, residual, emit_quant=True
        )
        assert isinstance(residual, ResidualInt8)
        assert residual.codes.dtype == torch.int8
        assert residual.scales.dtype == torch.float32
        assert residual.codes.shape == branch.shape
        assert residual.scales.shape == (*shape[:-1], shape[-1] // 128)
        storage = sum(t.numel() * t.element_size() for t in residual[:2])
        assert storage == branch.numel() * (1 + 4 / 128)
        errors = (_decode_fp32(residual) - expected_sum).abs().unflatten(-1, (-1, 128))
        assert torch.all(errors.amax(-1) <= residual.scales * 0.501 + 1e-6)
        torch.testing.assert_close(
            normal, expected_norm.to(dtype), atol=0.016, rtol=0.008
        )
        ndecoded = (nq.float().unflatten(-1, (-1, 128)) * ns.unsqueeze(-1)).flatten(-2)
        assert torch.all(
            (ndecoded - normal.float()).abs().unflatten(-1, (-1, 128)).amax(-1)
            <= ns * 0.501 + 1e-6
        )
        torch.testing.assert_close(
            materialize(residual), _decode_fp32(residual).to(dtype)
        )
    torch.testing.assert_close(
        final_norm(norm, branch, residual), residual_norm(norm, branch, residual)[0]
    )


def test_zero_and_outlier_groups():
    x = torch.zeros((2, 4096), dtype=torch.bfloat16, device=DEVICE)
    x[0, 0] = 1024
    x[0, 128:256] = 0.125
    residual = add_residual(x)
    assert torch.isfinite(residual.scales).all()
    assert torch.equal(materialize(residual), x)
    assert (residual.codes[1] == 0).all()


def test_wrong_carrier_rejected():
    x = torch.zeros((2, 4096), dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(ValueError, match="explicit GS128"):
        add_residual(x, x)
    malformed = ResidualInt8(x.to(torch.int8), x[:, :1].float(), x.dtype)
    with pytest.raises(ValueError, match="codes/scales"):
        materialize(malformed)


def test_native_norm_preserves_unrounded_sum():

    torch.manual_seed(73)
    norm = SimpleNamespace(
        variance_epsilon=1e-6,
        weight=torch.randn(4096, device=DEVICE, dtype=torch.bfloat16),
    )
    x = torch.randn(64, 4096, device=DEVICE, dtype=torch.bfloat16)
    residual = add_residual(torch.randn_like(x))
    summed = x.float() + _decode_fp32(residual)
    normalized = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
    expected = normalized.to(norm.weight.dtype) * norm.weight
    actual, _ = residual_norm(norm, x, residual)
    torch.testing.assert_close(actual, expected, atol=0.032, rtol=0.008)


def test_bolt_and_lilicorr_boundaries():
    from vllm.model_executor.models.lilicorr import LiLiCorrLayer
    from vllm.model_executor.models.nemotron_h import NemotronHMambaDecoderLayer

    x = torch.randn(8, 4096, dtype=torch.bfloat16, device=DEVICE)
    norm = torch.nn.RMSNorm(4096, eps=1e-6, device=DEVICE, dtype=x.dtype)
    layer = SimpleNamespace(
        norm=norm, mixer=lambda value: add_residual(materialize(value) * 0.25)
    )
    out, residual = NemotronHMambaDecoderLayer.forward(layer, x, None)
    assert isinstance(residual, ResidualInt8)
    out, residual = NemotronHMambaDecoderLayer.forward(layer, out, residual)
    assert torch.isfinite(final_norm(norm, out, residual)).all()
    # The two lattice layers must carry quantized skips across layer boundaries.
    norm = torch.nn.RMSNorm(1024, eps=1e-6, device=DEVICE, dtype=x.dtype)
    layer = SimpleNamespace(
        attn_norm=norm,
        mlp_norm=norm,
        attn=lambda value, bias: add_residual(materialize(value) * 0.25),
        mlp=lambda value: add_residual(materialize(value) * 0.5),
    )
    value = x[:, :1024].contiguous().view(2, 4, 1024)
    for _ in range(2):
        value = LiLiCorrLayer.forward(layer, value, None)
        assert isinstance(value, ResidualInt8)
    assert torch.isfinite(final_norm(norm, value)).all()


@pytest.mark.skipif(DEVICE != "cuda", reason="GPU graph replay gate")
def test_graph_replay_consumes_new_residual():
    norm = torch.nn.RMSNorm(4096, eps=1e-6, device=DEVICE, dtype=torch.bfloat16)
    x = torch.randn(64, 4096, dtype=torch.bfloat16, device=DEVICE)
    with torch.inference_mode():
        for _ in range(3):
            normal, residual = residual_norm(norm, x)
            residual_norm(norm, normal, residual)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            normal, residual = residual_norm(norm, x)
            actual, result = residual_norm(norm, normal, residual)
        x.copy_(torch.randn_like(x))
        graph.replay()
        normal, residual = residual_norm(norm, x)
        expected, expected_residual = residual_norm(norm, normal, residual)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            result.codes, expected_residual.codes, atol=0, rtol=0
        )


@pytest.mark.parametrize("shape", [(1, 4096), (256, 4096), (2, 8, 1024)])
def test_direct_normalized_payload(shape):
    torch.manual_seed(19)
    norm = torch.nn.RMSNorm(shape[-1], eps=1e-6, device=DEVICE, dtype=torch.bfloat16)
    branch = add_residual(torch.randn(shape, device=DEVICE, dtype=torch.bfloat16))
    residual = add_residual(torch.randn(shape, device=DEVICE, dtype=torch.bfloat16))
    normal, expected_r, q, s = residual_norm(norm, branch, residual, emit_quant=True)
    packed, actual_r = residual_norm(norm, branch, residual, quantized=True)
    assert isinstance(packed, ResidualInt8)
    torch.testing.assert_close(packed.codes, q, atol=0, rtol=0)
    torch.testing.assert_close(packed.scales, s, atol=0, rtol=0)
    torch.testing.assert_close(actual_r.codes, expected_r.codes, atol=0, rtol=0)
    assert packed.flatten_rows()[0].data_ptr() == packed.codes.data_ptr()
