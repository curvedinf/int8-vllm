# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from pathlib import Path

import torch

import vllm.envs as envs
from vllm.model_executor.layers.decode_int8 import (
    DecodeInt8Tensor,
    _gemm_fake,
    _norm_fake,
    _projection_fake,
)


def test_explicit_quantized_carrier_and_shapes():
    for rows in (1, 7, 42, 48, 64):
        x = torch.empty((rows, 4352), device="meta", dtype=torch.bfloat16)
        w = torch.empty((5120, 4352), device="meta", dtype=torch.int8)
        ws = torch.empty((5120, 34), device="meta", dtype=torch.float32)
        q, s = _gemm_fake(x, w, ws)
        intermediate = DecodeInt8Tensor(q, s)
        assert intermediate.shape == (rows, 5120)
        assert q.dtype == torch.int8 and s.dtype == torch.float32
        assert s.shape == (rows, 40)
        residual = torch.empty_like(q, dtype=torch.bfloat16)
        weight = torch.empty(5120, device="meta", dtype=torch.bfloat16)
        out, res = _norm_fake(q, s, residual, weight, 1e-6,
                              True, None, None, 0, 0)
        assert out.shape == res.shape == residual.shape
        assert out.dtype == res.dtype == torch.bfloat16


def test_compile_hash_separates_all_modes(monkeypatch):
    monkeypatch.setenv("VLLM_DECODE_INT8_INTERMEDIATES", "0")
    monkeypatch.setenv("VLLM_DECODE_INT8_RETAIN_GS128", "0")
    original = envs.compile_factors()
    monkeypatch.setenv("VLLM_DECODE_INT8_RETAIN_GS128", "1")
    reserved = envs.compile_factors()
    monkeypatch.setenv("VLLM_DECODE_INT8_INTERMEDIATES", "1")
    enabled = envs.compile_factors()
    assert original != reserved != enabled


def test_projection_carrier_contract_is_shape_independent():
    w = torch.empty((5120, 4352), device="meta", dtype=torch.int8)
    ws = torch.empty((5120, 34), device="meta")
    for rows in (1, 48, 64, 128, 2048):
        x = torch.empty((rows, 4352), device="meta", dtype=torch.bfloat16)
        q, s, dense = _projection_fake(x, w, ws, w, ws)
        assert q.shape == dense.shape == (rows, 5120)
        assert s.shape == (rows, 40)
        assert q.dtype == torch.int8 and dense.dtype == torch.bfloat16


def test_draft_conv_carrier_does_not_materialize_bf16():
    q = torch.empty((42, 5120), device="meta", dtype=torch.int8)
    s = torch.empty((42, 40), device="meta", dtype=torch.float32)
    x = DecodeInt8Tensor(q, s)
    delta = torch.empty((42, 2, 320), device="meta", dtype=torch.bfloat16)
    base = torch.empty((2, 5120), device="meta", dtype=torch.bfloat16)
    conv = x._replace(conv_delta=delta, conv_base=base,
                      conv_block_size=7, conv_group_size=16)
    assert conv.codes is q and conv.scales is s
    assert conv.conv_delta is delta and conv.conv_base is base


def test_draft_conv_strided_coefficient_view(monkeypatch):
    from vllm.model_executor.models.qwen3_dflash2 import DFlashGroupedConv

    monkeypatch.setenv("VLLM_DECODE_INT8_INTERMEDIATES", "1")
    conv = SimpleNamespace(
        base_kernel=torch.empty(2, 2, 5120), taps=2,
        block_size=7, group_size=16,
    )
    coefficients = torch.empty(42, 2, 2, 320)[:, 1]
    assert not coefficients.is_contiguous()
    x = DecodeInt8Tensor(torch.empty(42, 5120, dtype=torch.int8),
                        torch.empty(42, 40))
    result = DFlashGroupedConv._convolve(conv, x, coefficients, 1)
    assert result.codes is x.codes and result.scales is x.scales
    assert result.conv_delta.is_contiguous()
    assert result.conv_base.is_contiguous()


def test_decode_probe_uses_supported_mask_and_second_raw_row(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import eval_decode_int8

    request = {}
    second = {"token_id:42": -0.1, "token_id:17": -5.0}

    def completion(body):
        request.update(body)
        return {"token_ids": [17, 17], "logprobs": {
            "top_logprobs": [{"token_id:17": -4.0}, second]}}

    monkeypatch.setattr(eval_decode_int8, "post", completion)
    result = eval_decode_int8.probe(("test", [1, 2, 3], 17))
    assert request["allowed_token_ids"] == [17]
    assert "logit_bias" not in request
    assert result["raw"] is second
    assert request["max_tokens"] == request["min_tokens"] == 2
