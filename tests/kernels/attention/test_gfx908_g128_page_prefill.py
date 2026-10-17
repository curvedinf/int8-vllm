# SPDX-License-Identifier: Apache-2.0
"""Dedicated prefill must preserve the serial recurrence and generic fallback."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops import gfx908_g128_page_prefill as page
from vllm.v1.attention.ops import triton_unified_attention as ua
from vllm.v1.kv_cache_interface import KVQuantMode

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="gfx908 G128 page prefill"
)


def _inputs(lengths, prefixes, *, block_size=4256):
    torch.manual_seed(702)
    lens = [a + b for a, b in zip(lengths, prefixes)]
    width = (max(lens) + block_size - 1) // block_size + 1
    blocks = len(lengths) * width
    table = torch.randperm(blocks, device="cuda").to(torch.int32)
    table = table.reshape(len(lengths), width)
    q = torch.randn((sum(lengths), 12, 128), device="cuda", dtype=torch.bfloat16)[:, :8]
    k = torch.randint(
        -128, 128, (blocks, block_size, 1, 128), device="cuda", dtype=torch.int8
    )
    v = torch.randint(-128, 128, k.shape, device="cuda", dtype=torch.int8)
    sk = (torch.rand((blocks, block_size, 1, 1), device="cuda") * 0.02 + 0.001).half()
    sv = (torch.rand_like(sk) * 0.02 + 0.001).half()
    starts = torch.tensor([0] + lengths, device="cuda", dtype=torch.int32)
    starts = starts.cumsum(0).to(torch.int32)
    return dict(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=starts,
        max_seqlen_q=max(lengths),
        seqused_k=torch.tensor(lens, device="cuda", dtype=torch.int32),
        max_seqlen_k=max(lens),
        softmax_scale=128**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        kv_quant_mode=KVQuantMode.INT8_BLOCK_G128,
        g8_k_scale=sk,
        g8_v_scale=sv,
    )


def _pair(monkeypatch, data):
    from vllm.platforms import rocm

    if not rocm.on_gfx908():
        pytest.skip("requires gfx908")
    calls = []
    launch = page.page_prefill

    def record(*args, **kwargs):
        calls.append(kwargs)
        return launch(*args, **kwargs)

    monkeypatch.setattr(page, "page_prefill", record)
    reference = torch.empty_like(data["q"])
    with monkeypatch.context() as context:
        context.setattr(rocm, "on_gfx908", lambda: False)
        ua.unified_attention(out=reference, **data)
    out = torch.empty(
        (data["q"].shape[0], 9, 128), device="cuda", dtype=reference.dtype
    )[:, :8]
    ua.unified_attention(out=out, **data)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), reference.view(torch.int16))
    return out, reference, calls


@pytest.mark.parametrize(
    "lengths,prefixes",
    [
        ([65], [0]),
        ([129], [4255]),
        ([129], [4256]),
        ([129], [4257]),
        ([65], [8511]),
        ([65], [8512]),
        ([65], [8513]),
        ([7, 1, 4096], [4000, 4256, 0]),
        ([0, 65, 0, 7], [4256, 11, 0, 8512]),
        ([4096], [0]),
        ([4160], [0]),
        ([4096] * 8, [0] * 8),
    ],
)
def test_deployed_pages_match_generic_bits_and_preserve_sources(
    monkeypatch, lengths, prefixes
):
    data = _inputs(lengths, prefixes)
    saved = {
        name: value.clone()
        for name, value in data.items()
        if isinstance(value, torch.Tensor)
    }
    _, _, calls = _pair(monkeypatch, data)
    assert calls == [{"block_m": 64}]
    for name, value in saved.items():
        assert torch.equal(data[name], value), name


@pytest.mark.parametrize(
    "kind",
    [
        "decode",
        "verify",
        "small_prefill",
        "page64",
        "causal_window",
        "noncausal_window",
        "noncausal",
        "sinks",
        "alibi",
        "softcap",
        "scale_dtype",
        "explicit_tile",
        "explicit_queries",
        "explicit_warps",
        "explicit_stages",
        "per_seq_causal",
    ],
)
def test_unsupported_cases_keep_generic_dispatch(monkeypatch, kind):
    lengths = {"decode": [1] * 8, "verify": [7] * 8, "small_prefill": [64]}.get(
        kind, [129, 7]
    )
    data = _inputs(
        lengths, [192] * len(lengths), block_size=64 if kind == "page64" else 4256
    )
    if kind in ("causal_window", "noncausal_window"):
        data["window_size"] = (96, -1)
    if kind in ("noncausal", "noncausal_window"):
        data["causal"] = False
    if kind == "sinks":
        data["sinks"] = torch.randn(8, device="cuda")
    if kind == "alibi":
        data["alibi_slopes"] = torch.rand(8, device="cuda")
    if kind == "softcap":
        data["softcap"] = 30.0
    if kind == "scale_dtype":
        data["g8_k_scale"] = data["g8_k_scale"].float()
    if kind == "per_seq_causal":
        data["causal"] = torch.tensor([True, False], device="cuda")
    for selected, name, value in (
        ("explicit_tile", "VLLM_UA_PREFILL_TILE", "32"),
        ("explicit_queries", "VLLM_UA_PREFILL_BLOCKM", "16"),
        ("explicit_warps", "VLLM_UA_PREFILL_WARPS", "4"),
        ("explicit_stages", "VLLM_UA_PREFILL_STAGES", "1"),
    ):
        if kind == selected:
            monkeypatch.setenv(name, value)
    _, _, calls = _pair(monkeypatch, data)
    assert not calls


def test_graph_replay_keeps_dedicated_prefill_bits(monkeypatch):
    data = _inputs([129, 7], [4255, 8513])
    out, reference, calls = _pair(monkeypatch, data)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ua.unified_attention(out=out, **data)
    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), reference.view(torch.int16))
    assert len(calls) == 2
