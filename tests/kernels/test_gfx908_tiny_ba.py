# SPDX-License-Identifier: Apache-2.0
"""The GDN tiny projection must survive large-shape dynamic tracing."""

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.gfx908_tiny_ba import (
    tiny_ba_dispatch,
    tiny_ba_projection,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="ROCm GPU required",
)


def test_model_import_registers_op_before_cached_graph_load():
    import os
    import subprocess
    import sys

    env = dict(os.environ, VLLM_ROCM_USE_AITER="1")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; "
            "import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as m; "
            "assert m.GDN_AITER_TRITON_AVAILABLE; "
            "assert hasattr(torch.ops.vllm, 'gfx908_tiny_ba')",
        ],
        env=env,
        check=True,
        timeout=60,
    )


def test_large_trace_preserves_small_runtime_dispatch(monkeypatch):
    import vllm.model_executor.layers.mamba.gdn.gfx908_tiny_ba as module

    torch.manual_seed(113)
    weight = torch.randn((24, 5120), dtype=torch.bfloat16, device="cuda")
    seen = []
    original = module.tiny_ba_projection

    def record(x, w):
        seen.append(x.shape[0])
        return original(x, w)

    monkeypatch.setattr(module, "tiny_ba_projection", record)
    compiled = torch.compile(tiny_ba_dispatch, dynamic=True, fullgraph=True)
    # First trace is a large prefill, reproducing the original lost branch.
    for rows in (1664, 42, 7, 64, 65, 1, 0):
        x = torch.randn((rows, 5120), dtype=torch.bfloat16, device="cuda")
        actual = compiled(x, weight)
        expected = (
            original(x, weight)
            if 0 < rows <= 64
            else torch.nn.functional.linear(x, weight)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert 42 in seen and 7 in seen and 64 in seen and 1 in seen
    assert 1664 not in seen and 65 not in seen and 0 not in seen


@pytest.mark.parametrize("rows", [1, 7, 42, 64])
def test_tiny_projection_fp32_reference(rows):
    torch.manual_seed(117 + rows)
    x = torch.randn((rows, 5120), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((24, 5120), dtype=torch.bfloat16, device="cuda")
    actual = tiny_ba_projection(x, w).float()
    reference = torch.nn.functional.linear(x.float(), w.float())
    # One BF16 rounding step plus cancellation slack; no fp16 accumulation.
    torch.testing.assert_close(actual, reference, rtol=0.004, atol=0.002)


def test_graph_replay_uses_same_projection():
    x = torch.randn((42, 5120), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((24, 5120), dtype=torch.bfloat16, device="cuda")
    tiny_ba_dispatch(x, w)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = tiny_ba_dispatch(x, w)
    graph.replay()
    torch.testing.assert_close(output, tiny_ba_projection(x, w), rtol=0, atol=0)
