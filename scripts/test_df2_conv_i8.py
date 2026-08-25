#!/usr/bin/env python3
"""CPU unit checks for E4: DFlash2 base_kernel int8 dequant-on-add path."""

import os
import tempfile

import torch

from vllm.model_executor.models.qwen3_dflash2 import DFlashGroupedConv

H, TAPS, GSIZE, BLOCK = 64, 2, 16, 17  # BLOCK non-pow2 -> modulo branch
ENV = "VLLM_GFX908_DF2_CONV_I8"


def init_single_node_tp() -> None:
    """world-1 gloo TP so ReplicatedLinear can be built on CPU."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    fd, store = tempfile.mkstemp()
    os.close(fd)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{store}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
    os.unlink(store)


def make_module() -> DFlashGroupedConv:
    torch.manual_seed(0)
    m = DFlashGroupedConv(
        hidden_size=H,
        taps=TAPS,
        group_size=GSIZE,
        block_size=BLOCK,
        params_dtype=torch.bfloat16,
        prefix="test_conv",
    )
    m.base_kernel.data = torch.randn_like(m.base_kernel) * 0.05
    return m


def quant_planes(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Same per-[side,tap] symmetric int8 math as scripts/quant_df2_conv_i8.py."""
    sides, taps, hidden = w.shape
    planes = w.float().reshape(sides * taps, hidden)
    amax = planes.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scales = (amax / 127.0).to(torch.float32).reshape(sides, taps)
    q = (planes / amax * 127.0).round().clamp(-127, 127).to(torch.int8)
    return q.reshape(sides, taps, hidden), scales


def test_convolve_matches_bf16():
    os.environ.pop(ENV, None)
    m = make_module()
    h = (torch.randn(BLOCK, H) * 0.1).to(torch.bfloat16)
    delta = (torch.randn(BLOCK, TAPS, m.num_groups) * 0.02).to(torch.bfloat16)
    ref = [m._convolve(h, delta, side).float() for side in (0, 1)]

    m._bk_q, m._bk_s = quant_planes(m.base_kernel.data)
    for side in (0, 1):
        out = m._convolve(h, delta, side).float()
        rel = ((out - ref[side]).norm() / ref[side].norm()).item()
        print(f"  side {side}: output rel-L2 vs bf16 path {rel * 100:.4f}%")
        assert rel < 0.01, rel
    print("PASS convolve i8 vs bf16 within quant error")


def test_load_weights_flag_on():
    os.environ[ENV] = "1"
    m = make_module()
    bf16 = m.base_kernel.data.clone()
    q, scales = quant_planes(bf16)
    w_proj = torch.randn_like(m.kernel_projection.weight)
    weights = [
        ("base_kernel", bf16),
        ("base_kernel.i8", q),
        ("base_kernel.scale", scales),
        ("kernel_projection.weight", w_proj),
    ]
    loaded = m.load_weights(iter(weights))

    assert m._bk_q.dtype == torch.int8 and m._bk_s.dtype == torch.float32
    assert torch.equal(m._bk_q, q) and torch.equal(m._bk_s, scales)
    # bf16 base never materialized in this mode
    assert float(m.base_kernel.data.abs().sum()) == 0.0
    # everything else identical: projection loaded, names consumed
    assert torch.equal(m.kernel_projection.weight.data, w_proj)
    assert {"base_kernel.i8", "base_kernel.scale"} <= loaded
    print("PASS load_weights flag ON: int8 stashed, bf16 skipped, proj loaded")


def test_load_weights_flag_off():
    os.environ[ENV] = "0"
    m = make_module()
    bf16 = m.base_kernel.data.clone()
    q, scales = quant_planes(bf16)
    w_proj = torch.randn_like(m.kernel_projection.weight)
    weights = [
        ("base_kernel", bf16),
        ("base_kernel.i8", q),
        ("base_kernel.scale", scales),
        ("kernel_projection.weight", w_proj),
    ]
    loaded = m.load_weights(iter(weights))

    assert getattr(m, "_bk_q", None) is None
    assert torch.equal(m.base_kernel.data, bf16)
    assert torch.equal(m.kernel_projection.weight.data, w_proj)
    assert {"base_kernel.i8", "base_kernel.scale"} <= loaded
    print("PASS load_weights flag OFF: bf16 authoritative, i8 dropped")


def test_load_weights_flag_on_no_i8():
    os.environ[ENV] = "1"
    m = make_module()
    bf16 = m.base_kernel.data.clone()
    loaded = m.load_weights(iter([("base_kernel", bf16)]))
    assert getattr(m, "_bk_q", None) is None
    assert torch.equal(m.base_kernel.data, bf16)
    assert loaded == {"base_kernel"}
    print("PASS flag ON without .i8 in checkpoint: plain bf16 load")


if __name__ == "__main__":
    init_single_node_tp()
    test_convolve_matches_bf16()
    test_load_weights_flag_on()
    test_load_weights_flag_off()
    test_load_weights_flag_on_no_i8()
    print("4/4 PASS")
