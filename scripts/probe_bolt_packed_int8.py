#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Bounded packed-weight gate; does not establish serving fidelity.

Run with the repository ROCm environment and PYTHONPATH including the repo.
Pass --checkpoint to test official expert tensors in addition to synthetic
blocks. Full decoded matrices are constructed only by this reference probe.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vllm.model_executor.layers.quantization.utils.iq2_xxs_grid import IQ2_XXS_GRID
from vllm.model_executor.layers.quantization.utils.packed_g128_int8 import (
    make_grid,
    packed_gemm,
    quantize,
)


def decode_reference(packed):
    """CPU GGML-format decoder, vectorized independently of the GPU layout."""
    raw = packed.cpu().numpy().copy()
    scale = raw[..., :2].copy().view("<f2").astype(np.float32)
    data = raw[..., 2:].reshape(*raw.shape[:2], 8, 8)
    bits = data[..., 4:].copy().view("<u4")[..., 0]
    grid = np.array(IQ2_XXS_GRID, dtype="<u8").view(np.uint8).reshape(256, 8)
    out = np.empty((*raw.shape[:2], 8, 4, 8), dtype=np.float32)
    for lane in range(4):
        sign7 = (bits >> (7 * lane)) & 127
        parity = np.zeros_like(sign7)
        for bit in range(7):
            parity ^= (sign7 >> bit) & 1
        signs = sign7 | (parity << 7)
        sign = 1 - 2 * ((signs[..., None] >> np.arange(8)) & 1).astype(np.float32)
        out[..., lane, :] = (
            scale[..., None]
            * (0.5 + (bits >> 28)[..., None])
            * 0.25
            * grid[data[..., lane]]
            * sign
        )
    return torch.from_numpy(out.reshape(raw.shape[0], -1))


def group_quant(x):
    width = x.shape[-1]
    # Use IEEE division rather than torch's constant-divisor reciprocal
    # multiply: one scale ULP changes codes at exact half-integer ties.
    v = np.pad(x.float().cpu().numpy(), ((0, 0), (0, (-width) % 128)))
    v = v.reshape(x.shape[0], -1, 128)
    s = np.maximum(np.max(np.abs(v), axis=-1) / np.float32(127), np.float32(1e-30))
    q = np.rint(v / s[..., None]).astype(np.int8)
    return (
        torch.from_numpy(q.reshape(x.shape[0], -1)[:, :width].copy()).to(x.device),
        torch.from_numpy(s).to(x.device),
    )


def restore(q, s):
    return q.float() * s.repeat_interleave(128, -1)[:, : q.shape[-1]]


def check(name, packed, m, start, length, grid, *, dense_weight=None, **codec_args):
    torch.manual_seed(42)
    x = torch.randn((m, length), device="cuda", dtype=torch.bfloat16)
    q, sx = quantize(x)
    rq, rs = group_quant(x)
    torch.testing.assert_close(q, rq, rtol=0, atol=0)
    torch.testing.assert_close(sx, rs, rtol=2e-6, atol=1e-30)
    dense = decode_reference(packed) if dense_weight is None else dense_weight
    dense = dense[:, start : start + length].cuda()
    wq, ws = group_quant(dense)
    expected = torch.zeros((m, dense.shape[0]), device="cuda")
    for g in range(sx.shape[1]):
        sl = slice(g * 128, (g + 1) * 128)
        # FP32 products of int8 operands are exact at these bounded lengths.
        part = q[:, sl].float() @ wq[:, sl].float().T
        expected += part * sx[:, g, None] * ws[None, :, g]
    packed_gpu = packed.cuda()
    actual = packed_gemm(q, sx, packed_gpu, grid, start=start, **codec_args)
    rel = ((actual - expected).norm() / expected.norm().clamp_min(1e-20)).item()
    assert rel < 2e-5, (name, "group-scaled GEMM", rel)
    outq, outs = packed_gemm(
        q, sx, packed_gpu, grid, start=start, relu2=True, quant_out=True, **codec_args
    )
    activated = expected.clamp_min(0).square()
    activation_rel = (
        (restore(outq, outs) - activated).norm() / activated.norm().clamp_min(1e-20)
    ).item()
    assert activation_rel < 0.02, (name, "quantized epilogue", activation_rel)
    reference = x.float() @ dense.T
    fidelity = ((actual - reference).norm() / reference.norm().clamp_min(1e-20)).item()
    assert torch.isfinite(actual).all()
    # Capture must retain packed storage and use the same numerical path.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = packed_gemm(q, sx, packed_gpu, grid, start=start, **codec_args)
    graph.replay()
    torch.testing.assert_close(captured, actual, rtol=0, atol=0)
    print(
        json.dumps(
            dict(
                name=name,
                m=m,
                n=dense.shape[0],
                k=length,
                start=start,
                kernel_relative_error=rel,
                added_quantization_relative_error=fidelity,
                relu2_quantization_relative_error=activation_rel,
                packed_bytes=packed.numel(),
                graph="PASS",
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    grid = make_grid("cuda")
    rng = np.random.default_rng(42)
    raw = rng.integers(0, 256, (129, 10, 66), dtype=np.uint8)
    raw[..., :2] = np.full((129, 10, 1), 0.02, dtype="<f2").view(np.uint8)
    for m, start, length in [(1, 0, 1024), (8, 576, 576), (17, 1920, 640)]:
        check("synthetic", torch.from_numpy(raw), m, start, length, grid)
    if args.checkpoint:
        from safetensors import safe_open

        index = json.loads(
            (args.checkpoint / "model.safetensors.index.json").read_text()
        )
        for suffix in ["experts.0.up_proj.weight", "experts.0.down_proj.weight"]:
            key = next(
                k
                for k in index["weight_map"]
                if ".layers.1." in k and k.endswith(suffix)
            )
            with safe_open(
                args.checkpoint / index["weight_map"][key], framework="pt"
            ) as f:
                w = f.get_tensor(key)
            check(key, w, 8, 0, w.shape[1] * 256, grid)

        def read(key):
            with safe_open(
                args.checkpoint / index["weight_map"][key], framework="pt"
            ) as f:
                return f.get_tensor(key)

        config = json.loads((args.checkpoint / "config.json").read_text())
        layers = config["quantization_config"]["quantized_layers"]
        q8name = next(k for k, v in layers.items() if v["quant_algo"] == "Q8_0")
        packed = read(q8name + ".weight")[:129].contiguous()
        raw = packed.numpy()
        scales = raw[..., :2].copy().view("<f2").astype(np.float32)
        dense = torch.from_numpy((raw[..., 2:].view(np.int8) * scales).reshape(129, -1))
        check(
            q8name, packed, 8, 0, dense.shape[1], grid, dense_weight=dense, codec="q8_0"
        )
        fp4name = next(k for k, v in layers.items() if v["quant_algo"] == "W4A16_NVFP4")
        packed = read(fp4name + ".weight")[:129].contiguous()
        scales = read(fp4name + ".weight_scale")[:129].contiguous()
        global_scale = read(fp4name + ".weight_scale_2")
        code = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1).long()
        table = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
        )
        dense = (
            table[code] * scales.float().repeat_interleave(16, dim=-1) * global_scale
        )
        check(
            fp4name,
            packed,
            8,
            0,
            dense.shape[1],
            grid,
            dense_weight=dense,
            codec="nvfp4",
            weight_scales=scales.cuda(),
            global_scale=global_scale.cuda(),
        )
    print("PASS: packed G128 component gate", flush=True)


if __name__ == "__main__":
    main()
