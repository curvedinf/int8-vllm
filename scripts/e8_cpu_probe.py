"""CPU-only E8 probe for the DFlash ctx-KV W8A8 env gate.

Drives the real ``_build_context_kv_buffers`` / ``_project_context_kv`` code
from qwen3_dflash.py on stub modules with CPU stand-ins for the two GPU
kernels the gated path calls (pertoken_quant_rn Triton kernel and the
rocm_aiter_gemm_a8w8_ck custom op). Validates branch selection, buffer
build/free, shapes, dtype cast-back, bias handling, and numeric sanity of
the requant chain — not the GPU kernels themselves (those are gated on the
4x MI100 box).

Modes:
  nofake : do not register the fake custom op; env ON must fall back to the
           dense path with a warning (op-unavailable guard).
  fake   : register the CPU fakes; check env OFF == dense F.linear exactly,
           and env ON == requant path, dense copy freed, close to dense.
"""

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

MODE = sys.argv[1] if len(sys.argv) > 1 else "fake"

if MODE == "fake":
    # The op is defined wherever aiter imports (register_ops_once) but its
    # kernel sits under the CUDA dispatch key on the ROCm torch build; add a
    # CPU kernel (dequant-reference of gemm_a8w8_CK) so the probe can drive
    # the real custom op with CPU tensors.
    from vllm.utils.torch_utils import vllm_lib

    def _gemm_ck_cpu(
        x_q: torch.Tensor,
        w_q: torch.Tensor,
        x_s: torch.Tensor,
        w_s: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        return ((x_q.float() * x_s) @ (w_q.float() * w_s).t()).to(out_dtype)

    vllm_lib.impl("rocm_aiter_gemm_a8w8_ck", _gemm_ck_cpu, dispatch_key="CPU")

from vllm.model_executor.models import qwen3_dflash as qdf

# CPU stand-ins for the two GPU kernels the gated path touches.
import vllm._custom_ops as cops

_orig_rms = cops.rms_norm


def _ref_rms(out, x, w, eps):
    v = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    out.copy_((v * w.float()).to(x.dtype))


cops.rms_norm = _ref_rms

import vllm.model_executor.kernels.linear.mixed_precision.act_quant_rn as aqr


def _ref_rn(x):
    # mirrors _pertoken_quant_rn_kernel: s=amax/127 (1 if 0), floor(v+0.5)
    amax = x.float().abs().amax(dim=1, keepdim=True)
    s = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax)).float()
    q = (x.float() / s + 0.5).floor().clamp(-127, 127).to(torch.int8)
    return q, s


aqr.pertoken_quant_rn = _ref_rn

# get_current_vllm_config is imported at call time inside the build; point it
# at a spec-less stub so draft dtype falls back to the embedding dtype.
import vllm.config as vcfg

vcfg.get_current_vllm_config = lambda: SimpleNamespace(speculative_config=None)

torch.manual_seed(0)
H, Q, HD, NKV, L, M = 128, 32, 8, 2, 3, 24
KV = NKV * HD


def make_stub():
    layers = []
    for _ in range(L):
        qkv = SimpleNamespace(
            weight=(torch.randn(Q + 2 * KV, H, dtype=torch.bfloat16) * 0.05),
            bias=torch.randn(Q + 2 * KV, dtype=torch.bfloat16) * 0.01,
        )
        layers.append(
            SimpleNamespace(
                qkv_proj=qkv,
                q_size=Q,
                k_norm=SimpleNamespace(
                    weight=SimpleNamespace(
                        data=torch.randn(HD, dtype=torch.bfloat16)
                    )
                ),
            )
        )
    return SimpleNamespace(
        hidden_norm=SimpleNamespace(
            weight=SimpleNamespace(data=torch.ones(H, dtype=torch.bfloat16))
        ),
        embed_tokens=SimpleNamespace(weight=SimpleNamespace(dtype=torch.bfloat16)),
        _rms_norm_eps=1e-6,
    ), layers


def project(model, layers, ctx):
    qdf.DFlashQwen3Model._build_context_kv_buffers(model, layers, True)
    return qdf.DFlashQwen3Model._project_context_kv(model, ctx, M, L, NKV, HD)


# dedicated generator so the global stream stays identical across the two
# make_stub() calls (each preceded by manual_seed(0)) for a fair A/B
ctx = torch.randn(
    M, H, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(1234)
)

if MODE == "nofake":
    os.environ["VLLM_GFX908_DF2_CTXKV_W8A8"] = "1"
    model, layers = make_stub()
    # The op registers wherever aiter imports, so simulate its absence for
    # the build-time probe with a torch stand-in whose ops.vllm lacks it.
    real_torch = qdf.torch
    qdf.torch = SimpleNamespace(
        cat=torch.cat,
        float32=torch.float32,
        int8=torch.int8,
        empty=torch.empty,
        stack=torch.stack,
        Tensor=torch.Tensor,
        ops=SimpleNamespace(vllm=SimpleNamespace()),
    )
    try:
        qdf.DFlashQwen3Model._build_context_kv_buffers(model, layers, True)
    finally:
        qdf.torch = real_torch
    k, v = qdf.DFlashQwen3Model._project_context_kv(model, ctx, M, L, NKV, HD)
    assert model._ckv_q is None, "quant buffers built without the custom op"
    assert (
        model._fused_kv_weight.shape == (L * 2 * KV, H)
    ), "dense not kept under op-unavailable fallback"
    assert k.shape == (L, M, NKV, HD) and k.dtype == torch.bfloat16
    print("nofake: env ON + op unavailable -> dense fallback (warning above)")
else:
    # env OFF: byte-identical dense path
    os.environ.pop("VLLM_GFX908_DF2_CTXKV_W8A8", None)
    model, layers = make_stub()
    k_off, v_off = project(model, layers, ctx)
    dense_w = model._fused_kv_weight.clone()
    dense_b = model._fused_kv_bias.clone()
    assert model._ckv_q is None and model._fused_kv_weight.numel() > 0
    _ref_rms(_n := torch.empty_like(ctx), ctx, torch.ones(H), 1e-6)
    ref_flat = F.linear(_n, dense_w, dense_b)
    ref_k = ref_flat.view(M, L, 2, NKV, HD).permute(2, 1, 0, 3, 4)[0]
    assert torch.equal(k_off, ref_k.contiguous()), "env OFF changed dense output"
    print("fake/env-off: dense path byte-identical to F.linear")

    # env ON: requant path, dense freed (same weights as the env-off stub)
    os.environ["VLLM_GFX908_DF2_CTXKV_W8A8"] = "1"
    torch.manual_seed(0)  # redraw identical stub weights for a fair A/B
    model2, layers2 = make_stub()
    k_on, v_on = project(model2, layers2, ctx)
    assert model2._ckv_q is not None and model2._ckv_q.dtype == torch.int8
    assert model2._ckv_s.dtype == torch.float32
    assert model2._ckv_s.shape == (dense_w.shape[0], 1)
    assert model2._fused_kv_weight.numel() == 0, "dense copy not freed"
    assert k_on.shape == k_off.shape and k_on.dtype == torch.bfloat16
    rel_k = (
        (k_on.float() - k_off.float()).norm() / k_off.float().norm()
    ).item()
    rel_v = (
        (v_on.float() - v_off.float()).norm() / v_off.float().norm()
    ).item()
    assert rel_k < 0.05 and rel_v < 0.05, (rel_k, rel_v)
    print(
        f"fake/env-on: quant path OK (rel-L2 K={rel_k:.3%} V={rel_v:.3%} "
        f"vs dense, dtype={k_on.dtype}, dense freed)"
    )
