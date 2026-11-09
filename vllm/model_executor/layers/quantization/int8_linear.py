# SPDX-License-Identifier: Apache-2.0
"""Load-time int8 W8A8 conversion for bf16 checkpoint linears.

Used for surfaces whose checkpoints ship bf16 but which must serve int8
under the W8A8-everywhere directive (e.g. the DFlash2 draft `fc`, the
router gate). Weights are quantized per output channel once at load; the
activation uses the same per-token int8 quantizer as the CK W8A8 path, so
these surfaces run the exact production int8 GEMM.
"""

import types

import torch


def _int8_forward(self, x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1])
    if flat.dtype not in (torch.float16, torch.bfloat16):
        flat = flat.to(self._q_dtype)
    # Per-token int8 activation quant + CK A8W8 GEMM (the W8A8 path the
    # GPTQ linears use on gfx908).
    xq, xs = torch.ops.vllm.rocm_aiter_pertoken_quant_int8(flat.contiguous())
    out = torch.ops.vllm.rocm_aiter_gemm_a8w8_ck(
        xq, self._q, xs, self._s, x.dtype)
    return out.reshape(*x.shape[:-1], out.shape[-1])


def install_int8_linear(module, *, name: str = "") -> bool:
    """Quantize a ReplicatedLinear-style module's weight and swap its forward.

    Returns True when the module was converted. Idempotent.
    """
    if getattr(module, "_int8_installed", False):
        return True
    w = module.weight.data
    if w.dim() != 2 or not w.is_floating_point():
        return False
    wf = w.float()
    amax = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    s = (amax / 127.0).to(torch.float32).contiguous()
    q = (wf / s).round().clamp(-127, 127).to(torch.int8).contiguous()
    module._q = q
    module._s = s
    module._q_dtype = w.dtype
    module.forward = types.MethodType(_int8_forward, module)
    module._int8_installed = True
    return True
