# SPDX-License-Identifier: Apache-2.0
"""GPTQ int8 group MoE for gfx908 via the AITER grouped a8w8-blockscale kernel.

Replaces the WNA16 (A16W8) fallback with a true W8A8 path for routed
experts whose weights are GPTQ int8 with per-K-group scales. Kernel
conventions were validated on gfx908 (rel rms 0.016 vs a torch
reference at E=8/H=2048/I=512/top-2, GS128):

* weights are packed as ``[E, K, N]`` int8 with per-output-column
  ``[E, K/G, N]`` fp32 scales (the aiter kernel's default 128-wide N
  tiles are overridden with ``w_blockscale_n=1``);
* activations are int8 ``[M, K]`` with per-row ``[M, K/G]`` scales;
* gate-up uses ``gather_indx`` and ``apply_swiglu=False`` (the kernel's
  fused swiglu is interleaved and adds a residual, not flint's
  ``[gate|up]`` layout), the split/silu happens in torch;
* down uses ``scatter_indx`` and ``gammas=RoutingData.gate_scal`` (the
  renormalized routing weights in expert-sorted order).
"""

import os

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.auto_gptq import (
    AutoGPTQMoEMethod,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter

_G = 128
_PACK = 4  # 8-bit GPTQ packs 4 values per int32


def _unpack_i8(packed: torch.Tensor) -> torch.Tensor:
    """[E, K//4, N] int32 (uint8b128) -> [E, K, N] signed int8."""
    e, k4, n = packed.shape
    shifts = torch.arange(4, device=packed.device, dtype=torch.int32) * 8
    w = ((packed.unsqueeze(2) >> shifts[None, None, :, None]) & 0xFF)
    w = w.reshape(e, k4 * 4, n)
    return (w.to(torch.int16) - 128).to(torch.int8)


def _quant_act(x: torch.Tensor, g: int = _G):
    m, k = x.shape
    assert k % g == 0, f"activation K={k} not divisible by {g}"
    xf = x.float().view(m, k // g, g)
    scale = xf.abs().amax(dim=2, keepdim=True).clamp_min(1e-8) / 127.0
    q = (xf / scale).round().clamp(-127, 127).to(torch.int8).reshape(m, k)
    return q, scale.squeeze(2).to(torch.float32)


def _routing_from_topk(topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                       n_expts: int):
    """RoutingData + gather/scatter from topk ids/weights via the fused kernel.

    Reconstructs a logits matrix whose top-k softmax reproduces the router's
    weights (log(w) on the selected experts, -inf elsewhere) so aiter's fused
    ``routing()`` can build the structures without host-side loops (CUDA-
    graph safe; ``routing_torch``'s block_pid_map loop is not).
    """
    from aiter.ops.triton.moe.moe_routing.routing import routing

    _dbg = os.environ.get("VLLM_MOE_A8W8_DBG")
    if _dbg:
        ids = topk_ids.long()
        w = topk_weights.float()
        dup = 0
        for r in range(ids.shape[0]):
            dup = max(dup, int(ids[r].numel() -
                               ids[r].unique().numel()))
        print(f"[moe8t] M={ids.shape[0]} ids[{int(ids.min())},"
              f"{int(ids.max())}] w[{float(w.min()):.3g},"
              f"{float(w.max()):.3g}] dup={dup}", flush=True)
    # Padding/dummy rows carry expert id -1 (vLLM convention). A negative
    # index silently wraps in scatter_ and the fused routing kernel then
    # faults on the degenerate histogram; map those slots to expert 0 with
    # zero routing weight instead (their outputs are discarded upstream).
    valid = topk_ids >= 0
    ids = torch.where(valid, topk_ids,
                      torch.zeros_like(topk_ids)).long()
    w = torch.where(valid, topk_weights.float(),
                    torch.zeros_like(topk_weights, dtype=torch.float32))
    logits = torch.full((topk_ids.shape[0], n_expts), float("-inf"),
                        device=topk_ids.device, dtype=torch.float32)
    logits.scatter_(1, ids, torch.log(w.clamp_min(1e-30)))
    if _dbg:
        print("[moe8t] scatter-done", flush=True)
    rd = routing(logits, topk_ids.shape[1])
    if _dbg:
        print("[moe8t] routing-done", flush=True)
    return rd


class A8W8BlockscaleMoEMethod(AutoGPTQMoEMethod):
    """W8A8 grouped GEMM for GPTQ int8-group routed experts on gfx908."""

    def __init__(self, quant_config, moe) -> None:
        # Skip AutoGPTQMoEMethod.__init__: it selects a WNA16 backend, and
        # the WNA16 oracle rejects int8-group configs on ROCm (the very
        # reason this class exists).
        from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
            FusedMoEMethodBase,
        )

        FusedMoEMethodBase.__init__(self, moe)
        self.quant_config = quant_config
        self.input_dtype = None
        self._use_a8w8 = True
        from vllm.logger import init_logger

        init_logger(__name__).info_once(
            "gfx908 routed experts: GPTQ int8-group -> AITER grouped "
            "a8w8-blockscale (W8A8)")
        assert int(quant_config.group_size) > 0 and (
            int(quant_config.group_size) % _G == 0), (
                f"A8W8 MoE needs a {_G}-multiple group size, got "
                f"{quant_config.group_size}")

    def _to_kernel_layout(self, layer) -> None:
        e = layer.w13_qweight.shape[0]
        w13 = _unpack_i8(layer.w13_qweight.data)          # [E, K, 2I_p]
        w2 = _unpack_i8(layer.w2_qweight.data)            # [E, I_p, H]
        s13 = layer.w13_scales.data.float()               # [E, K/G, 2I_p]
        s2 = layer.w2_scales.data.float()                 # [E, I_p/G, H]
        replace_parameter(layer, "w13_qweight", w13)
        replace_parameter(layer, "w2_qweight", w2)
        replace_parameter(layer, "w13_scales", s13)
        replace_parameter(layer, "w2_scales", s2)
        # zero points are unused on the symmetric path
        for name in ("w13_qzeros", "w2_qzeros"):
            if getattr(layer, name, None) is not None:
                replace_parameter(layer, name, None)
        for name in ("w13_bias", "w2_bias"):
            if getattr(layer, name, None) is not None:
                replace_parameter(layer, name, None)
        assert w13.dtype == torch.int8 and w2.dtype == torch.int8, e

    def get_fused_moe_quant_config(self, layer):
        # No modular kernel / FusedMoEQuantConfig in this path.
        return None

    def process_weights_after_loading(self, layer) -> None:
        if not self._use_a8w8:
            return super().process_weights_after_loading(layer)
        self._to_kernel_layout(layer)

    @torch.no_grad()
    def _prewarm(self, layer, hidden: int, topk: int, n_expts: int) -> None:
        """Compile every capture-size variant before CUDA graph capture.

        AITER's routing/MoE Triton kernels JIT on first use; a variant that
        first appears inside a graph capture faults on this stack.
        """
        if os.environ.get("VLLM_MOE_A8W8_NOPREWARM"):
            return
        if getattr(layer, "_a8w8_prewarmed", False):
            return
        if torch.cuda.is_current_stream_capturing():
            return
        dev = layer.w13_qweight.device
        dt = torch.bfloat16
        for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            xd = torch.randn(m, hidden, device=dev, dtype=dt) / 10
            ids = torch.randint(0, n_expts, (m, topk), device=dev)
            w = torch.rand(m, topk, device=dev)
            w = w / w.sum(-1, keepdim=True)
            self._run(layer, xd, w, ids, n_expts)
        layer._a8w8_prewarmed = True

    def _run(self, layer, x, topk_weights, topk_ids, n_expts):
        from aiter.ops.triton.moe.moe_op_gemm_a8w8_blockscale import (
            moe_gemm_a8w8_blockscale,
        )

        _dbg = os.environ.get("VLLM_MOE_A8W8_DBG")
        _cap = (torch.cuda.is_current_stream_capturing() if _dbg else False)
        if _dbg:
            print(f"[moe8r] {getattr(layer, 'prefix', '?')} M={x.shape[0]} "
                  f"cap={_cap} enter", flush=True)
        rd, gather_idx, scatter_idx = _routing_from_topk(
            topk_ids, topk_weights, n_expts)
        if _dbg:
            print(f"[moe8r] cap={_cap} routing-done", flush=True)
        xq, xs = _quant_act(x)
        kw = dict(routing_data=rd, out_dtype=x.dtype, per_row_x_scale=True,
                  w_blockscale_n=1)
        h2 = moe_gemm_a8w8_blockscale(
            xq, layer.w13_qweight, xs, layer.w13_scales,
            gather_indx=gather_idx, apply_swiglu=False, **kw)
        if _dbg:
            print(f"[moe8r] cap={_cap} gateup-done", flush=True)
        gate, up = h2.chunk(2, dim=-1)
        h = (F.silu(gate.float()) * up.float()).to(x.dtype)
        hq, hs = _quant_act(h)
        y = moe_gemm_a8w8_blockscale(
            hq, layer.w2_qweight, hs, layer.w2_scales,
            scatter_indx=scatter_idx, gammas=rd.gate_scal,
            apply_swiglu=False, **kw)
        if _dbg:
            print(f"[moe8r] cap={_cap} down-done", flush=True)
        return y

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts,
              shared_experts_input):
        if not self._use_a8w8:
            return super().apply(
                layer, x, topk_weights, topk_ids, shared_experts,
                shared_experts_input)
        from aiter.ops.triton.moe.moe_op_gemm_a8w8_blockscale import (
            moe_gemm_a8w8_blockscale,
        )

        assert getattr(layer, "expert_map", None) is None, "expert_map unsupported"
        m, hidden = x.shape
        if os.environ.get("VLLM_MOE_A8W8_DBG"):
            print(f"[moe8] layer={getattr(layer, 'layer_name', getattr(layer, 'prefix', '?'))} M={m} "
                  f"capturing={torch.cuda.is_current_stream_capturing()}",
                  flush=True)
        n_expts = layer.w13_qweight.shape[0]
        topk = topk_ids.shape[1]
        self._prewarm(layer, hidden, topk, n_expts)
        return self._run(layer, x, topk_weights, topk_ids, n_expts)
