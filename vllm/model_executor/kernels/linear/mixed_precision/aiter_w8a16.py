# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER W8A8 INT8 dynamic blockscale kernel for ROCm gfx908.

The public class keeps its historical ``AiterW8A16LinearKernel`` name for
kernel-registry compatibility. GPTQ GS128 layers use AITER A8W8 at every M;
the A16W8 implementation remains only as a compatibility fallback for other
group sizes.
"""

import os

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

# AITER Triton kernels. Keep at module level to avoid repeated import overhead
# inside the hot apply_weights path.
from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import (
    gemm_a16w8_blockscale,
)
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale,
)

from vllm.model_executor.layers.rms_norm_int8_quant import PREQUANT_ATTR

_AITER_W8A16_SUPPORTED_QUANT_TYPES = [scalar_types.uint8b128]
_AITER_W8A16_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128, 256]

# The Direwolf production contract is AITER W8A8 at every batch shape.
# Keep this named threshold for compatibility with existing probes.
_W8A8_DISPATCH_MIN_M = 1


def _get_aiter_w8a8_config(M: int, N: int, K: int, group_size: int):
    """Select a gfx908-suitable AITER a8w8_blockscale config.

    For W8A8 the activation and weight scales are block-wise along K with
    BLOCK_SIZE_K=128, matching the GPTQ group_size. Microbench shows the same
    large-M config as W8A16 works well.
    """
    if M <= 16:
        block_m, block_n, num_warps, num_stages = 16, 64, 4, 2
    elif M <= 32:
        block_m, block_n, num_warps, num_stages = 32, 64, 4, 2
    elif M <= 64:
        block_m, block_n, num_warps, num_stages = 64, 64, 4, 2
    else:
        block_m, block_n, num_warps, num_stages = 128, 128, 8, 1

    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "cache_modifier": ".cg",
        "NUM_KSPLIT": 1,
        "SPLITK_BLOCK_SIZE": 2048,
    }


def _quantize_activation_per_block(x: torch.Tensor, block_k: int = 128):
    """Quantize FP16/BF16 activation to int8 with per-block K scales.

    Returns (x_q, x_scale) where x_scale shape is [M, K//block_k].
    """
    M, K = x.shape
    assert K % block_k == 0, f"K={K} not divisible by block_k={block_k}"
    x_blocks = x.reshape(M, K // block_k, block_k)
    absmax = x_blocks.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax)).to(
        x.dtype)
    x_q = (x_blocks / scale).clamp(-128, 127).round().to(torch.int8)
    return x_q.reshape(M, K), scale.squeeze(-1)


def _get_aiter_w8a16_config(M: int, N: int, K: int, group_size: int):
    """Select a gfx908-suitable AITER a16w8_blockscale config.

    The heuristic mirrors the one used for AITER a16w16 on MI100: small M
    decode uses BLOCK_SIZE_M=16, prefill / larger M scales up to keep the
    120 CUs busy. BLOCK_SIZE_K is fixed at 128 because all model shapes are
    divisible by 128 and the GPTQ group_size is 128.
    """
    if M <= 16:
        block_m, block_n, num_warps, num_stages, waves_per_eu = 16, 64, 4, 2, 2
    elif M <= 32:
        block_m, block_n, num_warps, num_stages, waves_per_eu = 32, 64, 4, 2, 2
    elif M <= 64:
        block_m, block_n, num_warps, num_stages, waves_per_eu = 64, 64, 4, 2, 2
    else:
        # Large-M prefill: microbench shows BLOCK_SIZE_N=128 and waves_per_eu=1
        # wins substantially on the Qwen3.6-27B TP4 projection shapes.
        block_m, block_n, num_warps, num_stages, waves_per_eu = 128, 128, 8, 1, 1

    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "waves_per_eu": waves_per_eu,
        "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg",
        "NUM_KSPLIT": 1,
        "SPLITK_BLOCK_SIZE": 2048,
    }


class AiterW8A16LinearKernel(MPLinearKernel):
    """AITER A8W8 selector for GPTQ GS128 on ROCm (gfx908).

    The class name is retained for registry and blocklist compatibility.
    """

    SUPPORTED_QUANT_TYPES = _AITER_W8A16_SUPPORTED_QUANT_TYPES

    # Per-process cache so we only JIT-compile each (kernel, M, N, K, gs) once.
    _WARMUP_CACHE: set[tuple[str, int, int, int, int]] = set()

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "AiterW8A16LinearKernel only targets ROCm"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type {c.weight_type} not supported; "
                f"supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "Only float16/bfloat16 activations are supported"

        if c.zero_points:
            return False, "Zero points are not supported by the AITER blockscale path"

        if c.has_g_idx:
            return False, "Activation reordering (g_idx) not supported"

        gs = c.group_size
        if (
            gs not in _AITER_W8A16_SUPPORTED_GROUP_SIZES
            and gs != c.full_weight_shape[0]
        ):
            return (
                False,
                f"Group size {gs} not supported; "
                f"supported: {_AITER_W8A16_SUPPORTED_GROUP_SIZES}",
            )

        K = c.partition_weight_shape[0]
        eff_gs = gs if gs != -1 else K
        if K % eff_gs != 0:
            return False, f"Input features {K} not divisible by group_size {eff_gs}"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert GPTQ packed weights to the AITER blockscale layout.

        Checkpoint layout from create_weights:
          qweight: [K//4, N] int32  (input_dim=0, output_dim=1, packed_dim=0)
          scales:  [K//G, N] fp16   (output_dim=1)

        AITER expects:
          qweight: [N, K] int8   (signed, zero-point bias of 128 removed)
          scales:  [N, K//G] fp16
        """

        def repack_w_q(x: BasevLLMParameter) -> BasevLLMParameter:
            # Checkpoint layout: [K//4, N] int32 (4 uint8 weights per int32).
            w = x.data
            K4, N_dim = w.shape
            K_dim = K4 * 4

            # Unpack along the packed K dimension -> [K, N] uint8.
            shifts = torch.arange(4, device=w.device, dtype=torch.int32) * 8
            w_u8 = ((w.unsqueeze(1) >> shifts[None, :, None]) & 0xFF).reshape(
                K_dim, N_dim
            )

            # Convert uint8b128 -> signed int8 (subtract the zero-point bias).
            wt = self.config.weight_type
            bias = wt.bias if wt.has_bias() else 128
            w_i8 = (w_u8.to(torch.int16) - bias).to(torch.int8)

            # AITER expects [N, K] int8.
            x.data = w_i8.t().contiguous()
            return x

        def repack_w_s(x: BasevLLMParameter) -> BasevLLMParameter:
            # scales are [K//G, N]; transpose to [N, K//G].
            x.data = x.data.t().contiguous()
            return x

        self._transform_param(layer, self.w_q_name, repack_w_q)
        self._transform_param(layer, self.w_s_name, repack_w_s)

        # DFlash fused context-KV consumer: same contract as
        # TritonW8A16LinearKernel — dense dequant of the AITER-layout
        # weights ([N, K] int8 x [N, K//G] scales), rows sliceable.
        def _dequant_rows(_unused) -> torch.Tensor:
            w_q, w_s, _, _ = self._get_weight_params(layer)
            # w_q: [N, K] int8 (already de-biased), w_s: [N, K//G]
            gs = self.config.group_size
            # Emit the model dtype, not the checkpoint scale dtype: GPTQ
            # scales are commonly stored fp16 even for bf16 models, and a
            # bf16-trained DFlash drafter must not round its dense KV rows
            # through fp16 (overflow at |w| > 65504/scale granularity).
            from vllm.config import get_current_vllm_config

            spec_cfg = get_current_vllm_config().speculative_config
            out_dtype = (
                spec_cfg.draft_model_config.dtype
                if spec_cfg is not None and spec_cfg.draft_model_config is not None
                else self.config.act_type
            )
            if gs == -1:
                return (w_q.float() * w_s.float()).to(out_dtype).contiguous()
            return (
                (w_q.float().view(N_local, K_local // gs, gs)
                 * w_s.float().unsqueeze(-1))
                .reshape(N_local, K_local)
                .to(out_dtype)
                .contiguous()
            )

        N_local, K_local = self._get_weight_params(layer)[0].shape
        layer.dequant_kv_rows = _dequant_rows

        # CK per-channel requant (gfx908): per-token x scales pair with
        # [N,1] weight scales in gemm_a8w8_CK. Skipped for TP-sharded
        # row-parallel layers where input_size_per_partition != input_size
        # (the CK contract needs whole-K per rank; sharded rows keep the
        # blockscale path).
        if (
            os.environ.get("VLLM_GFX908_CK_W8A8", "1") == "1"
            and current_platform.is_rocm()
        ):
            try:
                dense = _dequant_rows(None).float()
                wmax = dense.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
                layer._ck_s = (wmax / 127.0).to(torch.float32).contiguous()
                layer._ck_q = (
                    (dense / layer._ck_s).round().clamp(-127, 127).to(torch.int8).contiguous()
                )
            except Exception:
                layer.__dict__.pop("_ck_q", None)
                layer.__dict__.pop("_ck_s", None)

        # Pre-compile the AITER kernels for the shapes this layer will see so
        # the first real inference does not pay JIT compilation latency.
        self._warmup(layer)

        # With the CK copy live and warmed, the GS128 originals are dead
        # GPU weight (several GiB per rank across the target) — except QKV
        # layers, whose dequant_kv_rows hook reads live params for the
        # DFlash fused ctx-KV build, and layers without _ck_q (blockscale
        # fallback). Free the rest; the memory returns to KV cache.
        # QKVParallelLinear carries head_size; q_size lives on the parent
        # attention module, not the linear itself.
        if (
            hasattr(layer, "_ck_q")
            and not hasattr(layer, "head_size")
            and os.environ.get("VLLM_GFX908_CK_FREE_GS128", "1") == "1"
        ):
            for pname in (
                self.w_q_name,
                self.w_s_name,
                self.w_zp_name,
                self.w_gidx_name,
            ):
                if pname is not None and getattr(layer, pname, None) is not None:
                    replace_parameter(
                        layer,
                        pname,
                        torch.nn.Parameter(
                            torch.empty(0, device=layer._ck_q.device),
                            requires_grad=False,
                        ),
                    )

    def _warmup(self, layer: torch.nn.Module) -> None:
        """JIT-compile the production AITER A8W8 configs.

        Non-GS128 layers retain the compatibility A16W8 warmup and fallback,
        but both published Direwolf checkpoints are GS128.
        """

        w_q, w_s, _, _ = self._get_weight_params(layer)
        N, K = w_q.shape
        gs = self.config.group_size
        device = w_q.device
        dtype = self.config.act_type

        if gs == 128:
            warmup_shapes = (1, 17, 33, 65, 80, 96, 256, 1024, 2048, 4096)
        else:
            warmup_shapes = ()

        if gs == 128:
            # Warm the CK kernel (the production path), not the blockscale
            # fallback: graph capture and first inference otherwise pay the
            # aiter-JIT config lookup per production shape. The pertoken
            # quant + CK gemm pair mirrors apply_weights exactly.
            if hasattr(layer, "_ck_q"):
                for M in warmup_shapes:
                    key = ("ck", M, N, K, gs)
                    if key in self._WARMUP_CACHE:
                        continue
                    self._WARMUP_CACHE.add(key)
                    try:
                        xq = torch.randint(
                            -127, 127, (M, K), dtype=torch.int8, device=device
                        )
                        xs = torch.rand(M, 1, dtype=torch.float32, device=device) * 0.1
                        from aiter import gemm_a8w8_CK

                        gemm_a8w8_CK(xq, layer._ck_q, xs, layer._ck_s, None, dtype)
                    except Exception as e:
                        import warnings

                        warnings.warn(
                            f"CK warmup failed for (M={M}, N={N}, K={K}): {e}"
                        )
            else:
                for M in warmup_shapes:
                    key = ("a8w8", M, N, K, gs)
                    if key in self._WARMUP_CACHE:
                        continue
                    self._WARMUP_CACHE.add(key)

                    x = torch.empty((M, K), dtype=dtype, device=device)
                    cfg = _get_aiter_w8a8_config(M, N, K, gs)

                    try:
                        x_q, x_s = _quantize_activation_per_block(x, block_k=128)
                        gemm_a8w8_blockscale(
                            x_q,
                            w_q,
                            x_s,
                            w_s,
                            dtype=dtype,
                            config=cfg,
                        )
                    except Exception as e:
                        import warnings

                        warnings.warn(
                            f"AITER W8A8 warmup failed for (M={M}, N={N}, K={K}, gs={gs}): {e}"
                        )
            return

        for M in (1, 17, 33, 65):
            key = ("a16w8", M, N, K, gs)
            if key in self._WARMUP_CACHE:
                continue
            self._WARMUP_CACHE.add(key)

            x = torch.empty((M, K), dtype=dtype, device=device)
            cfg = _get_aiter_w8a16_config(M, N, K, gs)

            try:
                gemm_a16w8_blockscale(
                    x,
                    w_q,
                    w_s,
                    dtype=dtype,
                    config=cfg,
                )
            except Exception as e:
                import warnings

                warnings.warn(
                    f"AITER W8A16 compatibility warmup failed for "
                    f"(M={M}, N={N}, K={K}, gs={gs}): {e}"
                )

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, _, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        M, K = x_2d.shape
        N = w_q.shape[0]

        # The published target and DFlash2 checkpoints are GS128, so every
        # decode and prefill shape uses true AITER INT8 A8W8 compute.
        # Dispatch: the CK kernel (gemm_a8w8_CK, per-token scales) measured
        # 1.2-3.4x faster than the Triton blockscale at every M on gfx908,
        # but requires per-channel weight scales — the per-128-group
        # checkpoint scales are preserved for the blockscale fallback and
        # the fused context-KV dequant. We requantize per-channel at load
        # (cached) only when the CK path is enabled. The activation quant
        # and the CK gemm go through registered custom ops (wrapping the
        # same aiter calls) so torch.compile pattern matchers can see the
        # quant, and the aiter-JIT config lookup inside gemm_a8w8_CK
        # stays opaque to fullgraph tracing.
        if c.group_size == 128:
            if hasattr(layer, "_ck_q"):
                x_f16 = x_2d.to(torch.float16)
                quant_op = getattr(
                    torch.ops.vllm, "rocm_aiter_pertoken_quant_int8", None
                )
                gemm_op = getattr(torch.ops.vllm, "rocm_aiter_gemm_a8w8_ck", None)
                if quant_op is not None and gemm_op is not None:
                    # The fused-norm path stashes (q, scale) on the normed
                    # tensor; consume it instead of re-quantizing. Opaque to
                    # dynamo: the compiled path fuses at the graph level
                    # (GFX908RMSNormInt8QuantFusionPass) instead.
                    prequant = (
                        None
                        if torch.compiler.is_compiling()
                        else getattr(x, PREQUANT_ATTR, None)
                    )
                    if (
                        prequant is not None
                        and prequant[0].shape == x_2d.shape
                        and x_2d.dtype in (torch.float16, torch.bfloat16)
                    ):
                        x_q, x_s = prequant
                    else:
                        x_q, x_s = quant_op(x_f16)
                    # The CK kernel supports fp16/bf16 outputs only; the
                    # profile dummy run feeds fp32 activations whose dtype
                    # would otherwise flow through as Y (unsupported).
                    out_dtype = (
                        x_2d.dtype
                        if x_2d.dtype in (torch.float16, torch.bfloat16)
                        else torch.float16
                    )
                    output = gemm_op(x_q, layer._ck_q, x_s, layer._ck_s, out_dtype)
                else:
                    from aiter import gemm_a8w8_CK, pertoken_quant

                    x_q, x_s = pertoken_quant(x_f16, quant_dtype=torch.int8)
                    out_dtype = (
                        x_2d.dtype
                        if x_2d.dtype in (torch.float16, torch.bfloat16)
                        else torch.float16
                    )
                    output = gemm_a8w8_CK(
                        x_q, layer._ck_q, x_s, layer._ck_s, None, out_dtype
                    )
                if os.environ.get("VLLM_SPEC_DEBUG_DUMP") and not (
                    torch.cuda.is_current_stream_capturing()
                ):
                    print(
                        f"[SPEC-DBGC] ck M={M} N={N} K={K} "
                        f"in_abs={x_2d.abs().max().item():.3f} "
                        f"xq_abs={x_q.abs().max().item()} "
                        f"xs_abs={x_s.abs().max().item():.4f} "
                        f"wq_abs={layer._ck_q.abs().max().item()} "
                        f"ws_abs={layer._ck_s.abs().max().item():.4f} "
                        f"out_abs={output.abs().max().item():.3f}",
                        flush=True,
                    )
                from vllm import quant_audit_recorder as _qa

                if _qa._enabled() and not torch.cuda.is_current_stream_capturing():
                    _qa.record_gemm(
                        getattr(layer, "prefix", type(layer).__name__),
                        x_2d[: min(M, 64)],
                        x_q[: min(M, 64)],
                        x_s[: min(M, 64)],
                        N,
                        K,
                    )
            else:
                x_q, x_s = _quantize_activation_per_block(x_2d, block_k=128)
                cfg = _get_aiter_w8a8_config(M, N, K, c.group_size)
                if os.environ.get("VLLM_SPEC_DEBUG_DUMP") and not (
                    torch.cuda.is_current_stream_capturing()
                ):
                    print(
                        f"[SPEC-DBGC] blockscale M={M} N={N} K={K} "
                        f"in_abs={x_2d.abs().max().item():.3f} "
                        f"xq_abs={x_q.abs().max().item()} "
                        f"xs_abs={x_s.abs().max().item():.4f} "
                        f"wq_abs={w_q.abs().max().item()} "
                        f"ws_abs={w_s.abs().max().item():.4f} "
                        f"out_abs={float('nan')}",
                        flush=True,
                    )
                output = gemm_a8w8_blockscale(
                    x_q,
                    w_q,
                    x_s,
                    w_s,
                    dtype=x_2d.dtype,
                    config=cfg,
                )
        else:
            cfg = _get_aiter_w8a16_config(M, N, K, c.group_size)
            output = gemm_a16w8_blockscale(
                x_2d,
                w_q,
                w_s,
                dtype=x_2d.dtype,
                config=cfg,
            )

        if bias is not None:
            output.add_(bias)

        return output.reshape(out_shape)
