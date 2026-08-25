# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom normalization layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import kernels
import vllm.kernels  # noqa: F401
from vllm import envs, ir
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.batch_invariant import rms_norm_batch_invariant

logger = init_logger(__name__)


def poly_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, variance_epsilon: float
) -> torch.Tensor:
    from vllm import _custom_ops as ops

    out = torch.empty_like(x)
    ops.poly_norm(  # type: ignore[attr-defined]
        out,
        x,
        weight,
        bias,
        variance_epsilon,
    )
    return out


# --8<-- [start:rms_norm]
@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    # --8<-- [end:rms_norm]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        weight_dtype = dtype or torch.get_default_dtype()
        self.has_weight = has_weight
        self.weight = torch.ones(hidden_size, dtype=weight_dtype)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)

        # When has_weight=False, pass weight=None so implementations that
        # support a weightless path can skip the per-channel multiply.
        # Implementations that require weight (e.g. oink) fall back via IR
        # op priority when weight=None is unsupported.
        self.pass_weight = self.has_weight
        self.pass_weight_add = self.has_weight

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        if (
            envs.VLLM_GFX908_FUSED_NORM_QUANT
            and not torch.compiler.is_compiling()
            and self._fused_norm_quant_ok(x)
        ):
            return self._forward_native_fused_int8(x, residual)
        if residual is None:
            return ir.ops.rms_norm(
                x,
                self.weight.data if self.pass_weight else None,
                self.variance_epsilon,
                self.variance_size_override,
            )
        else:
            return ir.ops.fused_add_rms_norm.maybe_inplace(
                x,
                residual,
                self.weight.data if self.pass_weight_add else None,
                self.variance_epsilon,
                self.variance_size_override,
            )

    def _fused_norm_quant_ok(self, x: torch.Tensor) -> bool:
        from vllm.platforms import current_platform

        if not current_platform.is_rocm():
            return False
        if self.variance_size_override is not None or not self.pass_weight:
            return False
        if x.dim() != 2 or x.dtype not in (torch.float16, torch.bfloat16):
            return False
        return (1 << (x.shape[-1] - 1).bit_length()) * x.element_size() <= 65536

    def _forward_native_fused_int8(
        self, x: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Eager gfx908 fast path: one launch for norm + per-token int8 quant.

        The gfx908 serving stack runs eager (torch.compile is platform-
        disabled), so this is the production seam: the quant that CK W8A8
        linears would recompute is emitted here and stashed on the returned
        tensor. Native numerics keep the normed values within tl.sum
        reduction order of the ir-native chain.
        """
        from vllm.model_executor.layers.rms_norm_int8_quant import (
            PREQUANT_ATTR,
            rms_norm_int8_quant,
        )

        weight = self.weight.data
        if residual is None:
            out, q, scale = rms_norm_int8_quant(
                x, weight, self.variance_epsilon, native_numerics=True
            )
            setattr(out, PREQUANT_ATTR, (q, scale))
            return out
        out, res_out, q, scale = rms_norm_int8_quant(
            x,
            weight,
            self.variance_epsilon,
            residual=residual,
            native_numerics=True,
        )
        setattr(out, PREQUANT_ATTR, (q, scale))
        return out, res_out

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if envs.VLLM_BATCH_INVARIANT:
            assert self.variance_size_override is None, (
                "Batch invariance is not supported for variance_size_override"
            )
            pass_weight = (
                self.pass_weight_add if residual is not None else self.pass_weight
            )
            return rms_norm_batch_invariant(
                x,
                self.weight.data if pass_weight else None,
                self.variance_epsilon,
                residual=residual,
            )

        return self.forward_native(x, residual)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_cuda(x, residual)

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """ROCm path: use AITER Triton RMSNorm for large-M tensors.

        Microbenchmark shows AITER Triton RMSNorm is ~4x faster than the
        inductor-native path for prefill-sized inputs (M>=~512) and only
        slightly slower for tiny decode inputs. We dispatch on M to avoid any
        decode regression.

        With VLLM_GFX908_FUSED_NORM_QUANT (default on), the large-M path
        additionally emits the per-token int8 quant of the normed output in
        the same launch and stashes (q, scale) on the returned tensor for
        CK W8A8 linear consumers.
        """
        from aiter.ops.triton.normalization.rmsnorm import (
            rms_norm as aiter_rms_norm,
            rmsnorm2d_fwd_with_add as aiter_rmsnorm_add,
        )

        # AITER kernels expect 2D input.
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        m = x_2d.shape[0]

        # Crossover where AITER wins on gfx908 (microbench: ~M=512).
        if m < 256:
            return self.forward_native(x, residual)

        weight = self.weight.data
        eps = self.variance_epsilon

        if envs.VLLM_GFX908_FUSED_NORM_QUANT and x_2d.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            from vllm.model_executor.layers.rms_norm_int8_quant import (
                PREQUANT_ATTR,
                rms_norm_int8_quant,
            )

            if (1 << (x_2d.shape[-1] - 1).bit_length()) * x_2d.element_size() <= 65536:
                residual_2d = (
                    residual.reshape(-1, orig_shape[-1]).contiguous()
                    if residual is not None
                    else None
                )
                if residual_2d is None:
                    out_2d, q, scale = rms_norm_int8_quant(x_2d, weight, eps)
                    out = out_2d.reshape(orig_shape)
                else:
                    out_2d, res_out_2d, q, scale = rms_norm_int8_quant(
                        x_2d, weight, eps, residual=residual_2d
                    )
                    out = out_2d.reshape(orig_shape)
                    res_out_2d = res_out_2d.reshape(residual.shape)
                setattr(out, PREQUANT_ATTR, (q, scale))
                if residual_2d is None:
                    return out
                return out, res_out_2d

        if residual is None:
            out_2d = aiter_rms_norm(x_2d, weight, eps)
            return out_2d.reshape(orig_shape)

        # fused add + rmsnorm
        residual_2d = residual.reshape(-1, orig_shape[-1]).contiguous()
        out_2d = torch.empty_like(x_2d)
        res_out_2d = torch.empty_like(residual_2d)
        aiter_rmsnorm_add(
            out_2d,
            x_2d,
            residual_2d,
            res_out_2d,
            weight,
            eps,
        )
        return out_2d.reshape(orig_shape), res_out_2d.reshape(residual.shape)

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


# --8<-- [start:gemma_rms_norm]
@CustomOp.register("gemma_rms_norm")
class GemmaRMSNorm(CustomOp):
    """RMS normalization for Gemma.

    Two differences from the above RMSNorm:
        1. x * (1 + w) instead of x * w.
        2. (x * w).to(orig_dtype) instead of x.to(orig_dtype) * w.
    """

    # --8<-- [end:gemma_rms_norm]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        weight = self.weight.float() + 1.0
        if residual is None:
            return ir.ops.rms_norm(x, weight, self.variance_epsilon)
        return ir.ops.fused_add_rms_norm(x, residual, weight, self.variance_epsilon)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(x, residual)

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """ROCm path: use AITER Triton RMSNorm for large-M tensors.

        Mirrors RMSNorm.forward_hip but applies the Gemma (1 + w) weight
        adjustment before dispatching to AITER.
        """
        from aiter.ops.triton.normalization.rmsnorm import (
            rms_norm as aiter_rms_norm,
            rmsnorm2d_fwd_with_add as aiter_rmsnorm_add,
        )

        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        m = x_2d.shape[0]

        if m < 256:
            return self.forward_native(x, residual)

        weight = (self.weight.float() + 1.0).to(x_2d.dtype)
        eps = self.variance_epsilon

        if residual is None:
            out_2d = aiter_rms_norm(x_2d, weight, eps)
            return out_2d.reshape(orig_shape)

        residual_2d = residual.reshape(-1, orig_shape[-1]).contiguous()
        out_2d = torch.empty_like(x_2d)
        res_out_2d = torch.empty_like(residual_2d)
        aiter_rmsnorm_add(
            out_2d,
            x_2d,
            residual_2d,
            res_out_2d,
            weight,
            eps,
        )
        return out_2d.reshape(orig_shape), res_out_2d.reshape(residual.shape)


# --8<-- [start:rms_norm_gated]
@CustomOp.register("rms_norm_gated")
class RMSNormGated(CustomOp):
    """RMS Normalization with optional gating.

    This is a native PyTorch implementation that supports:
    - Standard RMS normalization
    - Group RMS normalization
    - Optional gating with SiLU activation
    """

    # --8<-- [end:rms_norm_gated]

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        activation: str = "swish",
    ):
        """Initialize RMSNormGated.

        Args:
            hidden_size: Size of the hidden dimension
            eps: Epsilon for numerical stability
            group_size: If not None, do GroupNorm with each group
                        having group_size elements.
                        group_size=None is equivalent to group_size=hidden_size
                        (i.e. there's only 1 group).
            norm_before_gate: If True and z is provided: out = norm(x) * silu(z)
                              If False and z is provided: out = norm(x * silu(z))
            device: Device to create parameters on
            dtype: Data type for parameters
            activation: Activation function name for gating
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    @staticmethod
    def forward_static(
        x: torch.Tensor,
        z: torch.Tensor | None,
        weight: torch.Tensor,
        epsilon: float,
        orig_dtype: torch.dtype,
        group_size: int | None = None,
        norm_before_gate: bool = True,
        activation: str = "swish",
    ) -> torch.Tensor:
        """Pure-PyTorch RMS normalization with optional gating.

        This static method contains the full native logic so that both
        ``forward_native`` and ``MatcherRMSNormGated`` (used by the
        compilation pattern matcher) can share the same implementation.

        If *z* is not None and *norm_before_gate* is True:
            ``out = rms_norm(x) * act(z)``
        If *z* is not None and *norm_before_gate* is False:
            ``out = rms_norm(x * act(z))``
        """
        x = x.float()
        weight = weight.float()
        if z is not None:
            z = z.float()

        assert activation in ["silu", "sigmoid", "swish"]
        act_fn = F.sigmoid if activation == "sigmoid" else F.silu

        if z is not None and not norm_before_gate:
            x = x * act_fn(z)

        if group_size is None:
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + epsilon)
            out = x_normed * weight
        else:
            from einops import rearrange

            x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
            variance = x_group.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_group * torch.rsqrt(variance + epsilon)
            out = rearrange(x_normed, "... g d -> ... (g d)") * weight

        if z is not None and norm_before_gate:
            out = out * act_fn(z)

        return out.to(orig_dtype)

    def forward_native(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return self.forward_static(
            x,
            z,
            self.weight,
            self.eps,
            x.dtype,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            activation=self.activation,
        )

    def forward_cuda(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
            rmsnorm_fn,
        )

        return rmsnorm_fn(
            x,
            self.weight,
            self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            activation=self.activation,
        )

    def forward_xpu(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.forward_cuda(x, z)


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(
            x.float(), (self.dim,), self.weight, self.bias, self.eps
        ).type_as(x)
