# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native HY4 affine W4A8 target and symmetric W8A8 companion weights."""

from typing import NamedTuple

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs


def canonical_module(name):
    name = name.removeprefix("language_model.").removeprefix("model.")
    if name.startswith("dfly.model.layers."):
        name = "dfly_layers." + name.removeprefix("dfly.model.layers.")
    elif name.startswith("dfly.model."):
        name = "dfly_layers." + name.removeprefix("dfly.model.")
    return name.removesuffix(".routed_experts")


class HY4NeUQIConfig(QuantizationConfig):
    def __init__(self, modules):
        super().__init__()
        self.modules = {canonical_module(name): bits for name, bits in modules.items()}
        if any(bits not in (4, 8) for bits in self.modules.values()):
            raise ValueError(
                "HY4 weights require INT4 target or INT8 companion storage"
            )

    @classmethod
    def get_name(cls):
        return "hy4_neuqi"

    def get_supported_act_dtypes(self):
        return [torch.float16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        if config.get("group_size") != 128:
            raise ValueError("HY4 weights require group_size=128")
        return cls(config["quantized_modules"])

    def get_quant_method(self, layer, prefix):
        name = canonical_module(prefix)
        bits = self.modules.get(name)
        if isinstance(layer, RoutedExperts):
            if bits != 4:
                raise ValueError(
                    f"HY4 routed experts missing from checkpoint manifest: {name}"
                )
            return HY4Experts(layer.moe_config)
        if isinstance(layer, LinearBase):
            return HY4Linear(bits) if bits else UnquantizedLinearMethod()
        return None


def _gemm_options():
    from vllm.config.vllm import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    return config.additional_config if config is not None else {}


def _prepare_wave(layer, bits, prefix="", layout=1):
    from aiter.ops.gemm_packed_native import pack_affine_weights

    weight = getattr(layer, prefix + "weight")
    scale = getattr(layer, prefix + "weight_scale")
    zero = getattr(layer, prefix + "weight_zero") if bits == 4 else None
    packed, scales, zeros = pack_affine_weights(
        weight, scale, zero, vector=layout == 2, overlap=layout == 3
    )
    weight.data = packed
    scale.data = scales
    if zero is not None:
        zero.data = zeros


def _dense_split(m, n, k):
    # Partition K on narrow, short matrices to expose independent work to the
    # 120 compute units. Large prefill tiles already fill the machine.
    groups = k // 128
    row_tiles = (m + 15) // 16 if m <= 16 or n <= 512 else (m + 31) // 32
    tiles = row_tiles * ((n + 127) // 128)
    wanted = min(16, groups, max(1, (256 + tiles - 1) // tiles))
    return 1 << (wanted.bit_length() - 1)


# Only these local TP4 shapes cleared the simultaneous four-card comparison.
# Zeros keep the original packet GEMM on this same packed allocation.
_FLY_G128_SPLITS = {
    (4, 128, 3072): {640: 8, 1024: 4, 2048: 2},
    (4, 2048, 3072): {48: 4, 96: 2, 320: 1},
    (4, 3072, 4096): {48: 4, 96: 2, 320: 1},
    (4, 4096, 3072): {48: 2, 96: 2},
    (4, 6912, 3072): {96: 1},
    (8, 1152, 1152): {640: 1},
    (8, 1152, 4352): {256: 2, 640: 1},
    (8, 1536, 6144): {640: 1},
    (8, 1600, 3072): {48: 8, 96: 4, 320: 1},
}


class G128ProjectionInput(NamedTuple):
    """One activation quantization shared by projections of the same token rows."""

    codes: torch.Tensor
    scales: torch.Tensor
    sums: torch.Tensor | None
    dtype: torch.dtype

    @property
    def shape(self):
        return self.codes.shape

    @classmethod
    def from_rows(cls, x):
        codes, scales, sums = _activation(x, 4)
        return cls(codes, scales, sums, x.dtype)


def _activation(x, bits):
    from aiter.ops.triton.gemm.basic.gemm_packed_g128 import quantize

    from vllm.model_executor.layers.residual_int8 import ResidualInt8

    if isinstance(x, G128ProjectionInput):
        return x.codes, x.scales, x.sums
    if isinstance(x, ResidualInt8):
        q, s = x.flatten_rows()
        sums = (
            q.reshape(q.shape[0], -1, 128).sum(-1, dtype=torch.int32)
            if bits == 4
            else None
        )
        return q, s.float().contiguous(), sums
    x = x.reshape(-1, x.shape[-1]).contiguous()
    if bits == 4:
        return quantize(x, with_sums=True)
    q, s = quantize(x)
    return q, s, None


def _weight_parts(bits):
    yield "weight", 2 if bits == 4 else 1, torch.uint8 if bits == 4 else torch.int8
    yield "weight_scale", 128, torch.float16
    if bits == 4:
        yield "weight_zero", 128, torch.float16


class HY4Linear(LinearMethodBase):
    def __init__(self, bits):
        self.bits = bits
        options = _gemm_options()
        self.pack_wave = bool(options.get("hy4_wave_gemm", True))
        self.wave = False
        self.tune = bool(options.get("hy4_split_gemm", True))
        self.packet = bool(options.get("hy4_packet_gemm", True))
        self.direct_fp16 = bool(options.get("hy4_direct_fp16_output", False))
        self.fly = bool(options.get("hy4_flydsl_gemm", False))
        self.fly_splits = {}
        self.fly_swiglu_rows = ()
        self.dfly_pair_norm = bits == 8 and bool(
            options.get("hy4_flydsl_dfly_norm", False)
        )
        self.vision_gelu = bool(options.get("hy4_flydsl_vision_mlp", False))
        self.gelu_rows = ()

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **attrs,
    ):
        self.padding = -input_size_per_partition % 128
        if self.padding and (self.bits != 8 or input_size != input_size_per_partition):
            raise ValueError(
                "HY4 shards need whole G128 groups. Use data-parallel vision."
            )
        stored_k = input_size_per_partition + self.padding
        for name, divisor, dtype in _weight_parts(self.bits):
            param = ModelWeightParameter(
                data=torch.empty(
                    (sum(output_partition_sizes), stored_k // divisor), dtype=dtype
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=attrs["weight_loader"],
            )
            layer.register_parameter(name, param)

    def _prepare_vision_gelu(self, layer):
        from aiter.ops.flydsl.kernels import gfx908_g128_gelu  # noqa: F401
        from aiter.ops.gemm_packed_native import pack_affine_weights

        # The next W8 projection pads its 4,304 inputs to 34 complete groups.
        # Explicit zero channels give its fused GELU input exactly the same grid.
        weight = torch.nn.functional.pad(layer.weight.detach(), (0, 0, 0, 48))
        scale = torch.nn.functional.pad(layer.weight_scale.detach(), (0, 0, 0, 48))
        packed, scale, _ = pack_affine_weights(weight, scale, overlap=True)
        layer.register_buffer("gelu_weight", packed, persistent=False)
        layer.register_buffer("gelu_scales", scale, persistent=False)
        layer.register_buffer(
            "gelu_bias",
            torch.nn.functional.pad(layer.bias.detach(), (0, 48)),
            persistent=False,
        )
        self.gelu_rows = (
            144,
            196,
            324,
            400,
            576,
            640,
            768,
            784,
            900,
            936,
            960,
            988,
            1024,
            1536,
            2048,
        )

    def process_weights_after_loading(self, layer):
        if self.dfly_pair_norm and getattr(layer, "prefix", "").endswith(
            (".mtp_fusion.eh_proj", ".hidden_correction.gate_up_proj")
        ):
            from aiter.ops.flydsl.kernels import gfx908_g128_dual_rms  # noqa: F401

        if (
            self.vision_gelu
            and self.bits == 8
            and layer.weight.shape == (4304, 1152)
            and layer.bias is not None
            and layer.bias.dtype == torch.float16
            and getattr(layer, "tp_size", 1) == 1
            and getattr(layer, "prefix", "").endswith(".dense_h_to_4h")
        ):
            self._prepare_vision_gelu(layer)
        self.wave = self.pack_wave and layer.weight.shape[0] % 16 == 0
        if self.wave:
            n = layer.weight.shape[0]
            k = layer.weight_scale.shape[-1] * 128
            if self.fly and self.packet:
                self.fly_splits = _FLY_G128_SPLITS.get((self.bits, n, k), {})
                if getattr(layer, "prefix", "").endswith(".gate_up_proj"):
                    self.fly_swiglu_rows = {
                        (4, 6912, 3072): (96,),
                        (4, 768, 3072): (2048,),
                    }.get((self.bits, n, k), ())
                if self.fly_splits or self.fly_swiglu_rows:
                    # Register the opaque operation before compiling the model.
                    from aiter.ops.flydsl.kernels import gfx908_g128  # noqa: F401

                    if self.fly_swiglu_rows:
                        from aiter.ops.flydsl.kernels import gfx908_g128_swiglu  # noqa: F401

                    self.wave = 3
            _prepare_wave(layer, self.bits, layout=self.wave)

    def apply_gelu(self, layer, x):
        codes, scales, _ = _activation(x, 8)
        codes, scales = torch.ops.aiter.gfx908_g128_gelu(
            codes,
            scales,
            layer.gelu_weight,
            layer.gelu_scales,
            layer.gelu_bias,
            4,
        )
        return G128ProjectionInput(codes, scales, None, x.dtype)

    def apply_swiglu(self, layer, x):
        q, s, sums = _activation(x, self.bits)
        codes, scales, totals = torch.ops.aiter.gfx908_g128_swiglu(
            q,
            s,
            sums,
            layer.weight,
            layer.weight_scale,
            getattr(layer, "weight_zero", None),
        )
        return G128ProjectionInput(codes, scales, totals, x.dtype)

    def dequantize_weight(self, layer, dtype=torch.float16):
        from aiter.ops.gemm_packed_native import dequantize_affine_weights

        weight = dequantize_affine_weights(
            layer.weight,
            layer.weight_scale,
            getattr(layer, "weight_zero", None),
            wave_format=self.wave,
            dtype=dtype,
        )
        return weight[:, : -self.padding] if self.padding else weight

    def apply(self, layer, x, bias=None):
        from aiter.ops.gemm_packed_native import affine_g128

        if self.padding and not isinstance(x, G128ProjectionInput):
            x = torch.nn.functional.pad(x, (0, self.padding))
        q, s, sums = _activation(x, self.bits)
        m, k = q.shape
        n = layer.weight.shape[0]
        dot8 = self.tune and self.bits == 4 and m == 1 and n >= 512
        split = _dense_split(m, n, k) if self.tune and not dot8 else 1
        tiles = None
        if self.packet and self.wave and not dot8:
            if m <= 16:
                tiles = (16, 64 if n <= 1024 else 128, 4)
            elif m <= 128 and n >= 2048:
                tiles = (16, 32, 2)
            elif m <= 512:
                tiles = (32, 64 if n <= 1536 else 128, 4)
            else:
                tiles = (64, 32, 2) if n <= 768 else (64, 128, 4)
            split = (
                min(8, k // 128)
                if self.tune
                and (m <= 16 or (m <= 128 and n <= 1536) or (m <= 512 and n <= 768))
                else 1
            )
        fly_split = self.fly_splits.get(m, 0)
        if fly_split:
            out = torch.ops.aiter.gfx908_g128_overlap(
                q,
                s,
                sums,
                layer.weight,
                layer.weight_scale,
                getattr(layer, "weight_zero", None),
                fly_split,
            )
        else:
            # Unsplit packet GEMMs can round the accumulator at the store.
            # Bias and split-K reduction must still see the FP32 accumulator.
            direct_fp16 = (
                self.direct_fp16
                and x.dtype == torch.float16
                and tiles is not None
                and split == 1
                and bias is None
            )
            out = affine_g128(
                q,
                s,
                sums,
                layer.weight,
                layer.weight_scale,
                getattr(layer, "weight_zero", None),
                split=split,
                dot8=dot8,
                wave_format=self.wave,
                triton_tiles=tiles,
                packet_load=True,
                output_dtype=torch.float16 if direct_fp16 else torch.float32,
            )
        if bias is not None:
            out += bias
        return out.to(x.dtype).view(*x.shape[:-1], layer.weight.shape[0])


class HY4Experts(FusedMoEMethodBase):
    def __init__(self, moe):
        super().__init__(moe)
        options = _gemm_options()
        self.pack_wave = bool(options.get("hy4_wave_gemm", True))
        self.wave = False
        self.waves = int(options.get("hy4_expert_waves", 4 if self.pack_wave else 2))
        self.layout = int(options.get("hy4_expert_layout", 2))
        self.packet = bool(options.get("hy4_packet_gemm", True))
        self.fused = bool(options.get("hy4_fused_moe", True))
        self.pair_threshold = int(options.get("hy4_pair_threshold", 768))
        if self.layout not in (1, 2):
            raise ValueError("HY4 expert layout must be 1 or 2")
        if moe.moe_parallel_config.ep_size != 1:
            raise ValueError("HY4 expert checkpoint uses tensor parallelism")

    def maybe_roundup_sizes(
        self,
        hidden_size,
        intermediate_size_per_partition,
        act_dtype,
        moe_parallel_config,
    ):
        return hidden_size, intermediate_size_per_partition

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **attrs,
    ):
        self.local_i = intermediate_size_per_partition
        if self.local_i % 128 or hidden_size % 128:
            raise ValueError(
                "HY4 expert shards must preserve full input quantization groups"
            )
        for leaf, divisor, dtype in _weight_parts(4):
            for which, n, k in (
                ("w13", 2 * self.local_i, hidden_size),
                ("w2", hidden_size, self.local_i),
            ):
                param = Parameter(
                    torch.empty((num_experts, n, k // divisor), dtype=dtype),
                    requires_grad=False,
                )
                set_weight_attrs(param, {"weight_loader": self.weight_loader})
                layer.register_parameter(f"{which}_{leaf}", param)

    def weight_loader(
        self, param, value, weight_name, shard_id, expert_id, return_success=False
    ):
        local = param[expert_id]
        rank = self.moe.tp_rank
        if shard_id == "w2":
            value = value.narrow(1, rank * local.shape[1], local.shape[1])
        elif shard_id in ("w1", "w3"):
            local = local.narrow(
                0, self.local_i if shard_id == "w3" else 0, self.local_i
            )
            value = value.narrow(0, rank * self.local_i, self.local_i)
        else:
            raise ValueError(f"Invalid HY4 expert shard {shard_id}")
        if local.shape != value.shape or local.dtype != value.dtype:
            raise ValueError(
                f"Invalid {weight_name}: shape {value.shape}, dtype {value.dtype}"
            )
        local.copy_(value)
        return True if return_success else None

    def process_weights_after_loading(self, layer):
        self.wave = self.layout if self.pack_wave else 0
        if self.wave:
            for prefix in ("w13_", "w2_"):
                _prepare_wave(layer, 4, prefix, self.wave)

    def get_fused_moe_quant_config(self, layer):
        return None

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        from aiter.ops.gemm_packed_native import affine_g128

        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )
        from vllm.model_executor.layers.quantization.utils.packed_g128_int8 import (
            combine_experts,
        )

        if layer.apply_router_weight_on_input:
            raise ValueError("HY4 applies route weights to expert outputs")
        experts = layer.w13_weight.shape[0]
        tokens = topk_ids.shape[0]
        tiles = None
        tile_m = 16
        if self.packet and self.wave:
            # Match the active expert occupancy at TP4. Larger route blocks
            # amortize one G128 weight packet across the prefill rows.
            tile_m = 64 if tokens >= 2048 else 32 if tokens >= 768 else 16
            tiles = (tile_m, 64 if tokens <= 16 else 128, 4)
        ids, exp, count = moe_align_block_size(
            topk_ids, tile_m, experts, pad_sorted_ids=True
        )
        plan = (ids, exp, count)
        topk = topk_ids.shape[-1]
        q, s, sums = _activation(x, 4)
        fused_activation = self.fused and self.moe.activation.value == "silu"
        paired = (
            fused_activation
            and tiles is not None
            and self.pair_threshold > 0
            and tokens >= self.pair_threshold
        )
        up = affine_g128(
            q,
            s,
            sums,
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w13_weight_zero,
            plan=plan,
            topk=topk,
            wave_format=self.wave,
            waves=self.waves,
            triton_tiles=tiles,
            packet_load=True,
            gated_quant=paired,
            clamp_limit=self.moe.swiglu_limit,
        )
        if paired:
            q, s, sums = up
        elif fused_activation:
            from aiter.ops.triton.gemm.basic.gemm_packed_g128 import quantize

            q, s, sums = quantize(
                up, with_sums=True, swiglu=True, clamp_limit=self.moe.swiglu_limit
            )
        else:
            mid = torch.empty(
                (up.shape[0], self.local_i), dtype=torch.float16, device=up.device
            )
            apply_moe_activation(
                self.moe.activation,
                mid,
                up.half(),
                activation_config=ApplyMoEActivationConfig(
                    clamp_limit=self.moe.swiglu_limit
                ),
            )
            q, s, sums = _activation(mid, 4)
        down = affine_g128(
            q,
            s,
            sums,
            layer.w2_weight,
            layer.w2_weight_scale,
            layer.w2_weight_zero,
            plan=plan,
            topk=topk,
            down=True,
            wave_format=self.wave,
            waves=self.waves,
            triton_tiles=tiles,
            packet_load=True,
        )
        return combine_experts(down, topk_weights, torch.float16)


HY4Experts.weight_loader.supports_moe_loading = True
