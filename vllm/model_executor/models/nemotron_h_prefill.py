# SPDX-License-Identifier: Apache-2.0
"""Forward-scoped native norm and GS128 input reuse for packed MoE prefill."""

import torch
from torch.nn.modules import module as module_hooks

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.fused_moe.runner import moe_runner as runner_module
from vllm.model_executor.layers.fused_moe.runner.moe_runner import _layer_name_type
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExpertsOrder,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.modelopt_gfx908 import (
    PackedG128LinearMethod,
    PackedG128MoEMethod,
)
from vllm.model_executor.models.nemotron_h import NemotronHMLP, NemotronHMoE
from vllm.platforms.rocm import on_gfx908
from vllm.utils.torch_utils import _USE_LAYERNAME, direct_register_custom_op

logger = init_logger(__name__)


def _prefill_experts(hidden, q, scales, router_logits, layer_name, latent_dim):
    if (
        hidden.ndim != 2
        or hidden.shape[0] < 256
        or hidden.shape[1] != 4096
        or q.shape != hidden.shape
        or q.dtype != torch.int8
        or scales.shape != (hidden.shape[0], 32)
        or scales.dtype != torch.float32
        or q.device != hidden.device
        or scales.device != hidden.device
        or not q.is_contiguous()
        or not scales.is_contiguous()
    ):
        raise ValueError("Invalid forward-scoped GS128 prefill payload")
    runner = runner_module.get_layer_from_name(
        runner_module._resolve_layer_name(layer_name)
    )
    first = runner.routed_input_transform
    latent = first.quant_method.apply_quantized(first, q, scales).to(hidden.dtype)
    if latent.shape[1] != latent_dim:
        raise ValueError("Latent output width does not match the prefill schema")
    runner.routed_experts._ensure_moe_quant_config_init()
    shared = runner._shared_experts._layer
    up, down = shared.up_proj, shared.down_proj
    mid, mid_scales = up.quant_method.apply_quantized(
        up, q, scales, relu2=True, quant_out=True
    )
    shared_output = down.quant_method.apply_quantized(down, mid, mid_scales).to(
        hidden.dtype
    )
    weights, ids = runner.router.select_experts(
        hidden_states=latent,
        router_logits=router_logits,
        topk_indices_dtype=runner._quant_method.topk_indices_dtype,
        input_ids=None,
    )
    routed = runner.routed_experts.forward_modular(
        x=latent,
        topk_weights=weights,
        topk_ids=ids,
        shared_experts=None,
        shared_experts_input=None,
    )
    return shared_output, routed


def _prefill_experts_fake(hidden, q, scales, router_logits, layer_name, latent_dim):
    return torch.empty_like(hidden), hidden.new_empty((hidden.shape[0], latent_dim))


def _prefill_experts_op(
    hidden: torch.Tensor,
    q: torch.Tensor,
    scales: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: _layer_name_type,
    latent_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _prefill_experts(hidden, q, scales, router_logits, layer_name, latent_dim)


direct_register_custom_op(
    op_name="gfx908_moe_prefill_quant",
    op_func=_prefill_experts_op,
    fake_impl=_prefill_experts_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _resident_linear(layer):
    return (
        isinstance(layer.quant_method, PackedG128LinearMethod)
        and layer.bias is None
        and layer.weight.dtype == torch.int8
        and hasattr(layer, "weight_gs128")
        and layer.weight_gs128.dtype == torch.float32
    )


def can_use_prefill(decoder, x, residual):
    """Select only the existing plain gfx908 TP4 latent/shared topology."""
    if (
        torch.compiler.is_compiling()
        or torch.is_grad_enabled()
        or not _USE_LAYERNAME
        or not on_gfx908()
        or x.ndim != 2
        or x.shape[0] < 256
        or x.shape[1] != 4096
        or x.dtype not in (torch.bfloat16, torch.float32)
        or not x.is_contiguous()
        or envs.VLLM_GFX908_FUSED_NORM_QUANT
    ):
        return False
    if residual is not None and (
        residual.shape != x.shape
        or residual.dtype not in (torch.bfloat16, torch.float32)
        or not residual.is_contiguous()
        or residual.device != x.device
    ):
        return False
    norm, mixer = decoder.norm, decoder.mixer
    if (
        type(norm) is not RMSNorm
        or getattr(norm._forward_method, "__func__", None) is not RMSNorm.forward_hip
        or norm.variance_size_override is not None
        or not norm.has_weight
        or not norm.pass_weight
        or not norm.pass_weight_add
        or type(mixer) is not NemotronHMoE
        or not mixer.use_latent_moe
        or mixer.is_sequence_parallel
        or mixer.enable_eplb
    ):
        return False
    runner = mixer.experts
    cfg = runner.moe_config
    wrapper = runner._shared_experts
    shared = mixer.shared_experts
    if (
        cfg.tp_size != 4
        or cfg.ep_size != 1
        or cfg.dp_size != 1
        or cfg.pcp_size != 1
        or cfg.sp_size != 1
        or cfg.is_sequence_parallel
        or cfg.skip_final_all_reduce
        or cfg.is_lora_enabled
        or cfg.hidden_dim != 1024
        or getattr(runner, "enable_dbo", False)
        or cfg.moe_parallel_config.enable_eplb
        or cfg.moe_parallel_config.use_fi_nvl_two_sided_kernels
        or runner.gate is not None
        or runner.shared_expert_gate is not None
        or isinstance(runner.router, ZeroExpertRouter)
        or not isinstance(runner._quant_method, PackedG128MoEMethod)
        or runner._quant_method.moe_kernel is not None
        or runner._quant_method.has_unpadded_output
        or wrapper is None
        or wrapper.enable_dbo
        or wrapper._output != [None, None]
        or wrapper._layer is not shared
        or type(shared) is not NemotronHMLP
        or not shared.packed_relu2
        or wrapper._determine_shared_experts_order(x) != SharedExpertsOrder.NO_OVERLAP
        or runner.routed_input_transform is not mixer.fc1_latent_proj
        or runner.routed_output_transform is not mixer.fc2_latent_proj
    ):
        return False
    first, last = mixer.fc1_latent_proj, mixer.fc2_latent_proj
    if not all(_resident_linear(v) for v in (first, shared.up_proj, shared.down_proj)):
        return False
    if (
        first.weight.shape != (1024, 4096)
        or last.bias is not None
        or shared.up_proj.weight.shape[1] != 4096
        or shared.down_proj.weight.shape != (4096, shared.up_proj.weight.shape[0])
        or shared.down_proj.reduce_results
        or module_hooks._global_forward_hooks
        or module_hooks._global_forward_pre_hooks
    ):
        return False
    modules = (
        norm,
        mixer,
        mixer.gate,
        first,
        last,
        shared,
        shared.up_proj,
        shared.down_proj,
        runner,
        wrapper,
    )
    return not any(v._forward_hooks or v._forward_pre_hooks for v in modules)


def try_prefill(decoder, hidden_states, residual):
    if not can_use_prefill(decoder, hidden_states, residual):
        return None
    from aiter.ops.triton.normalization.rmsnorm_g128 import rms_norm_g128

    mixer = decoder.mixer
    runner = mixer.experts
    normal, residual_out, q, scales = rms_norm_g128(
        hidden_states, decoder.norm.weight.data, decoder.norm.variance_epsilon, residual
    )
    logits, _ = mixer.gate(normal)
    shared, routed = torch.ops.vllm.gfx908_moe_prefill_quant(
        normal, q, scales, logits, runner._encode_layer_name(), 1024
    )
    reduced = runner._fused_output_is_reduced
    routed, reduced = runner._maybe_reduce_routed_output_before_transform(
        routed, reduced
    )
    shared = runner._maybe_reduce_shared_expert_output(shared, reduced)
    shared, routed = runner._maybe_apply_routed_scale_to_output(shared, routed)
    routed = runner.apply_routed_output_transform(routed)
    result = runner._maybe_reduce_final_output(shared + routed, 4096, reduced)
    logger.info_once("gfx908 native norm GS128 MoE prefill active", scope="process")
    return runner._maybe_add_zero_expert_output(result), residual_out
