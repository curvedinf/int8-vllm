# SPDX-License-Identifier: Apache-2.0
"""Explicit GS128 branch path for the deployed Bolt latent/shared MoE."""

from vllm import envs
from vllm.distributed.device_communicators.residual_g128_all_reduce import all_reduce
from vllm.model_executor.layers.decoder_int8 import projection
from vllm.model_executor.layers.residual_int8 import (
    ResidualInt8,
    add_residual,
    materialize,
)


def moe(mixer, normal):
    runner = mixer.experts
    cfg = runner.moe_config
    if (
        not isinstance(normal, ResidualInt8)
        or cfg.tp_size != 4
        or cfg.ep_size != 1
        or cfg.dp_size != 1
        or cfg.is_sequence_parallel
        or cfg.skip_final_all_reduce
        or mixer.enable_eplb
        or not mixer.use_latent_moe
        or runner._fused_output_is_reduced
        or mixer.shared_experts is None
        or runner.shared_expert_gate is not None
    ):
        raise ValueError("Direct decoder INT8 requires the deployed TP4 latent MoE")
    # Router logits are a control surface, not the decoder activation stream.
    if envs.VLLM_GFX908_GATE_I8_PREFILL and normal.shape[0] >= 256:
        from vllm.model_executor.models.nemotron_h_prefill import _gate_logits_i8

        logits = _gate_logits_i8(mixer, *normal.flatten_rows())
    else:
        logits, _ = mixer.gate(materialize(normal))
    latent = projection(mixer.fc1_latent_proj, normal)
    weights, ids = runner.router.select_experts(
        hidden_states=normal.codes,
        router_logits=logits,
        topk_indices_dtype=runner._quant_method.topk_indices_dtype,
        input_ids=None,
    )
    runner.routed_experts._ensure_moe_quant_config_init()
    routed = runner._quant_method.apply(
        runner.routed_experts, latent, weights, ids, None, None
    )
    routed = all_reduce(routed)
    shared = all_reduce(mixer.shared_experts(normal))
    factor = runner.routed_scaling_factor
    if factor != 1.0:
        routed = routed._replace(scales=routed.scales * factor)
    routed = projection(mixer.fc2_latent_proj, routed)
    return add_residual(shared, routed)
