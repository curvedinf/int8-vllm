# SPDX-License-Identifier: Apache-2.0
"""Explicit GS128 branch path for the deployed Bolt latent/shared MoE."""

import os

import torch

from vllm import envs
from vllm.distributed.device_communicators.residual_g128_all_reduce import (
    all_reduce_pair,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.decoder_int8 import projection
from vllm.model_executor.layers.residual_int8 import (
    ResidualInt8,
    ResidualPair,
    add_residual,
    materialize,
)

logger = init_logger(__name__)

# Persistent side stream/events per device. Created during eager warmup, before
# the decode graph capture; the graph then records the fork/join.
_SHARED_STATE = {}


def _shared_state(device):
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    entry = _SHARED_STATE.get(index)
    if entry is None:
        entry = (
            torch.cuda.Stream(device=device),
            torch.cuda.Event(),
            torch.cuda.Event(),
        )
        _SHARED_STATE[index] = entry
    return entry


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

    # The shared-expert MLP depends only on `normal`, exactly like the routed
    # branch. At decode M the GEMMs are launch/latency-bound and the microbench
    # measured near-perfect co-residency. This side-stream fork is kept off by
    # default: graph capture currently trips the native custom AR (illegal
    # access at 6/8 captures). A separate G128 AR instance per stream is the
    # follow-up; the env remains for eager A/B work.
    overlap = (
        os.environ.get("VLLM_GFX908_SHARED_OVERLAP", "0") == "1"
        and normal.shape[0] <= 64
        and normal.codes.is_cuda
    )
    shared = None
    if overlap:
        main = torch.cuda.current_stream(normal.codes.device)
        side, fork, join = _shared_state(normal.codes.device)
        fork.record(main)
        with torch.cuda.stream(side):
            side.wait_event(fork)
            shared = mixer.shared_experts(normal)
            join.record(side)

    # Router logits are a control surface, not the decoder activation stream.
    # Prefill and decode share the int8 gate so the routing path does not change
    # with batch size; the launcher's gate-i8 flag is the rollback.
    if envs.VLLM_GFX908_GATE_I8_PREFILL:
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
    if overlap:
        main.wait_event(join)
        logger.info_once("GS128 shared/routed overlap active", scope="process")
    else:
        shared = mixer.shared_experts(normal)
    # One collective for both branches: the latent and shared payloads are
    # adjacent and their scales stay one-per-128 in the concatenation.
    routed, shared = all_reduce_pair(routed, shared)
    factor = runner.routed_scaling_factor
    if factor != 1.0:
        routed = routed._replace(scales=routed.scales * factor)
    routed = projection(mixer.fc2_latent_proj, routed)
    if os.environ.get("VLLM_GFX908_MOE_PAIR", "1") == "0":
        return add_residual(shared, routed)
    # Defer the branch sum to the next norm; it fuses both dequantizations,
    # the skip add, and the output quantizations in one pass.
    return ResidualPair(shared, routed, shared.compute_dtype)
