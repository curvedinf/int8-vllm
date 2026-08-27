# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)

logger = init_logger(__name__)


def _resolve_dflash_attention_backend(
    draft_backend: AttentionBackendEnum | None,
    target_backend: AttentionBackendEnum | None,
) -> AttentionBackendEnum | None:
    if draft_backend is not None:
        return draft_backend
    # DFlash draft pages are unified with the target's in one KV cache, so the
    # drafter's backend must be able to index the target's page layout. Inherit
    # the target's backend instead of platform auto-selection, which can pick a
    # backend that cannot index KV pages by block stride (e.g. ROCM_ATTN on
    # gfx908) and fails page-size unification.
    if target_backend is not None:
        logger.info_once(
            "Using the target model's %s attention backend for the DFlash "
            "drafter.",
            target_backend.name,
        )
    return target_backend


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import (
        dflash_has_any_non_causal,
        dflash_target_rope_is_neox_style,
    )

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # The drafter must rotate Q/K the way its target does. Take that from the
    # built target before super() constructs the draft.
    is_neox_style = dflash_target_rope_is_neox_style(target_model)
    if is_neox_style is not None:
        draft_model_config.hf_config.is_neox_style = is_neox_style
    # "auto" KV dtype is resolved inside Attention against
    # get_current_vllm_config().model_config — still the TARGET's ModelConfig
    # here (fp16 under --dtype half), so a bf16 draft would get an fp16 cache
    # tensor while the flash-cache writer stores bf16 bits raw into it and the
    # reader reinterprets them as fp16: silently corrupted draft attention
    # (layer-0 attn_out rel L2 ~1.9, acceptance ~13%; see
    # HANDOFF_DFLASH_DIAGNOSTIC.md). Resolve "auto" against the drafter's own
    # dtype instead; explicit dtypes (production: int8_per_token_head) and
    # inherited quantized dtypes pass through unchanged.
    draft_kv_dtype = (
        speculative_config.kv_cache_dtype
        if speculative_config.kv_cache_dtype is not None
        else vllm_config.cache_config.cache_dtype
    )
    if draft_kv_dtype == "auto":
        draft_kv_dtype = {
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
        }.get(draft_model_config.dtype, "auto")
    logger.info_once(
        "DFlash draft KV cache dtype: %s (draft compute dtype %s)",
        draft_kv_dtype,
        draft_model_config.dtype,
    )
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=_resolve_dflash_attention_backend(
                speculative_config.attention_backend,
                vllm_config.attention_config.backend,
            ),
        ),
        cache_config=replace(vllm_config.cache_config, cache_dtype=draft_kv_dtype),
    )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    # MuseGlimmerForCausalLM marks its inner MuseGlimmerModel as the language
    # model, so get_language_model() already returns the inner module and has
    # no .model of its own.
    target_inner = getattr(target_language_model, "model", target_language_model)
    draft_inner = dflash_model.model

    # Skip embedding sharing under PP — each rank owns its own embedding.
    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
            target_inner, "embedding", None
        )
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            dflash_model, "has_own_embed_tokens", draft_embed, target_embed
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    return dflash_model
