# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from .utils import AutoWeightsLoader, maybe_prefix


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )

    def process_weights_after_loading(self) -> None:
        """W8A8 (gfx908 int8 doctrine) for the projection.

        The conv coefficients this GEMM produces multiply the residual
        stream; per-channel weight quant at the 1/254 bound perturbs them
        far below the bf16 conv arithmetic's own rounding.
        """
        if os.environ.get("VLLM_GFX908_DF2_W8A8", "1") != "1":
            return
        from vllm.model_executor.layers.quantization.utils import replace_parameter

        w = self.kernel_projection.weight.data.float()
        wmax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        self._kp_s = (wmax / 127.0).to(torch.float32).contiguous()
        self._kp_q = (
            (w / self._kp_s).round_().clamp_(-127, 127).to(torch.int8).contiguous()
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Consume `base_kernel[.i8|.scale]` (E4); forward the rest unchanged.

        With VLLM_GFX908_DF2_CONV_I8=1 and int8 planes shipped alongside the
        bf16 tensor, stash `_bk_q`/`_bk_s` for dequant-on-add in `_convolve`
        and leave the bf16 `base_kernel` param unloaded (zeroed) so the conv
        never reads a materialized bf16 base weight. Without the flag (or
        without the int8 tensors) the bf16 path is byte-identical to before.
        """
        items = dict(weights)
        loaded: set[str] = set()
        q = items.pop("base_kernel.i8", None)
        scale = items.pop("base_kernel.scale", None)
        if q is not None or scale is not None:
            if os.environ.get("VLLM_GFX908_DF2_CONV_I8", "0") != "1":
                # Flag off: the bf16 base_kernel loaded below is authoritative.
                loaded |= {"base_kernel.i8", "base_kernel.scale"}
            elif q is None or scale is None:
                raise ValueError(
                    "VLLM_GFX908_DF2_CONV_I8=1 needs both 'base_kernel.i8' "
                    "and 'base_kernel.scale' in the checkpoint."
                )
            else:
                shape = self.base_kernel.shape
                if q.shape != shape or scale.shape != shape[:2]:
                    raise ValueError(
                        f"base_kernel.i8 {tuple(q.shape)} / .scale "
                        f"{tuple(scale.shape)} do not match base_kernel "
                        f"{tuple(shape)}."
                    )
                self._bk_q = q.to(torch.int8).contiguous()
                self._bk_s = scale.to(torch.float32).contiguous()
                self.base_kernel.data.zero_()
                items.pop("base_kernel", None)
                loaded |= {"base_kernel.i8", "base_kernel.scale"}
        loaded |= AutoWeightsLoader(self).load_weights(items.items())
        return loaded

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "_kp_q"):
            flat = hidden_states.reshape(-1, hidden_states.shape[-1]).to(
                torch.float16
            )
            x_q, x_s = torch.ops.vllm.rocm_aiter_pertoken_quant_int8(flat)
            out = torch.ops.vllm.rocm_aiter_gemm_a8w8_ck(
                x_q, self._kp_q, x_s, self._kp_s, flat.dtype
            )
            return out.reshape(*hidden_states.shape[:-1], -1).to(
                self.base_kernel.dtype
            )
        return self.kernel_projection(hidden_states)

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        base = self.base_kernel[side]
        if getattr(self, "_bk_q", None) is not None:
            # E4 dequant-on-add: rebuild this side's plane from per-[side,tap]
            # int8 + fp32 scales instead of reading a bf16 base weight.
            base = (
                self._bk_q[side].to(torch.float32)
                * self._bk_s[side].unsqueeze(-1)
            ).to(delta.dtype)
        return _grouped_conv(
            hidden_states,
            delta,
            base,
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self._project(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self._audit_layer_idx = layer_idx
        draft_config = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            # Query tokens per request: the bonus token plus the mask tokens.
            block_size=1 + speculative_config.num_speculative_tokens,
            params_dtype=speculative_config.draft_model_config.dtype,
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv")
        )
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        dbg = os.environ.get("VLLM_SPEC_DEBUG_DUMP") and not (
            torch.cuda.is_current_stream_capturing()
        )

        def _dbg(tag, t):
            if dbg:
                print(
                    f"[SPEC-DBGA] {id(self) % 1000} {tag} "
                    f"nan={int(t.isnan().sum())}/{t.numel()} "
                    f"absmax={float(t.abs().max()):.3f}",
                    flush=True,
                )
            if os.environ.get("VLLM_DFLASH_AUDIT"):
                from vllm import quant_audit_recorder as _qa

                _qa.record_dflash_stage(
                    f"layer{self._audit_layer_idx}_{tag}", hidden=t
                )

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        _dbg("attn_conv_prep", hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        _dbg("attn_out", hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)
        _dbg("attn_conv_fin", hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        _dbg("mlp_conv_prep", hidden_states)
        hidden_states = self.mlp(hidden_states)
        _dbg("mlp_out", hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        _dbg("mlp_conv_fin", hidden_states)
        return hidden_states, residual


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )


@support_torch_compile
class CandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def process_weights_after_loading(self) -> None:
        """W8A8 for the hidden projection (codebooks stay float — audited).

        The 5120->256 projection runs once per drafted step; the codebooks
        it feeds are candidate-row gathers whose ranking drives acceptance
        and remain bf16 per the audit's explicit exception.
        """
        if os.environ.get("VLLM_GFX908_DF2_W8A8", "1") != "1":
            return
        w = self.hidden_projection.weight.data.float()
        wmax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        self._hp_s = (wmax / 127.0).to(torch.float32).contiguous()
        self._hp_q = (
            (w / self._hp_s).round_().clamp_(-127, 127).to(torch.int8).contiguous()
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self, "_hp_q"):
            flat = hidden_states.reshape(-1, hidden_states.shape[-1]).to(
                torch.float16
            )
            x_q, x_s = torch.ops.vllm.rocm_aiter_pertoken_quant_int8(flat)
            hidden = torch.ops.vllm.rocm_aiter_gemm_a8w8_ck(
                x_q, self._hp_q, x_s, self._hp_s, flat.dtype
            ).reshape(*hidden_states.shape[:-1], -1).to(
                self.predecessor_codebook.dtype
            )
        else:
            hidden = self.hidden_projection(hidden_states)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
        )


class DFlash2Qwen3Model(DFlashQwen3Model):
    decoder_layer_cls = DFlash2Qwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        draft_config = self.config.dflash_config
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )
        # Without its own tag the selector shares the draft head's compile cache.
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                params_dtype=vllm_config.speculative_config.draft_model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DFlash2Qwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        draft_config = self.config.dflash_config
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self.candidate_logits_processor = LogitsProcessor(
            vllm_config.model_config.get_vocab_size(),
            scale=float(draft_config.get("output_multiplier", 1.0)),
            soft_cap=softcap if softcap > 0 else None,
        )

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.candidate_selector.top_k
        )


EntryClass = DFlash2Qwen3ForCausalLM
