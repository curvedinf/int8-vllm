# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with PagedAttention and Triton prefix prefill."""

from typing import ClassVar

from dataclasses import replace

import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.utils.torch_utils import get_dtype_size, is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionLayer, AttentionType, MultipleOf
from vllm.v1.attention.backends.rocm_attn import (
    RocmAttentionBackend,
    RocmAttentionImpl,
    RocmAttentionMetadata,
    RocmAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_per_token_head_quant,
)
from vllm.v1.kv_cache_interface import (
    KVQuantMode,
    get_kv_quant_mode,
    kv_cache_uses_per_token_head_scales,
)

logger = init_logger(__name__)


class RocmAiterUnifiedAttentionBackend(RocmAttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "int8_per_token_head",
    ]

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # Mirrors TritonAttentionBackend: the UA kernels read the cache
        # through the tensor's own strides (see _ensure_scale_caches /
        # get_kv_cache_shape — logical (B, H, N, 2*hs) with num_blocks as the
        # outermost physical dim), so declaring the layout here enables
        # indexes_kv_by_block_stride and with it page-size padding for mixed
        # spec-decode KV specs (int8-PTH target + fp16 draft).
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            return (1, 0, 3, 2, 4)
        elif cache_layout == "NHD":
            return (0, 2, 1, 3)
        elif cache_layout == "HND" and include_num_layers_dimension:
            return (1, 2, 0, 3, 4)
        return (0, 1, 2, 3)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return 64

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_UNIFIED_ATTN"

    @staticmethod
    def get_impl_cls() -> type["RocmAiterUnifiedAttentionImpl"]:
        return RocmAiterUnifiedAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        # K and V are packed into the content dim: logical (B, H, N, 2*hs).
        if kv_cache_uses_per_token_head_scales(cache_dtype_str):
            # Pad each half of the content dim by
            # sizeof(float32)/sizeof(cache_dtype) so the per-(token, head)
            # scale fits inline after the quantized data (mirrors
            # TritonAttentionBackend; see _ensure_scale_caches).
            from vllm.utils.torch_utils import (
                STR_DTYPE_TO_TORCH_DTYPE,
                get_dtype_size as _get_dtype_size,
            )

            cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype_str]
            scale_pad = _get_dtype_size(torch.float32) // _get_dtype_size(
                cache_dtype
            )
            return (num_blocks, num_kv_heads, block_size, 2 * (head_size + scale_pad))
        return (num_blocks, num_kv_heads, block_size, 2 * head_size)

    @classmethod
    def customize_spec(cls, spec: "AttentionSpec") -> "AttentionSpec":
        """Per-token-head modes pack inline fp32 scales after each half's
        data, so the content is (K data + K scale + V data + V scale)."""
        mode = spec.kv_quant_mode
        if spec.state_content_bytes is not None or not mode.is_per_token_head:
            return spec
        hs_k, hs_v = spec.head_size, spec.head_size_v
        if mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            hs_k, hs_v = hs_k // 2, hs_v // 2
        scale_bytes = get_dtype_size(torch.float32)
        content = (hs_k + hs_v) * get_dtype_size(spec.dtype) + 2 * scale_bytes
        return replace(spec, state_content_bytes=content)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["RocmAttentionMetadataBuilder"]:
        return RocmAttentionMetadataBuilder

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """RocmAiterUnifiedAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )


class RocmAiterUnifiedAttentionImpl(RocmAttentionImpl):
    # Per-token-head quant: scale views carved from inline head padding.
    _k_scale_cache: torch.Tensor | None = None
    _v_scale_cache: torch.Tensor | None = None
    _k_data_cache: torch.Tensor | None = None
    _v_data_cache: torch.Tensor | None = None

    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

    def _ensure_scale_caches(self, kv_cache: torch.Tensor) -> None:
        """Extract per-head scale views from the padded content dimension.

        The KV cache is packed as logical shape
        ``(num_blocks, nkv, block_size, 2 * (hs + pad))`` where
        ``pad = sizeof(float32) / sizeof(cache_dtype)``.  The content dim holds
        ``[K(hs) | K_scale(pad) | V(hs) | V_scale(pad)]`` per (head, slot); the
        last ``pad`` elements of each half hold one float32 scale.  We create
        strided float32 views over those bytes.  ``kv_cache`` must be the
        packed logical tensor (call before any transpose), but may have HND or
        NHD physical strides.

        Scale shape: ``(num_blocks, block_size, num_kv_heads)``
        """
        if self._k_scale_cache is not None:
            return

        num_blocks, nkv, block_size, content = kv_cache.shape
        dtype_sz = kv_cache.element_size()
        scale_pad = get_dtype_size(torch.float32) // dtype_sz  # e.g. 4
        padded_hs = content // 2
        hs = padded_hs - scale_pad

        raw = kv_cache.untyped_storage()
        base_f32 = torch.tensor([], dtype=torch.float32, device=kv_cache.device).set_(
            raw
        )

        def to_f32_units(elements: int) -> int:
            nbytes = elements * dtype_sz
            assert nbytes % 4 == 0
            return nbytes // 4

        # Actual strides (in float32 units) from the tensor. The logical cache
        # may be physically NHD, so do not assume C-contiguous HND layout.
        strides = kv_cache.stride()
        block_f32 = to_f32_units(strides[0])
        head_f32 = to_f32_units(strides[1])
        slot_f32 = to_f32_units(strides[2])
        # Scale sits at byte offset hs within each (K, then V) content half.
        base_off_f32 = to_f32_units(kv_cache.storage_offset())
        k_scale_off_f32 = base_off_f32 + to_f32_units(hs)
        v_scale_off_f32 = base_off_f32 + to_f32_units(padded_hs + hs)

        # K scales (first content half)
        self._k_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(block_f32, slot_f32, head_f32),
            storage_offset=k_scale_off_f32,
        )
        self._k_scale_cache.fill_(1.0)

        # V scales (second content half)
        self._v_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(block_f32, slot_f32, head_f32),
            storage_offset=v_scale_off_f32,
        )
        self._v_scale_cache.fill_(1.0)

        key_cache, value_cache = self._split_kv_cache(kv_cache)
        self._k_data_cache = key_cache[..., :hs]
        self._v_data_cache = value_cache[..., :hs]

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks,
        )
        logger.info_once(
            "Using aiter unified attention for RocmAiterUnifiedAttentionImpl"
        )
        from aiter.ops.triton.unified_attention import unified_attention

        self.unified_attention = unified_attention
        self.supports_quant_query_input = True

        self._kv_quant_mode = get_kv_quant_mode(kv_cache_dtype)
        self._is_per_token_head_quant = self._kv_quant_mode.is_per_token_head

    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs)).  Split on the
        # actual content dim: per-token-head modes pad each half, so the
        # content is 2 * (hs + scale_pad), not 2 * hs.
        padded_hs = kv_cache.shape[-1] // 2
        return kv_cache.transpose(1, 2).split(padded_hs, dim=-1)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for RocmAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        key_cache, value_cache = self._split_kv_cache(kv_cache)

        softmax_scale = self.scale
        if self._is_per_token_head_quant:
            self._ensure_scale_caches(kv_cache)
            key_cache = self._k_data_cache
            value_cache = self._v_data_cache
            q_descale = None
            k_descale = None
            v_descale = None
            k_scale_cache = self._k_scale_cache
            v_scale_cache = self._v_scale_cache
        elif is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
            q_descale = layer._q_scale if query.dtype == self.fp8_dtype else None
            k_descale = layer._k_scale
            v_descale = layer._v_scale
            k_scale_cache = None
            v_scale_cache = None
        else:
            q_descale = None
            k_descale = None
            v_descale = None
            k_scale_cache = None
            v_scale_cache = None

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        if attn_metadata.causal:
            self.unified_attention(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                sinks=self.sinks,
                output_scale=output_scale,
                k_scale_cache=k_scale_cache,
                v_scale_cache=v_scale_cache,
            )
        else:
            # The aiter kernel is causal-only. Non-causal cross-attention
            # (ENCODER_DECODER, e.g. Whisper) falls back to the vLLM Triton
            # unified kernel, which shares this layout and honors the flag.
            from vllm.v1.attention.ops.triton_unified_attention import (
                unified_attention as triton_unified_attention,
            )

            descale_shape = (cu_seqlens_q.shape[0] - 1, key_cache.shape[2])
            triton_unified_attention(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                # Without this the kernel defaults to KVQuantMode.NONE and
                # reads an int8-PTH cache as raw bytes (no scale load) —
                # the DFlash2 noncausal draft collapses to garbage scores.
                kv_quant_mode=self._kv_quant_mode,
                q_descale=q_descale,
                k_descale=(
                    k_descale.expand(descale_shape)
                    if k_descale is not None
                    else None
                ),
                v_descale=(
                    v_descale.expand(descale_shape)
                    if v_descale is not None
                    else None
                ),
                sinks=self.sinks,
                output_scale=output_scale,
                k_scale_cache=k_scale_cache,
                v_scale_cache=v_scale_cache,
            )

        return output


    def _kv_readback_check(self, key, value, key_cache, value_cache, slot_mapping):
        """Reference-quantize K/V, snapshot target slots before the kernel,
        run the kernel, read back, compare. Logs KVREADBACK lines."""
        import torch as _t
        valid = slot_mapping >= 0
        slots = slot_mapping[valid]
        if slots.numel() == 0:
            return
        # Deduplicate slots keeping the LAST occurrence: spec-decode
        # rewrites rejected slots in the same call (last write wins), so a
        # first-occurrence reference would false-positive on every rewrite.
        sm = slot_mapping.clone()
        seen = {}
        order = []
        for i, sv in enumerate(sm.tolist()):
            if sv < 0:
                continue
            if sv in seen:
                order[seen[sv]] = -1
            seen[sv] = len(order)
            order.append(i)
        keep_idx = [i for i in order if i >= 0]
        if not keep_idx:
            self._rb_pending = None
            return
        keep = _t.tensor(keep_idx, device=sm.device, dtype=_t.long)
        n, nkv, hs = key.shape
        pad = key_cache.shape[-1] - hs  # inline scale slots
        def quant_ref(x):
            sc = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127.0
            q = _t.clamp((x.float() / sc).round(), -127, 127).to(_t.int8)
            return q, sc.squeeze(-1)
        keyv = key[keep]
        valuev = value[keep]
        slots = sm[keep]
        kq, ksc = quant_ref(keyv)
        vq, vsc = quant_ref(valuev)
        bs = key_cache.shape[1]
        rows = (slots // bs).long()
        cols = (slots % bs).long()
        # snapshot pre-write contents at those slots (first head only)
        pre_k = key_cache[rows, cols, 0, :].clone()
        self._rb_counter = getattr(self, "_rb_counter", 0) + 1
        idx = getattr(self, "_rb_idx", None)
        if idx is None:
            idx = _t.arange(slots.numel(), device=slots.device)
            self._rb_idx = idx
        # stash reference for post-kernel compare (kernel runs after return)
        self._rb_pending = (rows, cols, kq, ksc, vq, vsc, slots, pre_k, key_cache)

    def _kv_readback_finalize(self):
        """Compare the cache against the stashed reference (call after the
        kernel has run)."""
        import torch as _t
        p = getattr(self, "_rb_pending", None)
        if p is None:
            return
        rows, cols, kq, ksc, vq, vsc, slots, pre_k, key_cache = p
        self._rb_pending = None
        got_k = key_cache[rows, cols, 0, :kq.shape[-1]]
        ref_k = kq[:, 0, :]
        # STALE-DETECTOR: keep the previous step's reference for the same
        # impl instance; a mismatch that equals LAST step's K at this slot
        # means the write never landed (stale), vs unrelated bytes.
        prev = getattr(self, "_rb_prev", None)  # {slot: ref bytes}
        diff = (got_k.int() - ref_k.int()).abs()
        # Tolerate off-by-one rounding-mode differences (kernel rounds
        # half-to-even; torch.round is half-away).
        mism = (diff > 1).any(dim=-1)
        n_mism = int(mism.sum())
        if n_mism:
            worst = int(diff.max())
            stale = 0
            if prev is not None:
                for j in mism.nonzero().flatten().tolist():
                    sv = int(slots[j])
                    if sv in prev and (got_k[j].int() - prev[sv]).abs().max() <= 1:
                        stale += 1
            logger.warning(
                "KVREADBACK MISMATCH layer=%s mismatches=%d/%d worst=%d stale_prev=%d slots=%s",
                getattr(self, "_rb_layer", "?"), n_mism, slots.numel(),
                worst, stale, slots[mism][:6].tolist())
        new_prev = {int(sv): got_k[j].clone()
                    for j, sv in enumerate(slots.tolist())}
        self._rb_prev = new_prev

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        key_cache, value_cache = self._split_kv_cache(kv_cache)

        if self._is_per_token_head_quant:
            self._ensure_scale_caches(kv_cache)
            # Pass the padded halves: the kernel writes head_size data
            # elements plus the inline scale at offset head_size within
            # each half (mirrors TritonAttentionBackend).
            import os as _os
            if _os.environ.get("VLLM_KV_READBACK") and not torch.cuda.is_current_stream_capturing():
                # Write-then-read-back audit: quantize the incoming K/V with
                # a torch reference, run the real kernel, then re-read the
                # written slots and compare. Any mismatch is direct evidence
                # the cache holds different bytes than the kernel was given
                # (the temp-1.0 garble's last unobserved surface).
                try:
                    self._kv_readback_check(
                        key, value, key_cache, value_cache, slot_mapping
                    )
                except Exception as e:
                    logger.warning("KV-READBACK check failed: %s", e)
            from vllm import quant_audit_recorder as _qa

            if _qa._enabled() and not torch.cuda.is_current_stream_capturing():
                _qa.record_kv(
                    getattr(layer, "layer_name", "kv"),
                    key[:64].float(),
                    value[:64].float(),
                    None,
                    None,
                    self._k_scale_cache,
                    self._v_scale_cache,
                )
            triton_reshape_and_cache_flash_per_token_head_quant(
                key,
                value,
                key_cache,
                value_cache,
                self._k_scale_cache,
                self._v_scale_cache,
                slot_mapping,
                kv_quant_mode=self._kv_quant_mode,
            )
            if getattr(self, "_rb_pending", None) is not None:
                self._kv_readback_finalize()
            return

        # Reshape the input keys and values and store them in the cache.
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        if self._is_per_token_head_quant:
            return False
        return rocm_aiter_ops.is_enabled()

    def fused_qk_norm_rope_kvcache_supported(self):
        if self._is_per_token_head_quant:
            # The fused op writes unquantized fp16 K into the cache.
            return False
        return rocm_aiter_ops.is_enabled()

    def do_qk_norm_rope_kvcache_update(
        self,
        layer: AttentionLayer,
        qkv: torch.Tensor,
        q_out: torch.Tensor,
        k_out: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        rms_norm_eps: float,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        rocm_aiter_ops.do_qk_norm_rope_kvcache_update(
            qkv=qkv,
            q_weight=q_weight,
            k_weight=k_weight,
            cos_sin_cache=cos_sin_cache,
            positions=positions,
            num_heads_q=self.num_heads,
            num_heads_k=self.num_kv_heads,
            head_dim=self.head_size,
            is_neox=is_neox,
            rms_norm_eps=rms_norm_eps,
            q_out=q_out,
            k_out=k_out,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=layer_slot_mapping,
            k_scale=layer._k_scale_cpu,
            v_scale=layer._v_scale_cpu,
            kv_cache_dtype=self.kv_cache_dtype,
            use_shuffle_layout=False,
        )

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        flash_layout = True

        is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
