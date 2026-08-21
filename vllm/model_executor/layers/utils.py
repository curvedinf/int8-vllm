# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility methods for model layers."""

import os
from collections.abc import Callable

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

MOE_LAYER_ROUTER_GATE_SUFFIXES = {
    "gate",
    "router",
    "router_gate",
    "shared_expert_gate",
    "expert_gate",
}


def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros(
        (num_seqs, vocab_size + 1), dtype=torch.long, device=tokens.device
    )
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def apply_penalties(
    logits: torch.Tensor,
    prompt_tokens_tensor: torch.Tensor,
    output_tokens_tensor: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    """
    Applies penalties in place to the logits tensor
    logits : The input logits tensor of shape [num_seqs, vocab_size]
    prompt_tokens_tensor: A tensor containing the prompt tokens. The prompts
        are padded to the maximum prompt length within the batch using
        `vocab_size` as the padding value. The value `vocab_size` is used
        for padding because it does not correspond to any valid token ID
        in the vocabulary.
    output_tokens_tensor: The output tokens tensor.
    presence_penalties: The presence penalties of shape (num_seqs, )
    frequency_penalties: The frequency penalties of shape (num_seqs, )
    repetition_penalties: The repetition penalties of shape (num_seqs, )
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(
        prompt_tokens_tensor, vocab_size, num_seqs
    )
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs
    )

    # Apply repetition penalties as a custom op
    from vllm._custom_ops import apply_repetition_penalties

    apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return torch.nn.functional.linear(x, weight, bias)


def use_aiter_triton_gemm(n, m, k, dtype):
    if (
        not rocm_aiter_ops.is_triton_gemm_enabled()
        # MI300's - fp8nuz=True
        or current_platform.is_fp8_fnuz()
        or dtype not in [torch.float16, torch.bfloat16]
    ):
        return False

    # use hipblaslt for the larger GEMMs
    if n > 2048 and m > 512:
        return False
    return (
        (m == 5120 and k == 2880)
        or (m == 2880 and k == 4096)
        or (m == 128 and k == 2880)
        or (m == 640 and k == 2880)
        or (m == 2880 and k == 512)
        # Qwen3.6-35B-A3B lm_head: M=1, N=62080 — AITER 1.74x vs rocBLAS w/ BEST_CFG
        # (lm_head replicated across TP ranks, not column-split). Other unquantized
        # shapes in this model lose to rocBLAS at AITER's ~50μs floor — keep them off.
        or (m == 62080 and k == 2048)
    )


# gfx908 small-M tuning: AITER's default _get_config picks M_LEQ_64 for our shapes,
# which wastes blocks at M=1. This M=1 N≥1024 K=2048 config wins ~1.74x for lm_head.
_AITER_GEMM_M1_BEST_CFG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg",
    "NUM_KSPLIT": 1,
    "SPLITK_BLOCK_SIZE": 2048,
    "kpack": 1,
}


def rocm_unquantized_gemm_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    from vllm.platforms.rocm import on_gfx1x, on_gfx9, on_gfx950, on_gfx1250

    n = x.numel() // x.size(-1)
    m = weight.shape[0]
    k = weight.shape[1]

    cu_count = num_compute_units()

    # Next ^2 of n
    N_p2 = 1 << (n - 1).bit_length()
    # With 64 Ms per CU (each of 4 SIMDs working on a 16x16 tile),
    # and each working on a 512-shard of K, how many CUs would we need?
    rndup_cus = ((m + 64 - 1) // 64) * ((k + 512 - 1) // 512)
    # How many of 4 waves in a group can work on same 16 Ms at same time?
    # This reduces the Ms each group works on, i.e. increasing the number of CUs needed.
    GrpsShrB = min(N_p2 // 16, 4)
    # Given the above, how many CUs would we need?
    CuNeeded = rndup_cus * GrpsShrB
    # Deterministic reduction stores one float workspace value per K shard.
    fits_wvsplitkrc = (
        N_p2 * m * ((k + 512 - 1) // 512)
    ) <= 128 * 1024 * 12  # deterministic
    fits_wvsplitkrc &= CuNeeded <= cu_count

    skinny_operands_compatible = weight.is_contiguous() and (
        bias is None or bias.is_contiguous()
    )

    use_skinny_reduce_counting = (
        envs.VLLM_ROCM_USE_SKINNY_GEMM
        and on_gfx950()
        and x.dtype in [torch.float16, torch.bfloat16]
        and x.dim() == 2
        and (
            10 <= n <= 128
            and k % 8 == 0
            and k > 512
            and m % 16 == 0
            and fits_wvsplitkrc
            and skinny_operands_compatible
        )
    )

    if use_skinny_reduce_counting:
        x_view = x.reshape(-1, x.size(-1)).contiguous()
        return ops.wvSplitKrc(x_view, weight, cu_count, bias)

    # gfx1250's aiter gemm_a16w16 uses the gluon backend, which requires
    # K % 256 == 0 (it walks K with fixed-size descriptors and won't pad a
    # partial last tile). Some whitelisted shapes have K=2880 (e.g. gpt-oss-120b
    # hidden), so skip aiter there and fall back to the torch GEMM path below.
    if use_aiter_triton_gemm(n, m, k, x.dtype) and not (on_gfx1250() and k % 256 != 0):
        from aiter.ops.triton.gemm_a16w16 import gemm_a16w16

        if x.dtype != weight.dtype:
            x = x.to(weight.dtype)
        # gfx908: pass M=1 small-M config for lm_head shape (1.74x vs default config)
        cfg = _AITER_GEMM_M1_BEST_CFG if n == 1 and m == 62080 and k == 2048 else None
        if cfg is not None:
            return gemm_a16w16(x, weight, bias, config=cfg)
        return gemm_a16w16(x, weight, bias)

    use_skinny = (
        envs.VLLM_ROCM_USE_SKINNY_GEMM
        and (on_gfx9() or on_gfx1x())
        # build (gfx9/gfx11 ISA); fall back to torch GEMM there.
        # TODO GFX1250: Include once skinny GEMM is supported on gfx1250
        and x.dtype in [torch.float16, torch.bfloat16]
        and k % 8 == 0
        and skinny_operands_compatible
    )

    if use_skinny:
        # The skinny kernels assume contiguous K elements. A shape-preserving
        # reshape can retain a transposed activation's non-contiguous strides.
        x_view = x.reshape(-1, x.size(-1)).contiguous()
        if m > 8 and 0 < n <= 5:
            cu_count = num_compute_units()
            out = ops.wvSplitK(weight, x_view, cu_count, bias)
            return out.reshape(*x.shape[:-1], weight.shape[0])
        elif m % 4 == 0 and n == 1 and k <= 8192 and bias is None:
            out = ops.LLMM1(weight, x_view, 4)
            return out.reshape(*x.shape[:-1], weight.shape[0])

    if rocm_aiter_ops.is_tgemm_enabled():
        from aiter.tuned_gemm import tgemm

        return tgemm.mm(x, weight, bias)

    if x.dtype != weight.dtype:
        x = x.to(weight.dtype)
    return torch.nn.functional.linear(x, weight, bias)


def rocm_unquantized_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return weight.new_empty((*x.shape[:-1], weight.shape[0]))


def rocm_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.rocm_unquantized_gemm(x, weight, bias)


direct_register_custom_op(
    op_name="rocm_unquantized_gemm",
    op_func=rocm_unquantized_gemm_impl,
    fake_impl=rocm_unquantized_gemm_fake,
)


# Above this weight size, oneDNN's onednn_mm consistently matches or beats
# the SGL AMX kernel once M grows past decode-sized batches, and is within
# noise of it at decode-sized M -- so larger weights default to oneDNN
# rather than SGL. 1 MiB comfortably covers MoE router/gate weights (e.g.
# (2048, 128) .. (2880, 32) bf16/fp16, 180-720 KiB) while staying well below
# any dense qkv/o_proj/gate_up/down/lm_head projection in practice. This
# threshold is derived from bf16/fp16 unquantized dense-GEMM benchmarks only,
# so it does not apply to the int8 scaled_mm path below.
_CPU_SGL_GEMM_MAX_WEIGHT_BYTES = 1 * 1024 * 1024


def check_cpu_sgl_kernel(n: int, k: int, dtype: torch.dtype) -> bool:
    if not torch.cpu._is_amx_tile_supported() or dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.int8,
    ):
        return False
    if dtype == torch.float16 and not torch.cpu._is_amx_fp16_supported():
        # AMX-BF16/INT8 (amx_tile) and AMX-FP16 are separate CPU ISA
        # extensions -- e.g. Sapphire/Emerald Rapids expose the former but
        # not the latter -- and can_use_brgemm<at::Half> (gemm.h) always
        # attempts brgemm for fp16 regardless of M, so this needs its own
        # capability check rather than piggybacking on amx_tile.
        return False
    if dtype == torch.int8:
        # int8_scaled_mm_with_quant requires the packed weight to stay int8
        # (gemm_int8.cpp); convert_weight_packed's N < TILE_N fallback
        # returns a float32 tensor instead (gemm.cpp), which would trip
        # that check, so N must be a full TILE_N tile here.
        return k % 32 == 0 and n % 16 == 0
    if n * k * dtype.itemsize > _CPU_SGL_GEMM_MAX_WEIGHT_BYTES:
        return False
    if n < 16:
        # convert_weight_packed transposes to fp32 instead of VNNI-packing
        # when N < TILE_N (gemm.cpp), and weight_packed_linear detects that
        # (via the packed weight's dtype) and routes to its fp32/brgemm
        # fallback kernel -- no N/K alignment required in that regime.
        return True
    return k % 32 == 0 and n % 16 == 0


def dispatch_cpu_unquantized_gemm(
    layer: torch.nn.Module,
    remove_weight: bool,
) -> None:
    # skip for missing layers
    if layer.weight.is_meta:
        layer.cpu_linear = torch.nn.functional.linear
        return

    # Skip CPU GEMM dispatch for non-2D weights (e.g. MoE 3D expert weights).
    # These layers are handled by their own specialized methods.
    if layer.weight.ndim != 2:
        # this is not a linear layer
        # For now it should be a causal_conv1d op or MoE 3D expert weights
        # The C++ causal_conv1d kernels use VDPBF16PS (no AMX tiles), so the
        # VNNI weight prepack applies to any AVX-512BF16 CPU, not just AMX
        # (e.g. AMD Zen5/Turin).
        if torch.cpu._is_avx512_bf16_supported() and hasattr(
            ops, "causal_conv1d_weight_pack"
        ):
            # prepack conv weight
            unpacked = (
                layer.weight.view(
                    layer.weight.size(0),
                    layer.weight.size(2),
                )
                .contiguous()
                .clone()
            )
            # Stash the un-packed (dim, width) weight so the speculative-decode
            # GDN path (which uses torch conv, not the C++ kernel) can use it.
            layer._cpu_unpacked_conv_weight = unpacked
            layer.weight.data = ops.causal_conv1d_weight_pack(unpacked)
        return

    N, K = layer.weight.size()
    dtype = layer.weight.dtype

    # Zen CPU path: zentorch_linear_unary with optional eager weight prepacking.
    if current_platform.is_zen_cpu() and hasattr(
        torch.ops.zentorch, "zentorch_linear_unary"
    ):
        zen_weight = layer.weight.detach()
        is_prepacked = False

        if envs.VLLM_ZENTORCH_WEIGHT_PREPACK and hasattr(
            torch.ops.zentorch, "zentorch_weight_prepack_for_linear"
        ):
            zen_weight = torch.ops.zentorch.zentorch_weight_prepack_for_linear(
                zen_weight
            )
            is_prepacked = True

        layer.cpu_linear = lambda x, weight, bias, _p=is_prepacked: (
            torch.ops.zentorch.zentorch_linear_unary(
                x, zen_weight, bias, is_weight_prepacked=_p
            )
        )
        if remove_weight:
            layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        logger.debug_once(
            "CPU unquantized GEMM dispatch: using zentorch_linear_unary (prepacked=%s)",
            is_prepacked,
        )
        return

    # Small weights (e.g. MoE router/gate projections, where N is the expert
    # count rather than a hidden-size-scaled dimension) never reach oneDNN's
    # compute-bound regime, no matter how large the batch gets: SGL's lower
    # per-call dispatch overhead wins consistently across the full measured
    # M range. Larger dense projections (qkv/o_proj/gate_up/down/lm_head)
    # cross over to favoring oneDNN once batch size grows past decode-sized
    # M, so they keep using oneDNN below.
    if check_cpu_sgl_kernel(N, K, dtype):
        packed_weight = torch.ops._C.convert_weight_packed(layer.weight)
        if getattr(layer, "bias", None) is not None:
            bias_f32 = layer.bias.to(torch.float32)
        else:
            bias_f32 = None
        layer.cpu_linear = lambda x, weight, bias: torch.ops._C.weight_packed_linear(
            x, packed_weight, bias_f32 if bias is not None else None, True
        )
        if remove_weight:
            layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        logger.debug_once(
            "CPU unquantized GEMM dispatch: using sgl-kernel weight_packed_linear"
        )
        return

    if (
        ops._supports_onednn
        and current_platform.get_cpu_architecture() != CpuArchEnum.POWERPC
    ):
        try:
            origin_weight = layer.weight
            handler = ops.create_onednn_mm(origin_weight.t(), 32)
            layer.cpu_linear = lambda x, weight, bias: ops.onednn_mm(handler, x, bias)
            if remove_weight:
                layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
            logger.debug_once("CPU unquantized GEMM dispatch: using oneDNN onednn_mm")
            return
        except RuntimeError as e:
            logger.warning_once(
                "Failed to create oneDNN linear, fallback to torch linear."
                f" Exception: {e}"
            )

    # fallback case
    layer.cpu_linear = lambda x, weight, bias: torch.nn.functional.linear(
        x, weight, bias
    )
    logger.debug_once(
        "CPU unquantized GEMM dispatch: using torch.nn.functional.linear (fallback)"
    )


def cpu_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return layer.cpu_linear(x, weight, bias)


def rocm_unquantized_gemm_gfx908_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """gfx908 dispatch implementation (registered as torch custom op below).

    Routes through a `direct_register_custom_op` wrapper so torch.compile /
    inductor sees one opaque graph node and does NOT inline this Python down
    to `aten::mm` (rocBLAS). That ensures QKV/QKVZ/o_proj inside compiled model
    forwards still hit our LLMM1/wvSplitK dispatch, not just the eager-path
    MoE block / lm_head.

    Dispatch priority on gfx908 for fp16/bf16 weight (k % 8 == 0):
      1. LLMM1   for n==1, m % 4 == 0, k <= 8192, bias is None (fastest at M=1)
      2. wvSplitK for m > 8 and 0 < n <= 4 (skinny-M decode)
      3. AITER gemm_a16w16 for whitelisted lm_head shape (Stage 2)
      4. F.linear fallback (rocBLAS for M >= 8)

    Microbench (gfx908 single-GPU, 2026-04-24):
      - LLMM1   2.1-2.7x faster than rocBLAS for our M=1 hot shapes
      - wvSplitK 2.1-6.7x faster than rocBLAS (wins on (1, 1, 2048))
      - LLMM1 also beats AITER for (1, 62080, 2048) lm_head: 244us vs 267us
    """
    n = x.numel() // x.size(-1)
    m = weight.shape[0]
    k = weight.shape[1]
    debug = os.environ.get("VLLM_GFX908_DEBUG_DISPATCH") == "1"

    # Skinny GEMM dispatch (PR adapted from larkinwc/vllm-gfx908#4 microbench).
    # Required conditions for both: weight.is_contiguous() (LLMM1/wvSplitK
    # assume contiguous), fp16/bf16 dtype, k % 8 == 0 (vectorized loads).
    skinny_ok = (
        x.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype in (torch.float16, torch.bfloat16)
        and k % 8 == 0
        and weight.is_contiguous()
    )
    if skinny_ok:
        x_view = x.reshape(-1, x.size(-1))
        if n == 1 and m % 4 == 0 and k <= 8192 and bias is None:
            if debug:
                import sys as _sys
                print(f"[LLMM1] n={n} m={m} k={k}",
                      file=_sys.stderr, flush=True)
            if x.dtype != weight.dtype:
                x_view = x_view.to(weight.dtype)
            out = ops.LLMM1(weight, x_view, 4)
            return out.reshape(*x.shape[:-1], weight.shape[0])
        if m > 8 and 0 < n <= 4:
            if debug:
                import sys as _sys
                print(f"[wvSplitK] n={n} m={m} k={k}",
                      file=_sys.stderr, flush=True)
            cu_count = num_compute_units()
            if x.dtype != weight.dtype:
                x_view = x_view.to(weight.dtype)
            out = ops.wvSplitK(weight, x_view, cu_count, bias)
            return out.reshape(*x.shape[:-1], weight.shape[0])

    if use_aiter_triton_gemm(n, m, k, x.dtype):
        from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
        if debug:
            import sys as _sys
            print(f"[AITER_DISPATCH] n={n} m={m} k={k} dtype={x.dtype}",
                  file=_sys.stderr, flush=True)
        if x.dtype != weight.dtype:
            x = x.to(weight.dtype)
        cfg = _AITER_GEMM_M1_BEST_CFG if n == 1 and m == 62080 and k == 2048 else None
        if cfg is not None:
            return gemm_a16w16(x, weight, bias, config=cfg)
        return gemm_a16w16(x, weight, bias)
    # rocBLAS fallback for M >= 8 (no skinny-GEMM applies). Now reachable
    # from inside compiled forwards too (Stage 5h custom-op wrapper below).
    return torch.nn.functional.linear(x, weight, bias)


def rocm_unquantized_gemm_gfx908(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """gfx908 dispatch — torch custom op wrapper.

    Routes through `torch.ops.vllm.rocm_unquantized_gemm_gfx908` so inductor
    treats it as a single opaque graph node. This is what makes the LLMM1 /
    wvSplitK dispatch fire for QKV/QKVZ/o_proj inside compiled model forwards
    (round-3 Stage 5h). Without this wrapping, inductor inlines our Python and
    lowers the trailing `F.linear` straight to `aten::mm` → rocBLAS, bypassing
    the dispatch.

    Note: round-4 lever K (hoisting the n>4 check up here so inductor could
    inline F.linear at high concurrency) was attempted on 2026-04-26 and
    REGRESSED c=1 by 17% — the Python-level shape conditional broke the
    inductor trace and forced eager Python execution on every GEMM call
    (~10 µs × ~129 calls/token ≈ ~1.3 ms/token overhead). Reverted. The
    c=64 (-4%) / c=128 (-9.3%) regression vs round-2 stands; recovery would
    require per-layer closure binding (decide dispatch fn at layer init based
    on weight.shape, no runtime conditional), which is invasive and doesn't
    help the c=1 throughput target. See docs/mi100_decode_opt/round4_candidates.md
    lever K for the full post-mortem.
    """
    return torch.ops.vllm.rocm_unquantized_gemm_gfx908(x, weight, bias)


def _gfx908_weight_can_use_custom_gemm(weight: torch.Tensor) -> bool:
    if weight.is_meta or weight.dim() != 2:
        return True

    m = weight.shape[0]
    k = weight.shape[1]
    dtype = weight.dtype

    if dtype not in (torch.float16, torch.bfloat16):
        return False
    if k % 8 != 0 or not weight.is_contiguous():
        return False

    # LLMM1 can fire for n==1 with no bias. wvSplitK can fire for n<=4.
    # Keep these potentially-fast shapes on the opaque custom op so compiled
    # decode forwards still reach the runtime skinny-kernel dispatch.
    if (m % 4 == 0 and k <= 8192) or m > 8:
        return True

    # AITER eligibility still depends on runtime n for the large-GEMM cutoff,
    # but if n==1 can use it then this layer is a possible fast path.
    return use_aiter_triton_gemm(1, m, k, dtype)


def bind_rocm_unquantized_gemm_gfx908(layer: torch.nn.Module) -> None:
    """Pre-bind the gfx908 unquantized GEMM route for a loaded layer.

    The dynamic n>4 wrapper hoist regressed because it put shape conditionals
    in the traced forward path. This binding keeps the decision static: layers
    with no possible LLMM1/wvSplitK/AITER route use direct F.linear, while
    possible fast-kernel layers keep the existing opaque custom op.
    """
    if _gfx908_weight_can_use_custom_gemm(layer.weight):
        layer._vllm_unquantized_gemm = rocm_unquantized_gemm_gfx908
    else:
        layer._vllm_unquantized_gemm = default_unquantized_gemm


direct_register_custom_op(
    op_name="rocm_unquantized_gemm_gfx908",
    op_func=rocm_unquantized_gemm_gfx908_impl,
    fake_impl=rocm_unquantized_gemm_fake,  # output shape == weight.new_empty((*x.shape[:-1], weight.shape[0]))
)


def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx908
        if on_gfx908():
            return rocm_unquantized_gemm_gfx908
        return rocm_unquantized_gemm
    elif current_platform.is_cpu():
        return cpu_unquantized_gemm
    else:
        return default_unquantized_gemm
