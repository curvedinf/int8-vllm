# INT8 Serving Audit Results

## Scope

This audit covers the canonical gfx908 TP4/C8 serving recipe for:

- target: `curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128`
- draft: `curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128`

The ranking below is based on expected end-to-end performance impact on gfx908
versus likely quality loss. It is a static dataflow and kernel audit, not a
benchmark result.

The target has 64 layers: 48 GatedDeltaNet linear-attention layers and 16 full
attention layers. The DFlash2 draft has five noncausal sliding-attention layers.

## Changes worth making, ordered by expected effect

### 1. Wire AITER fused all-reduce + RMSNorm + GS128 INT8 quant-out into the W8A8 consumer

**Expected effect: very large. Quality cost: no additional intended loss beyond
the existing W8A8 activation quantization.**

AITER already exposes `fused_ar_rms_int8_per_group_quant()`. Its useful contract
is:

- FP16/BF16 input partials and residual
- FP16/BF16 residual output
- INT8 normalized activation output
- FP16 per-group scales

This is the correct boundary at which to quantize: the persistent residual stays
floating point, while the normalized activation is emitted directly in the
GS128 format consumed by the following AITER W8A8 GEMM. Wiring this through the
vLLM fusion pass can eliminate separate all-reduce, residual-add, RMSNorm,
activation-quantization, global-memory traffic, and kernel launches repeatedly
throughout all 64 target layers.

The current vLLM wrapper and fusion patterns call the FP8-oriented
`fused_ar_rms_per_group_quant()` path. No vLLM call site currently references
`fused_ar_rms_int8_per_group_quant()`.

**Important:** AITER's existing INT8 function does not make the collective
transport INT8. It receives floating-point partials, performs the all-reduce in
that floating-point representation, and quantizes the normalized output after
the collective. Documentation must not describe this as "INT8 all-reduce."

### 2. Quantize the target `lm_head` to GS128 W8A8 INT8

**Expected effect: large, potentially very large during speculative
verification. Quality cost: modest but more direct than internal-layer
quantization.**

The target is untied and stores both `model.language_model.embed_tokens.weight`
and `lm_head.weight` as BF16 tensors of shape `[248320, 5120]`. The BF16
`lm_head` is about 2.54 GB globally, or roughly 635 MB of weight traffic per TP4
rank whenever the vocabulary projection is evaluated. Speculative verification
scores several positions at once.

The model's GPTQ metadata has `lm_head: false`, so the current artifact cannot
run this projection through the intended AITER GS128 W8A8 path. Fixing it
requires either a requantized model artifact or a deliberate runtime-quantized
head implementation; it is not achievable through the serving launcher alone.

### 3. Add a gfx908-optimized noncausal DFlash2 attention path

**Expected effect: medium to large. Quality cost: none intended if masking and
INT8 numerics are preserved.**

The DFlash2 draft declares `is_causal: false` and uses a 2048-token sliding
window. The current ROCm AITER unified-attention backend is causal-only, so the
draft falls back to vLLM's Triton unified-attention implementation. A noncausal
gfx908 path retaining `int8_per_token_head` KV and INT8 QxK would accelerate all
five draft layers without intentionally changing model behavior.

### 4. Pin the gfx908 INT8 embedding path explicitly

**Expected effect: negligible additional performance with current defaults;
operational value only.**

`VLLM_GFX908_INT8_EMBEDDING` currently defaults to true. The canonical launcher
should nevertheless set it explicitly to `1` so an environment or default
change cannot silently restore BF16 embedding storage/gathers. This is a
reproducibility guard, not a new optimization while the current default holds.

## Changes that should not be made

### Do not set the GDN convolution cache to INT8 with the current kernel

Keep `mamba_cache_dtype` at the model dtype. Do not add
`--mamba-cache-dtype int8` to the production recipe.

The current causal-convolution update performs `x = x.to(conv_state.dtype)`.
There is no scale tensor or INT8-specific reconstruction. Selecting an INT8
convolution state would round the incoming continuous activation directly to
integer values before convolution. That is uncontrolled quantization rather
than a scaled INT8 cache format.

The benefit is also limited relative to the risk. For TP4, C8, seven speculative
tokens, 48 GDN layers, convolution width four, and the target's dimensions, the
FP16 convolution caches are only about 20 MB per GPU.

### Do not use bare INT8 for the GDN recurrent state

The canonical launcher currently passes:

```text
--mamba-ssm-cache-dtype int8
```

This should return to FP16 unless a scaled INT8 GDN recurrent-state kernel is
added. The current branch contains no per-head or per-block recurrent-state
scale and no INT8-specific GDN recurrence path. The Triton recurrence loads and
stores the state using its declared dtype, so bare INT8 turns a continuous
recurrent matrix into integer-valued state.

The theoretical bandwidth opportunity is meaningful: the target recurrent
state is approximately 151 MB per GPU in FP16 at C8, and the state is accessed
continually during decode. However, without scales, the quality loss is
uncontrolled and fundamentally different from W8A8 GEMM quantization. A future
scaled INT8 format may be worth implementing, but the current flag is not that
format.

### Keep the GDN depthwise convolution weights and arithmetic floating point

`QwenGatedDeltaNetAttention.conv1d` is instantiated without `quant_config`, so
its weights and convolution remain floating point. This is appropriate. The
convolution width is four, its weight tensor is small, and it is not a large
dense GEMM. A special INT8 depthwise-convolution kernel plus activation scales
would provide little savings and would inject error before every recurrent GDN
update.

The large GDN projections `in_proj_qkvz`, `in_proj_ba`, and `out_proj` do receive
`quant_config` and are the appropriate operations to route through AITER W8A8.

### Keep `A_log`, `dt_bias`, decay, and gating calculations in FP32

These values control recurrent decay and update strength, so their errors
accumulate through time. They contain very little data and contribute
negligible bandwidth or arithmetic compared with the projections and recurrent
state. Their quality value is high and their performance cost is negligible.

### Keep softmax and attention probability accumulation floating point

Keep softmax in FP32 or with FP32 accumulation, probability-times-value in
FP16/BF16, and the attention output in FP16/BF16. INT8 QxK and INT8 KV capture
the useful bandwidth and dot-product reductions. Quantizing probabilities or
PxV carries much greater quality risk without a compelling gfx908 kernel
advantage.

### Keep the residual stream and nonlinear/norm state floating point

Residuals, RMSNorm arithmetic, SiLU, RoPE, gated outputs, and attention outputs
should remain FP16/BF16. Fuse them aggressively, but quantize only the normalized
activation immediately consumed by a W8A8 GEMM. Persisting the residual stream
as INT8 would repeatedly requantize accumulated information across 64 layers.

### Keep KV scale metadata in FP32

`int8_per_token_head` stores INT8 K/V contents with FP32 per-token/per-head
scales. Two FP32 scales are tiny compared with the 256 INT8 K/V elements they
describe. Reducing scale precision saves essentially nothing while directly
worsening reconstruction accuracy.

### Keep the DFlash2 candidate codebooks and selector floating point

The predecessor and successor codebooks are each BF16 `[248320, 256]`, but the
selector gathers only the rows associated with candidate IDs rather than
scanning the vocabulary. Quantizing them could perturb path ranking and draft
acceptance for little runtime gain. The selector's `5120 -> 256` hidden
projection is also too small to be an important decode bottleneck.

### Do not claim genuine INT8 collective transport without a new scaled implementation

A real INT8 all-reduce could reduce TP transport volume, but each rank's partial
output requires its own scales. A correct implementation must communicate those
scales, dequantize differently scaled rank contributions, accumulate safely,
and preserve a floating-point residual. A bare integer sum would combine
incompatible scales and can overflow.

Treat genuine scaled INT8 transport as experimental kernel work until its
numerics and acceptance quality are established. The recommended current
production dataflow is:

```text
FP16 partial outputs
  -> AITER FP16 all-reduce
  -> fused residual add + RMSNorm
  -> GS128 INT8 quant-out + FP16 scales
  -> AITER W8A8 GEMM
```

## Priority summary

1. Integrate AITER fused AR + RMSNorm + GS128 INT8 quant-out with the W8A8
   consumer.
2. Quantize the target `lm_head` to GS128 W8A8 INT8.
3. Implement optimized noncausal DFlash2 attention for gfx908.
4. Explicitly pin the already-defaulted INT8 embedding path.
5. Keep convolution state, recurrent state, recurrence parameters, nonlinear
   state, attention probabilities, and residuals floating point until scaled,
   kernel-supported formats justify changing them.

