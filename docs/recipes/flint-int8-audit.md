# Flint int8 forward audit — quantization sites, target + draft

Scope: the serving forward of the PTQR target (`vllm/models/flint`) and the
DFlash2 draft (`vllm/model_executor/models/qwen3_dflash*.py`), as configured
by `scripts/serve_recipe_flint.sh`. Question: where does the forward leave
int8 (or another dtype) on the table, especially **int8→bf16 up-quants
followed by a re-quantization**?

Legend: **inherent** = necessary for the math; **exception** = measured
bf16 surface; **fixable** = avoidable round trip; **fixed** = already fused.

## Target layer chain (per decoder layer)

| # | Site | Dtype flow | Verdict |
|---|---|---|---|
| 1 | `input_layernorm` (RMSNorm) | bf16 in → bf16 out (fp32 math) | inherent |
| 2 | `qkv_proj` (CK W8A8) | bf16 → per-token int8 act quant → int32 accum → **bf16 out** | inherent (GEMM must emit an activation) |
| 3 | fused q/k RMSNorm+RoPE (`qk_norm_rope_rocm`) | bf16 → bf16 | fixed (one kernel) |
| 4 | AITER unified attention | bf16 q/k/v; KV written `int8_block_g128`; KV **dequantized inside the attention kernel** | inherent (attention math is bf16; the dequant is in-kernel, not a separate pass) |
| 5 | `o_proj` (CK W8A8) | bf16 → int8 act quant → bf16 out; TP all-reduce in bf16 | **fixable**: the int8 GS128 payload collective exists (`residual_g128_all_reduce`) and our fused AR+RMSNorm+quant collective is built; residual stream is still bf16 |
| 6 | `post_attention_layernorm` | bf16 → bf16 | inherent |
| 7 | router gate (`GateLinear`, int8 tier) | prefill: bf16 → int8 GS128 GEMM → fp32 logits; decode: bf16 tier | enabled (PTQR recovers) |
| 8 | routed experts input quant | bf16 normed → per-group int8 + fp32 scales (fused with residual+norm) | fixed |
| 9 | expert `gate_up` (a8w8 blockscale) | int8 → int32 accum → **bf16 out** | inherent (SiLU needs float) |
| 10 | expert SiLU*up epilogue | bf16 gate/up → int8 q + scales in one kernel (`silu_mul_quant_rn`) | fixed |
| 11 | expert `down` (a8w8 blockscale) | int8 → **bf16 out**, scatter/reduce with gammas | **fixable**: Bolt keeps the routed output int8 through `combine_experts`; flint up-quants at the combine |
| 12 | shared expert `gate_up_proj` (CK W8A8) | bf16 → int8 → **bf16 out** | inherent |
| 13 | shared expert `silu_and_mul` | bf16 → bf16, then `down_proj` re-quantizes per-token | **fixable**: same fuse as the experts, via the `PREQUANT_ATTR` seam in `aiter_w8a16.apply_weights` (l.468-479) |
| 14 | shared expert `down_proj` (CK) | bf16 → int8 → bf16 | inherent |
| 15 | combine + residual add | bf16 | inherent while the residual stream is bf16 |
| 16 | dense MLP layers 0-1 | identical to 12-14 (`FlintMLP`) | **fixable** (same silu seam) |
| 17 | final norm | bf16 | inherent |
| 18 | lm_head (int8 GS128) | bf16 → per-group int8 quant → int8 GEMM → bf16/fp32 logits | fixed (enabled) |
| 19 | embedding | bf16 table lookup (trained with int8 fake-quant) | **fixable** (int8 table + dequant in lookup; Bolt ships `packed_embedding`) |

## Draft chain (DFlash2)

| # | Site | Dtype flow | Verdict |
|---|---|---|---|
| 1 | draft `qkv_proj` / `o_proj` (AutoGPTQ→CK W8A8) | bf16 → int8 → bf16 | inherent |
| 2 | draft attention | bf16 q/k/v, KV `int8_block_g128` dequantized in-kernel | inherent |
| 3 | draft MLP (`Qwen3MLP` = Qwen2MLP: gate_up → `SiluAndMul` → down) | bf16 → int8 → bf16 → int8 → bf16 | **fixable** (silu seam, same as target) |
| 4 | conv `kernel_projection` ×10, `fc`, `candidate_selector.hidden_projection` | bf16 | exception (Qwen draft float set) |
| 5 | selector codebooks | bf16 | exception (measured) |
| 6 | draft lm_head / candidate logits | shares the target head weights; the draft's own head module stays bf16 | exception |
| 7 | draft KV | `int8_block_g128` | fixed |

## Non-quantization dtypes that look like quantizations

- GPTQ `scales` are stored **fp16** on disk (`quant_pack`), converted to fp32
  in kernels. That is scale storage, not an activation up-quant.
- `int8_block_g128` KV scales are fp32 per (token, head) group; the UA kernel
  dequantizes with them in-register.
- Router logits are fp32 by design (top-k selection).
- Norm outputs are bf16 by design; rewriting them int8 would break the
  residual stream (Bolt also keeps the norm output bf16).

## Open up-quant round trips, priority order

1. **MoE down output stays int8 through the combine** (Bolt's
   `decoder_int8.combine_experts` / `ResidualInt8`). Requires an int8
   quant-out down GEMM (AITER `gemm_float_quant_out` exists for resident
   GS128 weights; the GPTQ per-channel CK path would need the same epilogue)
   plus the int8 residual stream. This is the largest structural item.
2. **Shared/dense/draft MLP silu → int8 stash** (`PREQUANT_ATTR`): removes
   one per-token quant kernel and the bf16 round trip per MLP layer. Cheap,
   no numerics change (same int8 grid the CK path would produce).
3. **Embedding int8 table** (packed lookup): halves the embedding read and
   removes the bf16 up-quant at load time.
4. **o_proj int8 payload collective** (`residual_g128_all_reduce` + fused
   epilogue): the collective is built and race-free; the in-model cost under
   graph replay is the open blocker (TPOT 67 vs 59 ms).
