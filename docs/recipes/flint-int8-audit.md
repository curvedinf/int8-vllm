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
| 1 | `input_layernorm` (RMSNorm) | int8 payload in → norm (fp32 math) → bf16 GEMM input; **residual stays an int8 GS128 payload** (`residual_norm`) | fixed (payload in, payload residual out) |
| 2 | `qkv_proj` (CK W8A8) | bf16 → per-token int8 act quant → int32 accum → **bf16 out** | inherent (GEMM must emit an activation); norm→int8 prologue stash is a pending fusion |
| 3 | fused q/k RMSNorm+RoPE (`qk_norm_rope_rocm`) | bf16 → bf16 | fixed (one kernel) |
| 4 | AITER unified attention | bf16 q/k/v; KV written `int8_block_g128`; KV **dequantized inside the attention kernel** | inherent (attention math is bf16; the dequant is in-kernel, not a separate pass) |
| 5 | `o_proj` (blockscale W8A8) | int8 GEMM with **quant-out epilogue** → GS128 int8 payload; TP all-reduce on int8 codes + fp32 scales (`residual_g128_all_reduce`) | fixed |
| 6 | `post_attention_layernorm` | bf16 → bf16 | inherent |
| 7 | router gate (`GateLinear`, int8 tier) | prefill: bf16 → int8 GS128 GEMM → fp32 logits; decode: bf16 tier | enabled (PTQR recovers) |
| 8 | routed experts input quant | bf16 normed → per-group int8 + fp32 scales (fused with residual+norm) | fixed |
| 9 | expert `gate_up` (a8w8 blockscale) | int8 → int32 accum → **bf16 out** | inherent (SiLU needs float) |
| 10 | expert SiLU*up epilogue | bf16 gate/up → int8 q + scales in one kernel (`silu_mul_quant_rn`) | fixed |
| 11 | expert `down` (a8w8 blockscale) | int8 → **bf16 out**, scatter/reduce with gammas | **fixable**: Bolt keeps the routed output int8 through `combine_experts`; flint up-quants at the combine |
| 12 | shared expert `gate_up_proj` (CK W8A8) | bf16 → int8 → **bf16 out** | inherent |
| 13 | shared expert `silu_and_mul` | fused: bf16 gate/up → bit-exact bf16 act + per-group int8 stash, consumed by the blockscale down GEMM | fixed |
| 14 | shared expert `down_proj` (CK) | bf16 → int8 → bf16 | inherent |
| 15 | combine + residual add | expert combine is bf16 (down GEMM output); the residual add is fused into the next norm and emits int8 payloads | residual fixed; expert combine int8 is the remaining round trip |
| 16 | dense MLP layers 0-1 | identical to 12-14 (`FlintMLP`), same fused silu stash | fixed |
| 17 | final norm | int8 payload in → bf16 out for the logits GEMM (inherent: the head quantizes in-prologue) | inherent |
| 18 | lm_head (int8 GS128) | bf16 → per-group int8 quant → int8 GEMM → bf16/fp32 logits | fixed (enabled) |
| 19 | embedding | int8 table at load + gather-then-dequant (fork path, `VLLM_GFX908_INT8_EMBEDDING` default on; per-row fp16 scales) | fixed |

## Draft chain (DFlash2)

| # | Site | Dtype flow | Verdict |
|---|---|---|---|
| 1 | draft `qkv_proj` / `o_proj` (AutoGPTQ→CK W8A8) | bf16 → int8 → bf16 | inherent |
| 2 | draft attention | bf16 q/k/v, KV `int8_block_g128` dequantized in-kernel | inherent |
| 3 | draft MLP (`Qwen3MLP` = Qwen2MLP: gate_up → `SiluAndMul` → down) | same fused silu stash as the target | fixed |
| 4 | conv `kernel_projection` ×10, `candidate_selector.hidden_projection` | int8 W8A8 at load (`VLLM_GFX908_DF2_W8A8` default on) | fixed |
| 4b | conv `base_kernel` planes | int8 per (side,tap) at load, dequant-on-add; bf16 base freed | fixed |
| 4c | draft `fc` | int8 CK W8A8 at load (`install_int8_linear`) | fixed |
| 5 | selector codebooks | int8 tables at load, gather-dequant per (row, table) (`VLLM_GFX908_DF2_CODEBOOK_I8`) | fixed |
| 6 | draft lm_head / candidate logits | shares the target's int8 GS128 head (dflash spec setup replaces the provisional bf16 module) | fixed |
| 7 | draft KV | `int8_block_g128` | fixed |

## Non-quantization dtypes that look like quantizations

- GPTQ `scales` are stored **fp16** on disk (`quant_pack`), converted to fp32
  in kernels. That is scale storage, not an activation up-quant.
- `int8_block_g128` KV scales are fp32 per (token, head) group; the UA kernel
  dequantizes with them in-register.
- Router logits are fp32 by design (top-k selection).
- Norm outputs that feed a GEMM are bf16 activations and are quantized by that
  GEMM's own prologue (CK or blockscale). The **residual stream itself is int8**
  (GS128 codes + fp32 scales) between every decoder layer; there is no bf16
  residual path in the Flint forward (failures raise rather than fall back).

## Directive (user, 2026-11-09): W8A8 everywhere under PTQR

PTQR retrains against the bf16 teacher on the serving grid, so bf16 GEMM
surfaces are no longer justified as "quality exceptions" - they are training
gaps. Current status: every GEMM-shaped surface is int8 W8A8 (target and
draft), the embedding is an int8 table, the router gate and lm_head are int8
GS128 at ALL batch sizes, and the **residual stream is an int8 GS128 payload
between every decoder layer**. The router gate prefill threshold is
diagnostic only. A failed int8 payload step raises; nothing falls back to
bf16.

## Remaining activation round trips (in priority order)

1. **MoE expert down output → combine**: the blockscale `down` GEMM emits
   bf16, the weighted expert combine runs in bf16, and the layer then
   re-quantizes to the int8 payload (`_to_payload`). Closing it needs a
   quant-out epilogue on the down GEMM plus an int8 weighted combine
   (Bolt's `decoder_int8.combine_experts` pattern). Largest remaining item.
2. **Attention-input norm → `qkv_proj`**: the norm emits bf16 and the CK
   W8A8 GEMM quantizes it in-prologue. The MoE path already emits an int8
   norm stash via `residual_norm(emit_quant=True)`; the attention path should
   consume the same stash (one prologue quant removed per layer).
3. **Attention output → `o_proj`**: the attention kernel emits bf16 and
   `o_proj` quantizes in-prologue; an attention-epilogue quant would remove
   that round trip.
4. **Final norm → lm_head**: the head quantizes its input in-prologue; fuse
   the final norm with the GS128 prologue quant.

Draft-side: every dense grid surface (`conv kernel_projection` x10, `fc`,
`candidate_selector.hidden_projection`) is already int8 at load; no bf16
GEMM remains there.

## Why a bf16 tensor still appears, per case

Three categories, and only the first is an accepted exception:

1. **Float math the kernel performs, not an int8 payload round trip.**
   RMSNorm reductions, attention softmax/QK^T/PV, SiLU, and the logits
   interface are evaluated in fp32/bf16 by the kernels themselves. Both the
   Qwen 27B and Bolt recipes run AITER attention in float with only the KV
   cache payload int8; Flint matches that (the sanctioned attention
   exception). Logits must be float for sampling. These are not int8→bf16
   up-quants: the values were produced in float and never had an int8 form.
2. **Prologue-fusion gap** - the producer already has the values in
   registers but stores bf16 and the consumer re-quantizes:
   - norm output -> `qkv_proj` (CK prologue `pertoken_quant_rn`). The MoE
     path already emits an int8 stash; the CK path needs the same per-token
     stash and must consume it in `ACT_QUANT=round` mode.
   - final norm -> `lm_head` (GS128 prologue quant).
   - embedding gather -> first norm (int8 table dequantized before the
     norm reads it).
3. **Epilogue-fusion gap** - a GEMM writes bf16 and the next kernel
   re-quantizes:
   - expert `gate_up` -> `silu_mul_quant_rn`; fuse SiLU*mul+quant into the
     GEMM epilogue and the bf16 gate/up tensor disappears.
   - `down` -> weighted expert combine -> `_to_payload`; this is the only
     true int8->bf16->int8 round trip left in the target layer. Needs a
     quant-out down epilogue plus an int8 combine (Bolt pattern). Largest
     remaining item.

Nothing in categories 2 and 3 is doctrine; they are scheduled work. The
residual stream, o_proj output, expert input, and silu stash are already
int8 payloads, and a failed payload step raises rather than falling back.
