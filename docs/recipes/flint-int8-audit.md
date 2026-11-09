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
| 9 | expert `gate_up` (a8w8 blockscale) | int8 GEMM with fused SwiGLU + per-128 int8 epilogue (interleaved gate/up weights); no bf16 gate/up tensor | fixed |
| 10 | expert SiLU*up epilogue | folded into the gate_up GEMM epilogue (exact `silu_mul_quant_rn` contract: bf16-rounded halves, fp32 silu, one product rounding, GS128 quant) | fixed |
| 11 | expert `down` (a8w8 blockscale) | int8 GEMM → float combine (gammas) → **quant-out in the reduce epilogue** (shared branch folded as an int8 residual) → TP int8 collective | fixed |
| 12 | shared/dense `gate_up` | resident [2N,K] int8 weights + `gemm_silu_quant_out`: SwiGLU and the GS128 int8 output in one kernel; no bf16 gate/up tensor | fixed |
| 13 | shared expert `silu_and_mul` | fused: bf16 gate/up → bit-exact bf16 act + per-group int8 stash, consumed by the blockscale down GEMM | fixed |
| 14 | shared expert `down_proj` (blockscale) | int8 GEMM with quant-out epilogue → GS128 payload; TP-reduced with the routed combine and folded as an int8 residual | fixed |
| 15 | combine + residual add | weighted expert combine + shared-branch fold + per-128 int8 quant in one reduce kernel (`reduce_grouped`); the runner's float AR is skipped and the payload is TP-reduced with the GS128 collective | fixed |
| 16 | dense MLP layers 0-1 | identical to 12-14 (`FlintMLP`), same fused silu stash | fixed |
| 17 | final norm | int8 payload in → bf16 out; the head quantizes only the selected logits rows in-prologue (selection happens after the model returns) | inherent |
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

1. **MoE routed block**: DONE end to end. gate_up = int8 GEMM with fused
   SwiGLU + GS128 quant epilogue (interleaved gate/up weights, bit-exact
   with the old `silu_mul_quant_rn` chain: 2e-5 code mismatch from
   accumulation order, scales exact). down = float combine in the reduce
   with the shared branch folded as an int8 residual, quantized in the
   same epilogue; the runner's float all-reduce is disabled
   (`skip_final_all_reduce`) and the combined payload is TP-reduced with
   `residual_g128_all_reduce`. No bf16 tensor crosses the MoE block.
2. **Norm → GEMM prologue quant**: DONE for both consumers.
   `residual_norm(emit_pertoken=True)` emits the CK per-token contract
   (bit-identical to `pertoken_quant_rn`, 0 mismatches at M=1..1024) for
   `qkv_proj`; `residual_norm(emit_quant=True)` emits the GS128 per-group
   contract (bit-identical to `packed_g128_int8.quantize`) and both the
   expert prologue and the router gate consume it. One fewer quant launch
   per consumer per layer.
3. **Attention output → `o_proj`**: the attention kernel emits bf16 and
   `o_proj` quantizes in-prologue; an attention-epilogue quant would remove
   that round trip.
4. **Final norm → lm_head: NOT VIABLE as a norm epilogue.** vLLM selects
   the logits rows *after* the model returns
   (`hidden_states[logits_indices]` advanced indexing in the gpu runner),
   which drops tensor attributes; a stash emitted by the final norm cannot
   reach the head, and quantizing every row for the handful of selected
   ones is extra work. The head's own prologue already quantizes only the
   selected rows, which is the efficient order. Measured: head stash hits
   0/4 workers by construction.

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
   - norm output -> `qkv_proj` (CK prologue `pertoken_quant_rn`) — DONE for
     the payload layers (per-token stash consumed under `ACT_QUANT=round`);
     layer 0's plain RMSNorm still takes the CK prologue.
   - final norm -> `lm_head` — not viable: the runner selects logits rows
     after the model returns, so no stash can ride along (see item 4 above).
   - embedding gather -> first norm (int8 table dequantized before the
     norm reads it).
3. **Epilogue-fusion gap** - a GEMM writes bf16 and the next kernel
   re-quantizes:
   - expert `gate_up` -> `silu_mul_quant_rn` — DONE (folded into the GEMM).
   - `down` -> weighted expert combine -> `_to_payload` — DONE (quant-out in
     the reduce epilogue, shared branch folded in int8).

Nothing in categories 2 and 3 is doctrine; they are scheduled work. The
residual stream, o_proj output, expert input, and silu stash are already
int8 payloads, and a failed payload step raises rather than falling back.
