# Flint int8 recipe — design map (goal work, 2026-11-06)

Target: RecursiveSI/flint (39.8B-A2.5B MoE) + incoai/Flint-DFlash2 draft,
official fork recipe at C8, int8 PTQR both, full int8 opt-in engine.
Foundation: upstream v0.33.0 merged (d8a365e164) — Flint model lives at
vllm/models/flint/ (upstream PR #62938).

## Flint architecture (from config + vllm/models/flint/model.py)

- 49 layers: 2 dense (mlp_only_layers [0,1]) + 47 MoE
- MoE: 256 routed experts × 6/token + 1 shared expert; sigmoid router,
  norm_topk_prob, router_aux_loss_coef 5e-4
- Attention: 32Q/4KV heads, head_dim 128 (GQA 8:1); per-head QK RMSNorm +
  learned per-head attention sinks; 35 SWA layers (window 513) + 14 full;
  RoPE only on SWA layers (no_rope_layer_indices = the full-attn ones)
- hidden 2048, moe_intermediate 512, shared_expert 1024, dense intermediate
  6144; vocab 150,144; max ctx 262,144; bf16 checkpoint (~80 GB)
- lm_head untied

## Linear surface map (W8A8 candidates → GPTQ GS128 + AITER CK)

| Surface | Module | Class | Fork int8 path |
|---|---|---|---|
| Attention QKV | model.layers.*.self_attn.qkv_proj | QKVParallelLinear | AITER W8A8 (GS128) |
| Attention out | o_proj | RowParallelLinear | AITER W8A8 |
| Dense MLP (L0-1) | gate_up_proj/down_proj | Merged/Row | AITER W8A8 |
| Shared expert (L2-48) | shared_expert.gate_up/down | Merged/Row | AITER W8A8 |
| Routed experts | experts (FusedMoE) | w13/w2 | Bolt pattern: INT8 GS128 (aiter moe / fork packed) |
| Router | gate (Replicated weight) | sigmoid gate | dense (exception — Qwen/Bolt doctrine: router/gating dense) |
| lm_head | lm_head | ParallelLMHead | int8 W8A8 (VLLM_GFX908_INT8_LM_HEAD) |
| Embedding | embed_tokens | VocabParallelEmbedding | int8 gather |
| QK norm weights | per-head RMSNorm | elementwise | bf16 (norm — not a GEMM) |
| Attention sinks | per-head learned scalar | elementwise | bf16 (exception candidate — verify with KLD gate) |
| KV cache | — | — | int8_block_g128 target (SWA window 513 → block granularity OK; full-attn 14 layers long-ctx gate needed) |

## Flint-DFlash2 draft (config)

- model_type "qwen3" (DFlash2DraftModel arch) — SAME wrapper family as the
  fork's Qwen3.8 DFlash2 draft; 5 layers, all sliding_attention (window
  2048), 32Q/8KV, head_dim 128, hidden 2048
- dflash_config: block_size 8 (7 draft tokens/verify), conv 2-tap
  (conv_group_size 16), selector_rank 256, selector_top_k 16,
  target_layer_ids [5,15,26,36,46], mask_token_id 150019
- Compatibility CONFIRMED (2026-11-06, safetensors header fetch): tensor
  layout is exactly the fork's DFlash2 surface — candidate_selector
  (hidden_projection [256,2048], predecessor/successor_codebooks
  [150144,256] bf16), fc.weight [2048,10240] (= 5 target layers x 2048,
  matching target_layer_ids), hidden_norm, per-layer attention_conv +
  mlp_conv (base_kernel [2,2,2048] + kernel_projection [512,2048]), mlp
  gate/up/down, q/k norms, 8 KV heads (k_proj [1024,2048]). Loads through
  the fork's qwen3_dflash2.py machinery modulo head-count params.

## Exceptions (from Qwen 27B + Bolt measured findings)

1. bf16 draft residual/compute dtype (draft resolves own dtype; never force
   target dtype — 2026-08-24 overflow fix).
2. Selector codebooks bf16; draft ctx-KV projection bf16-dense (W8A8 lever
   E8 exists, off by default).
3. Router/gating dense.
4. QK-norm + sinks bf16 (sink quantization must be KLD-gated if attempted).
5. int8 KV only via int8_block_g{G} family; long-context gate at target
   context (Flint doc context 262k — gate at 40k+ like Qwen; g128 default,
   fall back g16/g8 if the garble class reappears — same table applies).
6. lm_head stays dense if quantization OOMs (Qwen: lm_head=False in GPTQ,
   int8 at runtime via VLLM_GFX908_INT8_LM_HEAD W8A8 instead).

## PTQR plan (mirror qwen38 recipe)

CONSTRAINT (verified 2026-11-06): Flint has NO transformers modeling code
(quant-venv transformers 5.15.1 lacks it; gptqmodel therefore cannot load
it). The fork's PTQR suite (scripts/ptqr_train_target.py 76KB — embedded
Qwen student/teacher forward, attn_impl=rocm_triton, distill loss, STE
fake-quant, --no_ptqr control, DAgger rollout legs) and the draft trainers
drive their OWN model implementations extracted from the vllm model files.
=> Flint pipeline is FORK-NATIVE: embed a Flint teacher (bf16 exact) +
student (int8 GS128 fake-quant + act-quant-RN + int8-KV emulation) forward
in a ptqr_train_flint.py, ported from vllm/models/flint/model.py semantics
(MoE sigmoid router + shared expert + sinks + SWA513/full mix + per-layer
no-rope + QK norm). One-shot GPTQ base (ptqr_make_base_ckpt + a
vllm-forward GPTQ pass or direct PTQR-from-bf16 as rung-0) then the same
rung ladder (R10S60 analog) with:
- LONG-CONTEXT training: token corpus (SDGraft data, train/val_tokens.pt
  format) + rollout DAgger legs + fresh synthetic long prompts; Flint is a
  coding-agent model — agentic/tool-use corpora with its chat template.
- Draft: ptqr_train_draft.py already extracts from
  vllm/model_executor/models/qwen3_dflash{,2}.py — the Flint-DFlash2 draft
  has the SAME tensor surface (verified), so port = head-count/config
  params + the flint tokenizer/vocab.
- Gates: fidelity top-1 == bf16 (draft), KLD + greedy agreement (target),
  acceptance ±1pt at TP4/C8.

- Target: GPTQ bits=8 gs=128 sym true_sequential, PTQR R10S60-style
  retraining vs frozen bf16 reference, LONG-CONTEXT training samples
- Draft: quantize_dflash2_int8.py-style + drafter remap; PTQR rung-1 with
  acceptance gate (TP4/C8 greedy, ±1pt of bf16 baseline)
- Both published as curvedinf/*-GPTQ-INT8-W8A8-GS128-PTQR

## Engine plan (full int8 opt-in)

- Opt-in flag (e.g. VLLM_FLINT_INT8 / recipe serve script env) selecting:
  AITER W8A8 for every linear above, int8 KV g128, int8 embedding gather,
  int8 lm_head, ACT_QUANT=round, AITER unified attention for SWA+full mix.
- New kernels where the merged upstream flint ops assume CUDA/flashinfer:
  decode_attention.py / moe.py / moe_gemv.py / qkv_fused.py are upstream
  CUDA/CuTe paths — route to aiter equivalents (fork has gfx908 G128
  attention cores + aiter moe). Need: SWA decode + full decode + verify
  variants on the UA/Gluon path with sinks support (upstream sinks support
  exists in aiter UA per the FA backend merge).

## Open questions

1. Sinks in aiter unified_attention for the SWA layers (fork FA backend has
   sink_ptr; verify UA path).
2. Flint-DFlash2 draft tensor compatibility with qwen3_dflash2.py loader.
3. MoE int8: aiter moe W8A8 GS128 vs Bolt packed-experts pattern (256
   experts × small intermediate 512 — check which path the fork MoE
   machinery prefers at TP4; per-rank 64 experts).
4. Build: extension rebuild post-merge pending (stale caches cleaned).
