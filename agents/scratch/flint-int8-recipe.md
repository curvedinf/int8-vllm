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

## PTQR trainer port structure (scripts/ptqr_train_flint.py, planned)

REUSABLE AS-IS from scripts/ptqr_train_target.py: act_quant_fake /
kv_block_quant_fake (exact deployed semantics), PTQRLinear +
replace_linears_with_ptqr (any nn.Linear tree), fake_quant_embedding_,
attach_kv_fake_quant, SR-SGD + deferred updates, anneal_tau, eval_kld
(TP vocab-parallel), rollout DAgger loaders, --no_ptqr control.

NEW for Flint (build_flint_model): plain-decoder teacher/student (no GDN,
simpler than Qwen): per-layer qkv/o PTQRLinear, per-head QK RMSNorm,
RoPE only on SWA layers (cos_sin from rope_parameters), attention-sink
bias (added to scores), SWA window 513 mask vs full attention, sigmoid
router top-6 renormalize (dense), shared expert PTQRLinear MLP, routed
experts int8 GS128 (per-expert PTQRLinear, TP-sharded 64/rank; hidden
2048 / moe_inter 512 → small expert mats). fp32 logits accumulate.
Training attention: SDPA/flex or curvedinf triton flash (no FLA).

## Draft int8 surface list (user-confirmed 2026-11-06: draft IS int8-ified)

Same doctrine as dflash2-ptqr-r1 + Bolt text-path findings:

| Draft surface | Treatment |
|---|---|
| q/k/v/o_proj, mlp gate/up/down (5 layers) | GPTQ INT8 GS128 + AITER W8A8 |
| attention_conv + mlp_conv kernel_projection | per-channel W8A8 (DFlashGroupedConv requant, VLLM_GFX908_DF2_W8A8 default ON) |
| attention_conv + mlp_conv base_kernel | bf16 dense (dequant-grid surface) |
| candidate_selector.hidden_projection | W8A8 (selector projections quantized, as Qwen draft) |
| predecessor/successor codebooks | bf16 (measured exception, audited) |
| fc.weight (ctx projection, 5x2048) | bf16 dense (dtype ladder 71.2 > 69.2 > 66.0% acceptance; E8 W8A8 lever VLLM_GFX908_DF2_CTXKV_W8A8 exists, off) |
| q/k norm, hidden_norm, layer norms | bf16 (norms) |
| Draft KV cache | int8_block_g128 (explicit in speculative config) |
| Draft compute dtype | bf16 from its own checkpoint (never the target dtype) |

PTQR: draft rung-1 recipe (ptqr_train_draft*.py port to flint tokenizer/
config; fidelity gate top-1 == bf16; acceptance ±1pt at TP4/C8).

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

## Pipeline status (2026-11-06 20:16)

- Corpus BUILT: SDGraft/data/flint_longctx 4.22M train / 42.6K val tokens
  (ids within vocab 150144; code+c4+agentic-chat+4-16k long legs).
- Trainer: single-GPU path green (tiny smoke + oracle). TP4 sharding
  scaffold committed; NEXT: partial-output all-reduce wiring in
  FlintRef.forward (o_proj/mlp/MoE partials, vocab-parallel logits
  gather) + 4-rank CPU equivalence check vs unsharded tiny model, then
  rung-0 (PTQR from bf16 RTN init) real run.
- Merge validation: micro PASS, battery 4/4; serve boot on attempt 5
  (fixes: orjson dep, dup op registration, g_idx getattr, phantom
  4-tuple unpacks, AIS 4-tuple, w_gidx_name default) — now in final AIS
  repack-cache load phase.
- Boot gate key note: /etc/llama/llama-api.key missing; used
  VLLM_API_KEY env override (read_api_key supports it) for gate boots.

## RUNG-0 SMOKE PASS (2026-11-06 23:59)

TP4 real-Flint trainer smoke COMPLETED: 3 steps, all 4 ranks identical
loss (323.79 @step0, tau=1.0, 4-experts-kept smoke config), SR-SGD updates
applied, clean exit. Full pipeline exercised: 80GB ckpt stream + name-map
+ expert EP assign -> student PTQR swap (573 surfaces) -> TP4 sharding of
teacher AND student -> deployed-quant forward (W8A8 GS128 fake-quant +
int8-block KV fake-quant post-rope) -> CE+KD loss -> backward -> SGD.
High absolute loss expected (4/256 experts + tau=1 dither; wiring gate,
not quality). Debug trail to green (7 commits): experts-before-mlp loader
branch order; shared-expert own-intermediate slicing (then superseded by
name-map-only loader); router rows follow reduced experts; ProcessGroup
out of module attrs (deepcopy); shard AFTER student build; PTQR-aware MLP
lockstep slicing (48-vs-12 scale broadcast); rope tables follow .to();
student unfreeze after deepcopy; autograd-aware TP gather (grad scatter).
Full-run blockers remaining: layer-streamed teacher (20GB/rank bf16 x4
OOMs 61GB host + won't fit 32GB MI100 alongside student), then rung
ladder + gates.

## FULL-EXPERT RUNG-0 DEBUG LEDGER (2026-11-07, CLOSED)

The full-expert streamed TP4 run now COMPLETES the entire build phase on
GPU (swap #6208 at 6.6G alloc — masters freed as replaced). Fix trail
(all committed): torch.device(meta) ctx for skeleton (else CPU kaiming
materializes 80G); set_default_dtype(bf16) (meta ctx alone = fp32 shard
37G); loader up_proj/sink whole-tensor slices by rank offset (presplit
asymmetry); replace_linears_with_ptqr collects (parent,attr) targets
(list(named_modules) pinned every old linear: old+new masters 37G).
SIGUSR1 faulthandler armed in the trainer for future stack dumps.

FORWARD SHAPE BUG (g10/g11, root-caused 2026-11-07 ~03:35): "mat1 and
mat2 shapes cannot be multiplied (1024x2048 and 256x2048)" at the FIRST
qkv forward. PTQR_SHAPE_AUDIT=1 dump (trainer env flag, writes
/tmp/ptqr_audit_rN.txt pre-loop) showed ALL 6213 PTQRLinears with
w=(2048,256) s=(2048,2) — every swapped linear cloned ONE tensor:
layers.48.mlp.shared.down (the last nn.Linear visited by the collection
loop). Cause: commit 70e81e70ff dropped `child` from the collected
targets but left `child.weight` in the swap loop — `child` was a stale
loop-leak binding. The 4-expert rung-0 smoke passed only because it
predates that commit. Fix: `child = getattr(module, child_name)` in the
swap loop. Also fixed en passant: presplit gate_up cat used `u[g:]`
(empty slice — files on disk carry gate-only entries; the loader
re-fetches up itself, so no regeneration needed); shard_flint_tp qkv
PTQR branch fell through to a redundant second weight assignment (now
properly else-scoped); save_ptqr_checkpoint vocab_range now takes
vocab_size override (Flint shrinks lm_head rows in place; derived range
was 4x off for export stitching). Trainer now has eval_kld_flint
(tau=0 held-out KLD + teacher NLL, token-chunked) and
save_ptqr_checkpoint wiring (save_every + final).

Status: g12 (full-expert rung-0, 3 steps, seq 1024, --rank_shards) —
audit shapes verified correct (qkv [1280,2048], 3107x2048/3008x512/
49x1024/47x256/2x1536 class counts all match); 3-step green check
pending at ledger-write time.

## STREAMED-TEACHER IO CAMPAIGN (2026-11-07 04:1x-05:3x, CLOSED)

g13 (first teacher exercise) exposed four never-run defects in
StreamedFlintTeacher._layer (fixed + bit-exact vs resident at B=1 and
B=3): unsliced full-ckpt q/k/v/o (dead weight-split); einsum with output
subscripts absent from inputs; missing input/post_attention layernorm;
flattened B*S attention leaking causal masks across batch rows.

g14 exposed the FATAL architecture problem: _t() materializes FULL
checkpoint tensors then slices -> the teacher reads the 80GB ckpt per
batch per step; 4 ranks = ~320 GB/step aggregate; all GPUs idle in page
waits at ~1 MB/s (bcache). FIX (commit dd44710973): teacher reads the
presplit rank shard (20GB/rank; presplit-aware getters — fused qkv entry,
gate/up SEPARATE (canonical format = on-disk files; the earlier "fused
gate_up" presplit writer fix was reverted — the old files' empty-cat bug
was benign), o/down pre-sliced); refactored _stage_layer/_apply_layer;
sweep() = layer-major over many batches (weights stage once per rung).
Trainer: --kd_cache 20 stage-replay (sweep once over M train + E val
batches -> rank0 spills fp16 logits to out_dir/kdcache (one shared
page-cache copy, 4-rank RAM copies would be ~49 GB); steps replay
torch.load(mmap).to(dev)). KD-cache path vs live path switchable
(--kd_cache 0 = live). eval_kld_flint takes both sources.

Per-rung cost model now: teacher sweep = ONE 20GB/rank sequential read
(~2-5 min) + M×0.6 GB fp16 spill; training steps are student-bound.

NEXT: g15 smoke (KD cache 4, eval_every 2, out ptqr_flint_smoke) then
rung 1 via scripts/ptqr_flint_rungs.sh 1 (kd_cache default 20, 60 steps,
seq 2048, save/eval every 20, chained --init_from for rungs 2-3).

## IO CAMPAIGN ROUND 2 (2026-11-07 06:2x-07:5x) — WON at g19/g20

g15 crash: sweep got CPU input ids → _embed now moves ids to device.
g16/g17: sweep still crawled (2.3 MB/s). MEASURED: single-stream dd =
553 MB/s — the disk is FAST; the pattern (mmap tensor faults, 4
concurrent ranks) is the problem. Fixes landed in sequence:
1. fadvise WILLNEED per-layer prefetch parsed from the safetensors
   header (helped 3/4 ranks; the coldest rank then streamed alone at
   1.4 MB/s inside the collective lockstep — g17 killed).
2. warm_shard(): staggered reverse-order 64MB sequential pre-read behind
   barriers (114s for all four) — but sweep-time mmap faults under the
   SATURATED page cache still degrade to page-by-page direct reclaim
   (g18 crawl).
3. _t() presplit path now pread()s tensors in 16MB chunks (sequential
   read path) + zero-copy frombuffer; shapes/dtypes parsed at init.
   g19: "[kd] cached 4 train + 10 val teacher sweeps in 421s" — KD
   precompute GREEN. _precache now does ONE sweep per split (chunking
   re-staged every layer per chunk — 4-chunking would have cost 5x on
   rung 1's 20 batches).
g20 (expandable_segments): KD precompute GREEN and FASTER ("cached 4
train + 10 val in 323s"); crash moved to the SAME place —
loss.backward() hipblasCreate ALLOC_FAILED on rank3 (the rank deepest
into backward). Genuine backward-peak memory pressure (18.7G masters +
fwd activations + fp32 logits/KD transients + per-linear wq recompute).
NEXT (exact fix): port the Qwen trainer's chunked lm_head+loss into
one_batch — FlintRef needs a hidden-state accessor (split forward into
forward_hidden -> h [B,S,H] + head(h)); then per-256-token chunks:
head(h[chunk]) -> CE + KLD vs cached teacher chunk -> backward the
chunk immediately (grads accumulate; lm_head grads deferred-apply
already supports it) — this is the Qwen "16 partial grads per step"
pattern (ptqr_train_target one_batch/loss chunking; logits_chunk arg
exists unused). Also chunk the KD p_t upcast. Re-smoke at seq 1024
(3 steps + eval + save), then LAUNCH RUNG 1: scripts/ptqr_flint_rungs.sh 1
(60 steps @1e-5, seq 2048, kd_cache 20 → ONE sweep ≈ 10-15 min, save/
eval every 20; rungs 2-3 chain via --init_from). Then draft cache
(243 seqs) + draft PTQR on spare GPU headroom during the ladder; then
engine extras + serve recipe C8 (fork Flint ROCm path exists: sinks/
SWA on unified attention, eager qk-norm+rope — commit 8ce35980d1;
remaining: MoE int8 route, GPTQ loader wiring, serve script).

## QUALITY BLOCKER (2026-11-07 08:0x) — trainer semantics vs real ckpt

g22: the FULL PIPELINE IS GREEN mechanically (KD cache 302s incl warm;
step 0 trains: loss 92.48 ce 74.75 kd 8.86 all ranks identical; cached
eval works: kld 30.3; per-rank checkpoint save works). BUT teacher_nll
85.6 — the TEACHER's own logits are garbage: row0 max -14.3 min -221
std 20, top1 acc 0 at any lag (probe on kdcache/train_0.pt). Even
uniform logits would give NLL ~11.9. The earlier 4-expert smoke
(resident teacher from FULL ckpt, loss 323) was the same class —
so the bug is in FlintRef/StreamedFlintTeacher SEMANTICS on the real
checkpoint, NOT in streaming. The tiny-config checks were CIRCULAR:
teacher vs resident implement the same (wrong) semantics, and tiny
sinks are ZEROS (init) — the sink path was never exercised.

DECISIVE GATE (next session, do FIRST): ground-truth the trainer
forward against UPSTREAM vLLM Flint (merged v0.33.0 model.py) on the
real checkpoint — TP4 offline LLM generate (37-token prompt): coherent
text ⇒ serving path right + trainer semantics wrong. Then diff layer
outputs: run the serving model's first N layers vs FlintRef on the
same ids (or bisect per-layer hidden norms in a sweep_states pass
looking for the explosion point — layer-0 output norm vs expected).

Audit list (real-vs-tiny differences, in suspicion order):
1. sinks: real learned values (tiny zeros); check sink INDEXING by
   head under TP (kv-head-group order) and add-point (scores post-scale
   pre-mask matches the CUDA kernel's virtual-key semantics — verified
   in form, NOT in the real values' scale: are softmax_offset values
   sane? print them from the ckpt).
2. layer_types/no_rope lists: 14 full-attn layers = no_rope indices
   [0,1,5,9,...,48]; verify against config exactly.
3. dense layers [0,1] vs MoE; inter 6144; shared 1024; moe 512.
4. router: sigmoid + top6 renorm (norm_topk_prob).
5. rope theta 1e4, tables [262144, 64] fp32; hd 128; 32Q/4KV.
6. embed/lm vocab slicing + gather.
7. q/k norm per-head [128] pre-rope.
Also print real softmax_offset stats + a real config cross-check
(load_flint_config vs the HF config.json keys — esp. any field the
SimpleNamespace misses, e.g. attention_bias, tie_word_embeddings).

Everything else is in place: once semantics are fixed + smoke re-runs,
launch rung 1 (scripts/ptqr_flint_rungs.sh 1).

Draft port committed (a99b184ccf): ptqr_flint_draft_cache.py (sweep_
states, exits from the DRAFT's dflash_config) + ptqr_train_draft_
flint_ptqr.py (CFG mutated pre-construction; load_draft takes the dir
EXPLICITLY — its default arg binds DRAFT_DIR at def time; CPU load
check PASS 710.1M params). Cache build TODO: 243 train seqs (one
sweep_states call, chunked 64).

## SEMANTICS HUNT (turn 3, 2026-11-07 ~09:4x) — two fixes landed, more remain

g23 (standard RMSNorm w*x — Flint ops/norm kernel is s*rs*w; my (1+w)
was Gemma-style): teacher NLL 85 -> 14.25. g24 (sink = virtual key in
the softmax denominator via cat+slice, NOT a broadcast bias on every
key — matches the upstream kernel l+exp(sink-m)): NLL 14.25 -> 13.83,
top1 0.3% — still ~uniform. MORE BUGS REMAIN. Both were invisible to
the tiny checks (zero sinks; circular teacher-vs-student; ones-init
norms now identity).

NEXT-SESSION AUDIT (continue here): teacher NLL target ~2-4, top1
>25%. Remaining candidates, unverified: (1) rope split form inside
rot() — neox half-split assumed; upstream asserts is_neox_style but
VERIFY the actual rot dim pairing; (2) GQA q/kv head INTERLEAVE under
TP slicing (h0*qh head order vs the checkpoint's head layout — if
k-head slices land on wrong heads, per-head k_norm/sinks scramble);
(3) dense-vs-MoE dispatch + shared-expert weight application (any
shared-expert gate/scale?); (4) router top-6 renorm + expert weight
application order; (5) embed mask/gather + lm_head gather order;
(6) window semantics off-by-one (>= window vs > window). DECISIVE
GATE if inspection stalls: run upstream vLLM Flint (vllm/models/flint,
TP4, bf16, offline generate, 37-token prompt) — coherent text proves
the serving path; then diff its layer-0..N hidden states vs
FlintRef's on the same ids (hook the serving model) to localize the
divergence in ONE pass. g24 run artifacts: /tmp/ptqr_g24.log,
smoke kdcache at /home/curved/models/ptqr_flint_smoke/kdcache.
Commits: 9137322454 (pread), 4ecde2103a (warm), 3d74854956 (ledger),
chunked-loss + eval-list + norm 5790e5922b-adjacent chain; see git log.

## GROUND-TRUTH GATE (turn 4, 2026-11-07 ~08:0x) — AIS fixed, boot hang next

- AIS per-rank read plan ROOT-CAUSED + FIXED (commit af1f3fb591):
  _iter_checkpoint_tensors unpacked 3 values from the fork's 4-tuple
  _prepare_weights return -> always fell back to full-tensor reads
  (4x80GB thrash, 8 min/shard). Now 43 s/shard (15 shards ~10 min),
  first boot writes the repack cache (~/.cache/vllm/ais_repack).
- GT4 (first upstream-vLLM Flint boot on the ROCm port): loads clean,
  dies at determine_available_memory — a WORKER HUNG during KV-cache
  profiling forward; engine shm_broadcast timed out ("cancelled"). No
  worker traceback surfaced. NEXT: (a) compare boot env/knobs with
  scripts/serve_recipe_qwen38.sh (the same stack boots daily there —
  JIT warmup / timeouts / max_num_seqs interplay); (b) suspect the
  eager qk-norm+RoPE -> unified_attention_with_output profile call
  (sinks+SWA on gfx908 UA — first exercise; VLLM_ROCM_USE_AITER_*
  set) or cold aiter JIT exceeding the init RPC timeout — retry once
  with longer timeout env (see VllmConfig/RPC timeout knobs) and
  enforce_eager to bisect compile vs kernel hang; (c) if UA+sinks is
  the hang, temporarily boot with the non-UA Triton attention path to
  isolate. GT script: scripts/flint_groundtruth.py (log /tmp/
  flint_gt4.log). GPUs were cleaned after the failed boot.
- Reminder: teacher NLL still 13.8 (norm + sink fixes landed); the GT
  generation remains the decisive reference for the remaining trainer
  semantic bugs.

## INPUTS FULLY VERIFIED + SERVING ALSO GARBAGE (turn 6, 2026-11-07 09:0x)

Boot chain fixed through GT11 (commits af1f3fb591..): AIS 4-tuple unpack
(fast loads, 43s/shard + repack cache), flint ops/norm Triton kernel
register-alloc failure on gfx908 -> standard fork RMSNorm/SiluAndMul,
eager path o_proj 3D-flatten, NCCL int8 AllGather needs
NCCL_ALGO=Ring NCCL_PROTO=Simple (production pins). GT11 BOOTS AND
GENERATES END-TO-END — but word salad, SERVING prompt NLL 20.4 on
chat-templated text.

Inputs verified byte-identical to HF: config.json (field-by-field),
tokenizer.json (sha256 match), weights (q_proj tensor byte-verified via
HF range-request), presplit rank0 content (all tensor classes equal to
full ckpt slices), corpus tokens decode to clean English. => BOTH the
upstream serving path (fork ROCm port) AND my streamed teacher produce
~uniform logits from verified inputs.

sglang/fa4-learned-sink-fix.patch (fetched from the HF repo) CONFIRMS
the sink semantics: virtual key in the softmax denominator, POST-scale
units (exp(sink) vs exp(qk*scale)) — matches my post-fix teacher. The
sglang bug it fixed was a missing O-rescale, not semantics.

GT12 in flight: VLLM_ATTENTION_BACKEND=TRITON_ATTN (no sink support ->
effectively sink-less) to bisect whether the serving garbage lives in
the fork's AITER-UA sink/SWA path. If TRITON_ATTN is coherent-ish =>
fix UA sink path; teacher's remaining bug (NLL 13.8) is separate but
likely the same area. Teacher candidates left: none by inspection —
if GT12 implicates sinks, re-audit how the FORK UA consumes sinks vs
the reference and mirror whatever divergence exists in the teacher.

## MEASUREMENT BUG FOUND + VENDOR REFERENCE (turn 8, 2026-11-07 10:2x)

CRITICAL: build_dataloader SHUFFLES UNSEEDED — every standalone teacher-
NLL probe compared cached logits against a DIFFERENT random window. All
pre-g25 NLL numbers were INVALID. Fixed: _precache now stores {"logits",
"y"} per batch (training pairing was always correct). g25 re-measured:
teacher NLL 10.9999, top1 3.2% — STILL broken (healthy ~2-4/>25%).

Vendor oracle: RecursiveSI/flint ships sglang/flint_sglang-0.1.4 wheel
(unzipped at /tmp/flint_sglang_src — source-only, no install). Its
model.py CONFIRMS every semantic I implemented: standard RMSNorm
(Qwen3Moe base), qk-norm -> rope(unless skip_rope = layer in
no_rope_layer_indices) -> attention w/ sinks; sigmoid top-6
(scores+bias, bias zeros when absent) renormalize; ungated shared
expert added AFTER routed output; window = config-1 in sglang units ==
my inclusive-513 mask. Nothing left to misread by inspection.

Presplit files FULLY re-verified byte-exact (lm_head/embed/final_norm +
qkv/o for L0,1,25,48 across all 4 ranks). Serving garbage (NLL 20.4)
is deterministic across backends; sink-dead test still pending (needs
worker-side patch — sitecustomize import hook broke boot (registry
ordering); GT script now has a generic recursive module search, but TP
workers are separate processes even with V1 multiprocessing off).

HYPOTHESES (ranked, next session):
1. Both impls derive from the same READING — but vendor matches that
   reading, so a shared misreading is unlikely now. Instead suspect
   REAL-SCALE mechanics the tiny checks can't see: per-rank kvh=1 GQA
   repeat, window 513, real sink VALUES in the virtual-key cat, or the
   sweep's tp_sum ordering at 4 ranks with real shapes.
2. Teacher ablation sweep (cheap, one 12-min GPU cycle each): disable
   sinks / window / rope / invert no_rope; watch NLL. A collapse to
   ~3-5 localizes the broken primitive.
3. Serving-side per-layer hidden dump (in-process mode works; GT15
   reached generation) vs teacher sweep_states on the same ids — first
   diverging layer pinpoints the shared bug.
4. Check chunked prefill/window interaction in serving (UA prefill
   window semantics) separately if teacher-side ablation finds nothing.

## ABLATION RESULTS (turn 9, 2026-11-07 ~11:1x)

FLINT_ABLATE env added (commit chain). NLL via matched-targets probe:
- baseline (g25): 10.9999, top1 3.2%
- sinks-dead:     26.065  top1 0.8%  => sinks are LOAD-BEARING and
  correctly implemented (live sinks help a lot)
- norope_invert:  12.167  top1 0.1%  => rope assignment (rope on SWA
  layers, none on full-attn) is CORRECT (inverting hurts)
Remaining ablations: rope-off-everywhere, window-off.
OBSERVATION: baseline logits are ALL strongly negative (row max -12.9,
min -82.5) while sinks-dead shifts them UP (+11.7 max) — live sinks
suppress the whole logit vector ~25 units. Consistent with reference
semantics (sink competes in e-units with scaled scores; values +2..+10
dominate typical scaled scores) but worth cross-checking against the
serving model's logit scale when the in-process oracle lands.
Also exonerated: weight layout (q_proj [4096,2048] is forced [out,in];
transposed impossible given hidden 2048).
NEXT: (a) remaining 2 ablations; (b) serving per-layer hidden dump vs
sweep_states on identical ids (the definitive divergence localizer);
(c) if teacher reaches NLL ~3-5, IMMEDIATELY run the 3-step smoke and
launch rung 1 (scripts/ptqr_flint_rungs.sh 1).

## ABLATIONS COMPLETE + DIVERGENCE LOCALIZED (turn 10, 2026-11-07 11:1x)

All four ablations (NLL, matched targets): baseline 11.00 | sinks-dead
26.07 | rope-off 11.50 | norope-invert 12.17 | window-off 10.84.
=> Attention SEMANTICS fully exonerated (sinks correct+load-bearing;
rope assignment correct; window nearly neutral at seq 1024).

MECHANICS ISOLATION (agents/tmp/layer_ref_check.py + sweep_capture.py):
world=1 LayerRef w/ full-ckpt tensors vs world=4 presplit sweep on the
same 64 random ids (seed 7), exits [0, 2]:
- layer 0 (DENSE, no MoE): ALREADY DIVERGES — ref rms 0.026 vs sweep
  rms 0.065 (rel diff 2.4x). /tmp/layer_sweep_tp4.pt, /tmp/layer_ref_*.pt
=> bug bounded to layer-level MECHANICS, not MoE math: candidates =
TP head slicing of qkv under GQA (kvh=1/rank; repeat_interleave maps
q-head h -> kv-head h//8), per-rank sink slicing sink[h0:h0+qh], o_proj
col slicing, dense-MLP inter slicing, or tp_sum ordering.
NEXT (fresh session): within-layer bisect — compare x+attn(norm1(x))
and the MLP separately between sweep and ref at layer 0 (extend
sweep_states to capture mid-layer, or replicate the sweep's per-rank
attn math manually on 1 GPU). The layer-2 compare needs exits fix
(capture used [0,2]; compare indexed wrongly once — rerun with the
saved files). Then fix, re-run g25 NLL gate (target 2-4), smoke, and
LAUNCH RUNG 1 (scripts/ptqr_flint_rungs.sh 1).

## LAYER-0 BISECT STATE (turn 11, 2026-11-07 12:0x) — contradiction to resolve

Sim bugs fixed along the way: (1) per-rank _embed partials must be
SUMMED before the layer (tp_sum=identity in sims); (2) sim must sum
LAYER INCREMENTS (_apply_layer(x) - x), not outputs (residual 4x).

VERIFIED MATCHING (same-process, same inputs): embed exact; normed q/k
per rank vs full (rounding); ATTENTION A vs B-per-rank-sum max 0.0039;
MLP A vs B-per-rank-sum max 0.0078; LayerRef module == hand-compose
==bitwise; module-attn == hand-attn ==bitwise in-process.

CONTRADICTION: A layer-out rms 0.0258 (attnA rms 0.0848, mlpA rms
0.0580 — parts cancel) vs B increment-sum rms 0.109/hB 0.0607; and
the C (4-rank sweep capture) rms 0.0646. If both component matches
hold, hB MUST equal hA — measured max|hA-hB| 2.20. ALSO: two probes
disagreed on A-attention rms (0.0920 vs 0.0848) with "identical"
inputs — one external reimplementation is lying (suspect the first
attn probe's x: it used xs_sum; today's used F.embedding — embed
verified equal, yet outputs differed => one probe's inputs actually
differed; most likely RNG: probe ORDER of teacher-creation vs randint).

NEXT (fresh session, ~5 min): copy _apply_layer's body into a probe
script verbatim and print per-rank attn_partial/mlp_partial rms
INSIDE the real code path (no external reimplementation), then diff
against layer.attn/layer.mlp on the same x. Then: g25-style NLL gate
on the FIXED path, smoke, LAUNCH RUNG 1. Artifacts: /tmp/attn_ab.pt,
/tmp/layer_ref_0.pt, /tmp/layer_sweep_tp4.pt; probes in agents/tmp/
(layer_ref_check.py, sweep_capture.py, rank_sim_check.py).

## TURN 12 (2026-11-07 12:1x): RNG ghost killed; attention A-vs-B STILL differs

ids drawn FIRST (before teacher/layer creation) in one process; x
verified BITWISE equal (rms 0.112). Verbatim-copied _apply_layer
attention per rank vs LayerRef.attn on the same x:
  A attn rms 0.0848 | B attn-sum 0.0920 | max|diff| 0.42
  A mlp rms 0.0580 | B mlp-sum 0.0281 | max|diff| 2.06
(the earlier "0.0039 attention match" probe had different actual inputs
— RNG ordering; its result is VOID.)
Both code blocks are textually identical (normed q/k verified equal;
sink slice, mask, repeats, o-slicing all re-checked). Remaining
suspects for the A-vs-B attention diff: (1) RmsNorm module vs _rms
bit-level — compare normed q/k BITWISE in the same process (earlier
probe: max|dq| 0.001 — but that probe is also suspect!); (2) the o_proj
partial summation (compare per-rank out vs module's head-slices of
`out` BEFORE o_proj); (3) GQA mapping: interleaved (h//rep) vs TILED
(h%rep) — both A and B use interleaved HERE, but if the REAL model is
TILED, per-rank presplit kv-head assignment (rank r -> kv-head r) is
wrong for q-heads [8r..8r+8) — CHECK THE VENDOR: sglang RadixAttention
GQA convention and the upstream flint kernels' kv-head index math
(kvh*_GQA + eh in decode_attention.py = INTERLEAVED ✓ — so presplit is
right; skip).
NEXT (fresh session): single-process probe printing (a) bitwise normed
q/k equality module-vs-teacher for rank0 heads 0-7, (b) module out
head-slices vs teacher rank out, (c) attention probs row 0 head 0
both paths. The first diverging tensor is THE bug. Then NLL gate ->
smoke -> RUNG 1.

## TURN 13 (2026-11-07 12:3x): ghosts identified; one clean test remains

1. GEMM shape non-determinism: F.linear with [5120,2048] vs [1280,2048]
   weights gives different bf16 results for the SAME rows (ROCm
   accumulation order) — explains ALL the ULP-level "divergences"
   (0.004-0.015 weight-value diffs, normed q/k "mismatches", the
   staged-weight ghost probe). NOT bugs.
2. The serial rank-sim is INVALID for the MLP: with tp_sum=identity,
   rank r's silu-MLP ran on x+attn_partial_r; the real path all-reduces
   attention first (mlp on the SAME full x). mlpA(0.058) vs mlpB(0.028)
   rms gap = artifact. Only the distributed path is testable there.
3. presplit file, pread, staged weights: ALL bitwise-verified (again).
REMAINING REAL QUESTION, the only one: LayerRef world=1 layer-0 (rms
0.0258) vs the REAL 4-rank sweep capture (rms 0.0646) on identical ids
— 2.5x rms gap is NOT noise. All cross-process compares are now
suspect (RNG order ghosts twice already).
CLEAN TEST (next session, ONE 4-rank job): extend agents/tmp/
sweep_capture.py — after sweep_states, on RANK 0 ONLY, build the
LayerRef world=1 layer-0 from full-ckpt tensors (safe_open, in-process,
ids already fixed), run it, and compare against the captured state IN
THE SAME PROCESS. No cross-process anything. If they differ -> inspect
where (the C-side _apply_layer ran distributed; A-side single) — dump
both attn/mlp partials from inside real code. If they MATCH -> the
teacher was correct all along and the NLL 11 problem lives elsewhere
(e.g., the TRAINING-side student or eval path pairing).

## TURN 14 (2026-11-07 12:5x): divergence CONFIRMED in-process

agents/tmp/sweep_capture2.py (one 4-rank job, same process on rank 0,
same ids): A(LayerRef w1) rms 0.02576 vs C(distributed sweep) rms
0.06464, max|A-C| 1.105. REAL, ghost-free. Artifacts:
/tmp/same_process_l0.pt (hA, hC, ids).
The distributed layer-0 emits ~2.5x the residual energy of the
single-rank reference. Attention math verified equivalent (ULP);
MLP partials in the real path run on the SAME post-reduce input as A.
=> next: instrument INSIDE _apply_layer at layer 0 on rank 0 in the
same job (env-gated prints of attn_partial/mlp_partial rms + the
post-tp_sum x), and cross-check each against the A-side equivalents
computed from /tmp/same_process_l0.pt ids. The doubled energy smells
like a DOUBLE-COUNT (e.g., a partial summed twice, or o_proj/mlp
partials overlapping across ranks). Suspect list for double-count:
the tp_sum in _apply_layer runs all_reduce on a tensor that ALREADY
includes a rank-local residual copy; or _TP_STATE group semantics
(all_reduce on the WORLD group vs tp subgroup applying to all ranks
of a different comm) — CHECK the actual x each rank carries AFTER
layer 0 across ranks (they must be IDENTICAL post-reduce; if they
already differ pre-reduce in composition, the bug is upstream of the
reduce).

## TURN 15 (2026-11-07 13:3x): dense-tp_sum A/B + the real remaining suspect

Capture-harness artifact identified: sweep_capture*.py never set
_TP_STATE, so ALL capture-script sweeps ran with tp_sum=IDENTITY
(rank3 zeros = no ids in its vocab range). The in-trainer sweep DOES
reduce (student's shard call sets _TP_STATE). So the A-vs-C 2.5x was
my harness's artifact — BUT the missing dense-tp_sum was still a real
trainer bug (fixed; toggle FLINT_NO_DENSE_SUM=1 kept for A/B).

Clean A/B (--data_seed 123, identical windows, kd_cache 2):
  OLD (no dense reduce): w0 NLL 10.00 top1 2.5% | w1 9.14 top1 4.3%
  NEW (dense reduced):   w0 NLL 27.64 top1 0.1% | w1 10.29 top1 1.4%
The fix made it WORSE — and both remain garbage (uniform = 11.9).
Dense layers are 2/49; the dominant surface is the 47 MoE layers.
=> The remaining bug is in the MoE layer path (or affects everything
equally: e.g., qkv GEMM staging, rope tables on real positions, or
the residual composition). NEXT (fresh session): per-layer NLL probe —
teacher sweep_states through an EXIT at EVERY layer boundary on a
natural-text window, computing per-layer "hidden->lm_head" NLL decay
to find the first layer where predictability collapses; also A/B the
MoE branch (replace routed+shared with shared-only / routed-only) —
one of them should dominate; and verify the shared expert's gate/up
slices in the presplit against the full ckpt (only rank0 L2 verified
so far). Serving-side oracle (vLLM) still available for hidden-dump
comparison if needed.

## TURN 16 (2026-11-07 ~14:0x): NLL-DECAY CURVE (agents/tmp/nll_decay.py)

Teacher sweep_states, exits at ALL 49 layer boundaries, natural-text
window (corpus[:1025]), rank-0 final-norm+lm_head at each depth:
NLL ~46-48 FLAT at every depth (uniform = 11.9 — hiddens are
ANTI-informative from layer 0), rms 0.099@L0 -> 0.559@L48 (residual
barely grows; last 6 layers do the only writing: NLL 51->30, rms
0.25->0.56). Depth-48 NLL 30 on this window (window-dependent 9-30).
INTERPRETATION: hiddens carry ~no token info from the very first
layer, and true tokens get LARGE NEGATIVE logits — sign/inversion-
class error affecting everything, not a per-layer MoE bug. Since
attention semantics + all weight slices are verified, the prime
remaining suspects are SHARED-BY-BOTH-paths assumptions: (1) the
fused qkv ROW ORDER (q|k|v assumed; HF could store interleaved per-
head q/k/v — CHECK by verifying qkv block structure: q rows should
have per-head norm-scale patterns matching q_norm); (2) rope table
INDEXING on real positions (cos built for max_pos 262144; verify a
few rotations numerically vs formula); (3) embed row order. NEXT
(fresh session): (a) in-process serving-model per-layer hidden dump
on the SAME window vs sweep_states (the serving boot works; A/B at
layer 0-2 localizes shared-vs-unique bugs); (b) if shared: dump the
vendor sglang wheel's attention forward (already unzipped at
/tmp/flint_sglang_src) for the qkv split order and rope details vs
mine. NOTE the flat curve says probing MID layers was uninformative
because hiddens never carry info — the bug is at or before layer 0's
INPUT: embed lookup, positions, or the qkv projection feeding it.

## TURN 17 (2026-11-07 ~14:4x): probe bugs fixed; the layer-0 anomaly

1. nll_decay.py (and ALL capture probes) had tp_sum=identity (missing
   _TP_STATE) — now set explicitly in the probe. REDUCED decay curve
   STILL flat anti-informative: NLL 44@L0 -> 25@L48, rms 0.022 -> dips
   0.009@L6 -> 2.8@L48.
2. Sink-units fix (raw-qk, scaled by 1/sqrt(d) — kernel formula) is
   CORRECT per the FA4 source but is a measured NO-OP: identical NLL
   to 4 decimals. Real scaled-score maxima dwarf sinks (+10), so the
   sink term underflows in my softmax either way. Sinks: CLOSED.
3. THE anomaly: depth-0 (after layer 0 = no-rope full-attn + dense
   MLP, ALL components and weights verified) SHRINKS the residual
   (0.11 -> 0.022 rms) and produces anti-informative hiddens. The
   attention+MLP increments must be ANTICORRELATED with the embed —
   characteristic of misread o_proj/down_proj weights or a sign-flip,
   yet bytes are verified...
NEXT (fresh session, the one un-done oracle step): dump the serving
vLLM model's LAYER-0 hidden on the SAME corpus window (in-process GT
boot works; hook model.model.layers[0] forward output) and diff
against sweep_states[0]. Serving NLL is also garbage (20.4) — if
serving layer-0 MATCHES the teacher's, the bug is in something BOTH
share that is NOT checkpoint bytes (e.g., HF checkpoint stores
SOFTMAX-NEGATED or transposed o_proj convention, or tie/scaling in
generation_config); if serving differs, the side with sane decay wins.
Also cheap to try: SKIP attention entirely at L0 (identity attn) in
the probe — if depth-0 NLL becomes ~12 (uniform-ish, informative
embed), the corruption is inside the attention block; likewise skip
MLP.

## TURN 18 (2026-11-07 ~15:1x): skip-bisect + embed/head exonerated

FLINT_SKIP0 bisect (exits [0,1], natural window, reduced):
  skip L0 attention: depth0 NLL 46.9 rms 0.109 (=embed+MLP only)
  skip L0 MLP:       depth0 NLL 45.8 rms 0.061
  => BOTH components individually skippable and NLL stays ~46:
     neither block alone causes it; the anti-informativeness appears
     as soon as ANY trained transform touches the embed.
CPU probes (no GPU needed): raw embed -> lm_head NLL 12.85 (sane,
~uniform — embed/lm_head alignment FINE, cos(embed,lm rows) ~0 as
expected untied); embed + final-norm(w*x) -> NLL 47.0; (1+w) -> 110.
Norm weight stats: final-norm w in [0.007,1.77] mean 0.71; L0
input_layernorm w in [1.2,11.3] mean 4.25 (gains >1 — plausible
trained, but high). mid-layer hiddens through final-norm+head giving
44+ at EVERY depth remains the signature.
=> The remaining space: something EVERY layer does. Per-layer:
rmsnorm(w*x verified vs kernel), attn (verified vs vendor+kernel),
rope (neox, theta 1e4, tables built fresh — VERIFY numerically vs
formula on a few positions!), mlp/moe (verified). NEXT (fresh):
(1) numeric rope-table check (cos/sin vs closed form at pos 0,1,513,
1024) — a wrong inv_freq (e.g., computed with float32 arange bug or
262144-length truncation) corrupts EVERY sliding layer while no-rope
full-attn layers 0/1 should then be CLEAN — but depth-0 NLL 44
contradicts... unless the anomaly at L0 is different: with FLINT_SKIP0
BOTH set (skip attn AND mlp) run once: if depth0 NLL == embed-only
(47) then layer 0 was NEVER the problem and the flat curve is just
"uninformative hiddens" at all depths (which changes interpretation:
maybe hiddens only become predictive after MORE context/the full
stack, and the true end-to-end NLL ~25-30 IS the bug surface);
(2) serving-model hidden dump at depths 0,1,2,48 vs sweep_states.

## TURN 19 (2026-11-07 ~15:3x): flat curve is BASELINE, not the bug

Skip-BOTH L0 control: depth0 NLL 47.05 == CPU embed+finalnorm+head
baseline (47.04) — the flat ~44-47 decay curve is just the embed
readout baseline (mid-layer probes are UNINFORMATIVE-BY-BASELINE in
this arch). NOT diagnostic. Rope table verified numerically OK (exact
<=1024; 1e-3 fp32 noise only at pos 100k).
THE REAL SIGNAL: depth-48 NLL 24.9 vs uniform 11.9 vs expected ~2.5 —
the full stack actively destroys next-token info. Residual rms growth
0.11->2.8 looks healthy. Everything per-component verified.
=> NEXT (fresh session): serving-model hidden dump at depth 48 (and
0,1,2) on the SAME window vs sweep_states — the last un-done
differential. If they MATCH, the bug is shared and upstream-of-both
(e.g., a checkpoint interpretation both get wrong — try transposed
o_proj/down_proj conventions [verified shapes forbid it], negative
sink sign, k_norm applied post-rope, or a Qwen3-style ATTENTION
OUTPUT scaling). If they DIFFER, the saner side wins and localizes.
ALSO: quick corpus sanity — feed the teacher a MEMORABLE sentence and
greedy-argmax a few tokens (is ANY local structure predicted? e.g.
repeated headers) to see whether SOME contexts work.

## TURN 20 (2026-11-07 ~16:1x): serving hidden dump — API absent; pivot

return_hidden_states does NOT exist in this fork's v1 engine
(SamplingParams rejects it; AuxOutputConfig only routes experts).
PIVOT for the serving-vs-teacher differential: use the sitecustomize
import-hook (agents/tmp/worker_patch/sitecustomize.py — imports OK
standalone; the earlier boot failure was the hook breaking vllm's
registry import ordering — fix by hooking AFTER first model use
instead: patch FlintDecoderLayer.forward inside the hook to (a) print
per-layer hidden rms for the first forward, (b) torch.save layer
0/1/2/48 hiddens to /tmp/serving_states_r{rank}.pt on the LAST
prefill chunk — then compare vs /tmp/teacher_states_r0.pt (teacher
capture of exits [0,1,2,48] on the same window already wired in
agents/tmp/nll_decay.py, NOT yet run — run it first, 4-rank, ~6 min).
Teacher logits on this window: NLL 24.9 (depth 48). Compare per-layer
rms AND state vectors; first divergent layer = the bug (or shared
garbage => checkpoint-interpretation class).
The GT hidden script (scripts/flint_gt_hidden.py) kept for when the
hook dump works.

## TURN 21 (2026-11-07 ~16:5x): serving-vs-teacher FIRST differential

Worker-side serving dump WORKED at runtime ([serve L] prints): layer
rms L0 0.0202 / L1 0.0042 / L2 0.0011 — the serving model's residual
COLLAPSES through the dense layers almost exactly like the teacher's
(0.0227 / 0.0069 / 0.0056). Two independent implementations agree on
the collapse => either REAL model behavior (residual shrinks by
design; last layers rebuild to 2.8) or a misread BOTH share. The
saved state files came out near-zero (hook save raced dummy-run /
graph capture overwrites) — vector-level cos/diff INCONCLUSIVE; redo
the save keyed on the LAST forward only (flag off after first real
prefill; skip dummy runs by checking input_ids sum or a capture count
> 3).
NEXT: fix the dump capture; if vectors then MATCH (cos>0.9) the two
impls agree numerically and the bug is shared-and-upstream (top
candidate: BOTH read layernorm weights correctly but the true model
divides the residual by something — e.g. per-layer residual scaling
or embed*sqrt(d) init convention — CHECK the vendor wheel's
FlintDecoderLayer.forward residual line). If they DIVERGE at L0
already, wire the LayerRef single-process L0 (verified equivalent
hand-math) as the tiebreaker.

## TURN 22 (2026-11-07 ~17:2x): vendor residual = standard

Vendor FlintDecoderLayer: standard Qwen3Moe pre-norm residual via
sglang LayerCommunicator (norms + scatter; no per-layer scaling, no
sqrt(d) embed convention). So the L0-2 residual collapse is NOT a
missed residual scale in the vendor reference either. State:
- teacher NLL 24.9 (depth48), serving NLL ~20 (chat) — both garbage,
  both collapse L0-2 identically, all components verified vs vendor.
REMAINING HYPOTHESES (ordered):
1. The checkpoint at <models>/flint-bf16 — byte-identical to HF — is
   NOT the weights the vendor benchmarks (e.g., HF repo hosts a
   DIFFERENT (quantized/base) variant; model card says benchmarks ran
   via sglang+FA4 on B300 with nvfp4 assets...). CHECK: the HF repo
   file list — model-*.safetensors ~80GB bf16 = the full model. The
   sglang serve scripts use THIS repo. So weights right.
2. THE ROUTER: vendor `sigmoid_routing` selects top-k on
   scores+bias but weights = raw scores; renormalized. Mine matches.
   BUT expert_bias defaults zeros AND the checkpoint has NO
   mlp.gate.bias tensor — vendor comment says "Checkpoints without a
   router bias leave this at zero" — fine.
3. q_norm/k_norm placement: vendor applies to FLAT [T, H*D] q/k
   (apply_qk_norm reshapes per-head, norm, flatten). Mine: per-head on
   [B,S,H,D]. Equivalent.
4. UNTESTED: attention on the LAST token pattern / causal mask off-by-
   one in MY eval only (teacher NLL probe uses logits[i] vs y[i] —
   aligned by construction).
5. The REAL possibility everyone missed: the corpus window's first
   tokens start MID-DOCUMENT (no BOS!) — tok[:1025] raw slice. A model
   expecting <|begin_of_text|> may produce garbage NLL on BOS-less
   text! The chat-templated serving probe (which HAD proper specials)
   gave NLL 20 — but that's the serving path (broken backend?). THE
   TEACHER was NEVER tested on BOS-prefixed natural text!! NEXT (2-min
   CPU probe): prepend 150000 (bos) to the window in nll_decay and
   rerun the NLL at depth 48. ALSO try a repeated-pattern string.

## TURN 23 (2026-11-07 ~17:3x): BOS probe — no effect

FLINT_BOS=150000 window: depth0/1/2 NLL 43.97/45.78/47.84 ==
BOS-less (43.9/45.5/47.8). BOS not transformative (depth-48 line
didn't print — probe ended early; early depths identical anyway).
HYPOTHESIS LIST after 23 turns (all components verified vs vendor
wheel; weights/config/tokenizer byte-verified vs HF; both my impls
AND upstream-vLLM-on-this-fork produce garbage that AGREE with each
other): the divergence must be in a convention ALL FOUR readers share.
Remaining un-verified conventions: (a) qkv fused ROW ORDER inside
q_proj? — no, separate tensors; (b) WINDOW SEMANTICS on SWA layers
(mask inclusive/exclusive — off-by-ones degrade not destroy);
(c) THE CHECKPOINT ITSELF may require sglang's fa4-learned-sink-fix
patch semantics — i.e., the HF WEIGHTS were TRAINED/EXPORTED with a
sink convention that ONLY the patched FA4 kernel implements (the
vendor SHIPS the patch as REQUIRED: "not in any SGLang release" —
maybe the sinks need the rescale form on the OUTPUT side: attn_out
*= sink_rescale = exp(old_max - new_max) — a per-head OUTPUT SCALE I
never applied!). THE SINK RESCALE AFFECTS OUTPUTS even when the sink
term itself underflows: attn_out must be multiplied by
exp(max(scores) - max(max(scores), sink)) when sink > score-max.
With sinks (+2..+10 RAW units = +0.2..+0.9 scaled) vs score maxima
~30-50, sink < max => rescale = 1 — no effect. VERDICT: not it
either, UNLESS score maxima are small in trained nets (recheck
actual score max stats, not assumed).
NEXT SESSION: (1) dump actual per-head score maxima + sink values
co-located (one layer, one window) to settle the sink-rescale
question numerically; (2) fix the serving dump capture (last-forward
keying) and get the L0/L48 vector cosine; (3) if both still agree and
stay garbage, escalate to the user: four agreeing implementations on
verified inputs suggests either the HF weights need a not-public
serving recipe, or something in the config semantics (e.g.
softmax_type "learnable" has additional machinery beyond the sink
vector — check HF config docs/upstream PR #62938 discussion).

## TURN 24 (2026-11-07 ~17:5x): sink-rescale CLOSED numerically

FLINT_DBG_SCORES at L2 (real window, rank 0): per-head score max
min 2.0 / mean 21.6 / max 44.0 vs raw sinks 0.61-5.81 (scaled
0.05-0.52). Sinks are NEGLIGIBLE in the softmax denominator; the FA4
patch's output rescale is a no-op here. Sink machinery: FULLY closed.
ALL self-testable hypotheses now exhausted. Standing evidence: four
implementations (teacher, LayerRef, fork-ROCm-vLLM, vendor-semantics)
agree numerically (collapse L0-2 signatures identical); inputs
byte-verified vs HF; every component cross-checked against vendor
source; NLL depth-48 ~25-30 (worse than uniform). REMAINING: (a)
serving-vector cosine with last-forward-keyed dump (mechanical fix);
(b) external: upstream vLLM PR #62938 discussion / RecursiveSI
confirmation of the HF checkpoint's serving convention (the vendor's
own NOTICE/README may state whether the HF safetensors need their
sglang build; their serve scripts use THIS repo so weights should be
right — but nothing else is left). If (a) shows agreement (expected),
escalate to user with the evidence table.

## TURN 25 (2026-11-07 ~18:0x): vendor kit README confirms assumptions

sglang/README: serves THIS HF repo (revision-pinned to latest sha);
"Flint weights loaded: 585 parameters matched the checkpoint";
requires the fa4-learned-sink-fix ("fixes learned-sink
normalization") — the PATCHED semantics, which we implement (and
measured negligible). No special checkpoint convention documented.
B300/H200 only, sglang 0.5.16, bf16. => weights + semantics should be
right in our implementations; the mystery is NOT a documented serving
convention.
STATUS: all local and documented-external hypotheses exhausted. Next
turn: (1) fix serving-dump keying for the final vector cosine; (2) if
agreement holds, fetch upstream vLLM PR #62938 (Flint model) for any
numerics notes; (3) then escalate to user with the full evidence
table (four agreeing implementations, verified inputs, NLL 25-30).

## TURN 26 (2026-11-07 ~18:2x): keyed dump missed; turn budget end

1024-token key never matched (chunked prefill splits the prompt;
layer-hook input shape differs per chunk) — 0 dumps, NLL line lost to
an early exit. Vector cosine still unobtained. NEXT SESSION, in
order: (1) key the hook on "input includes our exact first token id
pattern" OR simply dump EVERY forward to unique files and pick the
one whose rms matches the [serve L] prints of the real prefill (the
last batch of 50 [serve L] lines before generation); (2) cos vs
teacher at L0/L48; (3) fetch vLLM PR #62938 for numerics notes;
(4) escalate to user with evidence table if agreement holds. All
other deliverables (trainer/KD-cache/exporter/draft/AIS/serving-boot/
ladder) are committed and ready.

## TURN 27 (2026-11-07 ~18:4x): repetition probe — DECISIVE negative

8-token random pattern x128 periods (1024 tokens): top-1 cycle acc
after 2+ periods = 15.9% (chance 1/150k; "wrong-position pattern
token" = 12.5%). The teacher stack cannot track a period-8 cycle —
deep model-wide corruption, not distribution shift. Combined with:
four agreeing implementations, byte-verified inputs, vendor-confirmed
weights+semantics, every component verified — this is beyond local
hypothesis space. FINAL PATH: (1) serving vector cosine (mechanical),
(2) vLLM PR #62938, (3) escalate. agents/tmp/repeat_probe.py is the
cleanest single reproduction (any environment, 4 GPUs, 6 min).

## TURN 28 (2026-11-07 ~18:5x): upstream PR 62938 — FP32 RESIDUAL clue

PR numerics notes: "FP32 fused add + RMSNorm, rounded once. FP32
gate/up for the routed experts. FP32 RoPE table, BF16 router logits.
Untied LM head produces FP32 logits." My teacher keeps the residual
stream in BF16 — with early-layer residual rms collapsing to
0.001-0.005, bf16 accumulation/adds there may catastrophically
cancel (8-bit mantissa at 1e-3 scale vs per-layer increments).
NEXT (concrete, next session): teacher run with FP32 residual stream
(_apply_layer keeps x fp32; weights upcast per-GEMM; _rms fp32
already) + fp32 lm_head — rerun repeat_probe (target: cycle acc
>90%) and NLL gate. This is the first hypothesis matching the
collapse signature that was NOT yet tested.

## TURN 29 (2026-11-07 ~19:0x): FP32 residual — dead; investigation terminal

FLINT_FP32 teacher (fp32 residual stream, fp32 staged weights, fp32
lm_head — exactly the PR's reference numerics): repeat-probe cycle
acc 16.6% vs 15.9% bf16. Numerics eliminated. EVERY hypothesis now
measured-dead: norms, sinks (units+rescale+values), rope (tables,
assignment, theta), window, GQA mapping, router, shared expert, BOS,
dense-MLP reduce, fp32 accumulation. Four implementations agree;
inputs byte-verified vs HF (the exact bytes the vendor serves); all
documented conventions cross-checked. Remaining unknown requires
EXTERNAL info: upstream PR #62938 review discussion, RecursiveSI
contact, or a reference GPU environment (B300/H200+sglang) to compare
per-layer hiddens. Web search quota-blocked (HTTP 403) this session.
Reproduction: agents/tmp/repeat_probe.py (4 GPU, 6 min, acc 16%).
Deliverables ready: trainer+KD-cache, exporter, draft port/cache/
trainer, AIS fix, booting Flint ROCm serving path, rung ladder —
all committed; ladder unblocks on the teacher NLL gate.

## TURN 30 (2026-11-07 ~19:1x): links processed — reference VALIDATED

User links: vllm blog + PRs 63957 (glm47 parser fix) / 63986
(silu_and_mul TP-shard crash fix). Key facts: (1) WoosukKwon
validated BF16 RecursiveSI/flint at GSM8K 0.952 on ONE GB200, TP=1,
Flint-kernel path => weights + reference semantics PROVEN GOOD.
(2) Flint kernels target single Blackwell GPU; other configs fall
back to vLLM stock layers (the fallback is NOT what was validated).
(3) My tree's flint ops/* + configs.py are BYTE-IDENTICAL to
upstream main; model.py differs only in my ROCm port lines.
=> The mystery narrows to THE unexplained in-process contradiction
from turn 14: LayerRef(world=1, full ckpt) L0 rms 0.0258 vs teacher
(TP4 presplit sweep) L0 rms 0.0646 — 2.5x, same ids, same process,
components hand-verified matching (attn 0.0039, mlp 0.0078) yet
compositions differ. tp_sums are increment-only (verified by code
reading). THIS is the bug: something in the TP4 sweep path makes the
L0 composition wrong despite correct components — prime suspects:
(a) NCCL all-reduce touching x-carrying tensors somewhere (the _embed
reduce? double-count when combined with _apply_layer reduces), (b)
presplit staging aliasing (_pread frombuffer views shared between
ranks? no — per-process files), (c) the dbg runs themselves ran
unreduced in capture scripts (known artifact — the 2.5x WAS from
sweep_capture2 which sets _TP_STATE... VERIFY: that script created
teachers directly WITHOUT shard_flint_tp — _TP_STATE set? The script
did NOT call _TP_STATE.update!! => the 'in-process divergence' test
itself ran UNREDUCED = artifact!! The REAL never-done test: teacher
TP4 with _TP_STATE correctly set vs LayerRef world=1, same process.
NEXT SESSION: rerun sweep_capture2 with _TP_STATE.update(world=4)
after teacher creation; expect A==C; then NLL gate directly.

CORRECTION (same turn): repeat_probe.py DOES set _TP_STATE (verified
in the script) — so its 16% repetition failure stands as a REAL,
properly-reduced TP4 failure. The sweep_capture2 artifact explains
the turn-14 'divergence' only. Standing state: reduced TP4 teacher
fails repetition; its semantics match upstream (byte-identical
reference, GSM8K 0.952 upstream on the SAME weights at TP=1 Flint-
kernel path). The fork's serving (stock-layer fallback at TP4) also
fails — the validated path was TP=1+Flint-kernels, NOT the stock-
layer path. DECISIVE remaining experiment: teacher TP4 vs a KNOWN-
GOOD numeric on identical input. Options: (a) upstream stock-layer
path on any NVIDIA GPU (TP4) — user help needed; (b) port my teacher
to run TP=1 streaming (40GB/R... infeasible 32GB GPU); (c) TP=2
(40GB host/RAM infeasible)... So (a) or: (d) numerically verify the
TP4 sweep against LayerRef TP4 (shard_flint_tp on LayerRef — both
TP4, same sharding, in one job — if THEY differ, the bug is in my
teacher's sweep/staging; if equal, both share a semantic divergence
from upstream that survived inspection). (d) is runnable NOW and is
the next session's first action.

## TURN 31 (2026-11-07 ~19:3x): staging A/B in flight

agents/tmp/decisive_ab.py (rewritten): two StreamedFlintTeachers in
one 4-rank job — t(presplit+pread) vs t2(FULL-ckpt mmap path), same
ids (seed 7, 64 tok), _TP_STATE set, per-layer [0,1,2,48] state
compare printed + /tmp/decisive.pt. First attempt (resident
load_flint_shards) died on a known full-ckpt indexing quirk
(experts.91.up_proj). The t2 sweep uses the mmap full-tensor path =
known slow crawl (each rank reads every tensor; ~10-20 min for one
64-tok batch) — run is in flight without timeout (bash-0ffc4lhb),
verdict lands in /tmp/decisive2.log. INTERPRETATION GUIDE: presplit
!= full => pread/staging bug (fix, re-gate NLL); equal => staging
fine, divergence is in shared _apply_layer semantics vs upstream —
then the comparison target is upstream-on-NVIDIA (user help) since
fork serving (stock-layer TP4 path) is unvalidated upstream anyway.

## TURN 32 (2026-11-07 ~19:5x): PRESPLIT STAGING BITWISE-CLEAN

decisive_ab.py (layer-0-only, same 4-rank job, _TP_STATE set):
presplit+pread teacher vs full-checkpoint+mmap teacher, layer 0 on
64 seed-7 tokens => max|d| 0.00000, rms 0.03022 both. Staging paths
and presplit files FULLY exonerated. The teacher is self-consistent;
the divergence from upstream's validated path (TP=1 + Flint kernels,
GSM8K 0.952) is consequently in EITHER (a) my reimplementation's
semantics (survived every inspection) or (b) TP=4 vs TP=1 itself —
notably the stock-layer TP>1 fallback was NEVER validated upstream
(PR 63986 fixed a startup crash for TP4/TP8 only later; the fallback
path may be untested/broken upstream). Next session: run the fork's
serving on the SAME 64 ids at TP4 and compare ITS prompt logprobs to
the teacher's (already-measured serving NLL ~20 vs teacher ~25 on
different windows — a matched-window comparison settles shared-vs-
different quickly). If matched: both broken the same way, and the
only remaining target is upstream TP=1 numerical comparison (user
GPU help). If different: the fallback path is the serving bug and the
teacher needs the same fix.

## TURN 33 (2026-11-07 ~19:2x): matched-window oracle — SERVING SIDE DONE

Serving matched-window (64 corpus tokens, prompt_logprobs):
**SERVING NLL = 25.98** (upstream-vLLM-on-fork-ROCm). Teacher side
stuck: agents/tmp/teacher_lp.py faults 3/3 (HSA_STATUS_EXCEPTION in
an index kernel, S=64-specific; repeat_probe with S=1024 on the SAME
sweep path runs fine — probe-harness issue, likely the final
advanced-indexing line or an S=64 edge; add staged prints / reuse
repeat_probe structure to fix (~10 min next session)). Comparison
script ready (reads /tmp/serving_lp.pt + /tmp/teacher_lp.pt).
Once teacher NLL on the same window lands: ~26 => both broken
identically (only upstream TP=1 oracle remains); <<26 or >>26 =>
localize. Also earlier insight from serving lp run: LLM.generate
takes token ids POSITIONALLY ([ids.tolist()], sp) — the old
prompt_token_ids kwarg was the cause of several 'engine core exited'
false alarms (engine was fine; my script threw).

## TURN 34 (2026-11-07 ~19:4x): MATCHED-CONTENT DIVERGENCE FOUND

Teacher (causal: first-63 of a 1024 window == a 64-token prompt):
NLL 8.47. Serving on the same 64 ids: 25.98. SAME contexts, SAME
targets => the fork's serving ROCm path is BADLY WRONG and the
teacher is much closer to correct. ("engine core exited" before was
my prompt_token_ids kwarg bug — generate() takes ids positionally.)
Teacher full-window levels: window@0 NLL 25.1 (bins 10->31 rising,
looked positional), window@100000 NLL 28.2 (bins 34..20..31
non-monotonic) => the rise is CONTENT, not position. Teacher absolute
level varies 8.5-31 by content; still unhealthy vs 2-4 target.
THE PAIRING for future work: teacher = training data generator;
serving = deployment path; they MUST agree. Teacher is 3x better;
serving has its own bug (candidate: the ROCm eager path's rotary
rounding — vLLM rotary rounds cos/sin to bf16 (documented in
qk_norm_rope.py!) while the reference kernel uses fp32 tables; my
teacher uses fp32 tables) — PLUS maybe more.
NEXT: (1) serving-side fix: swap the eager `self.rotary_emb(...)`
call in the ROCm path for an fp32-table rotary (match the reference
kernel's numerics); re-run matched-window serving NLL (target: drop
toward 8-10 and beyond). (2) teacher-side: chase 8.5->2-4 (per-token
analysis on window-1 head; compare vs fp32-residual/other flags).

## TURN 35 (2026-11-07 ~20:0x): serving fp32-rope patch — INERT?!

Added fp32-table RoPE to the ROCm eager path (commit in log); serving
NLL on the 64-token window is BITWISE identical to pre-patch
(25.982864781031534 both) — 15 significant digits. Patch confirmed
present in the imported module source (inspect check). => The eager
rope block either (a) is dead on this execution path (check which
forward actually runs: maybe unified_attention_with_output's KV write
uses its own q/k before ours? no...), (b) positions/cos_sin indexing
produces values identical to the op's (implausible at 15 digits), or
(c) something upstream of the patch makes both variants equal (e.g.
rotary_emb None for all layers on serving — but then the old code
would have skipped too and outputs identical BECAUSE ROPE NEVER RAN —
CHECK: on serving, is cos_sin_cache present? If get_rope returns a
rotary whose cos_sin_cache is EMPTY or the op no-ops...). Next
session (cheap): add an env-gated print of positions.min/max +
cos_sin_cache.shape + a checksum of q before/after rope in the eager
path; run once. If rope is dead on serving, that IS the serving bug
(no positional encoding at all in SWF layers!) and explains serving
25.98 vs teacher 8.47.
Also note: spawn method issue — heredoc scripts die under spawn;
always run file-based scripts for vLLM.

## TURN 36 (2026-11-07 ~20:4x): alignment bug in MY probe fixed; serving lead

Teacher pairing was off-by-one in probes (row t predicts ids[t+1]; I
paired lt[1:] with ids[1:] — corrected to lt[:-1]). Corrected:
teacher first-63 NLL 10.13 vs serving 25.98 SAME targets (corr -0.20
— near-independent outputs). Divergence confirmed; serving ~2.6x
worse. Teacher full-window ~24.9 (bins 10.7 -> 30.8, content-driven).
PROMPT_LOGPROBS RUN THROUGH PREFILL — the aiter kernel inspected
(pa_decode_sparse) is the DECODE path. NEXT: read the PREFILL/extend
kernel's sink math in ../aiter (unified_attention triton kernels,
extend_attention.py; the prefill kernel folds sinks at line ~201
`sink_ptr + query_offset_1` — VERIFY virtual-key semantics there).
Primary serving hypothesis stays: sinks misapplied on prefill. Test
lever: run serving with the softmax_offset tensors zeroed via a
worker-side hook (fix the earlier sitecustomize: patch
FlintDecoderLayer.__init__ to zero attn.softmax_offset AFTER
construction — the checkpoint overwrites init values, so patch the
module's own softmax_offset post-load like the load_weights wrapper
did — but WITHOUT breaking the import registry: hook only when
`VLLM_IS_WORKER` env set...). Compare NLL: if zeroing sinks fixes
serving (drops toward 10-15), sinks/prefill is the bug.

## TURN 37 (2026-11-07 ~21:0x): prefill sink math = virtual-key (correct)

aiter unified-attention PREFILL kernel (used by prompt_logprobs):
M initialized to sink*log2(e), L=1.0 — the standard flash virtual-key
init, semantically correct; q-head indexing (kv_head*GQA+within)
correct. SINKS EXONERATED FOR SERVING TOO. Serving 26-vs-teacher-10
gap remains unexplained; candidates now: UA itself on this model
(GQA/window/KV-write path), the block-table/KV cache layout, or
chunked-prefill window semantics (chunk boundaries + SWA mask per
chunk). NEXT SESSION, with fresh budget: (1) serving with
`--enforce-eager` and chunked-prefill DISABLED (single-shot prefill;
rules out chunked-prefill window bugs — max_num_batched_tokens >=
1024); (2) if still 26: swap UA for a plain causal attention
implementation inside the fork model def for a one-off test (hack
the model's forward to use F.scaled_dot_product_attention with
explicit masks, no KV cache — pure-training-mode forward); the whole
serving-vs-teacher delta is then pinned to UA or model semantics in
ONE run. (3) Then the teacher gate (8-25 by content; needs 2-4) and
launch rung 1.
STATE: all code committed; probes in agents/tmp (gitignored);
serving lp saved /tmp/serving_lp{,_fp32rope}.pt; teacher /tmp/teacher_lp.pt.

## TURN 38 (2026-11-08): GOAL EXTENDED — Qwen27B + Bolt quality checks

New user requirement: "full quality checks on Qwen 27B and Bolt
configured according to the recipes before finishing." Scope to be
derived from docs/recipes/README.md (the canonical baseline) — see
next turns; likely: boot each recipe's serve script, run its
coherence + acceptance/KLD gates, record results (ledger rows
per the A/B doctrine). Do NOT finish the goal without this.
Also this turn: VLLM_FLINT_SDPA debug branch added (plain masked
attention, no KV write — isolates AITER UA vs model semantics);
import os fixed (worker NameError); eager/no-chunked-prefill serving
rerun in flight (gt_lp8).

## TURN 39 (2026-11-08 ~00:xx): eager verdict + SDPA run in flight

Serving --enforce-eager + no-chunked-prefill: NLL 25.73 vs 25.98
default — graphs/chunked-prefill EXONERATED. SDPA isolation run
(VLLM_FLINT_SDPA=1: plain masked attention w/ windows+sinks, no KV
write, bypasses AITER UA) in flight (gt_lp9). If it lands ~10 =>
AITER UA is the serving bug; ~26 => the bug is in the model/loader
path shared with UA. ALSO: goal extended with Qwen27B + Bolt full
quality checks (see turn-38 entry) — REQUIRED before finishing.

## TURN 40 (2026-11-08 ~00:2x): BREAKTHROUGH — AITER UA IS THE SERVING BUG

Serving with VLLM_FLINT_SDPA=1 (plain masked attention, windows+sinks,
no KV write, bypasses AITER UA): matched-window NLL **4.74** vs 25.98
with UA. Near-healthy! => the model/loader/ROCm port are CORRECT; the
AITER unified-attention path on this model (GQA kvh=1/rank, window
513, sinks, block 64, gfx908) is badly broken. Graphs/chunking were
already exonerated (eager 25.73).
ALSO: serving-SDPA 4.74 < teacher 10.13 on the same targets => the
TEACHER has its own ~2x gap to chase (smaller, real).
NEXT: (1) UA bug hunt: candidates = sink handling combined with
window, GQA head mapping at kvh=1, block-table/KV layout; probe by
disabling components (UA without sinks via zeroed offsets; UA without
window by pretending full_attention for all layers via cfg patch) —
each a serving run; OR diff UA vs SDPA outputs layer-0 directly
(hidden hook, single prefill). (2) teacher gap (10 vs 4.7):
attention-vs-next suspects. (3) Qwen27B+Bolt quality checks remain
REQUIRED before finishing.

## TURN 41 (2026-11-08 ~00:5x): UA toggles inert — structural bug

NOSINK: 25.982864781031534 (bitwise == default). NOWIN: same bitwise.
EXPLANATIONS: window toggle inert because the window is 64 tokens < 513
— windows never bind at this length (test must use a longer window);
sinks toggle inert because sink terms underflow numerically (as
measured everywhere). SDPA still 4.74 on the same 64 tokens => the
UA-vs-direct divergence at 64 tokens is STRUCTURAL: candidate = the
KV-cache write/read path inside UA prefill (auto=bf16 cache), GQA
kvh=1/rank mapping, or o_proj RowParallel under UA's layout — NOT
sinks/window. Read: UA backend DOES accept both (supports_sink True;
window_size wired at 3 call sites).
NEXT: (1) UA prompt_logprobs on a 1024-token window vs SDPA on the
same 1024 (windows bind >=513; two runs) — if they AGREE at 1024, the
bug is a single-block/64-token UA path (still fatal); if UA stays bad,
bisect the KV write: run with kv_cache_dtype bf16 explicitly, and
probe `attn` cache tensors after a prefill (hook). (2) teacher gap
(10 vs 4.7) continues. (3) Qwen27B+Bolt checks REQUIRED before finish.

## TURN 42 (2026-11-08 ~02:0x): UA pathologies multiply; backend pinning

UA on a 1024-token window HANGS (>15 min at 0% prefill; killed) — in
addition to garbage at 64 tokens (25.98 vs SDPA 4.74). So UA is both
wrong AND non-scaling on this model. NOSINK + VLLM_ATTENTION_BACKEND=
TRITON_ATTN still selects ROCM_AITER_UNIFIED_ATTN (log line "Setting
kv cache block size to 64 for ROCM_AITER_UNIFIED_ATTN") — the pinning
ignores both. NEXT LEVER: VLLM_ROCM_USE_AITER=0 entirely (fork's plain
ROCm triton path — qwen38-proven) + NOSINK; sinks are measured
negligible so the fork's non-sink backend is acceptable for the
recipe IF it computes sliding windows + GQA correctly. If that lands
NLL ~4-7, the Flint serving recipe answer = sinks off + fork triton
attention (documented deviation, journaled), avoiding the UA bug
entirely. Qwen27B+Bolt checks still REQUIRED; teacher gap (10 vs 4.7)
continues.

## TURN 43 (2026-11-08 ~02:2x): UA pin is hard; backend routing blocked

Even VLLM_ROCM_USE_AITER=0 + NOSINK + VLLM_ATTENTION_BACKEND=
TRITON_ATTN still boots ROCM_AITER_UNIFIED_ATTN for flint ("Setting
kv cache block size to 64 for ROCM_AITER_UNIFIED_ATTN"). The pin
survives every lever tried. For the RECIPE this means: a flint
deployment needs either (a) a fixed UA path (its bug: garbage at 64
tok, hang at 1024 tok — both model-specific; UA itself is fine for
qwen38), or (b) a real cached attention implementation in the fork
model (the VLLM_FLINT_SDPA branch needs KV-cache write/read added to
become deployment-viable; it currently proves correctness at NLL
4.74 on prefill), or (c) find the registry override path
(Attention layer's backend selection: check vllm/model_executor/
layers/attention/attention.py for the selector; the flint Attention
call may pass an explicit backend).
NEXT SESSION PRIORITY: (1) locate the registry/selector code that
forces UA (10-min read, no GPU); implement a supported override
(e.g., allow backend selection when sinks are None); (2) benchmark
the SDPA+cache route if override remains blocked. Teacher gap and
Qwen/Bolt quality checks remain queued.

## TURN 44 (2026-11-07 ~21:5x): BACKEND OVERRIDE FOUND — CLI flag works; env var is a no-op

`VLLM_ATTENTION_BACKEND` is NOT a vLLM env var in this tree (grep of
envs.py: zero hits) — that is why every env-var attempt in turns 42-43
silently kept the AITER unified-attention pin. The sanctioned lever is
the EngineArgs/CLI field `attention_backend` (arg_utils.py:811) which
sets `attention_config.backend`; the platform then takes the
"selected_backend" branch in rocm.py:get_attn_backend_cls and skips the
hard UA pin. Verified live: `LLM(attention_backend="TRITON_ATTN", ...)`
logs "Using TRITON_ATTN backend (selected via --attention-backend)" at
rocm.py:889. Matched-window serving NLL for TRITON_ATTN run in flight
(expect ~4.7 like the SDPA reference, not ~26 like UA).

Also found while reading the teacher: teacher `_apply_layer` scales the
sink by 1/sqrt(head_dim) (comment claims kernel uses raw qk units), but
the upstream decode kernel folds `sink * log2e` directly against the
*scaled* logits (decode_attention.py:689) — i.e. the sink is in natural
logit units and must NOT be scaled again. Serving-SDPA appends the raw
sink (match), teacher scales it (suspect). Teacher ablation
(FLINT_ABLATE=sinks) queued to test whether this closes the 10.13->4.74
teacher gap.

## TURN 45 (2026-11-07 ~22:0x): SCHEDULER REGRESSION found by the flint probe — fixed

The TRITON_ATTN serving run crashed at the first schedule step:
`AttributeError: 'AsyncScheduler' object has no attribute 'skipped_waiting'`
(scheduler.py:_prioritize_cached_waiter -> self.skipped_waiting). This is
the v0.33.0 upstream merge vs fork commit d01e4a253e ("bounded cached-turn
admission"): upstream 42dbe4578c reworked `skipped_waiting` into
`kv_holding_waiting` (queue) + `deferred_waiting` (set) but the fork's
prefix-fair-admission helper still read the removed attribute. It fires
whenever `VLLM_GFX908_PREFIX_FAIR_ADMISSION=1` (the serve scripts' default)
and the FCFS head is a cold prompt > 3*block_size (48 tokens) — i.e. every
qwen38/Bolt first prompt, and every flint probe.
Fix: `other = self.waiting if queue is self.kv_holding_waiting else
self.kv_holding_waiting` in scheduler.py; the fork profiler
(vllm/profiler/lifecycle.py) had the same stale attribute, updated to
`kv_holding_waiting`/`deferred_waiting`. Re-ran the flint TRITON_ATTN
probe with PREFIX_FAIR=1 + SHORT_PREFILL_COHORT=1 explicitly to exercise
the fixed path (result below when it lands).

Sink values are LARGE (checkpoint `softmax_offset`, fp32 per head):
L0 -9.0..9.9 mean 3.7; L2 -0.15..9.75; L20 1.97..7.28 mean 5.27;
L47 -0.75..5.9. exp(sink) is 1e1..1e4 — the learned sink is a real
"attend nothing" prior, not a rounding detail. Consequence: the old
`NOSINK` toggles being bitwise inert under fork-UA at 64 tokens means
the UA path was NOT applying the sinks at all — a leading suspect for
the UA corruption (SDPA-with-raw-sinks gives 4.74). Teacher fix applied:
removed the extra 1/sqrt(hd) on the sink in both `AttentionRef` and
`StreamedFlintTeacher._apply_layer` (kernel folds raw `sink*log2e` into
the scaled logits); re-measure queued. Next: SDPA+NOSINK A/B to prove
sink essentiality.

## TURN 46 (2026-11-07 ~22:0x): TRITON_ATTN == UA BITWISE — corruption is backend-independent

With the scheduler fix, `attention_backend="TRITON_ATTN"` booted cleanly
("Using TRITON_ATTN backend (selected via --attention-backend)",
rocm.py:889; PREFIX_FAIR=1 + SHORT_PREFILL_COHORT=1 exercised) and the
matched-window NLL is **25.982864781031534 — bitwise identical to UA**
(and to the older UA NOSINK/NOWIN runs). Two different attention kernels
producing bit-identical logits means the attention output is not
reaching the computation at all: `attn_output = torch.empty_like(q)`
filled by `unified_attention_with_output` is the prime suspect (impl
dispatch is by `self.attn.impl`, so the backend label does change);
SDPA (which computes `out` itself and copies into attn_output) is the
only branch that gives healthy numbers (4.74). Same signature explains
every previous bitwise-inert toggle (NOSINK/NOWIN/eager/chunking).
NEXT: teacher ablation `FLINT_ABLATE=attn_zero` (all attention
contributions zeroed) — if it lands ~25.98 on the same window, the
serving helper call is effectively returning zeros; then instrument
`attn_output.abs().max()` right after `unified_attention_with_output`
in the model (env-gated) to confirm and locate the real culprit.
Also: env-injection mystery resurfaces — `VLLM_BOLT_SHORT_PREFILL_COHORT`
and `VLLM_GFX908_PREFIX_FAIR_ADMISSION` were SET in engine workers even
though the launching shell had no VLLM_* vars; same class as the open
`VLLM_GFX908_RESIDUAL_INT8` mystery in upstream-v0330-merge.md.

## TURN 47 (2026-11-07 ~22:2x): CONFIRMED — serving attention contributes ~nothing

Teacher ablations on the SAME 64-token window (all inputs/targets
matched, teacher fixed-sink run since the sink rescale changed nothing:
10.1779 vs 10.13 at 1024 — sinks are real but not the teacher gap):
- default teacher: 10.18
- FLINT_ABLATE=attn_zero (every layer's attention output zeroed):
  **26.4425** ~= serving 25.9829 (UA and TRITON_ATTN, bitwise equal).
So the serving model is running with near-zero attention: the
`attn_output = torch.empty_like(q)` filled by
`unified_attention_with_output(...)` is effectively zeros, and the
remaining 0.46 NLL delta is the teacher-vs-serving MLP/rope/noise floor.
All three attention backends share the exact same
`if attn_metadata is None: return output.fill_(0)` branch, which also
explains every bitwise-inert toggle. The helper is a
`direct_register_custom_op` whose `get_attention_context()` reads
`forward_context.attn_metadata` — prime suspect: the flint layer names
or the custom-op context boundary make `attn_metadata` None during real
prefill. `VLLM_FLINT_DBG_OUT=1` (model.py, prints impl class, metadata
type, out_absmax, q_absmax on the first forward of each layer) run in
flight to confirm.
Teacher remaining gap (10.18 vs SDPA-serving 4.74) is still open and
SEPARATE; `attn_zero` bounds the serving side, not the teacher.

## TURN 48 (2026-11-07 ~22:2x): ROOT CAUSE — missing unified_kv_cache_update

vLLM declares `forward_includes_kv_cache_update = False` for UA, Triton
and ROCm backends alike (rocm_aiter_unified_attn.py:161, triton_attn.py:415,
rocm_attn.py:200). The KV write is therefore a SEPARATE op:
`Attention.forward` calls `unified_kv_cache_update(key, value, layer_name)`
(attention.py:555) which invokes `impl.do_kv_cache_update(...)`; only then
does `unified_attention_with_output` read the cache. My ROCm port called
only the attention helper (the comment claiming "UA owns the KV write" was
wrong — UA writes nothing in its forward; it has a separate
`do_kv_cache_update` at rocm_aiter_unified_attn.py:1492). So every
cache-based path read an all-zero cache: V=0 => attention output 0 =>
byte-identical logits for every backend, every sink/window/eager toggle
inert, SDPA (which bypasses the cache and computes from raw q/k/v)
healthy at 4.74. FIX: flint's ROCm eager branch now calls
`unified_kv_cache_update(k, v, self.attn.layer_name)` before
`unified_attention_with_output` (guarded by
`forward_includes_kv_cache_update`), passing the returned dep through.
This closes the whole serving saga (turns 34-46). Verification run next.

## TURN 49 (2026-11-07 ~22:2x): SERVING FIXED — NLL 4.66 on the production UA path

After the unified_kv_cache_update fix, the same 64-token probe (UA
backend, vLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1, no debug branch):
**SERVING NLL 4.6588** (was 25.9829) — matches the SDPA reference branch
(4.74) and the healthy band. The KV-cache write was the entire serving
saga: zero cache -> V=0 -> zero attention output; every backend and
toggle was inert because of it. Sanity from the instrumented run: the
profile-run forward legitimately reports attn_metadata=None and
out_absmax=0.0 (expected for the dummy profile pass), so the DBG_OUT
probe must gate on metadata presence to be useful for real forwards —
fixed next use. Also this run's `[dbgout]` lines came from TP0-TP3
profile passes (4 lines for layer 0), not the generate step.
Follow-ups queued: W=1024 long-window run (UA previously hung there),
teacher re-measure with fp32-table rope (10.18 -> ? expected ~4.7),
then the KD gate and rung ladder.

## TURN 50 (2026-11-07 ~22:3x): 1024-token UA works — NLL 1.39 (hang gone)

Same fixed stack, W=1024 prompt: **SERVING NLL 1.3898**, no hang (UA at
1024 previously stalled >15 min at 0% prefill on the zero-cache path).
64-token 4.66 / 1024-token 1.39 are healthy levels; window 513 and
block-64 cache paths are exercised. The serving saga (turns 34-49) is
closed: one missing `unified_kv_cache_update` call.
Remaining: the TEACHER is now the odd one out — 10.09 at the same 64
tokens (fp32 rope + raw sinks, both corrections in) and ~10.1 for the
first 63 of a 1024 window, vs serving 4.66/1.39. Teacher hidden-state
dump (sweep_states, exits=0..48) running now; serving hidden dump next;
per-layer RMS/corr comparison will name the first divergent layer.

## TURN 51 (2026-11-07 ~23:0x): hidden-dump probes iterated; teacher gap still open

Three probe iterations for the serving hidden-state dump (VLLM_FLINT_HIDDEN_DUMP):
(1) first attempt captured vLLM's zero-input warmup step (256 rows,
all-zero hidden states — the profile/warmup really does run with zero
inputs, which is why every layer's output is exactly 0 there);
(2) zero-check without a capture guard crashed in
profile_cudagraph_memory ("operation not permitted when stream is
capturing" on the host sync);
(3) fixing that consumed the dump flag on a capture pass (flag set even
though no save happened), so nothing was written.
Final probe: skip captures, zero-check for dummy steps, and set the
"dumped" flag only when a save actually happened. Run in flight.
Meanwhile W=1024/UA reconfirmed 1.39 and teacher stays at 10.09 for the
same targets (matched first-63 of the 1024 window: serving 4.69,
teacher 10.09), so the teacher is definitively the wrong side of the
gap and the per-layer diff is the way to localize it.

## TURN 52 (2026-11-07 ~23:3x): TEACHER BUG FOUND — the sink column was masked out

Per-layer hidden comparison (serving dump vs teacher sweep_states, same
64 ids) localised it: serving L0 matched an independent CPU fp32
reference of the *serving* math almost exactly (corr 0.9997, diff
4.8e-4), while the teacher's L0 matched that same reference computed
with **sinks disabled** (teacher L0 rms 0.02555 == ref-nosink 0.02554;
debug run: attn_partial rms 0.05565 == my sink-masked per-rank
computation 0.05564).
Root cause: `_apply_layer` builds the attention mask as
"True == masked" then appends `mask.new_ones(S, 1)` for the sink column
— i.e. the virtual sink key is masked to -inf in every layer. The
single-process `AttentionRef` uses the opposite (allowed) convention and
is correct. Introduced in 5790e5922b while reworking sink semantics.
Sinks in this checkpoint are large (L0 -9..9.9, mean 3.7; L20 mean 5.3),
so the teacher's attention was wrong at every layer -> NLL 10.09 vs
serving 4.66. Fix: append `new_zeros` (sink allowed); `dead` still masks
non-finite sinks. Teacher NLL re-measure in flight; expect ~4.7.
Note the earlier 0.107-rms "teacher" files were the FLINT_ABLATE=sinks
runs, where sink=-inf makes `dead` all-True and zeroes *all* attention —
not a valid sink ablation anymore.

## TURN 53 (2026-11-07 ~23:3x): TEACHER CLOSED — 4.6586 vs serving 4.6588

After allowing the sink virtual key, the teacher on the same 64-token
window: **NLL 4.6586**. Serving (UA + KV-write fix): **4.6588**. The
0.0002 delta is bf16 GEMM ordering. The teacher == serving now, so:
- the KD data generator is correct (this was the blocker for the ladder),
- the earlier "teacher gap" was entirely the masked sink column,
- sink values are essential; the `sinks` ablation now means "sink column
  -inf" and is a valid on/off switch again.
Serving status recap: 4.66 @64 tokens, 1.39 @1024 tokens, UA backend,
FP32 mamba N/A (no mamba in flint), int8 KV not yet enabled for flint.
NEXT: rung 1 of scripts/ptqr_flint_rungs.sh (60 steps, lr 1e-5), KD gate
(KLD + greedy agreement vs the bf16 teacher), then rung 2/3, draft PTQR,
int8 engine surfaces, recipe + publish, and the Qwen27B/Bolt quality
checks required by the objective.

## TURN 54 (2026-11-07 ~23:4x): RUNG 1 LAUNCHED

`scripts/ptqr_flint_rungs.sh 1` running (60 steps, lr 1e-5, seq 2048,
kd_cache 20 default, save/eval every 20, out models/flint-ptqr-r1).
First launch died instantly on `set -u` + unset LD_LIBRARY_PATH (script
bug, fixed: `${LD_LIBRARY_PATH:-}`); second was killed by the 600s
background-tool timeout mid-KD-save (15/20 files written); third runs
with no timeout. KD precache shape: 20 train + eval_steps val teacher
sweeps (one sweep each, seq 2048), rank 0 spills ~615 MB/batch files
(12.3 GB), teacher shard warm 107 s. Eval prints
`eval step N kld ... lm ...` — that KLD vs the bf16 teacher is the
rung-1 gate. After rung 1: rungs 2/3 chain via --init_from, then export
+ serving KLD/acceptance gates, then the draft.

## TURN 55 (2026-11-08 ~00:0x): QA gate map for the required Qwen/Bolt checks

Objective extension requires full quality checks on Qwen 27B and Bolt
"configured according to the recipes". Gate lists read from
docs/recipes/README.md (canonical) and docs/recipes/nemotron_bolt.md:

Qwen 27B (boot scripts/serve_recipe_qwen38.sh; port 8020):
  1. test_int8_kv_micro.py; 2. battery_gfx908.py 4/4;
  3. kld_probe_v2.py (real-artifact KLD; kld_gate_boot.sh <tag> wraps
     boot+probe);
  4. coherence curl;
  5. ua_live_soak.py -n 500 (stack assertions: AITER W8A8, UA, CAR=0,
     epilogue off, int8_block_g128 KV both models, fp32 mamba, TP4, C8,
     DFlash2; NOTE README gate text says NS=13 while the serve script
     defaults NS=6 — record the mismatch and run the gate with the
     script's configured pair).

Bolt (scripts/serve_recipe_bolt.sh; port 8020; key via VLLM_API_KEY or
/etc/llama/llama-api.key — the key file is not readable by this user,
script falls back to VLLM_API_KEY):
  1. same micro+battery; 2. C8 answer smoke acceptance ~4.55;
  3. 28,067-token retrieval cold + prefix replay;
  4. ua_live_soak.py --model nemotron-bolt --concurrency 8 -n 500
     (500/500 coherence);
  5. probe_bolt_quality.py --output logs/bolt-quality.json (52-prompt
     logprob corpus, 52/52 finite/readable, 10 strict-format);
  Bolt's recipe explicitly has NO matched official-reference KLD gate —
  implementation + coherence gates only.
Both need the 4 GPUs, so they run after the flint ladder; qwen38 serves
port 8020 and Bolt also 8020 — boot them one at a time.

Flint int8 surface plan (for the engine step): GPTQ GS128 W8A8 for
q/k/v/o, dense+shared MLP, and all 256 routed experts/layer (MoE — new
for this fork's dense-qwen stack, to be validated on boot); bf16
exceptions: router gate, qk norms, softmax sinks; embed/lm_head via the
existing loader int8 conversions (VLLM_GFX908_INT8_EMBEDDING /
INT8_LM_HEAD).

## TURN 56 (2026-11-08 ~00:0x): int8 MoE kernel identified — aiter grouped a8w8 blockscale

The flint experts (256/layer, top-6) are the only surface without an
existing fork int8 W8A8 path: AutoGPTQMoEMethod accepts int8 GPTQ with
group scales but selects a **wna16 (A16W8)** backend (`select_wna16_moe_backend`,
input_dtype None); the int8 MoE oracle supports per-channel int8
(`kInt8StaticChannelSym`) only. The sibling aiter checkout, however,
already ships the exact kernel semantics needed:
`aiter/ops/triton/moe/moe_op_gemm_a8w8_blockscale.py` —
`moe_gemm_a8w8_blockscale(x, w, x_block_scales, w_block_scales,
routing_data, gather_indx, scatter_indx, apply_swiglu=...)` with
BLOCK_SIZE_K=128 block scales over grouped experts (adapted from
triton-kernels matmul_ogs; routing via aiter moe_routing RoutingData).
Plan: flint MoE int8 W8A8 = per-128-K activation quant (reuse the dense
kernel's `_quantize_activation_per_block`) + expert weights packed as
[E,K,N] int8 with [E,K/128,N/128] scales + RoutingData from the existing
fused-MoE sort + this kernel for gate_up (swiglu) and down. Remaining
exceptions for flint: router gate, qk norms, sinks (bf16); embed/lm_head
via the loader int8 conversions; KV int8_block_g128; no mamba surfaces.

## TURN 57 (2026-11-08 ~00:0x): rung-1 KD crash (stale loader) + fix

First full rung-1 run completed the KD precache (20 train + 10 val
teacher sweeps, 531-538 s, 12.3 GB spill) then died at the first step:
`AttributeError: 'dict' object has no attribute 'to'` — the training
loop's `_kd_load` still returned the raw `torch.load` result while
`_precache` had been changed to write `{"logits":…, "y":…}` (the eval
path was already updated, the train loop was not). Fix: `_kd_load`
unwraps `obj["logits"]`. Relaunched. Precache is rebuilt on every run by
design (the unseeded dataloader shuffle makes old logits/`y` unmatchable
against a new iterator), so each attempt costs ~9 min before training;
add `--data_seed` + store `x` in the cache if this needs to be cheap.
Also this run: aiter grouped a8w8 blockscale MoE kernel parameterized
with `w_blockscale_n` (aiter commit 94a46f47d) so GPTQ per-column GS128
scales can be passed as [E, K/128, N] without re-quantizing experts.

## TURN 58 (2026-11-08 ~00:1x): rung-1 OOM at seq 2048 -> grad checkpointing

Second rung-1 attempt passed the KD-load fix and died at the first
training step in `FlintRef.forward_hidden` -> AttentionRef.forward:
CUDA OOM (30.8/32.0 GiB allocated, forward only). The student holds
18.5 GiB/rank of weights; the reference attention keeps fp32 [8, 2048,
2048] score/softmax tensors per layer for backward (~134 MB x 49 plus
matmul inputs ~7-10 GiB total) which does not fit next to the weights.
Fix: `--grad_checkpoint` (torch.utils.checkpoint, use_reentrant=False)
around the 49 decoder layers in FlintRef.forward_hidden, enabled by the
rung script. Forward peak becomes ~one layer + weights + KD logits;
backward recomputes layers (~30% slower). Relaunched.

## TURN 59 (2026-11-08 ~00:35-01:0x): rung 1 training; in-loop KLD moved UP

Rung-1 (with grad checkpointing, --grad_checkpoint) cleared the OOM and
is training: step 0 loss 2.347 (ce 2.245, kd 0.051); step 10 loss 1.771
(ce 1.634); step 20 loss 2.253 (ce 2.041) — per-batch CE is noisy.
In-loop eval (teacher-forced val KLD, tau=0):
  step 0  kld 0.080868  teacher_nll 3.2220
  step 20 kld 0.131807  teacher_nll 3.2220
KLD rose 63% after 20 steps. Two readings: (a) the scale-only PTQR
updates at lr 1e-5 are overshooting this model (per-batch CE at step 20
is also above step 10), or (b) the in-loop metric is not the gate — the
Qwen ledger recorded exactly this divergence (trainer-stack KLD said
improvement while the deployed semantics destroyed weights; the trusted
gate is export -> serve -> KLD/agreement, ledger PTQR_P1_* rows). Plan:
let rung 1 finish, export step 60 via scripts/ptqr_export_flint.py,
serve it, and compute the SER VING-gate KLD vs the bf16 teacher (the
teacher's val nll on the same windows is 3.222). Do not chain rungs 2/3
until the serving KLD says rung 1 is healthy; if rung 1 is worse than
the step-0 point, export step 0's checkpoint (already saved) as the
untrained RTN baseline and compare.
Also note checkpoint saves are 4 x ~12 GB (48 GB per save event, step
0/20/40/60) and take ~4-5 min each; disk 2.5 TB free.

## TURN 60 (2026-11-08 ~01:3x-01:5x): rung 1 done (artifact = step 40), exported

Rung 1 completed: 60 steps (step-40 eval kld 0.173934 vs step-0
0.080868 vs step-20 0.131807 — in-loop KLD worsened monotonically as
tau annealed to 0). The trainer's final-save condition skipped step 60
(loop exits at step==max before the modulo save); fixed to always save
the last step (last_saved tracking). So rung-1's kept artifact is
**step 40**; its step-0/20/40 scale files also remain, but the step
files are pruned to the latest, so a step-0 export would need the
pristine bf16 shards + step-0 scales (not attempted).
Export: `scripts/ptqr_export_flint.py --ckpt_dir models/flint-ptqr-r1
--step 40 --world 4 --src models/flint-bf16 --out
models/Flint-GPTQ-INT8-W8A8-GS128-PTQR-r1s40` -> 36,439 tensors packed
to GPTQ GS128, 35 shards, quantize_config bits=8 gs=128 sym (lm_head
false), config/tokenizer copied.
Serving-gate probe built (agents/tmp): teacher_val_topk.py (4-rank
teacher, top-20 logprobs + target NLL on a 1024-token val window at
TOK_OFF), serv_val_topk.py (vLLM prompt_logprobs=20 on the same ids,
top-20 + target NLL coverage), compare_val_topk.py (union-renormalized
per-position KLD, median/mean/p95, top-1 agreement, NLLs).
Teacher window: NLL 3.7097 (val window at offset 0).
Boots queued: bf16 flint (reference), then the int8 export.

Serving-gate reference (bf16 flint served, kv=auto, 1024-val window at
offset 0): median KLD 0.005459, mean 0.040420, p95 0.072471, top-1
agreement 947/1023 (92.6%), target coverage in top-20 73.6%. This is
the probe's bf16 noise floor (serving kernels vs the streaming teacher);
qwen's serving-gate numbers (0.0069 PTQR vs 0.0110 deployed) sit in the
same band, so the probe is calibrated. Int8 export boot in flight with
kv=int8_block_g128 (recipe target).

Int8 export first boot failed: `quantization=None` in the engine config,
so the flint linears were built unquantized and the loader died on
`model.layers.0.mlp.down_proj.qweight` not being a parameter. Cause:
vLLM 0.33 auto-detects the quant method from **config.json's
`quantization_config`** (the deployed qwen checkpoints carry gptqmodel's
block); `quantize_config.json` alone is not consulted for the method.
Fixed in the exporter (inject the GPTQ descriptor into config.json) and
patched the existing export in place. Re-boot in flight; next boot
surfaces the expert path (AutoGPTQMoEMethod int8 -> wna16 backend).

Int8 export boot status (2026-11-08 ~02:1x-02:4x):
- vLLM detects `auto_gptq` once config.json carries quantization_config
  (fixed in the exporter; existing export patched in place).
- Dense flint linears load through GPTQ; the 256-expert MoE layers warn
  "not supported by GPTQMoeMarlin. Falling back to Moe WNA16 kernels" and
  select the **TRITON WNA16** backend — i.e. the experts currently run
  A16W8 (weight-only int8, bf16 activations), NOT W8A8. This is the known
  gap the aiter grouped a8w8-blockscale kernel is meant to close.
- AIS loader fails on the new export (`FsGpuError: AIS hipFileRead
  failed: InvalidValue`) — probes use VLLM_FS_GPU=0 (plain safetensors,
  ~20 s/shard). Need to root-cause before the serve recipe (load speed).
- First full int8 boot (int8_block_g128 KV) loaded every shard, then
  crashed in the first forward with `hipErrorIllegalAddress` (surfaced at
  sample_hidden_states indexing; the async fault is earlier). Isolation
  boot with **bf16 KV** in flight — separates KV-path fault from
  weights/GEMM/MoE fault.
- Crashed boots leave ~28 GiB/GPU zombies; kill before the next attempt
  (this caused one false OOM).

## TURN 61 (2026-11-08 ~02:5x): EXPORT EXPERT-STITCH BUG — local names collide

The int8 export (first attempt) served garbage: with **bf16 KV** it boots
and runs (so the weights path itself doesn't fault), but median KLD vs
the teacher is **2.14** and top-1 agreement **18/1023 (1.8%)** on the
1024-token val window. Weight forensics: dense layers are faithful
(dequant rel err 0.7% vs the bf16 source: down_proj, o_proj, gate_proj),
but **every expert maps to the wrong slot**: export expert 1 -> HF
expert 0, expert 3 -> 0, expert 7 -> 4. Mechanism: the plan addresses
experts by their *local* index `loc = e // world` inside the rank's
expert list, and `rank_of(name)` returns the FIRST shard containing that
name. Every rank's shard contains local experts 0..63, so every group of
four globals read rank 0's copy: e in [4k,4k+3] -> global expert 4k.
Fix: whole-expert plan rows carry the holding rank explicitly
(`e % world`) and the getters take it. Re-export (v2) in flight.
Also confirmed: the int8_block_g128 KV path with the UA backend faults
(hipErrorIllegalAddress) on flint's GQA kvh=1 while bf16 KV is fine —
a separate kernel issue to fix after the weights are right.

Export faithfulness audit (v1, independent of the expert bug): q/k/v
projections dequant rel err 0.66-0.70%, dense gate/up 0.72%, router
bf16 exact, lm_head bf16 exact, embedding bf16 rel 0.85% (the trainer's
int8 fake-quant, intended), dense down/o 0.66-0.72%. Checkpoint expert
masters are bit-identical to the HF bf16 source (rel 0.0), so PTQR did
not move expert weights in 40 steps — expected, scales carry the
training. With the expert-rank fix the v2 export should be faithful
end-to-end; the earlier garbage was 100% the expert collision.

v2 export verified: experts now dequantize to their own HF counterparts
(experts 1/3/7/9 rel err 0.65% == uniform quantization noise; v1 had
expert 3 -> expert 0 etc.). Boot with bf16 KV in flight to measure the
serving-gate KLD of the corrected int8 weights. If it lands in the
0.005-0.02 band the int8 weight path is healthy and the remaining
blocker is the int8_block_g128 KV fault on flint (UA, kvh=1); if it is
still >>0.02 the in-loop KLD trend (0.081 -> 0.174 over rung 1) is
confirmed at the serving gate and the rung recipe needs changing
(frozen scales / different lr), with the step-0 scales+bf16 masters as
the RTN baseline.

## TURN 62 (2026-11-08 ~03:4x): EXPORT SINK BUG — head-sharded per rank

v2 export (experts fixed) serves KLD 1.52 median / 18.7% top-1
agreement — still broken. Class audit finds the exporter's `param` rows
read rank 0's copy via `rank_of`, which is fine for *replicated* params
(norms, qk norms, router: verified bf16-exact) but **the attention sink
is head-sharded**: each rank's ckpt holds only its 8-head slice
(`layers.N.attn.sink` [8]), so the export wrote an [8] sink as the
[32] tensor and every rank loaded rank-0's head slice. Shared-expert and
all quantized classes audited faithful (0.65-0.72%), so the sink was
the last mapping bug. Fix: new `cat0` export kind = rank-ordered
concatenation; v2's sinks repaired in place (rewrite the shards that
carry them) instead of a full 30-min re-export.

v2 sink repair verified: all 49 `self_attn.softmax_offset` tensors are
now [32] and bit-identical to the HF checkpoint (0.0 max diff). New boot
with bf16 KV in flight; this is the first faithful int8 export of the
rung-1 (step-40) PTQR payload.

## TURN 63 (2026-11-08 ~04:0x): INT8 EXPORT HEALTHY — serving-gate KLD 0.0095

With both export bugs fixed (experts by holding rank; sinks concatenated
across head shards), the rung-1 step-40 int8 payload measures at the
serving gate on the 1024-token val window (bf16 KV):

  variant                      median KLD  mean   top1 agree  coverage  NLL(cov)
  bf16 flint (probe floor)     0.005459    0.0404  92.6%       73.61%    1.6998
  int8 rung-1 step40           0.009465    0.0572  90.3%       73.70%    1.7196

i.e. the int8 weight stack costs ~0.004 median KLD over the serving bf16
floor and keeps top-1 agreement within 2.3 pt — quantization-noise scale,
well inside the qwen band (deployed GPTQ 0.0110, PTQR 0.0069 on their
probe). The in-loop KLD (0.081 -> 0.174) was a misleading metric, as in
the qwen program. Export lineage fixed: v2 dir + in-place sink repair =
current faithful artifact (models/Flint-GPTQ-INT8-W8A8-GS128-PTQR-r1s40-v2).
Remaining blockers for the recipe: int8_block_g128 KV fault on flint
(UA, kvh=1), W8A8 for the routed experts (currently TRITON WNA16),
then the draft, serve recipe at C8, and the Qwen/Bolt QA.

## TURN 64 (2026-11-08 ~04:1x): int8 KV — prefill OK, warmup/decode faults

With `enforce_eager=True` the int8_block_g128 KV path **works for
prefill**: same 1024-token val window, coverage 0.7429, NLL 1.7567 vs
bf16-KV 1.7196 — i.e. int8 KV costs ~2% NLL on this window. The
`VLLM_KV_DBG_SHAPE` dump shows the flint cache `(65759, 1, 64, 260)`
stride `(16640, 16640, 260, 1)` — nkv=1, block 64, one 128-element
scale group (G=128) — and the bounds audit passes.
Graph mode fails earlier: `compile_or_warm_up_model` -> warmup_kernels
(the multi-request prefill + 3 decode steps) dies with
hipErrorIllegalAddress *after* the 196 kvshape prints, i.e. in the warmup
forward itself, not the scale-view carving. Eager works, so the fault is
in a shape the warmup exercises but our 1024-token single prefill does
not: most likely the **decode/multi-request kernel with nkv=1** (qwen
runs 2 kv heads/rank). Decode probe (eager, 8 tokens) in flight to
separate decode from graph capture.

## TURN 65 (2026-11-08 ~04:3x): int8 KV fault ROOT-CAUSED — stale scale views

Clean-GPU bisect: eager int8 KV works (prefill NLL 1.7567, decode OK);
graph mode faults in `compile_or_warm_up_model` -> warmup_kernels after
196 kvshape prints. Disabling the *memory-estimate* capture
(`profile_cudagraph_memory`, which runs BEFORE the real KV cache is
allocated) makes graph mode + int8 KV work end-to-end: val probe
coverage 0.7351, NLL 1.6965, median KLD 0.0103 vs bf16-KV 0.0095 —
int8 KV costs ~0.0008 median KLD, negligible.
Mechanism: `_ensure_scale_caches` cached its as_strided scale views on
first call (inside the profiling capture, over the dummy/pre-allocation
cache) and never invalidated them; after the real cache replaced it the
views pointed at freed memory -> hipErrorIllegalAddress surfaced later.
Fix: remember the cache data_ptr and rebuild the views when it changes.
Debug knob `VLLM_DISABLE_CUDAGRAPH_MEM_PROFILE=1` added as a
workaround/diagnostic (also useful on its own: the estimate is only
applied when VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS is set).
Also noted: `VLLM_BOLT_SHORT_PREFILL_COHORT` appeared set again in a
probe's engine env with no setter in the launcher (the open env-injection
mystery from upstream-v0330-merge.md).

Fix verified: int8_block_g128 KV + FULL cudagraph mode + the int8 export
boots and generates with NO env workaround (scale-cache pointer
invalidation is the root fix; the kill switch remains only as a
diagnostic). Remaining engine gap: routed experts still run TRITON
WNA16 (A16W8) — the aiter grouped a8w8-blockscale kernel is the planned
W8A8 path.

## TURN 66 (2026-11-08 ~04:4x): W8A8 MoE kernel conventions VALIDATED on gfx908

Unit test (agents/tmp/moe_a8w8_unit.py, random E=8 H=2048 I=512 top-2
GS128): the aiter grouped a8w8-blockscale pipeline matches a torch
reference at rel rms 0.0157 (quantization noise). Exact conventions:
- Weights: w13 int8 [E, K, 2I] + scales [E, K/128, 2I] fp32 (per-output-
  column scales — the kernel's default 128-wide N tiles must be
  overridden with the new `w_blockscale_n=1` argument);
  w2 int8 [E, I, H] + scales [E, I/128, H].
- Activations: int8 [M, K] + per-row scales [M, K/128], pass
  `per_row_x_scale=True`.
- Gate-up call: `gather_indx=routing.gather_indx`, `apply_swiglu=False`.
  The kernel's fused swiglu is NOT flint's: it expects interleaved
  gate/up (`a[..., ::2]`, `a[..., 1::2]`) and adds +1 to the linear half
  (`swiglu_add_residual` default). Split the 2I output in torch instead:
  silu(first half) * second half (checkpoint layout is [gate|up]).
- Down call: `scatter_indx=routing.scatter_indx`,
  `gammas=routing.gate_scal` (the renormalized routing weights in
  expert-sorted order — do NOT rebuild from topk probs; the slot order
  is expert-sorted, not probability-sorted).
- Routing: `routing(logits, top_k)` with sm_first=False = softmax over
  the top-k logits; returns (RoutingData, gather_indx, scatter_indx).
Next: FusedMoEExperts subclass + GPTQ-int8-group quant-method branch to
load/shard/run this in the fork (weights currently arrive as GPTQ
qweight/scales; conversion to [E,K,N] int8 + [E,K/128,N] scales is a
per-expert unpack identical to the dense AiterW8A16 repack).

## TURN 67 (2026-11-08 ~04:5x): W8A8 MoE integration written

`vllm/model_executor/layers/quantization/gptq_moe_a8w8.py`:
A8W8BlockscaleMoEMethod (AutoGPTQMoEMethod subclass for gfx908 GPTQ
int8-group routed experts). It bypasses AutoGPTQMoEMethod.__init__ (the
WNA16 oracle rejects int8-group configs on ROCm — "No WNA16 MoE backend
supports the deployment configuration"), reuses the parent's
create_weights (GPTQ packed [E,K//4,N] int32 + [E,K/G,N] scales), and in
process_weights_after_loading unpacks to [E,K,N] int8 + fp32 scales.
apply(): builds aiter RoutingData from topk ids/weights by scattering
log(w) into a [M,E] logits matrix and calling aiter's fused routing()
(host-loop-free, CUDA-graph safe; routing_torch's block_pid_map loop is
not), quantizes activations per row/128K, runs the grouped gate_up GEMM
with gather + torch silu(gate)*up, quantizes the intermediate, runs down
with scatter + gammas=gate_scal. Hooked in AutoGPTQConfig.get_quant_method
for RoutedExperts when on_gfx908() + int8 + G%128==0
(VLLM_GFX908_MOE_A8W8=0 disables). Boot probe in flight.

W8A8 expert path result (eager): val window coverage 0.7361, NLL 1.7168,
median KLD 0.011010, top-1 agreement 90.0% — quality-identical to the
WNA16 (A16W8) path (0.0095/90.3%/1.6965) within probe noise, so the
grouped a8w8 blockscale experts are numerically sound. Graph mode with
the integration HSA-faults (warmup/capture), while eager works; suspect
Triton JIT variants compiling inside cuda-graph capture. Added a
one-time capture-size JIT prewarm (M 1..512, skipped during capture) in
the quant method; graph-mode probe in flight.

Prewarm did not fix it: graph mode still fails to init with the W8A8
experts (HSA exception at warmup/capture), while eager is stable. The
AITER grouped a8w8 MoE kernel path is therefore capture-incompatible on
this stack today — documented limitation; the fork keeps the WNA16
experts for graphed serving and the W8A8 experts for eager serving
(VLLM_GFX908_MOE_A8W8=1, default). Both pass the same quality gate.
Priorities now: (1) Qwen27B + Bolt recipe quality checks (explicit
objective requirement, GPU-only), (2) draft PTQR, (3) flint serve recipe
C8 + publish.

## TURN 68 (2026-11-08 ~05:2x): Qwen 27B recipe QA (explicit objective requirement)

Stack gates: test_int8_kv_micro PASS (prefill/decode/mixed/non-causal
rows); battery_gfx908 **4 PASS 0 FAIL**.
Boot: scripts/serve_recipe_qwen38.sh start (VLLM_API_KEY fallback; the
/etc/llama key file is not readable). Startup contract printed:
PTQR target+DFlash2 GS128; AITER W8A8/UA/custom-AR; INT8-G128 KV/fp32
Mamba/quant-out; TP4/C6; CPU KV tier 12 GiB. Health 200; coherence OK
("The capital of France is Paris.").
KLD gate (kld_probe_v2 capture vs the archived SPEC_BF16 reference dump):
median 0.016611, mean 1.6708, p95 11.11, greedy agree 30/52, 52/52
prompts ok. CAVEAT: SPEC_BF16.npz is a 2026-09-01 capture from the
pre-merge engine; the recorded R10S60 gate value was median 0.00694 /
42-of-52 on same-era captures. The fresh number may be engine drift in
the reference path or a real regression — needs a fresh same-era bf16
reference capture to attribute (no bf16 qwen checkpoint found under
models/ yet). 500-case soak running.

Qwen 27B soak: **500/500 coherent, 0 failures** (C8, 472 s) — the
recipe's main coherence gate passes on the current stack. Remaining for
the Qwen attribution: capture a FRESH same-era bf16 reference
(models/Qwen3.8-27B-bf16-ref exists) with the same serve recipe, then
compare the PTQR capture against it (the archived SPEC_BF16.npz is from
2026-09-01). After that: Bolt gates (boot, 500 soak, probe_bolt_quality,
28k retrieval) and the flint draft/recipe/publish.

Qwen bf16-reference recapture notes (2026-11-08 ~06:1x-06:4x):
- bf16 ref boot OOMs at the recipe's default KV sizing (bf16 weights 2x);
  KVMEM=6442450944 boots.
- First reference capture (bf16 target + PTQR int8 draft, SPEC ON) fails
  every other prompt with HTTP 400: "Out of range float values are not
  JSON compliant: nan" — i.e. NaN logprobs in the response at 256-token
  generations (64-token requests are fine). Likely the spec-verify path
  with an int8 draft against a bf16 target (the deployed PTQR target
  never shows this). Re-capturing with SPECOFF=1 (target-only reference)
  and the same int8 KV dtype; archive note: SPEC_BF16.npz from Sep 1 was
  a bf16 target + bf16 draft capture.

Fresh-reference KLD attempt 1 (PTQR spec-on capture vs bf16 no-spec
reference): median 0.0214, 29/52 — NOT apples-to-apples (the reference
path was generated without spec decode, so the probe compares across
different trajectories; the archived 0.0069/42-of-52 came from
same-config captures). Capturing the PTQR target with SPECOFF=1 for a
clean target-only comparison; also note the bf16 target + int8 draft NaN
logprobs at 256-token generations is a real spec-path finding worth
keeping (deployed PTQR target has never shown it).

## Qwen 27B QA — FINAL (2026-11-08)

Clean same-config serving gate (PTQR target, SPECOFF, int8_block_g128 KV,
TP4/C6) vs a fresh same-era bf16 reference captured identically:
**median KLD 0.007540**, mean 1.785, p95 11.85, greedy agree 36/52,
52/52 prompts. Recorded R10S60 same-era value was median 0.00694
(spec-on captures, 42/52) — the KLD gate is within ~9%, no regression.
Full Qwen 27B recipe QA summary: int8-KV micro PASS; battery 4/4;
serve-contract boot OK; coherence OK; 500/500 coherence soak; KLD
0.0075 median. Side finding: bf16 target + int8 PTQR draft with SPEC ON
returns NaN logprobs at 256-token generations (spec-off is clean; the
deployed PTQR target never showed it) — worth a follow-up outside this
goal. Qwen QA COMPLETE; Bolt next.

## Bolt boot fix (2026-11-08): merge dropped the gfx908 W4A16-NVFP4 path

Bolt could not boot on the merged branch: mamba `in_proj` (W4A16_NVFP4)
went through the new generic `ModelOptLinearMethod` ->
`init_nvfp4_linear_kernel(use_a16=True)`, whose ROCm kernel list only has
the emulation kernel (filtered out for W4A16) -> "Failed to find a kernel
that can implement the NVFP4 linear layer". The merge's
`build_linear_method` dispatcher bypassed the fork's
`ModelOptNvFp4W4A16LinearMethod` (packed_g128 = on_gfx908()).
Fix (two parts): register `LINEAR_METHOD_BUILDERS["W4A16_NVFP4"]` with a
builder that returns the packed method on gfx908 (generic path
otherwise), and decorate that method with
`@register_weight_loader_v2_supported_method` (without it the legacy
loader shape-asserts on the packed NVFP4 tensors).
Verified: Bolt boots (weights 8.07 GiB/rank in 22 s, CPU KV tier, GS128
shared/routed overlap, prefix-cache geometry, mixed-load prefill), health
200, coherent reply. Gates running: 52-prompt quality probe, 500-case
soak, 28k retrieval/prefix replay.

Bolt gates (2026-11-08): 52-prompt quality probe **PASS 52/52**
(coherence + finite logprobs, logs/bolt-quality.json). 28k retrieval
(28,067 prompt tokens): cold 32.4 s, created_cache 28,064 tokens ->
"ORCHID CEDAR 4829"; prefix replay 1.95 s with cached_tokens 28,064 and
the IDENTICAL answer ✓ (the optional /reset_prefix_cache step 404s
because the diagnostic endpoint needs VLLM_SERVER_DEV_MODE=1, which the
recipe says not to enable in production). 500-case soak running.

## TURN 69 (2026-11-08 ~09:4x): W8A8 MoE GRAPH CAPTURE FIXED — root cause was padding ids

The capture crash was NOT a kernel limitation: vLLM's warmup/dummy and
padded decode batches carry **expert id -1** in `topk_ids`. My routing
reconstruction did `logits.scatter_(1, topk_ids.long(), log(w))`; a
negative index silently wraps to the LAST expert, so the whole batch
piled onto expert 255 -> degenerate histogram -> the aiter fused routing
kernel faulted (HSA 0x1016). Debug prints showed `ids[-1,-1] dup=5` at
M=8 immediately before the crash. Fix: mask invalid slots
(`ids < 0` -> expert 0, weight 0) before building the synthetic logits.
Result: **W8A8 experts + int8_block_g128 KV + full CUDA graphs** server
boots (init engine 12.2 s, capture 100%), coherent replies.
Flint C8 speed on that full int8 stack (TP4, C8, no draft — the DFlash2
draft is not trained yet), vllm bench serve openai, random corpus:
- 32 in / 256 out, 64 reqs @ C8: output **171.7 tok/s**, req 0.67/s,
  TTFT 514 ms, TPOT/ITL **44.75 ms** (22.3 tok/s per stream).
- 2048 in / 64 out, 16 reqs @ C8: TTFT 5.66 s (≈5.8k prefill tok/s),
  TPOT 68.8 ms, output 43.8 tok/s.
No 16-bit fallback: this is the int8 path end to end (AITER W8A8 dense,
grouped a8w8-blockscale MoE, int8 KV).

## Bolt QA — FINAL (2026-11-08, with the gfx908 W4A16 fix)

- Stack: micro PASS, battery 4/4.
- Boot: healthy; recipe contract active (packed INT8 GS128 official
  weights, fused expert cache 40 layers, CPU KV tier, GS128 shared/routed
  overlap, prefix-cache geometry, mixed-load prefill, short-prefill
  cohort).
- 52-prompt quality probe: **PASS 52/52** (coherence + finite logprobs,
  logs/bolt-quality.json).
- 28,067-token retrieval: cold 32.4 s -> "ORCHID CEDAR 4829"; prefix
  replay 1.95 s, cached_tokens 28,064, **identical answer** ✓.
- 500-case soak: **492/500 coherent, 8 failures** — all
  "repeat tail x7" on the bash-count-lines prompt family, with the
  replies opening with literal `<think></think>` markers (the template's
  enable_thinking=False rendering plants an empty think block).
  Re-running the failing seeds flips pass/fail -> stochastic at temp 0.7,
  not a fixed regression, but BELOW the recipe's recorded 500/500 gate.
  Recorded as a coherence regression to investigate (candidates: the
  empty-think template convention vs the merged engine's template
  rendering; presence penalty path is untouched upstream code).

## TURN 70 (2026-11-08 ~10:4x-11:4x): INT8 PTQR DRAFT + RECIPE + AIS

Draft PTQR export (`scripts/ptqr_export_flint_draft.py`) — first run
asserted on intentionally-removed `.weight` keys; the sanity check now
allows exactly the 35 GPTQ-replaced names and asserts qweight/qzeros/
scales exist. Export audit: 35 GPTQ linears mean dequant err 0.66 %
(worst 0.70 %, same class as the target export), 12 bf16-grid surfaces
0.16 %, 34 untouched tensors byte-identical to the bf16 draft
(151 tensors, 518 MB).

Draft acceptance gate (same random C8 protocol, TP4, int8 KV, NS=7,
`vllm bench serve` 32 in/128 out, 16 reqs @ C8):
- bf16 draft baseline: acceptance length **2.27** (rate 18.21 %),
  TPOT 30.06 ms.
- int8 PTQR draft: acceptance length **2.25** (rate 17.93 %),
  TPOT 30.02 ms.
Delta -0.02 length / -0.28 pp -> PASS (Qwen draft doctrine is +-1 pt).
Coherent chat replies with the int8 draft.

Serve recipe `scripts/serve_recipe_flint.sh` (peer to qwen38/bolt):
TP4, C8, bf16 compute, max-model-len 262144, int8_block_g128 KV on
target AND draft, DFlash2 NS=7, FULL_DECODE_ONLY graphs, AITER W8A8
dense + gfx908 a8w8-blockscale experts, CUSTOM AR, reasoning parser
qwen3, host-only API key check like the other recipes. Contract lines
verified in the boot log: AITER unified attention, "gfx908 routed
experts: GPTQ int8-group -> AITER grouped a8w8-blockscale (W8A8)",
int8 KV, EAGLE3 aux layers (6,16,27,37,47), capture 100 %,
4.36 M KV tokens (16.7x at 262k).

Recipe C8 numbers (int8 target + int8 draft, random corpus):
- 32 in / 256 out, 64 reqs @ C8: output **261.1 tok/s**, 1.02 req/s,
  TTFT 627 ms, TPOT 27.16 ms, acceptance length 2.49 (21.3 %).
  (Target only, same box: 171.7 tok/s -> the draft is +52 %.)
- 2048 in / 64 out, 16 reqs @ C8: TTFT 6.20 s (~5.3k prefill tok/s),
  TPOT 44.08 ms, acceptance length 1.67.

AIS on this export is FIXED: the reported `hipFileRead InvalidValue`
came from a stale/partial repack cache directory. Cleared it; the raw
38.50 GiB byte gate PASSES (109,620 chunks, scripts/fs_gpu_weightcheck.py)
and both the target and the int8 draft now load through AIS
(`VLLM_FS_GPU=auto` is the recipe default). First boot writes the
per-rank repack cache (41 GiB, ~15 min); later boots are HIT
(weight load 267-283 s). Draft cache 1.1 GiB.
Also registered `VLLM_GFX908_MOE_A8W8` in vllm/envs.py (removes the
"unknown env var" warning) and added the one-time W8A8-expert log line
so the startup report names the expert kernel.

## TURN 71 (2026-11-08 ~19:2x): 512k / C12 PREPARED (NOT BOOTED)

User asked for a 512k per-request cap and a C12 recipe, then "just
prepare it" (interrupted the restart).

- The HF model card and the Recursive announcement confirm 512k support
  (recommended <=256k); the checkpoint config documents only 262,144.
  Model card facts: sliding window 513 tokens (current + 512 preceding);
  RoPE on sliding layers, NO positional encoding on the 12 full-attention
  layers; documented context configuration 262,144.
- Code: the target and DFlash2 draft rope tables now extend to the
  runtime max_model_len (`get_current_vllm_config_or_none`), so positions
  >262k no longer read past the table. Identical tables at <=262,144.
- Recipe: `--max-model-len 524288`, `--max-num-seqs 12`, and
  `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` (the config guard raises above the
  declared max_position_embeddings).
- KV math: the measured 262k boot held 4,476,935 tokens; the 512k rope
  tables add ~0.27 GB/GPU, so expect ~4.4M shared tokens. C12 x 512k =
  6.29M > pool, so full-length concurrent requests will queue/preempt.
- Status: PREPARED, NOT VALIDATED. Live service was restored to the
  previous C8/262k configuration via env overrides. First 512k/C12 boot
  and a >262k long-context retrieval test are pending.

## TURN 72 (2026-11-08 ~19:4x): 512k/C12 BOOTED + TUNING BASELINES

Prepared 512k/C12 config validated: target "Using max model len 524288",
draft max_position overridden to 524288 by vLLM (its own rope override
exists; our table extension is belt-and-braces), GPU KV cache size
**4,536,102 tokens**, max concurrency for 524,288 tokens/request 8.65x
(C12 short requests fit; full-length concurrency is 8.65).
W8A8 expert log line, UA, int8 KV, EAGLE3 aux, graphs all present.

New harness `scripts/bench_flint_workloads.py` (token-id prompts,
ignore_eos, greedy, SSE timing): steady N at t=0 and staggered 6 + 6
with a delay.

Baselines on the 512k/C12 recipe (random token-id prompts, so draft
acceptance is worst-case):
- steady C12, 2048 in / 256 out: wall 16.26 s, **188.9 output tok/s**,
  prompt 1511 tok/s, TTFT mean 7.60 s (p95 13.28 s), TPOT 69.3 ms.
  The 12x2048 prefill is serialized through 2048-token chunks and
  interleaves with 96-row decode steps.
- staggered 6 + 6 at delay 2.0 s: wall 8.78 s, **350.0 output tok/s**,
  wave1 TTFT 2.64 s, wave2 TTFT 3.12 s (1.07 s after launch), TPOT
  89.7 ms. Overlapping prefill with decode nearly doubles throughput vs
  the steady C12 run.
- For reference (C8, 32/256, earlier boots): 261.1 tok/s, TPOT 27.2 ms.

These are the tuning targets for GEMM retune / fusion / residency work.

GEMM inventory for the retune (from the dispatch code):
- Dense linears (qkv, o, dense MLP 0-1, shared experts): CK W8A8
  (`gemm_a8w8_CK`, per-channel requant at load) with
  `aiter/configs/a8w8_tuned_gemm.csv` lookup; missing shapes fall back
  to splitK=0. Shapes (M at C12/NS7 = 96 verify rows; prefill chunk 2048):
  qkv N=6144 K=2048; o N=2048 K=4096; dense gate/up N=12288 K=2048 (TP4
  split); dense down N=2048 K=6144; shared expert gate/up N=2048 K=2048
  (1024-intermediate split over TP), shared down N=2048 K=1024.
- Routed experts: grouped a8w8-blockscale Triton kernel with a HARDCODED
  `get_kernel_config` heuristic (no tuned CSV). Local E=64, gate/up
  N=1024 K=2048, down N=2048 K=512, top-6; decode M=96, prefill M=2048
  per chunk. This is the primary retune target (47 layers x 2 calls/step).
- The warning seen at boot ("not found tuned config in a8w8_tuned_gemm.csv
  ... will use default config") is the CK path missing Flint shapes.

## TURN 73 (2026-11-08 ~20:2x): MOE QUANT FUSION + GEMM TUNE (slice 2)

Per-layer MoE breakdown at M=96 (decode) before the change: routing 337 us,
input quant 106 us, gate_up 293-345 us, silu+mul 23 us, silu+quant 104 us,
down 177 us -> full _run 1114 us; x47 layers ~52 ms of the ~90 ms step.

Changes (commit 64b6bcc483):
- `group_quant_rn` (one Triton pass per 128-group) replaces the eager
  torch quant chain: 103 -> 35 us.
- `silu_mul_quant_rn` fuses SiLU*up + quant (strided views handled):
  127 -> 40 us; int8 values match the old path (1-ulp fp32 scales from
  HIP fast division; q flips 0.013% of elements).
- gate_up tuned config (block_n=256, warps=8, stages=2, group_m=1)
  from scripts/tune_flint_moe_gemm.py: 293 -> 253 us (-14%). The down
  kernel's block_n=32 shrink is load-bearing; forcing 256 faults
  (hipErrorIllegalAddress), so only leaf knobs are candidates there.
Full _run 1114 -> 862 us (-23%).

Workload impact (first run per boot; repeats hit the prefix cache, so
A/B only on cold runs):
- steady C12 2048/256: 188.9 -> **212.7 output tok/s** (+12.6%),
  TTFT 7.60 -> 6.93 s, TPOT 69.3 -> 57.8 ms.
- staggered 6+6 delay 2.0 s: 350.0 -> **456.3 tok/s** (+30.4%),
  wall 8.78 -> 6.73 s, TPOT 89.7 -> 77.3 ms.
Next: the routing build (337 us/layer, 15.8 ms/step) is the largest
remaining fixed cost; then dense CK tuning and the norm/AR fusions.

## TURN 74 (2026-11-08 ~21:0x): ROUTING SHORTCUT — VALID BUT SLOWER (ABANDONED)

Idea: build the aiter Bitmatrix directly from the known top-k ids
(index_put of bit words) and call `sort_tokens`, skipping the redundant
logits build + fused topk (~135 us claim).

Result: structures needed by the GEMM (hist, token_offs_raw/pad,
block_pid_map) match the stock path EXACTLY; the gather/scatter/gate
permutations differ only within an expert, and the gate_up GEMM output
is bit-identical (max diff 0.0). But the shortcut measures 500 us vs
155 us for stock in the same process (three zeroed scratchpad/bitmatrix
allocations + index_put dominate at these tiny sizes). Abandoned.

Measurement caveat: stock routing measured 155 us here vs 337 us in the
per-layer breakdown process — the routing cost needs in-situ profiling
(rocprof on the serving process) before any rewrite is worth it.
NOTE: my first equivalence test used random ids WITH duplicates per row
(impossible from a real top-k); that produced spurious structure
mismatches until the ids were made distinct.

## TURN 75 (2026-11-08 ~21:3x): DENSE CK + DOWN-KERNEL RETUNE

CK dense (whole-K column-parallel shapes qkv 1280x2048, dense gate_up
3072x2048, shared gate_up 512x2048 at M=96/2048): splitK=0 (the missing-
config fallback) is already fastest for every shape except shared
gate_up M=2048, where splitK=1 gives 236 vs 245 us (-3.4%, ~0.4 ms per
prefill chunk over 47 layers). Nothing else to gain there; the boot
warning is cosmetic for these shapes. o_proj/down_proj are row-parallel
(sharded K) and stay on the Triton blockscale path, not CK.

MoE down kernel (I=512,N=2048,block_m=16): the heuristic shrinks
block_n to 32 for grid occupancy. Measured through the production _run
path (probe_flint_moe_down.py, one config per process): block_n=128,
warps=8, stages=2, group_m=1 is best at 867 vs 887 us (-2.2% full
layer); block_n=256 is ~876. split_k=2 changes the output (excluded).
The earlier full-sweep crash was an artifact of my fabricated harness
tensors, not a kernel limitation - the production path accepts every
block_n. Installed the down override in _TUNED_MOE_CONFIGS.

## TURN 76 (2026-11-08 ~20:4x): PROFILING ATTEMPT (ATTACH BLOCKED)

- rocprofv3 attach (1.3.2) fails two ways on this host: the target must be
  started with ROCP_TOOL_ATTACH=1, and even then ptrace is denied
  (PTRACE_SEIZE EPERM), so `--pid` profiling is unusable here.
- vLLM's torch profiler via `--profiler-config` (PROFILE_DIR lever added
  to the recipe) only produced a frontend trace (244k python_function
  events, no GPU kernels) and `/stop_profile` timed out (HTTP 000) after
  the bench; worker traces never appeared. Eager mode also costs 6x
  (TPOT 357 ms vs 57.7 ms graphed), so the profiled run is not
  representative anyway.
- Decision: stop investing in process-level profiling. Budget the step
  from known pieces: MoE ~0.86 ms x47 = 40 ms, dense CK linears ~7 ms,
  attention ~2 ms; the rest (~20-30 ms) is per-layer norm/rope/split/
  all-reduce glue, which the fusion work targets directly.

## TURN 77 (2026-11-08 ~21:0x): FUSED Q/K NORM+ROPE (ROCm) ACTIVE

New `qk_norm_rope_rocm` in vllm/models/flint/ops/qk_norm_rope.py: one
Triton kernel does per-head RMSNorm (fp32) + NeoX RoPE (fp32 from the
fp32 cache) for q and k, no paged-cache write (UA writes int8 KV
itself). Numerics match the eager reference except 1-3 elements out of
~110k (1-ulp). Microbench: eager chain 438 us -> fused 58 us per layer
at both T=96 and T=2048. Wired into FlintAttention.forward behind
VLLM_GFX908_QK_FUSION (default 1), eager path kept as fallback.

End-to-end, steady C12 2048/256 cold run: 212.7 -> **221.0 output
tok/s** (+3.9%), TTFT 6.93 -> 6.31 s, TPOT 57.8 -> 53.0 ms (-8.3%).
Coherent (17*3 -> 51). The model-level gain is smaller than the
microbench because the eager chain's allocator churn is amortized in
the real forward.

CORRECTION (important): the earlier staggered numbers (350/456 tok/s)
were PREFIX-CACHE ARTIFACTS - the staggered harness used the same
seed-0 prompts as the preceding steady run, so prefill was cached. A
cold staggered run (seed 1, delay 2.0 s) measures **198.4 tok/s**, which
is WORSE than the cold steady run (221.0). The "staggering nearly doubles
throughput" claim is retracted; the delay must be swept with a unique
seed per run before any stacking claim. Prefix caching also means every
A/B needs one cold run per configuration.

## TURN 78 (2026-11-08 ~21:2x): COLD STAGGERED SWEEP — NO OVERLAP GAIN

All runs use unique seeds (no prefix-cache reuse), steady/C12 reference.

Staggered 6+6, delay sweep (output tok/s):
- 0.5 s: 211.5, 208.8, 183.9
- 1.0 s: 205.5, 196.3, 136.9
- 1.5 s: 155.4
- 2.0 s: 200.3
- 3.0 s: 180.8
- 4.0 s: 174.1
Steady C12 reference (same box, cold): 221.0, 216.5, 157.2.

Findings:
1. Run-to-run variance is +/-25% even with identical configs (one steady
   run at 157 tok/s vs 221), so single runs cannot resolve differences
   under ~10%. Future A/Bs need >=3 repeats and medians.
2. Staggering 6+6 at delay <=1 s is indistinguishable from steady within
   that noise; longer delays simply add idle wall time (174-200 tok/s at
   3-4 s). The default chunked-prefill scheduler already interleaves
   prefill chunks with decode in the steady case, so the "delayed second
   wave" adds no throughput.
3. Conclusion: the simultaneous decode+prefill behaviour is the default;
   tuning stays oriented at the steady C12 2048/256 workload, and the
   staggered case is documented as equivalent-with-idle-delay.
