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
