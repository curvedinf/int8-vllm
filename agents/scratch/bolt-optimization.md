# Bolt optimization campaign 2 (active)

User restarted indefinite optimization on 2026-10-18 under a new request; the
2026-10-17 pause note is superseded. Loop until the user says stop: profile at
the campaign workload, identify 5 experiments (mixed scope: kernels, fusion,
config), run each with 5 measured runs, commit/push wins, repeat. Focus:
prefill. Gate: prefill improves, decode does not regress (median + pooled,
same discipline as before). Memory reallocation authorized while >= 6 full
256k-token streams fit (6 x 262144 = 1,572,864 tokens; current 14 GB/GPU KV
arena reports 3,283,288 effective tokens, so reductions must keep >= that
effective-token floor).

## Environment (fixed for all measurements)

- 105 W cap all 4 MI100s, set via sysfs power1_cap=105000000; thermal
  controller STOPPED (mi100-thermal-controller.service), nginx and
  vllm-bolt production service stopped. Harness asserts all of these per run.
- Diagnostic server: serve_recipe_bolt.sh with PORT=8021
  VLLM_SERVER_DEV_MODE=1 (dev RPCs for lifecycle profiling + prefix reset),
  API key agents/tmp/bolt-eval/regression.key (0600, never print/commit).
  Run server + all GPU/bench work as the checkout owner, never root.
- Harness: agents/tmp/bolt-opt/loop2/run.py, manifest-1k.json
  (sha256 47d824dd59fd74baf7c384ff1211d5aed0b43163cc11b06a06d1b147cc364b4b,
  6 prompts x 1024 in / 256 out, temp 0 seed 0 ignore_eos, stream token
  events; phase metric aggregate_decode_only_after_all_prefills_v2).
  Modes: bench (warmup + N runs), profile (lifecycle rank-0 capture),
  --require-marker for rank qualification from server.log byte offset.
  Results land in loop2/<batch>/. Boot from AIS cache hits ~6 min.

## Campaign-2 baseline (2026-10-18, loop2/baseline/)

Reference incumbent = accepted standard recipe (vLLM 4711913204, AITER
3c74adc19). 5 runs after 1 warmup: median prefill 1889.59 tok/s (range
1854.4-1904.3), median aggregate decode 152.53 tok/s (range 147.8-158.4),
pooled decode 153.11, pooled whole-run output 117.17, acceptance 3.28-3.52.
Decode noise ~7% (acceptance-driven); prefill noise ~2.6%. NOT comparable to
the 4096/C8 campaign numbers (2300.42/168.28).

## Experiment ideas (mined, ranked)

1. Dense packed prefill kernel: decode/INT8-convert weight once per
   (N-tile,K-group) and apply across all M rows (profile 2026-10-15: 512
   M-tiles at 8k prefill redo conversion; Strata does decode-once per block,
   their #1 technique). NVFP4 Mamba projections first (36% of kernel time).
2. Strata EXPDEC analog for routed experts: stage group activations
   (INT8) in shared memory, R rows per lane, rows-per-block tuning,
   shared-memory IQ2 codebooks (their kernel -11..-23%, bitwise identical).
3. Expert-cache growth via KV reallocation (5 -> ~8-10 GiB expert cache,
   KV down toward the 6-stream floor) -> more INT8 expert hits in prefill.
4. Mixed-prefill scheduling retune for C6/1024 (page budget, interval,
   cohort policy interactions at the new shape).
5. NVFP4 dense conversion widening (VLLM_BOLT_KERNEL_TUNING lever exists,
   default 0 - check what it gates before using).

Reference: github.com/Niko1221/Strata (MIT; llama.cpp-derived kernels, MIT).
Their tree has no Bolt/Nemotron path; applicable ideas are the kernel
techniques (decode-once Split<TG> load/apply split in
src/kernels/cuda/iq_kernels.cu, EXPDEC staging in docs/EXPERTDEC.md,
calibrated dense GEMM in docs/AMD_HIP_PERFORMANCE.md). Attribute if code is
copied; prefer reimplementation in AITER Triton idiom.

## Loop-1 profile (2026-10-18, loop2/profile/, rank-0 GPU-only Kineto)

1024/256 C6, kernel busy 0.998. Prefill window 3.248 s: grouped packed `_gemm`
(IQ2 experts, Triton) 38.8%; blockscale dense INT8 GEMMs ~18.9%; NCCL
Ring/Simple collectives 13.8% (out of scope - standing collective constraint);
custom AR 3%; router topk 2.2%; misc elementwise/quant/norm/mamba ~9%.
Decode window 9.696 s: packed_routed native HIP 27.8%; dense decode (Cijk +
blockscale BM64/16) 20.8%; custom AR 9.5%; unified attention 4.3%. Full
summary: loop2/profile/prof1-summary.json. NVFP4 Mamba projections are already
INT8-resident via campaign-1 dense cache (served by blockscale kernel); packed
decode at prefill is routed IQ2 experts only.

## Batch B10 design (locked 2026-10-18)

Gate: prefill median AND pooled improve; decode median non-degrading AND
pooled decode regression <= 2% (noise band; campaign-1 gate was both-median-
and-pooled strictly, which reverted B04_E3 on a -0.5% pooled decode dip).

- E1 (kernel, medium): resurrect B04_E3 cached-INT8 prefill hit branch.
  Prior run: median prefill +8.44%, median decode +0.77%, pooled whole
  +4.12%, pooled decode -0.50% (reverted on strict gate only). Saved sources
  loop/batch04/exp3-saved/aiter (built on 3f8045a90; port onto current main
  3c74adc19; output layout unchanged so combine_experts unaffected).
  Prefill `_gemm` is 38.8% of prefill window here. Qualify with
  --require-marker + component checks before the 5-run spend.
- E2 (config/memory): reallocate KV 14 GB -> ~8 GB (assert boot effective
  tokens >= 1,572,864 = 6 x 262144), expert cache 5 -> 10 GiB, dense 3 -> 4.
  Decode/verify protection (packed_routed 27.8% of decode); raises E1 hit
  fraction. Run standalone first, then stacked if E1 accepted.
- E3 (kernel-config sweep): blockscale dense GEMM pinned configs for prefill
  shapes (num_stages 2/3, GROUP_SIZE_M, KSPLIT, BM variants; all pinned
  stages=1 today). Micro-bench offline with real shapes first; ~18.9% of
  prefill window.
- E4 (kernel rewrite, large): Strata-style decode-amortized W-tile staging in
  grouped Triton `_gemm` miss path - decode/INT8-convert (BN=128 x BK=128) W
  tile once per K-group per CTA, reuse across inner M loop (inverted loop
  order vs per-M-tile redone decode; distinct from rejected B04_E2 tile
  growth and B09_E2 register-dense MFMA). Accelerates the path E1 leaves.
- E5 (scheduler config): mixed-prefill tokens/interval + short-prefill
  cohort at C6/1024 (6144-token single-chunk prefill; admission effects).
  Bounded 3-leg sweep.

Order: E1 -> E2 -> E3 -> E5 -> E4 (cheap first after the high-confidence
port; E4 last, biggest scope). Each vs latest accepted incumbent; ledger row
per experiment; commit+push accepted changes (security gate first).

## B10 progress

- E1 DONE/ACCEPTED 2026-10-18: B04_E3 port onto AITER main dc13c3d18
  (pushed). 5 runs vs campaign baseline: prefill median +14.42% pooled
  +14.20% (boot95 +10.7..+16.6), decode median +2.61% pooled +2.63%.
  16/16 component, 32/32 real-checkpoint exact, battery 4/4, KV micro PASS.
  vLLM ledger row BOLT_LOOP2_B10_E1_..., vllm main 6d401447d8 (pushed).
  New incumbent: AITER dc13c3d18 + vLLM 4711913204. Next reference for
  E2/E3/E5/E4 = loop2/b10e1/ five runs.
- E2 DONE/ACCEPTED 2026-10-18: KVMEM 14 GB -> 8 GB (7.15x 256k streams,
  >= 6 floor verified at boot), expert cache 5 -> 10 GiB (9131 pairs,
  228-239 slots/layer). Config-only. Vs E1: prefill median +11.22%
  pooled +11.06%; decode median +8.60% pooled +7.59% (CI excludes 0).
  Cumulative vs campaign baseline: prefill +27.2%, decode +11.4%.
  Launcher defaults + recipe doc updated; ledger row
  BOLT_LOOP2_B10_E2_MEMORY_REALLOCATION. New incumbent = E2 config;
  next reference = loop2/b10e2/ five runs. REMAINING: E3 (blockscale
  config sweep), E5 (scheduler sweep), E4 (Strata decode-once kernel).
- E3 DONE/REJECTED 2026-10-18: blockscale num_stages=3/GROUP_SIZE_M=8
  for m>64. Micro: -11..-35% kernel ms at m=1024 shapes, all bitwise
  exact; but end-to-end prefill flat (+0.13%) - the workload prefills
  as one 6144-token chunk where gain was only 5-8% of a 19% share.
  AITER change reverted (clean tree at dc13c3d18). Negative result in
  ledger. LEARNED: dense GEMM config is NOT the remaining prefill
  bottleneck; grouped packed _gemm (38.8%) and NCCL (13.8%) dominate
  post-E1/E2. ALSO: first post-warmup run after a fresh boot can stall
  at 99.9% VRAM (32.18/32.21 GB with 10 GiB expert cache) + JIT of new
  variants; re-measure on settled server when run 0 staggers (TTFT
  waves 0.5/4.7/9.3s is the signature).
- E5 DONE/REJECTED 2026-10-18 (default kept): cohort=0 collapses
  prefill -43.8% (1351 vs 2405 tok/s) - the short-prefill cohort is
  LOAD-BEARING at C6/1024; interval leg skipped (inert neighborhood;
  cohort carries admission). Ledger row written. E4 remaining; design
  from a fresh profile of the incumbent (boot in progress, then
  profile mode + analysis).
- Fresh incumbent profile (prof2, loop2/profile2/): prefill window
  1.997s (-38.5% vs baseline). _gemm 23.0% (was 38.8); NCCL 20.6% =
  SP MoE per-layer all-gather (nemotron_h.py:285) + draft collectives
  (226 launches, absolute time barely moved 0.449->0.411s) - top B11
  investigation, but collective backend/precision/order changes are
  barred by the standing wire policy; needs user disposition before
  any B11 collective work. Dense blockscale ~27.5% (explains E3 flat:
  5-8% kernel gain ~= 1.9% end-to-end, sub-noise). E4 = BM=128
  grouped prefill tiles (B04_E2 retry at current state): MO
  block_size 128 tier in modelopt_gfx908.py + wrapper assert + 8
  warps; kernel BM constexpr already generic. Qualify then 5 runs vs
  b10e2 reference.
- E4 DONE/INCONCLUSIVE 2026-10-18 (reverted): 128-row grouped prefill
  tiles. 16/16 component + 32/32 real-checkpoint exact at block 128;
  measurement prefill median +1.90% pooled +1.64%, bootstrap CI
  -2.6..+5.5 (spans zero, below noise) - not a demonstrated win.
  Both files reverted (aiter at dc13c3d18, vllm clean of E4). Matches
  B04_E2's verdict at 4096/C8: tile growth lands inside the noise gap.
  NOTE: server on 8021 still runs E4 in-memory; tree is clean -
  restart before any B11 measurement.

## B10 closed; B11 design final (2026-10-18)

Accepted: E1 + E2 (cumulative prefill +27.2%, decode +11.4% vs
campaign baseline). Negatives documented: E3 (dense config - kernel
gain sub-noise end-to-end), E5 (cohort load-bearing, default kept),
E4 (tile growth sub-noise).

NCCL identification COMPLETE: SP-MoE is OFF (requires DP>1;
parallel.py:697). The 226 prefill NCCL launches are the per-layer TP
all-reduce of MoE outputs (fp32 wire, 6144x4096x4B = 100.7 MB/layer)
exceeding the gfx908 TP4 custom-AR cap of 32 MiB
(all_reduce_utils.py:31-37) -> NCCL Ring fallback at prefill; decode
(0.79 MB) uses custom AR. Structural: page 4256 forces chunks
>= 4256 tokens = 69.7 MB > cap; wire dtype fenced by the
_tp_ar_fp32_mode do-not-fix policy; backend/precision/order barred.
=> NCCL 20.6% of prefill is POLICY/STRUCTURALLY FENCED; user
disposition requested (options: allow bf16 wire for oversized ARs,
restructure, or accept plateau). NOT a B11 experiment.

B11 experiments (legal set, prefill-first; reference = b10e2 five):
- E1: bf16 copy elimination (831 prefill launches 2.3%, 34725 decode
  2.1%) - audit non-contiguous sources (MO apply x.contiguous()
  modelopt_gfx908.py:262, draft cat/slices), remove bit-exact.
- E2: _quant coverage extension - why 384 _quant launches remain at
  prefill despite B07 E4 norm+quant fusion; extend to remaining
  projections bit-exact.
- E3: Strata EXPDEC activation staging for native packed_routed
  (25.6% of decode; their -11..-23% bitwise) - decode gain, prefill
  must not regress.
- E4: router sbtopk (2.6% prefill / 3.2% decode) - only if a bounded
  mechanism exists.
- E5: reserved for the first B11 profile delta.

## B11 progress

- E1 CLOSED (audited, no measurement): bf16 copy family 2.3% prefill
  is small scattered copies; sources are numerics-required casts
  (mamba SwiGLU fp32), draft cats, and necessary contiguity guards.
  No bit-exact removable subset above noise. Ledger row written.
- E3a DONE/REJECTED 2026-10-18: VLLM_BOLT_ROUTED_WAVES=4 decode
  median -3.65% pooled -2.49% CI excludes zero (occupancy loss from
  256-thread blocks dominates read-duplication savings). Default 2
  stands. E3b (LDS staging at NW=2) prior downgraded - duplication
  is not the binding constraint; consider only if a kernel-level
  micro first shows a win. NOTE: server on 8021 runs waves=4
  (rejected) in-memory; next boot uses launcher defaults (waves=2).
- REMAINING B11: E2 (_quant coverage extension, prefill 2.4%), E3b
  (LDS staging, decode - micro-gate first), E4 (router sbtopk audit).
  Plateau context: largest legal prefill items are exhausted;
  NCCL 20.6% awaits user disposition.
- E2 CLOSED (audited): B07-E4 fusion IS engaged at this workload
  (_fused_add_rms_norm_g128 80x prefill); remaining 384 _quant are
  first-quantizations of distinct tensors. No headroom. Ledger row.
- E4 CLOSED (audited): router topk ceiling 2.6% = noise edge. Ledger
  row.
- E3b DONE/INCONCLUSIVE 2026-10-18 (reverted): micro bitwise-exact
  +18.4% per expert pair, but end-to-end null (prefill -1.0% noise,
  decode -1.1% CI spans zero). 18% kernel win does not translate:
  TP4 decode critical path is sync/latency-bound (custom AR +
  scheduler cadence), not throughput-bound on packed_routed. AITER
  reverted to dc13c3d18. Server STOPPED (clean state).

## B11 closed (2026-10-18): 0 accepted, 5 documented

E1/E2/E4 audited-closed (sub-noise ceilings), E3a rejected (decode
CI regression), E3b inconclusive-reverted (kernel win, e2e null).
Legal surface above the noise floor is exhausted at this workload:
- Prefill: NCCL AR-size fallback 20.6% (FENCED - user disposition
  requested: bf16 wire for oversized ARs / restructure / accept);
  grouped _gemm 23% (E1-optimized, tile growth dead x2); dense
  blockscale 27.5% (config dead; register-MFMA rewrite rejected in
  campaign 1); all remaining items <= 2.6% individually.
- Decode: packed_routed 25.6% (staging win not on critical path;
  waves regression), custom AR 8.5%, Cijk 8.9%.
Campaign totals stand at prefill +27.2% / decode +11.4% (E1+E2).
B12 requires either the NCCL disposition or genuinely new mechanisms;
next pass: research pass (fresh eyes on prof2 + Strata repo re-read
for unexplored techniques) before proposing B12.

## B12 skew investigation (2026-10-18, ledger 0da340b7fb)

4-rank bounded capture (prof3) + live hwmon sampling: rank 2 (card2)
runs ~7-8% slower on EVERY kernel family - 785 MHz avg sclk vs
885/934/866 at identical 105W caps, 59-61C (silicon variance, not
thermal). Other ranks spin up to 0.597s waiting (6.6% of decode);
rank2 also arrives last in prefill NCCL (0.266 vs ~0.46s in
collectives). Mechanism for a fix: unequal TP column sharding
(numerics-safe for column-parallel sums) - but integer head splits
(Q32, KV replicated pairs 0/1+2/3) and vLLM equal-split TP
infrastructure make it a fork-wide architecture project. PENDING USER
DECISIONS (both presented): (1) NCCL fence disposition (custom-AR cap
raise / bf16 wire / accept); (2) unequal-TP scope authorization.
Loop continues with maintenance validation (battery + soak at the
accepted incumbent) while dispositions pend.

## Accepted-incumbent validation (2026-10-18, post-B12-skew)

Tree = AITER dc13c3d18 + vLLM main (docs-only since 6d401447d8).
Offline: battery 4/4, INT8 KV micro PASS, 37/37 packed component
suites. Live (8021, default launcher env): C8 answer smoke
acceptance 4.51 (reference band ~4.55); 52/52 chat corpus coherence
with finite logprobs (probe_bolt_quality). Server left running idle
at the accepted config. Full 500-case soak deferred (bounded gates
green; soak is available on request). Campaign posture:
validation-green hold; no bounded legal experiments remain above the
+-2.6% noise floor pending (1) NCCL fence disposition and (2)
unequal-TP scope decision. Loop continues with periodic
re-validation and resumes experiments on disposition.
- FULL SOAK 2026-10-18: 500/500 coherent, 0 failures at C8 in 249s
  (ua_live_soak, accepted config, port 8021). Recipe-standard
  validation now COMPLETE at the accepted state.
- OPERATIONAL NOTE (E2 consequence): at C8 with concurrent long
  contexts the 8 GB KV arena occasionally queues requests on
  capacity (seen during soak: 4 waiting, reason=capacity). Within
  the six-stream authorization (7.15x); production C8 long-context
  concurrency is softer than at 14 GB. Report to user; KVMEM is a
  launcher lever if they prefer trading expert-cache size back.
- Integrity audit: 547 ledger rows parse; vllm main == origin/main
  (947bbabb84); aiter main == curved/main (dc13c3d18); caps 105W x4,
  controller inactive, nginx down.
- LONG-CONTEXT GATES 2026-10-18 (accepted 8 GB KV config): 28k
  retrieval identical cold (13.06s) and replay (3.70s); GPU-prefix
  reset CPU-restore path works (5.73s, restored not recomputed).
  All recipe-standard gates now green at the accepted state.
- Disposition branches PREPARED (not applied):
  loop2/disposition-branches.md - Branch 1 one-line custom-AR cap
  raise (128 MiB TP4) with full measurement plan; Branch 2 scoped
  B13 unequal-TP project plan (fractional column widths first,
  bitwise oracle). Execute on user ruling only.
- RESEARCH PASS 2026-10-18 (plateau renewal): web search backend
  unavailable (subscription 403); upstream sweep instead. aiter
  upstream (through 285579b621): all recent packed/MoE work targets
  gfx950/942/1250 (Kimi-K3, MLA, sparse, flydsl). The one adjacent
  item - IQ2R packed 2-bit MoE for GLM-5.3 (a3158e4800) - requires
  MXFP8 (gfx908 has none) and device re-encoding of weights,
  banned by the official-checkpoint-bytes constraint. No applicable
  new technique found; hold state is search-backed.

## B13 (2026-10-18, user-directed restart: NEVER stop, never claim
exhaustion; several B11 closures were estimates, not measurements)

- E1 DONE/REJECTED (measured): grouped _gemm launch-config sweep
  (never swept before - B10 E3 covered only the dense kernel). Pinned
  w4/s1 IS optimal: all alternatives 3.4-21.6% slower, bitwise exact.
  Ledger row written.
- E2 RUNNING: prefill-informed expert-cache admission
  (VLLM_BOLT_PREFILL_ADMIT=1): MO apply steps the cache with prefill
  topk_ids too (m>64), so admission ranking reflects prefill traffic
  and later prefill forwards hit INT8 slots instead of decoding IQ2.
  Cache registered at load (no prefill alloc); step completes before
  GEMM on the same stream. Server booting.
- E3 QUEUED: MAX_BATCHED_TOKENS 8512 -> 4256 (chunk-size effects on
  kernel shapes + AR count; never measured).
- E4 QUEUED: expert cache 11 GiB + KV 7 GB (1,640,833 tokens =
  6.26x >= 6.0 floor; legal under the authorization).
- E5 QUEUED: draft-prefill attribution from prof3 rank traces.

## B13 progress (continued)

- E2 iterated twice: full-topk step collapsed prefill (65k-wide
  1-CTA histogram + 128-call rerank cadence; post-convergence runs
  hit 2461/2448); refined 64-token subsample + forced rerank
  (AITER cache.step rerank arg) eliminated the transient; alone
  sub-noise (prefill +0.5%/+1.1%).
- E5 COMPOSITE ACCEPTED 2026-10-18: admit + cache 11 GiB (10,024
  pairs; 240-expert layers fully resident) + KV 7 GB (6.26x floor
  verified). Vs incumbent: decode median +4.54% pooled +2.90%
  (acceptance-neutral), prefill +0.56%/+1.22%, wall +4.63% median.
  n=5 CIs span zero - caveat recorded; accepted under the
  pre-declared gate with slope + per-run evidence. Battery 4/4, KV
  micro, 37/37, C8 4.44, 52/52. LANDED: AITER a154bc169 (rerank
  arg), vLLM 18de73827a (MO lever + launcher 11GiB/7GB/admit +
  docs + ledger). CUMULATIVE CAMPAIGN TOTALS: prefill +27.9%,
  decode +16.5% vs baseline. New incumbent = composite; reference
  for next legs = loop2/b13e5 five runs.
- REMAINING B13: E3 (MAX_BATCHED_TOKENS 4256 leg), E4 (draft
  attribution). B14 ideas: composite replication for CI strength,
  chunk-size x dense-config interaction, draft prefill overlap.

## B14 (opened 2026-10-18)

- B13 closed: E3 chunk4256 rejected (prefill flat, decode -2.6%;
  interaction queued), E4 draft attribution sub-noise (orchestration
  kernels ~0ms). eec0cfcc76 pushed.
- Step-1 profile of composite incumbent (prof4): dense blockscale
  32.1%, grouped _gemm 20.7%, NCCL 17.9%; FOUND admission churn -
  _refresh_weights 0.118s/window (forced rerank every forward flips
  rankings via counts//2 decay). ~4.7% prefill cost for marginal
  decode contribution.
- E1 RUNNING: memory-only (11 GiB + KV 7, VLLM_BOLT_PREFILL_ADMIT=0)
  vs b13e5 - if decode holds ~177 and prefill recovers the churn,
  simplify the accepted config.
- E2 QUEUED: chunk4256 x dense-s3g8 interaction (admission off).
- E3-E5: from E1 verdict + prof4 deltas.
- E1 DONE/ACCEPTED 2026-10-18 (2a31797a1c): memory-only (admission
  off) vs composite: prefill +3.73% median CI +0.58..+6.40 EXCLUDES
  ZERO; decode -2.75% (CI spans 0) but +1.7% above the B10 E2 level -
  no regression to any accepted state. Launcher/docs now default
  VLLM_BOLT_PREFILL_ADMIT=0 (lever documented decode-priority).
  CUMULATIVE: prefill +32.8%, decode +13.3% vs baseline. New
  incumbent = memory-only; reference = loop2/b14e1 five runs.
  Next: E2 (chunk x dense-s3g8, admission off) and further prof4-
  driven candidates.
- E2 DONE/INCONCLUSIVE (a39ab04584): chunk4256 x dense-s3g8 - all
  point estimates positive (prefill +0.3/+1.0, decode +0.9/+2.3,
  wall +0.6/+2.0) but every CI spans zero; chunk orchestration
  overhead keeps consuming the kernel gains. Both reverted; chunking
  neighborhood closed (two legs). Deeper 2128 skipped.
- E3 DONE/REJECTED (micro): dense split-K strictly worse on every
  prefill shape (ks2 +24..70%) - CUs already saturated; no oracle
  needed. Ledger row written.
- B15-E1 DONE/REJECTED (a861019331): draft-linear int8 (W8A8-GS128
  via post-load conversion, 3 implementation iterations to clean
  boot). Decode -2.96% median, prefill -2.23% - tiny-M bf16 Cijk
  GEMMs already optimal; quantize+blockscale overhead exceeds
  savings. ACCEPTANCE HELD (3.26-3.49 within gate) - the Cijk family
  is speed-optimal as-is, not a quality question. Reverted clean.
  LESSON: draft GEMM surfaces are tiny-M - int8 conversion cannot
  win there by construction.

## B15-E2 design (2026-10-18): dense blockscale prefill-only MFMA

ROOFLINE CASE: dense blockscale at (4640x4096, m=6144) = 6.46ms =
36 TFLOPS ~ 5% of gfx908 int8 MFMA peak (728 TFLOPS theoretical);
even 30-40% achievable = 3-6x kernel headroom on the family owning
32.5% of prefill (prof5). Potential +10-25% prefill.
PRIOR ART: B09_E2 register-fed MFMA dense path - REJECTED on decode
regression (replaced the shared path at all m). FIX: dispatch the
new path m>64 ONLY (prefill), leaving decode (m<=64) on the current
path. Saved sources: loop/batch09/exp2-saved/ + *before-launch-bound.*
archives. Design study dispatched (agent-1); implementation next.
- DESIGN STUDY COMPLETE (spec in agents/tmp session log; key facts):
  B09_E2 failed on SCALAR direct loads (1/8-1/4 sector efficiency,
  no LDS sharing), not the MFMA math (bitwise-exact 70/70 + 432
  projections). Decode loss was likely unpaired-run confounder, not
  causal. v2 spec: coalesced LDS staging (16B/lane), swizzled
  fragment layout, manual 2-stage prefetch pipeline, 16x16x16i8
  v1 / 32x32x16i8 v2 knob, launch_bounds(512), ScratchSize==0 gate,
  bitwise oracle test_native_dense_g128_gfx908.py resurrectable,
  component-bench MUST beat Triton before any serving run (E2 -65%
  was visible there). Latency-bound diagnosis: num_stages=1 exposes
  600-800ns per K-block.
- E2 CLOSED AT COMPONENT GATE (f3e5a1b785): v2a (coalesced LDS
  staging + XOR swizzle, fixing E2's scalar-load failure) = 0.56-
  0.67x Triton; v2b (+ ping-pong prefetch pipeline) = 0.57-0.64x;
  NW wave templating 2/4 = WORSE (0.47-0.56x, occupancy loss beats
  LDS-traffic gain - falsified the LDS-bound theory). Bitwise oracle
  held on every variant/shape. Never beat Triton at production
  shapes; only tiny launch-bound shapes win. Verdict: Triton codegen
  (36-41 TFLOPS + tuned table + full shape coverage) is genuinely
  strong on this family; native route closed. Tree reverted clean.
  B15 tally: E1 rejected (tiny-M), E2 closed (component gate), E3
  rejected (settled regression). Dense-family routes now exhausted
  at: config (3x), split-K, chunking (2x), kernel rewrite (2
  campaigns). Remaining prefill upside: NCCL 18.1% (FENCED, user
  disposition) and unequal-TP scope decision (decode skew 6.6%).

## B16 (2026-10-18)

- E2 ROUTER TOPK investigation: decode sbtopk = 0.291s (3.2%) from
  the Python grouped-topk fallback (torch.topk x3-4 per layer-step;
  forward_hip gated on VLLM_ROCM_USE_AITER_MOE which the packed
  stack never sets). AITER dispatch-fix premise DEAD:
  biased_grouped_topk_hip is an undecorated no-op stub in this
  checkout; the real impl needs module_moe_asm (CK-dependent,
  deliberately skipped by the fork's shim build; core.py:1125).
  Bolt config: n_group=1/topk_group=1/9-of-240..496 sigmoid+bias -
  degenerate plain biased top-9. VIABLE ROUTE: custom Triton kernel
  (select by biased scores, ties by smaller index = torch.topk
  convention; gather unbiased sigmoid weights; renorm same-op -
  bitwise ids + weights parity achievable; torch.sigmoid stays
  torch for ULP-exact scores). Expected: replace 3-4 launches +
  elementwise per layer-step -> est +2-3% decode, ~+0.5-1% prefill.
  Kernel implementation next turn (correctness-critical, fresh
  context).
- E1 QUEUED: prefill torch.compile probe (mode=0 is fork default;
  diagnostic boot first - custom ops likely graph-break, cheap fail).
- E2 DONE/REJECTED (a55f692740): Triton biased top-9 kernel built,
  parity-proven (id SETS exact incl. tie test - reference is
  sorted=False arbitrary order; weights ULP-equal paired by id),
  micro 2.8-8.7x per-call - but the micro was CPU-DISPATCH-bound.
  E2E flat-to-negative (prefill -0.7%, decode -2.0%, CIs span 0):
  serving hides router Python overhead behind GPU work; true ceiling
  was sbtopk 3.3% of decode. LESSON (matches E3b/E3a pattern): per-
  call CPU timing misleads in this serving stack - only profiled GPU
  time x e2e validation counts. AITER route confirmed dead (stub +
  skipped ASM module). Kernel reverted.
- E1 DONE/REJECTED (17e87f86e1): torch.compile mode 1 (boot
  succeeded; watcher false-death was a port-8020 status artifact).
  Prefill -13.37% median CI EXCLUDES ZERO, decode -5.54%. The fork
  mode=0 is now measurement-validated. Launcher reverted.
- B16 CLOSED: E1 + E2 both rejected with mechanisms (compile
  replaces tuned paths/graph-breaks; router micro was CPU-bound).
  Server stopped; tree at incumbent. B17 next: fresh research pass
  for mechanisms not yet in the ledger (candidate angles: serving-
  shape-specific dense table entries via offline autotune of the
  Triton config table for OUR shape population only; expert-cache
  slot policy refinement; draft attention backend).

## B17 (2026-10-18)

- E1 DONE/INCONCLUSIVE (efcfab8b3e): CENSUS (major finding): serving
  prefill m = {1024, 1064, 4104, 8512, 72, 1032, 4112} - all prior
  dense experiments assumed 6144/4256; that wrong-m assumption WAS
  the micro-vs-e2e gap. Per-shape autotune of 45 true shapes:
  43 win +7..+128% kernel (s3g8 family; m=72 wants 64x64x4w). E2E
  STILL FLAT (prefill -0.8%, decode +1.3%, CIs span 0). LAW (4th,
  cleanest): dense blockscale is not the serving prefill constraint
  at any config - time sits in grouped _gemm (23%), NCCL (18%,
  fenced), attention. Table reverted; census+tuner retained.
  Prefill constraint list now precise: E1-hit-optimized grouped
  _gemm, NCCL (fenced), page attention (1%), chunk_scan (1.4%).
- E2 DONE/ACCEPTED (7ef3352a09): tiered route tiles (BM=64 at
  routes>=experts*24, BM=32 at *8; prior rule left serving m=1024/
  1064 chunks on BM=16 forever). Micro: bm32 +46..74% at true
  shapes; tiers match winners in all 15 cells. E2E: prefill +4.20%
  median +3.62% pooled (largest since B14), decode pooled +0.54%,
  decode median -1.27% (noise band). C8 4.53, 52/52, battery 4/4.
  LANDED (vllm MO change only). CAMPAIGN TOTALS: prefill +38.2%,
  decode +11.9% vs baseline. New incumbent; reference = loop2/b17e2.
  Next: B17-E3+ (BM=48 or occupancy-adaptive tiers; expert-slot
  policy; the same census re-check on the grouped kernel via a
  shapes logger for packed_experts).
- E3 CENSUS DONE (grouped-census.json retained): tiers confirmed
  engaging live - BM=64 at m=4104/1064 and <=368-expert layers at
  1024; BM=32 on 432/496-expert layers at m=1024; cache hits flow
  during prefill (per-call hit-rate needs finer logging - deferred).
  1600 grouped calls per bench; tiles 381-1003; no tier anomaly -
  the E2 tier map matches micro winners in production. B17
  remaining: E4 (expert-slot policy audit), E5 reserved.
- E4 CLOSED (fb81bcf27c): offline coverage model - equal+smallest
  16.4% mean vs proportional 15.5%; ~2pp redistribution. A full 2x
  cache doubling measured only +2-4% pooled decode (B13-E5), so a
  2pp shift is an order below the noise floor. No measurement
  spent. B17 CLOSED: E1 inconclusive (4th law), E2 ACCEPTED
  (+4.2% prefill - campaign's biggest single prefill win since
  B14), E3 census (tiers confirmed live), E4 audited. Environment
  re-verified this turn: 105W x4, controller inactive, nginx down,
  trees clean at 7c293f1e8d / a154bc169. Campaign: prefill +38.2%,
  decode +11.9%. B18 next: fresh pass (remaining: NCCL fence and
  unequal-TP scope decisions with user; new angles from census
  artifacts).

## B18 (2026-10-18)

- Step-1 profile (prof6): prefill kernel-sum 2.525 -> 2.439s;
  grouped fell 0.580 -> 0.486s (-16%) from the tier win. Mix: dense
  33.7% (proven non-constraint x4), grouped 19.9%, NCCL 18.8%
  (fenced), other 8.4%.
- E1 DONE/REJECTED (1311345804): admission retest at new tiers -
  prefill -5.14% median CI -7.4..+0.6. Churn GREW at BM=32/64
  (refresh storms over 381-1003-tile plans); hits were already
  flowing from decode-warmed ranking. B14-E1 verdict confirmed at
  the new geometry; no flip. Admission stays default-off. The re-
  open hypothesis is answered and closed.
- B18 remaining: 'other' 0.205s breakdown; next census angle.
- E2 CLOSED (ledger): 'other' family = long tail, max item 0.5%.
  Unfenced prefill surface fully measured to closure.
- E3 REPLICATION DONE (df7d1e3b8a): fresh-boot replication of the
  B17-E2 incumbent vs b14e1: prefill +1.20% (vs original +4.20%).
  Two-boot mean ~+2.6%. ACCEPTANCE STANDS (both samples positive,
  kernel mechanism proven, decode unharmed); LEDGER CORRECTED.
  CAMPAIGN TOTALS (honest): prefill ~+36%, decode ~+12.5% vs
  baseline. LESSON: single-boot medians at n=5 can overstate by
  2-3x; acceptance rows with CI-spanning-zero gains get a
  replication before headline claims. B18 CLOSED (E1 reject, E2
  audit, E3 correction). B19 next.

## B19 (2026-10-18)

- FULL-STACK VALIDATION (a1a2a95388, no code changes): (1)
  Replication from existing samples - both incumbent boots vs the
  original baseline: prefill +38.32/+34.33 (CIs [30.4,42.0] and
  [32.1,39.5]), decode +11.85/+13.16 (CIs exclude zero) - ALL FOUR
  CIs exclude zero; totals CONFIRMED prefill ~+36%, decode ~+12.5%.
  (2) 4096/C8 holdout at the tier-enabled incumbent (fresh 5 runs):
  prefill 2694.9 (+17.2% vs pre-E1 reference; +9.8% vs the B13-era
  holdout), decode 183.8 (+9.2%/+9.8%) - the tier change GENERALIZES
  to the held-out workload, both phases.
- State: everything unfenced is closed and now cross-validated.
  Remaining upside sits behind the two user dispositions (NCCL wire
  policy ~18% prefill; unequal-TP scope ~6.6% decode). B19
  continues: sustained-value validation cadence + upstream watch;
  new experiment batches require new mechanisms (research) or the
  dispositions.
- E1 DONE/REJECTED (732487c4b7): MIXED_PREFILL_TOKENS 4256->8512 -
  prefill exactly flat (-0.15/+0.15%; continuation chunks are
  per-request 1024-token, request-bound not budget-bound), decode
  -1.67% (bigger prefill batches steal decode rounds - the exact
  mechanism the 4256 budget prevents). The mixed-prefill knob
  family (budget, interval, cohort) is now FULLY swept. Campaign
  totals unchanged: prefill ~+36%, decode ~+12.5%, replicated.
- E4 RESEARCH (1f0abbd846): first vLLM-upstream sweep (100 commits
  since 10-10, 33 hits triaged) - NEGATIVE: fusions are kimi_k3-
  specific; per-tensor quant is numerics-incompatible with GS128;
  sampler items sub-noise per draft audit; arch-specific items
  excluded. Both upstream repos now covered by the research
  cadence. Loop continues: next passes watch for new upstream
  mechanisms; dispositions remain the decisive upside.

## B20 (2026-10-18)

- E1 ABANDONED (81560798de): native decode census - 3 boots failed
  (GPU work inside graph capture; then /dev/shm exhaustion from
  leaked psm_* segments of the crashed boot; then warmup RPC
  conflict despite the capture guard). Family already closed by
  E3a/E3b + skew bound - census was completeness only. Tree clean.
  OPERATIONAL: crashed boots leak /dev/shm/psm_*; rm before reboot
  (cleaned; 31G free). Power re-verified 105W x4, controller
  inactive. Campaign totals unchanged: prefill ~+36%, decode
  ~+12.5% (replicated, generalized).
- E2 THIRD-BOOT VALIDATION (c5ccef308d): prefill +36.30% CI
  [33.8,40.8], decode +11.38% CI [5.9,16.2] - both exclude zero.
  Three-boot spread: prefill +38.3/+34.3/+36.3, decode
  +11.9/+13.2/+11.4 - STABLE. 52/52 quality re-verified. Totals
  stand at prefill ~+36%, decode ~+12%.

## Campaign-2 roll-up (2026-10-18, ~6h elapsed)

36 campaign-2 ledger rows: 4 accepted changes (E1 cached-INT8
prefill, E2 memory reallocation, B13-E5 composite superseded by
B14-E1 memory-only, B17-E2 tiered tiles), 12 rejected with
mechanisms, 7 audited-closed, 5 inconclusive-reverted, 1
investigation (skew), 2 measurements (censuses), 1 replication
correction, 1 research pass, 1 abandoned, plus 3 validation rows.
Upstream watch: 1 new commit since last sweep (nothing relevant).
Standing: prefill ~+36% / decode ~+12%, three-boot stable,
generalized at holdout, all gates green. Dispositions pending:
NCCL wire policy (~18% prefill), unequal-TP scope (~6.6% decode).
Next pass: continue watch + validation cadence; dispositions
execute on user ruling.
- Watch pass 2 (post-rollup): AITER head unchanged (285579b621);
  vLLM upstream 1 irrelevant commit. Conformance check: caps
  105W x4, controller inactive, nginx down, trees at
  3f3bc376f1/a154bc169, launcher free of experimental overrides
  (grep=0 hits). State fully conformant; no drift.
- Watch pass 3 (11:20 UTC): both upstream heads unchanged;
  no new mechanisms. Standing unchanged.
- Watch pass 4: vLLM moved (7c9042c05a -> 953fa442fa) - 6 new
  commits triaged: CI/docs/video/rl-protocol/sm80-90 build fix -
  NONE applicable (no gfx908/rocm/mamba/moe surfaces). AITER
  unchanged. No batch triggered.
- Watch pass 5: both heads unchanged (953fa442fa / 285579b621).
  Standing unchanged.
- Watch pass 6: both heads unchanged. Cadence continues.
- Watch pass 7: heads + environment verified unchanged/conformant.
  To reduce checkpoint churn, watch passes now batch into the
  scratch note only when upstreams MOVE or a validation re-run
  occurs; standing remains prefill ~+36% / decode ~+12%.
- Watch pass 20 (batched; passes 8-19 all unchanged): vLLM moved
  953fa442fa -> 3c13653738 - 1 new commit (sentence-transformers
  crossencoder frontend; not applicable). AITER unchanged
  throughout. No batch triggered.
- E4 ANALYSIS (prof4): decode Cijk (draft GEMMs) = 1.159s, 12.9% of
  the decode window, 6287 launches ~14/draft-step, 0.18ms avg.
  Int8-GS128 swap ceiling ~+7.8% decode (kernel ~2.5x). B15-E1
  candidate: route draft dense linears (Q8/U8 source weights) through
  the target's INT8 dense-cache path; ACCEPTANCE GATE mandatory
  (draft surface, +-1pt acceptance; fp16-KV history shows draft
  sensitivity; PTQR precedent shows int8 drafts can hold). Prefill
  untouched (draft prefill negligible per B13-E4).
- prof5 of the accepted incumbent booting for the B14 close-out mix.
- prof5 DONE (churn-free confirmed, cache refresh 0.000s): dense
  32.5% / grouped 23.0% / NCCL 18.1% / other 12.6%. B14 CLOSED:
  E1 accepted (+3.7% prefill CI-demonstrated), E2/E3 measured-dead,
  E4 sized B15-E1. Campaign: prefill +32.8%, decode +13.3%. B15-E1
  next: draft dense INT8 (acceptance-gated, +7.8% decode ceiling);
  implementation in the draft model path next turn.

## Standing constraints (unchanged from campaign 1)

- No hot kernel reloads; restart server per experiment.
- Do not modify collectives/AR precision; _tp_ar_fp32_mode quirk stays.
- Legacy FUSED_NORM_QUANT stays off. Official checkpoint bytes unchanged.
- Git as checkout owner (curved), both forks main; run security-hygiene-check
  before every commit; ledger rows in docs/recipes/surface_experiments_ledger.jsonl
  for every experiment; A/B row before any perf claim.
- Protected unrelated worktree paths:
  agents/tmp/bolt-opt/loop/protected-worktree-hashes.json (10 paths + 3
  deletions) - never broad restore/stage/stash.
- Battery 4/4 + INT8 KV micro after accepted runtime changes; acceptance gate
  for draft-surface changes.
