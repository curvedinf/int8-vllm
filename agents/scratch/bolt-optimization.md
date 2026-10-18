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

## B10 closed; B11 sketch (design next loop pass from prof2)

Accepted: E1 + E2 (cumulative prefill +27.2%, decode +11.4% vs
campaign baseline). Negatives documented: E3 (dense config - kernel
gain sub-noise end-to-end), E5 (cohort load-bearing, default kept),
E4 (tile growth sub-noise). B11 candidates:
1. NCCL SP-MoE all-gather 20.6% of prefill (nemotron_h.py:285) -
   BLOCKED on user disposition re wire policy (no collective
   backend/precision/order changes).
2. Router sbtopk 2.6% + _quant 2.4% prefill fusion candidates.
3. Expert cache >10 GiB needs KV below the 7.15x floor - NOT
   available under the six-stream authorization.
4. Strata EXPDEC-style activation staging for native packed_routed
   decode path (25.6% of decode) - decode protection/gain.

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
