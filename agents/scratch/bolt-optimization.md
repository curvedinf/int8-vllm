# Bolt continuous optimization

Active user contract (2026-10-16): optimize indefinitely until manually stopped.
Each batch profiles4096input/256output/C8, designs FIVE experiments across
scopes including structural fusion, implements sequentially and benchmarks
each EXACTLY FIVE measured times. Those five define the result. No extensions
or extra performance repeats for uncertainty; mark inconclusive and move on.
Retain only demonstrated end-to-end wins without prefill or
decode degradation, commit/push wins, then repeat. No subagents authorized.

Report AGGREGATE prefill and decode regularly. Corrected v2 client uses one
shared monotonic clock: prefill ends at the latest first-token event; decode
counts only later token events / remaining generation-only duration. Record
remaining requests (5-6 under current admission); this is the decode-only tail,
not steady full-C8. Whole-run output throughput is separate. Old per-request
rates cannot establish aggregate decode. Synthetic clock/burst check passed.
Frozen manifest SHA256263da19de5a5bc8823fe38d7113553c08f00fa453127dbbd523c74622d0bcda1,
eight technical prompts exactly4096input/256output, seed0/temp0/ignoreEOS,
stream/tokenIDs. GPU+CPU prefixes reset before every run;8success/32768input/
2048output counters and returned token counts asserted. One full untimed warmup
per leg. No profile/micro/benchmark GPU overlap. Profile timings excluded.
Predefined gate: >=3% pooled whole-run gain; median AND pooled aggregate
prefill/decode no worse. Inconclusive comparisons are not adopted. Full
fidelity/256k validation remains deferred under the user contract.

Current incumbent: AITER15a3e419553d970169edbf386a4ff7166dcc2f2a, exact tree of
3b85a215 (requested memory changes, original packed Triton large-prefill path).
Published rollback of0d8d783ab cache-aware prefills. Root48a1e1e629 before
current documentation/ledger update; independent LiLiCorr fixture fix
eaa11b900b stays. Current five-run medians: aggregate prefill925.7928tok/s,
aggregate decode96.7054tok/s; pooleddecode92.4945; pooledwhole-run47.6546.
Corrected E2 candidate: medians1021.1976prefill/89.3540decode; pooleddecode
90.2872, pooledwhole50.3717. End-to-end+5.702%, pooledprefill+7.396%, but
pooleddecode-2.386%/mediandecode-7.602% fails phase gate. No proven causal
slowdown (wide intervals); conservative withdrawal. Historical +7.90% E2
whole-run evidence remains valid, old per-request phase gate was insufficient.
Raw five-run legs and comparison: loop/batch02/{incumbent-control-*,
pre-e2-control-*,e2-aggregate-requalification.json}; harness version
aggregate_decode_only_after_all_prefills_v2. Do not restore temporary0d8
backups in batch02/pre-e2-source; rollback is now the retained configuration.

The phase calculator is now also reviewable in
vllm/benchmarks/streaming_phases.py, with scripts/report_bolt_phases.py for raw
trial JSON. Nine CPU regression checks pass (shared-clock bursts, admission,
finished requests, absent tails and malformed timelines). Recalculation of23
recorded v2 trials matches every stored phase field exactly. CLI requires
PYTHONPATH="$PWD" when this venv resolves installed benchmark modules first.
Ongoing run.py remains byte-for-byte frozen and still uses its local helper;
public code is an equivalent validator, not a mid-experiment method change.

Requested memory settings remain accepted; do not resume old acceptance tests.
Dense ceiling3GiB, original packed GPU tensors/codec metadata released:
277projections/2,387,607,552INT8values/2.224GiB perGPU; originals freed1.389GiB.
Expert ceiling5GiB/slotsauto/refresh128, originals retained:4565pairs,
114-119slots/layer, payload5,368,184,832+175,944metadata bytes. GPUKV14e9/rank,
CPU24GiB withrank-local6GiBpin, context262144, TP4/C8/NS7, INT8GS128compute/KV,
BF16residuals,FP32Mamba,customAR. All4GPUcaps105W; thermalcontroller stopped
ANDdisabled. nginx stopped; no80/443listeners. Installed Bolt thermal dependency
renamed thermal-controller.conf.disabled-static-cache-test; do not reintroduce.
Production vllm-bolt stopped; never start alongside isolated diagnostic unit.

Diagnostic bolt-opt-loop.service is STOPPED (23:38UTC). E1candidate source
fully archived and restored; AITER clean15a3. Next implement E2native invariant
work elimination, component/micro/battery gates, fresh serve boot and EXACTLY
FIVE full workloads. No GPU micro/profile/benchmark overlap. Production stays
off. Diagnostic8021/logdirloop/runtime, lifecyclebatch02/host/gateactive,
chunk32steps/devRPCisolated. Do not perform another reference extension.

Batch02 reference profile:590501kernels/alllaunchcorrelated; prefill+mixed
23.3892rank0kernel-seconds, packedrouted10.3514s/44.257%;collectives3.8049s
(includespeerwaiting). reference-{csv/,profile-analysis.json,opportunities.json,
retained-sources.json} andhost/gpu/reference-retained-*. Profile timings excluded.
Preregistered experiments-plan.json against15a3 reference:
1Separate cached-only INT8 Triton prefills and guarded packed misses: DONE,
INCONCLUSIVE/reverted.2Native invariant reciprocal/base-scale elimination.
3Route-plan API/32-64row decoded-weight reuse.4Native UP/ReLU2/GS128 output
fusion.5Bounded initial single-page short-prefill admission; preserve long/mature
decoder cadence. Design exp5-admission-design.json. Implement sequentially.

E1 original FIVE: median1091.0658prefill/94.5625aggregate decode; pooled94.4056
decode/54.5436whole-run. Whole-run+14.456%, but median decode-2.216% vs96.7054
reference although pooled+2.066%; fails the predefined median phase gate.
Not adopted.31component checks, final10prefillchecks,32realcheckpoint checks
bit-exact (all4TPoffsets/4widths/m65,256); KVmicroPASS,battery4/4,ruffpass.
Source/harness hashes exp1-sources.json andexactbackup exp1-saved. Raworiginal
exp1-{0..4}.json/summary/first-five-comparison/decision. Candidate runtime and
tests now fully restored to15a3; no E1 performance code retained.

The agent incorrectly initiated an extension because median/pooled decode
signs differed. User explicitly corrected twice at23:37-38UTC: EXACTLY FIVE
measured runs, use those five as the metric and move on. Extra candidate five
had already completed by interruption; retained as UNREQUESTED exp1-repeat-*
and excluded from the originalfive decision. Extension canceled; no reference
repeat. Never treat five as a minimum or broaden perf repeats for uncertainty.
All earlier trial data preserved; do not selectively combine extras.

Batch01 completed. E1native-onlylargeprefill+1.67%whole/inconclusive/reverted.
E2cachedprefill initially accepted0d8, NOW WITHDRAWN under corrected phase gate.
E3denseepilogue+2.27%whole/below3%/inconclusive/reverted;64checks passed after
independent fixture cleanup fix, final10checks/micro/battery4/4.
E4all-groups-selected routing+0.985%whole/inconclusive/reverted;13cases expert
sets match and weightsrtol1e-6/atol1e-8. Compiled unsortedtopK can reorder
experts even in baseline, so exact eager-vs-compiled order is invalid oracle.
E5dense128x128/8warps/stage2M>64: micro4.52%held-out, full+0.071%whole,
prefill-0.155%/olddecodeproxy-0.016%;notadopted/reverted. All6variants exact
on6discovery shapes, winnerexact6held-out shapes;21integration,micro,battery4/4.
Batch01 sources, diffs, tests, raw five trials/hardware and failures retained
under loop/batch01/exp{1..5}-*. No perf candidate currently retained frombatch01.

Offline native evidence/prototype, NOT deployed: batch02/native/{incumbent.isa,
incumbent-metadata.txt,instruction-counts.json}. Cached alignedUP executes
precise reciprocal before hit/miss branch and8ushort scale loads, vs1aligned
DOWN;47VGPRs/8AGPRs/no spills. Prototype exp1-prototype.hip/.so CPUcompiles,
ISA exp1-codegen shows1aligned/2unaligned scale loads, same8MFMA; reciprocal
inside miss branch. Need real GPU tests with varying FP16 scales, allpartial
TPoffsets and hits/misses. Existing make_weights uses constant0x3000 scales;
that alone cannot catch wrong256-element block selection. Runtime csrc remains
unmodified. llvm-objdump --offloading EXTRACTS sidefiles; operate only on copies.
Native wrapper hashes source+hipccflags, so native edits recompile safely.
MT>1 requires explicit plan alignment and MT-aware COUNTguard; changing MT
alone with16-row plans is invalid. Cache-only separation/fusion must not read
unwritten hit partials; preserve orderedFP32group accumulation/INT8rounding.

Use Git mutations/push ascurved to avoid root-owned Git objects. AITER remote
curved; vLLM origin; bothmain. Preserve unrelated root edits including service
migration/startup-watch,AGENTS.md,bolt-support,serve_recipe_qwen38,services docs,
logs/README deletion,newQwen support doc. Fingerprint snapshot loop/
protected-worktree-hashes.json. No venv dependency changes. All Python .venv.
Offload diagnostic path issue resolved by explicit user direction: checkout-
relative flagdirectory, no disclosurepermission. Allthreeflags absent; behavior
unchanged. Before everycommit scan final stagedblobs/diff/metadata under
security-hygiene-check and security-policy.md; accountpath history unchanged.

Costly prior evidence: CPU24 full-region registrations exceed AMD hostmemory
accounting; rank-local pin/copy24GiBTP4 micro/fullboots passed. Oldwatchdog uses
production PIDfiles; never reuse for isolated operations. Atomic launcher edits
avoid running-shell shutdown corruption. Legacy speedprobe is C1cold8192/1
prefill andC8random32/1000decode (2warmups); incomparable to4k/256. Do not rerun
legacy baseline. Prior rejected dynamic32expert cache (1531/351vs1545/362),
singletonSDOT,quanttables,densepreshuffle,FP32routersplit8,IQ2padding66->68:
see priorledger/artifacts before repeating. Actual experts240-496, never512.
