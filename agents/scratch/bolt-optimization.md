# Bolt continuous optimization

Continue until the user manually stops. Each batch profiles frozen4096 input /
256 output /C8, registers FIVE experiments across scopes including architectural
fusion, and executes sequentially. EXACTLY FIVE measured runs per final candidate
are the entire metric budget. No extensions, extra controls, or new baselines;
reuse the latest accepted original five. One untimed warmup per boot is established.
Component tests and bounded profiles are separate from the five-run metric.
Keep positive end-to-end gain with median AND pooled aggregate prefill/decode no
worse; otherwise restore only owned changes. Commit/push winners, renew batches
automatically. Agent-added3% whole-run floor was explicitly removed atB05E2 to
honor user criterion; that correction and originalfloor miss are in its ledger
row. Earlier negative pooled-phase verdicts remain unchanged. No subagents.
Report aggregate prefill AND separately timed aggregate decode regularly; never
per-request rates or whole-run output rate as decode. Commentary within60seconds:
check clock, shorten reasoning, keep waits below30seconds.

## Current state and next action

Accepted incumbent: rootd70e9b6ed8694e77abbdccbcf5de97e9fcaf0146 and
AITERba264db0b19c6b058c981533bf424332659ed5fe, bothmain/pushed/remoteverified.
B05E2 exact routeweight/reduce/outputcast fusion atop B02E5shortcohort and
B03E1route64. Reference B05exp2-{0..4}.json ORIGINALFIVE: median2081.5140315
aggregate prefill/162.7157729aggregate decode; pooleddecode163.7370051,
pooledwhole74.6502256. Use exp2 forE4/E5, no newbaseline.

B05E3 COMPLETE/reverted afterfive: median2152.7672311prefill/165.0328814decode,
pooleddecode162.2629241(-0.900%), whole+1.843%.142component/regression checks,
41finalquantizer checks,64realcheckpoint cases andKV/battery4/4 pass; exact
arithmetic preserved. BM16 selectedfrom4/8/16/32 beforefive, weighted kernel
92.360->26.343ms. This component win didnotpass pooleddecode guard. All source,
originalfive andchecks inB05exp3-* /exp3-saved. AITER restored exactly toba264;
newquantizer test removed onlyafter snapshot verification. No E3code retained.

B05E4 COMPLETE/reverted: exactlyfive median2111.7576106prefill/164.3918404decode;
pooleddecode163.6306389(-0.06496%),whole+1.080%. Tinynegative isnotcausalproof;
no extra runs or gate changes. KV/battery4/4/Ruff,29components and78actual
checkpointcases exact.112operator timings+6.756%4096/+6.882%4160. Source/rawfive
exp4-saved/exp4-* preserved. Restored onlypacked_dense_cache.py/removednewtest;
AITERcleanba264. Server stopped, bench24441/monitor22433 exited0. B05E3 ledger/
notes e2686252a9 pushed/remoteverified. Acceptedreference stays exp2 originalfive.

B05E5 COMPLETE/reverted: exactlyfive median2096.9516635prefill/167.2895145decode;
pooleddecode163.2031650(-0.326%),whole+0.538%.113component/regressionchecks,
20actualcheckpointcases allTPoffsets,KV/battery4/4/Ruff pass. NativeFP32norm/
bothBF16rounds/INT8/F32scaleoutputs exact. Startupconfirmed fusion selected.
Prototype4.07xproducer didnot yield a phase-nondegrading stack. Restored only
5rootproductionfiles/2AITERfiles andremoved2snapshot-verifiednewtests. Allsource,
proofs/originalfive inB05exp5-saved/exp5-*; no E5code retained. AcceptedE2 stays.

ACTIVE B06: freshprofile complete; server stopped; profile81066/monitor9831,
converter58468/analyzer94412 exited0. Correct analyzer --gpu-csv takesDIRECTORY,
notkernelCSVfile (emptyinitialattemptwasoverwritten; actual298917kernels). Rank0
pid1636360,all298917 launch-correlated. Prefill/mixedsum15.2717rank0kernel seconds:
dense/otherGEMM29.320%,packed24.826%,collectives20.011%(includeswaiting),
attention9.909%,quant0.57735s. Profiletimes excludedfrommetric.

B06experiments-plan.json registersFIVE beforeimplementation:1exactintegerIQ2
quantLUT;2immutableAITERINT8preshuffle;3aligneduint16IQ2headerreads;4attention
metadatahoist/prefetch/unroll preservingmathorder;5latent/sharedinputquantreuse
viaexplicitforwardpayload,no globalcache. Allconstraints/validation inplan.
B06E1 COMPLETE/rejected/reverted afterexactlyfive: median2007.5566699aggregate
prefill/165.1814276decode; pooleddecode162.9159325(-.501%),prefill-3.553%,whole
-2.488%. Componentrate-14.827%/-14.709% alreadynegative; no postmicrotuning or
extraE2Eruns. ProofCPU149323776signedquant/3047328maxrecovery andGPU130023424
values zero;84components/64actualallTPcases PASS;KV/battery4/4/Ruff. LUT4KiB
reservation fitunchanged4565pairs/allGPUs. Allsource/proofs/originalfive retained
B06exp1-saved/exp1-*. Restoredonly4trackedAITERfiles andremovedbacked-upnewtest;
AITERcleanba264/acceptedcombine retained. Rootruntimeunchanged. Server stopped;
bench67897/monitor88109 exited0. New ledger/notes pendingcommit.

ACTIVE B06E2 immutable densepreshuffle: inactive prototypepreshuffle_exact_
prototype.py preparedfromAITERkernel withcurrentleftassociation/FPfusionOFF.
Micro session83216(preshuffle-micro.py) running onidleGPU; logexp2-preshuffle-
micro.log. No production E2edits yet. Samepayloadbytes,onepersistentINT8copy,
logical[N,K]metadata;eligibleN%16==0/K%128==0,otherweightsrowmajorfallback. New
resident_preshuffle cachekey cantracklayout withoutstaleaddressmarkers; clear
cachealso clearslayout. Wrapperusesrowmajoractscales; preserveFP32scale/group
order,outputcasts,cacheownership/releasedoriginals. Followplanvalidationthen
EXACTLY5 versusB06incumbent(originalB05E2five), no newbaseline/extensions.
RemainingE3uint16header/E4attentionmetadata/E5sharedinputquantreuse registered
inB06plan; no futureproductionedits duringcurrentcandidate'sfive.

## Fixed serving contract

Allfour MI100 caps105W. mi100-thermal-controller.service stopped AND disabled;
installeddependency dropin renamed thermal-controller.conf.disabled-static-cache-test.
nginx must stay down; productionvllm-bolt.service stopped. Never run production
alongside isolatedbolt-opt-loop.service. Staticdense3GiBceiling;277projections,
2387607552INT8values/2.224GiB perGPU; originalpackedGPUweights/codecmeta released
(1.389GiB freed). Expertcache5GiB/slotsauto/refresh128, originalIQ2packed experts
retained:4565pairs/114-119slots perlayer,5368184832payload+175944metadata bytes,
348344unused. GPUKV14e9/rank; CPU24GiB total/ranklocal6GiBpin;context262144;
TP4/C8/NS7;INT8GS128compute/KV,BF16residual,FP32Mamba,customAR/largefallback.
KVcapacity3283288tokens/12.52fullcontexts. User deferred fullfidelity/256k checks;
do not claim passed or resume accepted-memory-config testing. Legacy1500/400
probes used differentC1cold8192/1prefill vsC8random32/1000decode; incomparable.

Acceptedroute64: M>64/routes>=actualExperts*64 withmatchingUP/DOWN plan;
native/sparse16. Shortcohort shared4256prefill/8512total, eligibleinitialtext/
output+pending<=64 may enter offcadence;long/mature/MM/LoRA/preempt/stale retain
16roundcadence. Combine uses exactATen4accumulator9waysum withFPfusionOFF;
B256/W4 M<=64,B512/W4 otherwise. Preserve this API/rootexport when reverting.

## Evidence, infrastructure and rejected directions

Canonical ledger docs/recipes/surface_experiments_ledger.jsonl. Ignored raw/source
artifacts agents/tmp/bolt-opt/loop/batch01..batch05; preserve originalfive.
B05profile rank0pid1569371/299185kernels all launch-correlated. Prefill/mixed
rank0kernel sum15.6339s (notTP4wall): dense/otherGEMM28.639%,packed24.235%,
collectives20.968%(includeswaiting),attention9.689%,quant0.5707s. RawKineto grid
is a LIST; quant grid[0]=M/grid[1]=ceil(K/128). Activation widths1024/2048/4096/
24576 with88/40/128/1calls perprefill step. B05experiments-plan.json preregistered
allfive; E2 accepted, E1/E3/E4/E5 reverted; allcomplete.

B05E1 newfactorizedIQ2/cache-hit/header composition: whole+6.663%,medianprefill
+12.700%/decode+.827%, pooleddecode-.400% =>reverted. CPU149323776signedcases/
GPU130023424values zero;100components/32realchecks,48completeMLPtimings+53.963%.
Proof is distinctfromprior integer-IQ2proof; artifacts exp1-saved/exp1-*.
B04allfive reverted: two-pagecohort,route128,singlelaunchcacheprefill,denseUP-
quant/directBF16epilogue,ROCmsharedoverlap. E3whole+4.116% butpooleddecode-.502%;
E4whole+3.015% butpooleddecode-.181%; E5whole+.393%/decode-.392%,peak31.845GiB
(~159MiBheadroom). Do not shrink caches/KV tofit workspace. Currentpeak30.718GiB.
B03E1route64 retained; E2header8way,E3nativeUProwleader,E4full/groupMLPfusion,
E5temporaryINT8staging reverted. B02E5shortcohort retained. B02E1 UNREQUESTED
extra5 exp1-repeat-* are excluded; neverrepeat this mistake. B01cacheprefill0d8
WITHDRAWN in15a3 undercorrectphasegate; do notrestore.

Loop harness: frozenrun.py/compare.py/phase_metrics.py/manifest-4k.json inloop/.
bench NAME --batchbatchNN --repeat5; compare BASE CAND --batchbatchNN. Frozen
manifest263da19de5a5bc8823fe38d7113553c08f00fa453127dbbd523c74622d0bcda1:
eighttechnicalprompts4096 each/256actualtokenIDs,seed0,temp0,ignoreEOS. ResetGPU+
CPU prefixes eachrun; assert8success/32768input/2048output. Decodev2 counts token
events strictlyAFTER latestfirsttoken until lastgenerationevent. Prefill32768/
latestfirsttoken includes admission/interleaveddecode. Cohort keepsall8 intail.
Public streaming_phases.py/report_bolt_phases.py:9CPUtests/23recordedtrialsexact.
No GPUmicro/profile/benchmark overlap. benchmark.lock guards fullruns.
Profileonewarmup+boundedrank0capture32steps/chunk, activegate removedfinally;
profiletimes excludedfrommetric. monitor.py samples sysfs2s/noGPUcontext.

Isolatedbolt-opt-loop.service port8021,Usercurved,MemoryMax56GiB/Swap8GiB;
7-8minfreshboot. Runtime lifecycle dropinB05; activegate must beabsent before
bench. HotreloadcanOOM byduplicating~0.5GBscales. Stop143/failed is normal;
DO NOTreset-failed (GCs transientunit). If unit disappears,recreate identical
properties/envfile privately. Never print APIkey/fullenv; runner reads private
liveAPIchild environment. GPUwork AScurved, root.venv/bin/python only;
ROCM_PATH/LD_LIBRARY_PATH=/opt/rocm[/lib],gfx908,GPU_ARCHS,gfx908,VLLM_TARGET_DEVICE
rocm,PYTHONPATHroot:siblingaiter. No pip/venv/runtime changes. Servingtorch2.11
runtime deliberately patchedHIP/HSA7.2; never reinstall unreviewed.

## Preservation and publication

All Git mutations/push AScurved, bothmain. Rootremoteorigin/AITERremotecurved.
Stage only experiment-owned code andledger/thisnote; preserve unrelated root
AGENTS/services/Qwen/startupwatch work. Verify loop/protected-worktree-hashes.json
(10paths bytes+mtime), plus protecteddeletions; no broadrestore/stage/stash.
Before EVERY commit, afterallfinaledits/staging, run security-hygiene-check under
agents/scratch/security-policy.md; helper loop/security-gate.py. Fullstagedblobs/
diff/metadata; no disclosureexceptions. ExistingintentionalpublicGitattribution
accepted. Offloadflagpath changedtocheckout-relative byexplicituserchoice;
nohistoryrewrite/disclosuregrant. Re-read finalcandidate beforecommit.

Known pitfalls: importedautousefixtures neednoqaF401 (Ruffremovedone, caught
beforeperf/restored). Formattermayrewrite unrelatedoldcode; preserveoldblocks.
PartialTP128windows maycross256codecblocks; no commonDhoist acrossboundaries.
Keep FP32group order anddouble-rounding RTE(F32_RTE(v/scale)). Compiledunsorted
TopK mayreorderexperts: compareexpertsets/tightweights. Initialize serving's
producerstream before GPUstreamtests; current_stream createsit onfirstcall.
llvm-objdump --offloading extractssidefiles; useignoredcopies. NativeMT>1 must
match16*MTplan/COUNT. Actualexpertcounts240-496,never512. Maintain-scratch checked/
notdueOct15. Existing skill criteria alreadycover usercorrections; no duplicate
refinement added.
