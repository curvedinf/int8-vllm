# Bolt continuous optimization

User contract: continue indefinitely until manually stopped. Each batch profiles
4096 input /256 output /C8, then designs FIVE experiments across scopes including
architectural fusion. Implement sequentially. EXACTLY FIVE measured candidate
runs define each verdict; no extensions or extra controls for uncertainty. Reuse
the accepted originalfive as reference. Keep demonstrated end-to-end wins with
no prefill/decode regression, commit/push, then begin the next batch. No subagents.
Predeclared gate: >=3% pooled whole-run gain; median AND pooled aggregate
prefill/decode no worse. Inconclusive -> restore owned changes and move on.

Report AGGREGATE prefill and decode regularly, with separate phase clocks.
Frozen manifestSHA256263da19de5a5bc8823fe38d7113553c08f00fa453127dbbd523c74622d0bcda1:
eight technical prompts,4096tokens each,256outputs,seed0,temp0,ignoreEOS,stream
actualtokenIDs. One untimed fullwarmup perboot, prefixes GPU+CPU reset eachrun;
assert8success/32768input/2048output/256returnedIDs perrequest. No overlapping
GPU profile/micro/benchmark. Profile timings excluded. Decode v2 counts only
shared-clock token events AFTER latest firsttoken /remaining generation time;
prefill32768/latestfirsttoken includes admission/interleaveddecode. Accepted
cohort keeps all8requests in decode tail; oldreference had5-6. Whole-run output
rate is separate. Never report per-request or whole-run rates as aggregate decode.
Public calculator: vllm/benchmarks/streaming_phases.py, scripts/report_bolt_phases.py
(require PYTHONPATH="$PWD" to avoid installed-package shadowing). Nine clock/
burst/malformedtimeline checks passed;23oldv2raw trials matched exactly.

Accepted incumbent: root performance551af7862b plus AITER3f8045a90048c1f99d4a465f4fe230e13327f81b.
Rootlatercommits are ledger/notes only.
B02E5 initialshortcohort plus B03E1 64-row expertprefill reuse are accepted.
Reference for remaining B03 candidates: batch03/exp1-{0..4}.json, medians
1999.5434301 aggregateprefill /161.5315694 aggregatedecode tok/s; pooleddecode
161.9482114, pooledwhole72.7072659. Exactlyfive are the metric. B03E1 source
exp1-saved, hashesexp1-sources.json, compareexp1-vs-incumbent.json. Accepted
cohort root8084504576: shared4256prefill/8512total budgets; initialsingle-page
text cohort mayadmit offcadence whileallrunning/head eligible and output+pending
<=64. Long/mature/MM/LoRA/preempt/stale retainordinary16-round cadence. E1 uses
64-row matching UP/DOWN plans onlyM>64 androutes>=actualExperts*64; other/native
batches16. Do not lose this paired AITER API/root routing when reverting tests.

Fixed accepted memory/precision: dense3GiB ceiling,277projections,
2,387,607,552INT8values/2.224GiB perGPU; original packed GPU tensors/codecmetadata
released (1.389GiB freed). Expert5GiB ceiling/slotsauto/refresh128, packedoriginals
retained;4565pairs,114-119slots/layer,5,368,184,832payload+175,944metadata bytes.
GPUKV14e9/rank, CPU24GiB/rank-local6GiBpin,context262144,TP4/C8/NS7,
INT8GS128compute/KV,BF16residual,FP32Mamba,customAR/largefallback. All4caps105W;
thermalcontroller stopped ANDdisabled. Installed thermaldependency renamed
thermal-controller.conf.disabled-static-cache-test; do not reintroduce. nginx
muststaydown (runnerchecks everyrun); production vllm-bolt stopped. Never start
production alongside isolated bolt-opt-loop. Fullfidelity/256kvalidation deferred
by user; do not claim it passed. Do not resume old acceptedmemory testing.

CURRENT: B03 allfive experiments complete. E1route64 is the only retained
B03performance change; E2-E5 reverted. E5 originalfive median1889.1058prefill /
160.4105decode, pooledwhole-4.291%, pref-5.523% ->reject. 18prototype/64integrated/
32realchecks pass; synthetic sparsegain didnottransfer. Exactcode/rawfive/hashes
exp5-saved/exp5-sources.json andotherexp5-* retained. RestoredONLY3rootfiles and
removedONLY2backed-up AITERnewfiles; AITERclean3f8045a90. No E3/E4 restored.

Batch04freshprofile next, NOTa newbaseline. Runtime lifecycle dropin moved to
batch04 anddaemon-reloaded; directoryalreadycurved-owned. DiagnosticunitSTOPPED,
incumbentKV micro PASS,battery session6754 active, thenbootacceptedE1. Run
run.py profile incumbent --batchbatch04, monitorincumbent; record sessions when
launched. Profile includesoneuntimedwarmup andboundedrank0trace, not fiveperf
runs; keepacceptedB03E1 originalfive asmetric/reference. Analyze freshprofile,
identify currentcomputeopportunities anddesign FIVE newscopes witharchitectural
work beforeimplementation; thenexecuteeachsequentially/exactlyfive. No stop at
batchboundary. Currentmetrics1999.5434aggregateprefill/161.5316aggregatedecode.
E5ledger row/notes readyforsecuritygate/commit/push; pendingrootchanges are
notes/ledger only beyondprotected unrelatedwork.

Campaign artifacts under agents/tmp/bolt-opt/loop/ (ignored): frozenrun.py,
compare.py,phase_metrics.py,manifest-4k.json. benchmark.lock prevents overlap;
run.py bench NAME --batchbatchNN --repeat5. Profile usesRPC andbounded32steps/
rank0chunks; alwaysremove activegate. Keepraws/failures. Hotreload RPC canOOM
byduplicating~0.5GiB scales; preferfreshboot. Startup~7-8min. Monitor records
power/temp/freq/VRAM every2sec, noGPUcontext. E1peakobserved31.223GiB (~795MiB
nominalheadroom). NativeHIP loader hashes source+compiler+flags, JITsafe.
llvm-objdump --offloading EXTRACTSsidefiles: operateonlyondisposablecopy.

B03 profile:305783launch-correlated kernels,22.2209rank0prefill/mixedkernel-seconds,
packedrouted10.2139s/45.965%. Captured BEFORE E1; rawbatch02/host/gpu/
batch03-incumbent-* (oldruntimegate), CSV/analysis/opportunities underbatch03.
E1accepted +25.708% pooledwhole, medianprefill+40.954%/decode+3.126% versusB02E5.
B03E2 header8way broadcast exact/allfinitehalfproof, micro+35.8%; fiveprefill
+6.216% butdecode-1.271%, whole+2.642% below3 ->reverted. B03E3 rowleader native
UP/ReLU2/quant fusion50+32exactchecks; fivewhole-1.415% ->reverted. B03E4 fullMLP
fusionprefill spills; groupwise feedback no spills butextraYtraffic, fivewhole
-12.354%/bothphasesregress ->reverted. Alloriginalfive/source/codegen andfailure
logs retained exp{2,3,4}-*. Avoidrepeatwithoutaddressingmeasuredcost.

Older rejected directions: consult ledger andbatch01/02 before repeating.
Cache-aware prefill0d8 wasWITHDRAWN in15a3 under correctedaggregate decode gate;
never restoretemporary0d8backup. B02E1 originalfive only; unrequested extra
candidatefive exp1-repeat-* areEXCLUDED. Userexplicitlycorrected twice: five is
completebudget, notminimum. B02E2 nativeinvariants, E3 common-base integerIQ2,
E4 nativeUPquantfusion allbelowmaterial/phasegate; E5cohort won. IntegerIQ2proof
uses8/25/43magnitudes and48terms;37,329,768ratios/all31743positivefinitehalf bases
exact, explicitv_rcp_f32 needed (ordinaryTLdivision stillprecise). Preserved
next-integer-quant-* artifacts; no such kernelretained. FullMLP/groupwise native
prototypes andtests underB03 remaininactive afterE4revert. Architectural leads
needfreshprofile: shareidenticalinputactivationquant explicitly (neverpersistent
pointercache acrossgraphreplay); reevaluatedensefusion onlyifnewprofilejustifies.

Operational pitfalls: compiledunsortedtopK canreorderexpertseveninbaseline;
compareexpertsets/tightweights, notexacteagerorder. Actualexpertcounts240-496,
never512. MT>1 requires matching16*MT alignment/COUNT guards. PartialTPwindows
can span256codecblocks; do not hoist onebase multiplier acrosscrossingwindows.
PreserveorderedFP32 group sums anddouble-rounding RTE(F32_RTE(v/scale)). CPU24GiB
full-region registration exceedsAMDhostmemoryaccounting; rank-localpin required.
Legacy1500/400probe usedC1cold8192/1prefill vsC8random32/1000decode, different
acceptance; incomparable, neverrerunlegacybaseline. Fixtureimports needexplicit
noqa F401; lintremovedone inE4, caughtbeforeperf, restored/assertcachepresent.

Gitmutations/push ascurved (avoidroot-ownedobjects), bothmain; rootremoteorigin,
AITERremotecurved. BeforeEVERYcommit scanfinalstagedblobs/diff/metadata with
security-hygiene-check underagents/scratch/security-policy.md. Helperloop/
security-gate.py readscompletecandidates; noapproved disclosureexceptions.
Offloadflagpath was changedtocheckout-relative byexplicituserchoice, notpermission
todiscloseaccountpaths; historyunchanged. Preserveunrelatedroot edits/service
migration/startupwatch/AGENTS/Qwenfiles. Verifyloop/protected-worktree-hashes.json
(bytes+mtime), neverbroadrestore/stash/stage. AllPython .venv/bin/python, no pip
orvenv changes. GPUwork ascurved, PYTHONPATH root:siblingaiter;105W fixed.
