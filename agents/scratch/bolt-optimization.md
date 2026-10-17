# Bolt continuous optimization

Continue indefinitely until the user manually stops. Each batch profiles frozen
4096 input /256 output /C8, registers FIVE experiments across scopes including
architectural fusion, then executes sequentially. EXACTLY FIVE measured runs
per final candidate define the verdict. No extensions, additional controls, or
new baselines. Reuse the latest accepted original five. One untimed full warmup
per boot is established; component tests and bounded profiles are separate.
Keep positive end-to-end gain AND median+pooled aggregate prefill/decode no
worse; otherwise restore only owned changes and move on. Do not relax the gate
after seeing results. Commit/push winners, then renew the batch automatically.
No subagents. Report aggregate prefill AND separately timed aggregate decode
regularly; never per-request or whole-run output rate as decode. Keep updates
within60seconds (shorten reasoning/use time tools rather than long silent work).

Frozen manifest:263da19de5a5bc8823fe38d7113553c08f00fa453127dbbd523c74622d0bcda1,
eight technical prompts4096 each/256 actual output token IDs,seed0/temp0/ignoreEOS.
ResetGPU+CPU prefixes everyrun; assert8success/32768input/2048output/256IDs each.
Decodev2 counts shared-clock token events strictlyAFTER latestfirsttoken until
lastgenerationevent; prefill32768/latestfirsttoken includes admission/interleaved
decode. Accepted cohort keepsall8 in decode tail. Public calculator
vllm/benchmarks/streaming_phases.py andscripts/report_bolt_phases.py:9CPUtests,
23recordedv2trialsexact. No GPU micro/profile/benchmark overlap.

Accepted incumbent NOW: B05E2 routeweight/reduce/outputcastfusion on top of
B02E5shortcohort+B03E1route64. AITERba264db0b19c6b058c981533bf424332659ed5fe committed;
rootintegration/ledger/notes arecurrentstagedcandidate(nextrootcommit). Original
source/harnesshashesB05exp2-sources.json;sourceparentroot2bbe3aa625/AITER3f8045a90.
Currentreference B05exp2-{0..4}.json ORIGINALFIVE, medians2081.5140315 aggregate
prefill/162.7157729 aggregatedecode, pooleddecode163.7370051,pooledwhole74.6502256.
Use exp2 asreferenceforE3-E5; B05incumbent-* preservesolderB03E1fiveforcumulative
comparisons. DO NOTrestoreold3f8045a90whileremovinglatercandidates; keepnewcombine
API/rootexport+call paired. Route64M>64/routes>=actualExperts*64;matchingUP/DOWN
plan. Native/sparse16. Cohortshared4256prefill/8512total, offcadenceeligibleinitial
text/output+pending<=64;long/mature/MM/LoRA/preempt/staleretain16-roundcadence.

Protocolcorrection: agent-added3%overallfloor wasextra toexplicituserfive-run/
nondegradingphasecriterion. RemovedexplicitlyatB05E2adoption, withplannedfloor
miss documentedinledger. Positiveend-to-end ANDmedian+pooledphaseguardsremain.
Noextra runs. Earliernegativepooledphaseverdictsunchanged; do notrelabeltheir
plannedgates. Userinstructionwins overskill-inferredmaterialfloor.

Fixed user configuration: staticdense3GiB ceiling,277projections,
2,387,607,552INT8values/2.224GiB perGPU,originalpacked GPU tensors/codecmetadata
released(1.389GiB freed). Expertcache5GiB/slotsauto/refresh128, originalIQ2packed
experts retained:4565pairs,114-119slots/layer,5,368,184,832payload+175,944metadata
bytes (348,344unused). GPUKV14e9/rank,CPU24GiB/ranklocal6GiBpin,context262144,
TP4/C8/NS7,INT8GS128compute/KV,BF16residual,FP32Mamba,customAR/largefallback.
All4GPUcaps105W. Thermalcontroller stoppedANDdisabled; installeddependencydropin
renamed thermal-controller.conf.disabled-static-cache-test. nginx muststaydown,
productionvllm-bolt stopped. Neverrunproduction alongsideisolatedbolt-opt-loop.
Fullfidelity/256kvalidation deferredbyuser; do notclaimpassed orresumeoldconfig
testing. Existing1500/400legacyprobe useddifferentC1cold8192/1prefill vsC8random
32/1000decode/acceptance; incomparable, neverrerunlegacybaseline.

B05E1 originalfiveCOMPLETE/inconclusive/reverted: median2253.4939aggregateprefill/
162.8682aggregatedecode;pooledwhole+6.663%,medianprefill+12.700%/decode+0.827%,
butpooleddecode-0.400% ->predeclaredbothmedianANDpooledphasegate unmet. Notcausal
proof ofslowdown;no extra runs/gaterelaxation. RestoreONLY3trackedAITerfiles/
removeONLY2backed-upnewtests;AITERclean3f8045a90. Allcode/proofs/rawfive retained
B05exp1-saved/exp1-sources.json/exp1-*. CPU149323776signedcases/GPU130023424values
zero;100component/32realchecksPASS;48completeMLPtimings+53.963%/53.602% exact.
KV/battery4/4/Ruff/fivegatesPASS;peakVRAM31.222GiB. No E1code retained. Native
andallsettingsunchanged. B05firstverdictledger/notes pendingsecurity/commit/push.

B05E2 originalfiveCOMPLETE/ACCEPTED: median2081.5140prefill/162.7158decode,
+4.099%/+0.733%;pooledprefill+3.840%/decode+1.105%,whole+2.672%. All5gatesPASS,
source/harnessfrozen;peakVRAM30.718GiB. 97initial+4geometrychecks,34finaloperator,
64realallTP casesPASS;KV/battery4/4/Ruff. B256/W4 M<=64,B512/W4prefill selected
beforefive. FullFP32ATentreeandBF16roundexact. AITERcommitba264db0b19c
created/pushpending88374; rootcode2files+ledger/notes needfinalsecurity/commit/
push NOW. ServerrunningacceptedE2,monitor1406,bench70350exit0. StopunitbeforeGPU
E3work. Gate/outputB05. Sourceexp2-saved/exp2-sources.json. No E1code retained.

NEXT B05E3 multirowprefillactivationquant. Designinexperiments-plan.json;
implementnewmultirowkernel onlyM>64, preserveFP32max/div_rn/RTE androwmajorQ/scales,
native/decodeoriginal. BoundedBM4/8/16/32 choicebeforefive; validateallinputdtypes/
Ktails/boundaries/graphs/dense+expertconsumers/realweights. CompareEXACTLYfive
againstB05exp2 ORIGINALFIVE;no newbaseline/extensions. E4/E5plannedonly.

B05 profile COMPLETE:299185kernels/alllaunchcorrelated,rank0prefill/mixed15.6339
kernel-seconds;dense4.4773s/28.64%,packed3.7889s/24.23%,collectives3.2782s/20.97%
(includeswaiting),attention1.5148s/9.69%,quant0.5707s,copy0.5029s. Threechunks
rank0pid1569371 underB05host/gpu, CSV/analysis/opportunities adjacent. Profiletime
excludedfrommetric;no newbaseline. B05experiments-plan.json registeredFIVE before
implementation:1currentcache/codeccomposition;2exactMoEweightedreduce/BF16fusion;
3multirowprefillactivationquant;4staticdenseFP32scaleN-contiguouslayout;
5partialMambaoutputround/weight/quantproducerfusion. E2-E5 plannedonly.

Laterexperiment caveats: E2mustrecoverATen9-wayFP32reductionorder beforeclaiming
exactTLsum; preserveBF16round beforeexistingdownstreamquant. E3currentquantone
128element/4waveCTAperrow/group; multirowprefillrewrite retainsrowmajorQ/scales
andnative decode. E4onlystaticdenseFP32groups logical[1,N,ng] stridedphysical
N-contiguous; freeoldlayout, preservetotalbytes. Expertrawgroups/activations
muststayrowmajor. E5BoltusesMambaMixer2,groups8/TP4,localI2048/group1024,
eps1e-5,use_biasFalse. forward_cuda delegatesnativewhengroups!=1. Preserve
FP32gate/pow/mean/norm reductions exactly, thenfusefinalFP32->BF16round,
BF16weightmultiply/secondBF16round andGS128quant. PreserveRowParallelbias/rank0/
AR/SPsemantics. No fullnormreductionrewritewithoutATentreeparity.

History compacted: ledgerdocs/recipes/surface_experiments_ledger.jsonl plus
ignoredB01-B04artifacts retainalloriginalfive/source/verdicts. B03E1onlyretained
B03change (+25.708%whole,medianprefill+40.954%/decode+3.126% vsB02E5). B03E2
8wayheader micro+35.8%,whole+2.642%/decode-1.271% ->reverted. B03E3nativeUProwleader
fusionwhole-1.415%;E4full/groupMLPfusion-12.354%;E5temporaryINT8staging-4.291% all
reverted. B04allFIVEreverted:two-pagecohort,route128,singlelaunchcacheprefill,
denseepiloguefusion,ROCmsharedoverlap. E3whole+4.116%/medianprefill+8.444% but
pooleddecode-0.502%;E4whole+3.015% butpooleddecode-0.181% ->strictgate unmet, no
extra runs. Those small negativesarenotcausalproof. E5whole+.393%/decode-.392%,
peakVRAM31.845GiB (~159MiBheadroom);incumbent~31.22GiB. E1compositionnewalgebra
andcachemissintegrationaddressespriorindividuallysmallgains. B02E1UNREQUESTED
extra5 exp1-repeat-* areEXCLUDED; usercorrectedcount twice, fiveiscompletebudget.
IntegerIQ2priorproof37,329,768positivecases/all31743positivehalf isdistinctfrom
newalpha149Mproof. B01cacheprefill0d8WITHDRAWNin15a3; do notreintroduceit.

Infrastructure ignoredagents/tmp/bolt-opt/loop: frozenrun.py/compare.py/
phase_metrics.py/manifest-4k.json. bench NAME --batchbatchNN --repeat5; compare
BASE CAND --batchbatchNN. Profileonewarmup+boundedrank0capture32steps/chunk;
removeactivegatefinally. benchmark.lock preventsGPUoverlap. Monitor2s sysfs/no
GPUcontext. HotreloadcanOOMduplicating~0.5GBscales; freshboot~7-8min. Isolated
bolt-opt-loop.service port8021,Usercurved,MemoryMax56GiB/Swap8GiB. Lifecycle
runtime dropinB05. Stopreturns143/failednormally; DO NOTreset-failed (wouldGC
transientunit). IfGC occurs, regenerateidenticalunitproperties/envfile privately;
neverprintcredentials. NativeHIP loader hashes sources/compiler/flags.

Gitmutations/push AS curved, bothmain; rootremoteorigin,AITERremotecurved.
BeforeEVERYcommit scanfinalstagedblobs/diff/metadata withsecurity-hygiene-check
underagents/scratch/security-policy.md; helperloop/security-gate.py. No disclosure
exceptions. EstablishedintentionalpublicGitattribution accepted. Offloadflagpath
changedtocheckout-relativebyexplicituserchoice; nohistoryrewrite/disclosuregrant.
Protectunrelatedroot edits/services/startupwatch/AGENTS/Qwenfiles, neverbroad
restore/stage/stash. Verifyloop/protected-worktree-hashes.json(bytes+mtime).
AllPython root.venv/bin/python, no pip/venv/runtimechanges. GPUwork ascurved,
PYTHONPATHroot:siblingaiter,105Wfixed. ImportedautousefixturesneedexplicitnoqaF401;
lintremovedone inB03E4,caughtbeforeperf/restored. llvm-objdump--offloading extracts
sidefiles; useignoredcopy. Actualexpertcounts240-496,never512. PartialTPgroupscan
span256codecblocks; no commonDhoist there. PreserveFP32grouporder/double-rounding
RTE(F32_RTE(v/scale)). CompiledunsortedtopKcanreorderexperts; compareexpertsets/
tightweights ratherthanexacteagerorder. Maintain-scratchchecked/notdueOct15.
Self-improvement enabled; currentusercorrections alreadycoveredbyexistingrules.

LaterMamba research lead ONLY (notimplemented): fastest-dimFP32mean1024 uses
Reduce.cuh inputvectorization4,64lanes/row,4sequential256-valuechunks withfour
accumulators thenwarpDown32/16/8/4/2/1; verifyinput_vectorized_thread_reduce_impl
beforeassumingorder. A nativeHIPfullgate/norm/weight/quantfusioncouldmatchthat
exacttree andTorchOCMLexp/div/rsqrt semantics, retainbothBF16rounds. Existing
AITERctypesHIPloader avoidsfullvLLMbuild. Mustproveallmath bitwise; originalB05E5
partialepilogueplan staysuntil evidence supportsnewfullfusion. Group128quant
max associative exact, butSiLU exp/div andvariance tree requirecare. No future
sourcechangeswhileE2runs. Currentprotectedrootbytes/mtimes verified05:31.
