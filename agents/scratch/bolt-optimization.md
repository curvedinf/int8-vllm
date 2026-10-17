# Bolt continuous optimization

Continue until the user manually stops. Profile4096input/256output/C8, register
FIVE experiments across scopes including architecture/fusion, execute sequentially
and renew batches automatically. EXACTLY FIVE measured E2E runs per final valid
candidate are the entire metric budget: no extensions,extra controls,new baseline,
3%floor or additional runs unchanged candidates. One untimed warmup per boot.
Bounded profiles/component correctness are separate. Keep positive pooled whole-run
gain AND BOTH median/pooled aggregate prefill/decode nondegrading. Report ONLY
aggregate prefill and separately timed aggregate decode; never whole-output-asdecode.
No subagents. Check clock/commentary within60seconds; waits below30seconds.

## Resume now

As of2026-10-17 15:19:18UTC B09E3 COMPLETE/REVERTED afterEXACT5. Unitstopped,
allGPUjobsclosed. Bench9003/monitor68598/stop99838exit0;allsource/proof/harness/
token/counter/105W/ALL4qualified/protectedbytes+mtimePASS,coherentpreviews/fullsamples.
Median2319.0868prefill/166.1440decode,pooledprefill+0.3569%/decode-3.6841%/whole
-1.2147%. Component11-13%gain didnottranslateE2E. Peak31.706543GiB(~1.12abovebase),
allocationcause notisolated; no cache/memoryconfigchange. RestoredONLYlinear.py,
removed5hash-verifiednewfiles;AITER3c74clean. Originalreference retained,no extra runs.
Rawfive/30frozen sources/19proofs under B09exp3-* /exp3-saved,canonicalledger.
E3exactwireproof96actualallTP cases/64graphs/64wire-specialchecksPASS;customAR and
largePyNccl bothchecked,VllmConfig initcontext expired. InitialTP4fixture omitted
production ps.graph_capture lifecycle: illegal-address replay. CorrectedONLYfixture
context/stream, samecasespass; oldfailedscript/log retained before-capture-context.
54focused+20finalrootchecks(notdistinctcounts)PASS,KV/battery4/4. No E3code retained.

ACTIVE B09E4 NEXT,notimplemented. B09exp4-design.md researchdesign. Newseparate
BF16input nativeMoEnorm producer storesF32containingEXACToriginalTRITONBF16norm
value, reuses SAME roundedvalue forq/scales; oldnativeRMS API/kernels untouched.
GateLinear fallback force_fp32_compute storesF32weight; allspecializedflags false
onMI100. Guard exactGateLinear/F32weight/out_dtype/fallbackflags andoldaccepted
MoEtopology; F32input/unsupportedfeatures useoldpath. Do NOTread VllmConfig inforward.
Samegate/logits/router/routedscalefc2/collectives/BF16latent-sharedcasts/residual.
OriginalBF16hidden can be metadata ONLY to unchangedexplicitq/scales opaqueop;
poisonedhiddenvalues mustprove no valueconsumption. Separate guarded prefillentry
canreuseoldopaque schema ifproven, oldnativeproducer/runnerAPI remainsreference.
ImportantNaNdistinction: E4replacesTRITONBF16store,preserveitsBF16payload before
promotion; E3replacedTorchcast andusedTorchNaN0x7fc0. DoNOTblindlyreuseE3rounding.
Reuse actualallTP AISdense fixtures/norm+gate sourceweights andexisting
 tests/model_executor/test_nemotron_h_norm_g128_prefill.py protocolfixture;
oldrouted/collectives areidentity there, explicitlystate limitations. Needreal
GateLinear input/logitbits andsameconditionalrouting/fullopaqueoutputs proof,
sourceimmutability/residualalias/q/scales/graphs/fallback/KV/battery/ALL4path,
thenONEwarmup andEXACT5, ORIGINALB07exp4reference. NoE4GPUwork/code yet.

Accepted runtime root4711913204533dfe702bd7e9ed7d23d07a08503b,
AITER3c74adc198e55aeb5bcd41516bddfb3ccadf6725, BOTHmain/pushed. RootmetadataHEAD
f43d2173c5a85bf5b813ab3f761104af62594e69 pushedorigin/remoteverified15:00:41.
Keep B07E1 dedicated4256-pageattention(roota079) AND B07E4 nativeMoEnorm+GS128
fusion(root47119/AITER3c74), weighted9combine/B03route64/B02shortcohort beneath.

Reference ORIGINALB07exp4-{0..4}: median2300.4249954prefill/168.2838861decode,
pooleddecode170.7456560,pooledwhole80.5757898. B09incumbent exactbytecopies. No rerun.
Canonical decisions/provenance: docs/recipes/surface_experiments_ledger.jsonl.
Rawfive/sources/proofs: agents/tmp/bolt-opt/loop/batch01..batch09 (ignored).
B08all5 rejected. B09E1 traversalG8 rejected;E2nativeMT8 rejected afterEXACT5:
1415.9199/159.4661,pooledprefill-38.637%/decode-7.835%/whole-28.026%. E2restored/
AITERcleanbeforeE3. No accepted B08/B09runtime yet. Do not replay unchanged variants.

## Remaining registered B09 experiments

B09freshprofile/ALLFIVEbeforecode: experiments-plan.json,canonicalledger(root962e).
Rank0kernelSUM13.866725s(NOTTP4wall),281348ALLlaunch-correlated kernels,3chunks:
dense32.341%,packed27.285%,collectives20.494%(waitincluded),copies3.530%,
quant2.194%,normquant1.515%. ProfileEXCLUDEDmetric;originalfive reused.
E4NEXT afterE3: gate-ready F32 nativeMoEnorm output eliminatingBF16normal+upcast;
B09exp4-design.md. Preserve originalTRITONBF16 normal bits/payload beforeF32store
(notE3TorchNaNcanonicalization). SamegateF32GEMM/router; originalBF16 hidden used
ONLYshape/dtypemetadata for explicitq/scales opaqueop; prove withpoisonedmetadata.
Original latent/sharedBF16casts/scalefc2/AR/residual preserved; F32inputfallback.
E5: draftreplicatedFC4096x24576 output-rowsharding1024/rank, originalfullresident
weights retained, sameGEMM/dtypecast +bitwisecolumnallgather. LiLiCorrForCausalLM
inherits DFlashQwen3ForCausalLM.combine_hidden_states(qwen3_dflash.py),fc created
ReplicatedLinear(return_biasFalse),castdraftdtype beforefc. Need ALLrankactualinput/
weightidentity andTP4gatherbytegate first; followinghidden_norm/metadata/KV unchanged.
NoE4/E5implementation yet. Futurebatchresearch future-launch-bound-opportunity.md:
inspectoriginalnativepackedISA for default1024bounds vsactual512/NW*64; doNOTchange
originalkernels oraddB09variants. E2matchingactual512removed328B/threadscratch but
newdensearchitecturestilllost. Reprofile/registerlateronlyifheadroomproven.

## Fixed serving contract

All4MI100 caps105W; mi100-thermal-controller.service stoppedANDdisabled,dependency
dropin.disabled-static-cache-test. nginxdown; productionvllm-bolt/vllm units stopped.
Neveralongsideisolatedunit. Staticdense3GiBceiling:277projections/2387607552INT8values,
2.224GiB/rank;originalpacked/meta released1.389GiB. Dynamicexperts5GiB/slotsauto/
refresh128,ORIGINALIQ2retained:4565pairs/114-119slots/layer,5368184832payload+
175944metadata/348344unused bytes. GPUKV14e9/rank,CPU24GiBtotal/pin6GiBrank,
ctx262144,TP4/C8/NS7,INT8GS128compute+KV,BF16residual,FP32Mamba,customAR/large
PyNcclfallback. No workspace cache/KV shrinking. Peak~30.59GiB; oldoverlap31.845GiB.
Fullfidelity/256k userDEFERRED; do notclaimpassed orrestartmemory-configtesting.
Old1500/400 reports useddifferentworkloads(C1cold8192/1 vsC8random32/1000).

CRUCIAL _tp_ar_fp32_mode returns STRING"0" whenunset. BothGroupCoordinator and
functionalall_reduce useifmode,so"0"isTRUTHY: BF16ALREADYpromotesF32wire andcasts
back. Neverfixflag/change wireprecision/order withoutnewcontract. FUSED_NORM_QUANT
unset->False;legacyperrow/trunc producerincompatibleGS128. Namesarenotdtypes.

## Actual fixtures and retained pitfalls

B09dense_case_plan.py/dense_rank_support.py/exp1-rank-cache-proof.json reuse68
originalcheckpoint/AISbytegates,exact9projectiontypes/all4TP,includingextra probe
spans. AISmeta.shapeGLOBAL butruns/compactpayloadLOCAL; gatherlogicalbytewindows,
nevernaivequarterorwholecompactreshape. Mambafivegroups8192/8192/1024/1024/128 ->
local2048/2048/256/256/32=4640. AttentionQ32/KV2/head128 ->localQ1024+K128+V128=1280,
KVreplicatedpairs0/1&2/3. RowK1344 partial256codecwindows maycrossblocks.
NVFP4global_scale2 scalar expandedperrow; mergedQKV cat originals. Draftsingle
model.safetensors(noindex),fcsourceU8[4096,12288],scalesF8[4096,1536].
prepare(rank,key) usescurrentCUDAdevice; set_device/current_stream BEFORE tensors.
Fixturesisolateddensecache1GiB ONLY; production3unchanged. Independent oldhelper
B09dense_reference.py AST-identicaloriginaldensecache. E2source/proofs archived
exp2-saved and *before-launch-bound.*; no oldnative/genericbinary replacement.

## Harness, operation, publication

loop/run.pySHAd05ada8f44803882596a4cd1eb6db12fe459ea5cbe826bd220e95d99f7182802;
compareSHA678abddfd7c5f0ef2a3a14ecfa1383b1baf94f71abfbd53d574f4e99c2527e10;
phaseSHA452486851c8d596d89747a818cec5ad46d8191ed023db59861cd78563a299c58;
manifest263da19de5a5bc8823fe38d7113553c08f00fa453127dbbd523c74622d0bcda1.
Same8technical4096prompts/256actualIDs,temp0/seed0/ignoreEOS;resetGPU+CPUprefixes
EACHrun. Assert8success/32768input/2048output. Decodev2 ONLYeventsAFTERlatestfirst
until lastgenerationevent(all8tailcohort);prefill32768/latestfirst. compareEXACT5,
bootstrapdescriptiveonly. verify-frozen-result.py checksallfrozen/protected/cap/
counter/token/qualification withoutGPUwork; readoutputs separatelyforcoherence.
run.py optional --arm-sparse-qualification defaultFalse; oldB08E5runtimearmhooks
ALLreverted. Forfuturedata-dependentbranches requireactualwarmupepoch qualification,
notinit/dummycapture. Staticbranches explicitrankmarkers sufficient. Neverweaken
rankcorrelation: explicitWorker_TP orsamePIDmappedtoexactlyONEcurrentworker.

Isolatedunitbolt-opt-loop.service port8021/Usercheckoutowner,MemoryMax56GiB/Swap8GiB.
Freshboot6-7min, nohotreload(duplicates~.5GiBscales/OOM/biaseswarmcachestate).
DoNOTreset-failed(GCs transientunit). startnextwhenready; stopafter5 andawaitallGPU
jobs/monitorbefore next. monitor.py sysfs2s/nocontext;benchmark.lock preventsload
overlap. B09dropinhost/active,32profile-steps/chunk;gateMUSTbeabsentbeforebench.
Converter scripts/kineto_to_agent_csv.py --output CSV_DIR GPU_JSON_GZ...
Analyzer scripts/analyze_agent_profile.py --gpu-csv CSV_DIRECTORY --host HOST_DIR
--output JSON. CSVargDIRECTORY,rawargs.gridLIST. Profilingboundedrank0only/excluded.

AllPythonrepo.venv/bin/python; GPUworkascheckoutowner/HOMEowner,ROCm/gfx908env,
PYTHONPATHrepo:siblingAITER. No installs/runtimechanges: torch2.11rocm7.1wheel has
patchedHIP/HSA7.2. Nativecontentaddressmusthashsource+compiler+flags+localdeps.
AllGitmutations/pushascheckoutowner,rootremoteorigin/AITERremotecurved,bothmain.
StageONLYownedcode+ledger+thisnote. Protectedwork loop/protected-worktree-hashes.json
10pathsbytes+mtime and3deletions;never broadrestore/stage/stash. BeforeEVERYcommit
finalstaging then security-gate.py --repo ABS --label EXACT_MESSAGE, fullstaged
blobs/diff/metadata. agents/scratch/security-policy.md grantsnodisclosureexceptions;
accountpathsredacted/offloadflags checkout-relative byexplicituserchoice. Public
sourcecredits needverifiedattributionreview whenstagingattentioncopyrightfiles.
PublishAITERfirst/rootnext,verifyremote refs. Skillrefinementassessedpercompleted
experiment; avoidaddingone-offhardwaredetailrules. Maintain-scratchchecked/notdue
Oct15; routineupdates/compactiondo not resetortriggerstartup timer.
