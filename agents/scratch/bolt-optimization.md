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

Acceptedincumbent B07E4 nativeMoEnorm+GS128fusion plusB07E1 attention.
Rootmain fusion taskcommit(currentpublishedwork;parent6ff8cacea4);AITERmain
3c74adc198e55aeb5bcd41516bddfb3ccadf6725 companion. B07exp4-{0..4} ORIGINALFIVE
isE5reference: median2300.4249954aggregateprefill/168.2838861aggregate decode;
pooleddecode170.7456560,pooledwhole80.5757898. Do notoverwriteexp1originalfive
(previousreference)orincumbentoldB05copies;no newbaseline. Underlyingweighted
combine/rootd70e9b6,B03route64,B02shortcohort andB07E1attention retained.

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
bench67897/monitor88109 exited0. Ledger989cf1caeb pushed/remoteverified.

B06E2 COMPLETE/inconclusive/reverted afterexactlyfive: median2088.2307235prefill/
161.6718427decode,pooleddecode162.2723956(-.894%),medianprefill+.323%/decode
-.642%,whole+.299%.35component/78actualallTPchecks,KV/battery4/4/Ruff pass;
FP32/BF16/INT8/scalesexact.237eligible/40fallbackstatics preservedonecopy/bytes;
peak30.971GiB. Allsource/originalfive B06exp2-saved/exp2-* retained. Restoredonly
2AITERfiles/removedverifiednewtest; AITERcleanba264/acceptedcombine. Serverstopped,
bench75994/monitor39821 exited0. E2ledger1b492bcd63 pushed/remoteverified.

B06E3 COMPLETE/rejected/reverted afterexactlyfive: median2098.7817504prefill/
154.1559830decode,pooleddecode158.5602029(-3.162%),medianprefill+.830%,whole-.652%.
GPU150994944F32bitcomparisons/all65536halfpatterns inclNaN/Inf zero;93components/
64actualallTPcases exact,KV/battery4/4/Ruff PASS. ISA98ubyte ->80ubyte+9ushort;
completeMLP+3.009%4096/+3.593%heldout4160 didnotpassstackguard. Allsource/proofs/
originalfive B06exp3-saved/exp3-* retained. Restoredonlydecoder/removedverifiedtest;
AITERcleanba264/acceptedcombine retained. Serverstopped,bench52456/monitor73668
exited0;allcap/token/source/harness gatesPASS,peak30.7184GiB. Initialcomponent
attempt brieflyoverlappedfinishingbattery,terminated143 thenreran93afterbattery
exit0;no measuredtrial affected. E2ledger1b492bcd63 pushed/remoteverified.

B06E4 COMPLETE/inapplicable/reverted afterEXACTLY5: observedmedian2085.2863416
aggregateprefill/165.1609087decode,pooleddecode165.0625913,whole+.628%. Optimized
pathwasINACTIVE: actualKVpages4256tokens(13332-keytiles),componentassumed64.
TenlivebootcompiledTTIRs show4256constant/BM16/defaultMFMA. No speedclaimor
retentiondespitepositivephases;source/testrestoredtosnapshots/HEAD. Alloriginalfive,
frozensources/proofs B06exp4-saved/exp4-* retained. KV/battery4/4,30checks,
14originalsourcecomparisons/5nativeISAidentical arecomponentproofonly. Corrected
futureattention requiresactual4256geometry/pairstraddles anddeployedpath proof,
not anextraE4retry. Bench11554/monitor88661/stop12580 exit0,peak30.7184GiB.

B06E5 COMPLETE/inconclusive/reverted afterEXACTLY5: median2101.5895159aggregate
prefill/165.2888644decode,pooleddecode162.7684634(-.592%),whole+.570%,prefill
median+.964%/pooled+1.332%. All4ranks qualifiedactive BEFOREfirstmeasuredtrial;
noineligiblepathissue.43distinctchecksPASS(31new+12regression),16actualallTPcases
exactq/F32scales/BF16outputs/nativecallcounts,KV/battery4/4/Ruff. Completeprojection
component+7.431%4096/+7.445%heldout4160 didnotpasspooleddecodeguard. Initial7fixture
checks hadwrongscalerank[N,G]vs[1,N,G];26otherspassed,corrected7+10newprotocolPASS.
No arithmeticfixneeded. Allsource/proofs/originalfive B06exp5-saved/exp5-* retained.
Restoredonly3productionfiles/removedsnapshot-verifiednewtest;acceptedruntime
rootd70e9b6/AITERba264unchanged. Bench85411/monitor10435/stop32759 exit0;
alltoken/cap/source/harness gatesPASS,sampledpeak30.6185GiB. Noextra5. Possible
nativePythonoptionalargoverheadunquantified;do notreplayunchangedE5payloaddesign.

B06allFIVE complete, nonekept. ACTIVEB07 profileCOMPLETE;profile28048,
monitor44496/stop33812/converter5763/analyzer72000 exit0. Serverstopped/GPUidle.
Rank0pid1705452,312403kernels ALLlaunch-correlated,3chunks. Prefill/mixedrank0
kernelsum15.25728s(NOTTP4wall):dense29.561%,packed24.798%,collectives19.505%
(waitincluded),attention9.944%,quant.57984s,norm.12764s,copy.49604s. Profile
excludedfrommetric. B07incumbent-{0..4} exactORIGINALB05E2five;nonewbaseline.
Sources14files frozenincumbent-saved/incumbent-sources.json,rootc462b85448 /
AITERba264 acceptedruntime;E5ledgerc462pushed/remoteverified.

B07experiments-plan.json REGISTEREDFIVE beforecode:1dedicated4256page-outer
attention(BM16/64bounded,Ktile32/MFMA16,originalnativeJITunchanged);2factorIQ2
quantONLY,originalbyte/native/cachehit unchanged;3plainprefillbiasfreeBF16final
GEMMstore,native/defaultF32 unchanged;4nativeAITERnorm+GS128dualoutputfusion
withprefill-onlyexplicitpayload/nativelegacyop preserved;5groupTWOexisting
largefallbackPyNcclcollectives,separatebuffers/samedtypes/counts/noconcat.
All4livepathqualification requiredBEFOREeachfive. CPUresearch/details in
batch07-prep/research.md;actualKVpages4256/133tiles confirmed,notinitial64log.
SelectednumericenvAR_FP32unset->STRING0(TRUTHY,currentBF16messagespromotetoF32),
FUSED_NORM_QUANTunset->False;preserveactual
inputdtypes,do notinferfromcustom<float>name. NoB07runtimecodeyet. B07E1 COMPLETE/KEPT afterEXACTLY5: median2235.6512733prefill/165.5856116decode,
pooleddecode166.8156138(+1.880%),prefillmedian+7.405%/pooled+7.698%,whole+5.125%.
All4qualifiedactiveBEFOREfirsttrial;counter/token/cap/source/harness/coherencegates
PASS,sampledpeak30.7190GiB. Bench43199/monitor98559/stop43751 exit0,serverstopped.
KeeporiginalgenericJITsource andall6native/genericISAidentical.35productiontests,
26prototypebitchecks,14finalsourcebitchecks,KV/battery4/4/newRuffPASS. OnlyBM16/64
choices:actual4256-pageoriginal4096 11.88739ms->64 4.50232;heldout4160 12.26406->
4.64174. Frozen3source/testfiles exp1-saved/exp1-sources.json. Accepted3files+
ledger/note publishedunder a079b82640;finalstagedsecurity PASS withbothpubliccredit
reviews, eight unchanged/copiedpublicauthorcredits distinguishedfromprivatedata.

B07E2 COMPLETE/inconclusive/reverted afterEXACTLY5: median2273.6122105aggregate
prefill/165.3744938decode;pooleddecode166.128282(-.412%),medianprefill+1.698%/
decode-.127%,whole+.602%. ALL4qualifiedbeforefirsttrial;alltoken/cap/source/harness/
coherencegatesPASS,sampledpeak30.719116GiB. CPU149323776finiteproofreused;newGPU
184549376values/ALL65536halfpatterns/22offset-tailcaseszero.88components/84actual
allTPcheckpointcasesPASS;KV/battery4/4/Ruff. Two preregisteredvariants only: guarded
completeMLP+2.121%4096/+2.099%4160selected;unguarded+.059%/-.061%rejectedBEFOREfive.
Both184MproofsPASS. Originalbyte/native/cachefill/cachehit unchanged, dedicated
prefillkernel. Restoredonlyrootmarker/AITERwrapper,removedtwohash-verifiednewfiles;
acceptedE1attention/roota079 andAITERba264 retained. Bench45545/monitor13251/stop31285
exit0;serverstopped. Allsource/proofs/originalfive exp2-saved/exp2-* retained.

B07E3 COMPLETE/inconclusive/reverted afterEXACTLY5: median2314.4960458aggregate
prefill/166.4745794decode;pooleddecode163.8616413(-1.771%),medianprefill+3.527%/
decode+.537%,whole+.916%. ALL4qualifiedbeforefirsttrial;alltoken/cap/source/harness/
coherentpreviewgatesPASS,sampledpeak30.555286GiB.68components/130actualcheckpoint
cases(10surfaces/all4TP)PASS;196608BF16roundboundary/nonfinitebitchecks exact.
Initial63PASS/3FAIL caughtTritonNaN0x7fffvsPyTorch0x7fc0;correctedONLYnewprefill
store,66rechecks+2narrowfallbackPASS. ONEfixedperformancevariant:12pairedcomplete
projectionchecks/24timings +8.344%4096/+8.294%4160;sharedfusedUP/DOWNunchanged.
Originalnativehelper/JITbyteidentical,accum/traversal/signatureASTidentical except
newstore;KV/battery4/4/RuffPASS. Waitingclient24555terminated143 BEFOREANYwarmup/
requestwhilecheckingunprefixedstartupmarkers;all4Packedmarkersalreadyrank-prefixed,
no harness/runtimechange. Replacement85801 ONEwarmup/EXACT5,monitor36691/stop19805
all0. Source/proofs/originalfive exp3-saved/exp3-* preserved. RestoredONLY3rootprod
files/removenewroottest+3AITERnewfilesafterhashcheck;acceptedE1attention retained.

B07E4 COMPLETE/KEPT afterEXACTLY5: median2300.4249954aggregateprefill/
168.2838861decode;pooleddecode170.7456560(+2.356%),prefillmedian+2.897%/
pooled+2.739%,whole+2.676%. All4qualifiedBEFOREfirsttrial;alltoken/cap/source/
harness/coherentpreviewgatesPASS,sampledpeak30.562191GiB. 108productiontests/
96actualoperation-checkpointcasesPASS,KV/battery4/4/Ruff. 120producerprototype/
192actualprojectionvariantcases/12graphproofsPASS. TWOlayouts only:corrected
8pairedchecks/12timings4096reload+6.398%/register+8.085%,heldout4160+6.797%/+8.360%;
REGISTERselectedbeforeintegration/E2E. Initial12timingsexcludedbecausegraphinput
mutationnotrestored;equal-inputcorrectiononly,nofurther variant orE2E. Native
APIs/sourceunchangedexceptearlyMoEDecoderM>=256guard;newprefillopaqueoperation,
explicitq/scales,noTensorattr/globalstash,originalnormloop/defaultFPfusionTrue/
4warps/collectives/scalefc2/cachebytes preserved. 3oldprojectionquantcalls->1fc2
quantplusfusednormquant. New6filesexp4-saved/exp4-sources.json(11incloriginals).
Bench97395/monitor8760/stop20873all0,serverstopped. Firstwaitingclient80562stopped143
BEFOREANYrequest/warmup:earlymarkersomitTPprefix. QualifierONLYusescurrentjournal
JSON+uniquePID->rank;6CPUcases/actualall4PASS,ambiguous/missingrejected;timing/
workload/hardwareASTunchanged. NewrunSHA06026268cae372bbade6afde0ef8e819240bb9ebca21fec9938b29625d8ef04e;
oldrun-before-E4-journal-pids.py retained,manifestrefrozenBEFOREonewarmup/EXACT5.
AITERcompanion3c74adc198e55aeb5bcd41516bddfb3ccadf6725 stagedsecurityPASS;
publishAITERfirstthenrootcaller/ledger/noteswithfinalstagedsecurityandremoteverify.

ACTIVEB07E5 next: groupTWOexistinglargefallbacklatent/sharedPyNcclreductions,
separatebuffers/samemessages/order/noconcat/noquantboundarymove. CPUdesign
exp5-design.md,planregistered. NoE5runtimeedits/GPUworkyet. Currentactualdtypeaudit
12772exit0 exp5-current-ar-dtype-audit.json: _tp_ar_fp32_mode returnsSTRING0whenunset;
GroupCoordinator/functionalall_reduce useifmode,so0isTRUTHY. LogicalBF16ALREADY
promotestoF32beforedispatch/downcastsafter. DO NOTfixflagorswitchwireprecision.
E5groupsSAMEactualF32messages;originalpromotions+allocateBOTHoutputsBEFOREgroup_start,
originaldowncastsAFTERgroup_end. Checkallbackendeligibilityonactualpromotedmessages.
SinceE4KEPT,wirebothlegacyrunnercompletionANDnewnemotron_h_prefill completion,
otherwisewouldqualifyunusedpath. ReferenceORIGINALexp4five,NOTexp1orincumbent;
exact5/no extensions/newbaseline. Preserveall6acceptedfusionfileswhenrevertingE5.

E1precommitpubliccredits: originalanddedicatedsource retain4upstreamauthorlines
EACH. Securityhelper nowacceptsmultiple --public-attribution-review flags, checks
exactlabeledcredithashesagainstupstreamsnapshot+HEADorigin (fornewderivativefile,
original_attribution_file). B07exp1-{generic,dedicated}-public-attribution-review.json
andpublic-attribution-upstream.txt. No policyexception/permissionforothercontacts.
UsebothreviewflagsatFINALSTAGEDgatebeforepublishingwinner. No sourcechangesforprivacy.

E4unchangedpublicsourcecredits verifiedagainstofficialrawGithub; ignoredreview/
snapshot/hash retained. Securityhelper optional--public-attribution-review only
classifiesunchangedlabeledpubliccreditsmatchedagainstHEAD/upstream; nodisclosure
policyexception. NoE4sourcecommitted. compare.py guidance-onlynote corrected
AFTERE4frozenhashverification:exactfive,noextra controls. Arithmeticunchanged,
oldhelper retained B06compare-before-E5.py;freeze updatedhashforE5.

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
7-8minfreshboot. Runtime lifecycle dropinB07; activegate must beabsent before
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
