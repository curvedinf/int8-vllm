# Paused Bolt optimization

The user stopped the continuous loop on 2026-10-17 and requested the best
combination as the standard Bolt recipe/launcher, with standard quality gates.
Do not restart optimization automatically. The registered B09 E5 draft FC
sharding experiment was never implemented. Resume it only under a new request.

## Accepted state and operations

Runtime sources: vLLM `4711913204533dfe702bd7e9ed7d23d07a08503b`, AITER
`3c74adc198e55aeb5bcd41516bddfb3ccadf6725`. B07 E1 page attention + B07 E4
native MoE norm/GS128 fusion remain accepted, above weighted9 reduction,
B03 route64 and B02 short-prefill cohort. Canonical settings and routine-gate
results live in docs/recipes/nemotron_bolt.md and the experiment ledger.

The standard vllm-bolt.service is active/enabled on localhost8020. Its actual
process argv/env was checked against TP4/C8/NS7, graphs, INT8 KV, BF16 model,
FP32 state and 3GiB dense/5GiB expert cache settings. All four caps remain105W;
thermal controller stopped+disabled, nginx stopped. The isolated optimization
unit and its owned profiler drop-in were retired; no automatic optimizer runs.
Keep existing production credentials private. Do not hot-reload kernels: it
can duplicate scale caches and bias cache/memory state.

Reference metric remains ORIGINAL B07 exp4-{0..4}, bytecopies B09 incumbent:
median2300.4249954 aggregate prefill/168.2838861 separately timed aggregate
decode tok/s. Exactly five frozen4096-input/256-output/C8 trials, same eight
prompts, seed0/cache resets/105W; one untimed warmup. No new performance
baseline, extensions or extra controls. Retention requires positive pooled
whole-run gain and BOTH median/pooled phase rates nondegrading. Report only
aggregate prefill and separate aggregate decode, never whole-output as decode.
Earlier1500/400 reports used different workloads and are not comparable.

## Costly evidence and remaining uncertainty

Canonical decisions: docs/recipes/surface_experiments_ledger.jsonl. Raw
source snapshots/proofs/five-run results: ignored agents/tmp/bolt-opt/loop/
batch01..batch09. All B08 experiments and B09 E1..E4 rejected/reverted; no
runtime change from those batches is accepted. B09 E4 completed five before
wrap-up:2266.8257/158.5371 median rates; pooled prefill-1.4120%,decode-7.3924%,
whole-4.1829%. Frozen33sources/22proofs and all-four qualification/caps/counters
verified, then only its six hash-verified owned paths restored/removed.

Finalization artifacts: agents/tmp/bolt-final/. Keep failure observations and
subsequent corrected checks. The original short-answer soak exhausted64tokens
in reasoning and returned no final answer; fixture now disables reasoning,
checks final content, and retains a separate reasoning functional gate. Cases
0..51 completed serially; remaining52..499 ran atC8 without repeating completed
cases. Same prompts/seeds/budgets verified. One interrupted in-flight response
was excluded. New CLI count/concurrency validation and authenticated entrypoint
were checked. These are quality gates, not additional performance repetitions.

Code-answer spot checks still invent Math.clamp availability (localNode22 has
no API); this also appears in greedy reasoning/nonreasoning diagnostics. Origin
unisolated; no optimizer regression established. Coherence is not task accuracy.
Official-reference full-model fidelity and full256k remain unverified; do not
claim passed. The recipe/ledger record the limited numerical-corpus evidence. The raw
52-prompt completions capture is not a chat-quality gate; the Bolt adapter
captured the same52 through the actual chat template. Same52 plus500 replies
were rechecked after fixing whitespace/Markdown false positives; ten focused
checker regressions cover readable layouts, genuine garble and empty answers.

Critical wire policy: _tp_ar_fp32_mode() returns STRING"0" whenunset; it is
truthy, so BF16 partials already promote to F32 wire and cast back. Do not fix
that flag or change collective precision/order/backend. Legacy perrow/trunc
FUSED_NORM_QUANT stays off; the accepted native producer preserves the original
Triton BF16 norm (including NaN payload), residual and exact GS128 q/scales.

B09 E2 launch_bounds512 removed scratch but the new dense MFMA still regressed;
no unchanged reruns. E3 TP4 fixtures require ps.graph_capture lifecycle. E4
synthetic32-expert logits were nonrepeatable even on the original; deployed
counts240/304/368/432/496 were repeatable. Component proofs do not establish
full-model fidelity. E5 research only: exp5-design.md, draft replicated FC
row-sharding with bitwise gather; no implementation/testing authorized now.

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

## Resumption and ownership

All Python uses the serving .venv; GPU work runs as checkout owner with HOME
and ROCm/gfx908 env, PYTHONPATH repo:siblingAITER. No venv/runtime changes:
torch2.11rocm7.1 wheel deliberately carries patched HIP/HSA7.2. All Git writes
and pushes as checkout owner, both main; AITER remote curved, root remote origin.

The frozen benchmark harness and manifest hashes are in B09 incumbent-sources.
Its decode metric counts only events AFTER the latest first-token event until
the final generation event; prefill includes admission/interleaved decode.
verify-frozen-result.py verifies artifacts without new GPU work. Future branches
need actual all-rank runtime qualification before spending a five-run budget.
For data-dependent branches, initialization/capture markers are insufficient.

Unrelated working changes remain protected by agents/tmp/bolt-opt/loop/
protected-worktree-hashes.json:10paths bytes+mtime, plus3deletions. Never broad
restore/stage/stash. Preserve those pending changes; the final recipe commit is
scoped to this task. Follow AGENTS.md/security-policy before every commit and
verify remote refs. Maintain-scratch was checked/notdue Oct15; compaction and
ordinary task updates do not retrigger it.
