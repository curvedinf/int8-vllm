# Bolt continuous optimization

Active user contract (2026-10-16): run indefinitely until manually stopped.
Each batch profiles 4096 input /256 output at C8, identifies prefill compute
opportunities, designs five experiments spanning structural/fusion/kernel
scopes, and executes them sequentially against the accepted incumbent.
Benchmark every experiment FIVE times. Retain only demonstrated end-to-end
wins with no prefill or decode degradation; commit/push wins, then repeat.
Do not finish after a batch or plateau. No subagents authorized.

The requested memory configuration is accepted; stop its old acceptance tests.
Fixed contract: static dense INT8 ownership (original GPU packed tensors and
codec metadata released), dense ceiling3GiB; dynamic experts keep packed
originals, expert ceiling5GiB, slots=auto, refresh128. Actual277 dense
projections/2,387,607,552INT8 values and4565expert pairs/114–119slots perlayer.
GPU KV14e9bytes/rank, CPU24GiB local pin6GiB/rank, context262144, TP4/C8/NS7,
INT8GS128 compute/KV, BF16 residual, FP32 Mamba, custom AR. All four GPU caps
105W; thermal controller stopped AND disabled; nginx stays down. Its installed
thermal-controller.conf dependency is renamed .disabled-static-cache-test.
Current optimization owns isolated bolt-opt-loop.service on127.0.0.1:8021,
LOG_DIR agents/tmp/bolt-opt/loop/runtime, lifecycle capture directory
agents/tmp/bolt-opt/loop/batch01/host, chunk32steps, diagnosticRPC enabled.
Production vllm-bolt is stopped (exit143 reportedfailed); do not start both.

Batch01: profile complete, five-run incumbent complete: median43.297s, pooled47.329outputtok/s,
median928.430prefilltok/s and14.005decode tok/s/request; acceptance3.367–3.559.
Experiment1 native-only prefill INCONCLUSIVE: five runs pooled48.120outputtok/s
(+1.67% vs47.329), median42.418s/957.035prefill/14.427decode; below3% minimum.
Restored original dispatch; no adoption. Experiment2 cache-aware prefill
implemented in AITER (get existing projection; no allocations/admission changes);
KV microPASS/battery4/4 and31GPUtests passed (10large-prefill plus21expert cache).
E2 RETAINED, AITER0d8d783ab committed/push in progress: candidate51.294outputtok/s
vs matched fresh control47.540 (+7.90%); pooled prefill+9.15%, decode+6.17%.
All5 corresponding trials improve each metric; arithmetic exact. Five fresh
control results retained alongside original post-profile baseline. Source/test
backups in loop/batch01/exp2-saved. Incumbent nowroot8a0e598d0e/AITER0d8d783ab.
E3 epilogue fusion next; diagnostic server stopped for implementation/checks.
Rank0 trace648950kernels, all launch-correlated; packed routed prefill10.327s /
23.162s in prefill-containing steps (44.58%). Dense GEMMs and collectives follow.
Artifacts loop/batch01/{profile-analysis.json,csv/,host/gpu/}. Frozen eight
4096-input/256-output natural technical prompts are loop/manifest-4k.json,
SHA256263da19de5a5bc8823fe38d7113553c08f00fa453127dbbd523c74622d0bcda1.
run.py resets GPU/CPU prefixes and asserts exact8/32768/2048 request/token
metrics for every run. Prefill=32768/maxTTFT; decode=sum tokens after first
burst /sum per-request generation spans; end-to-end=2048/whole-run walltime.
Profile timings are excluded from speed comparisons. Each experiment has an
untimed workload warmup and five measured runs. Aim for >=3% end-to-end gain
beyond observed noise with neither prefill nor decode median/pooled rate worse;
mark uncertain comparisons inconclusive. Full-context/fidelity remain deferred.

Batch01 hypotheses (execute sequentially, no simultaneous candidate edits):
1. Native register-local HIP routed kernels for M>64 eliminate the expensive
   Triton packed conversion path. Preserve existing small-M execution; compare
   real geometry/partial slices at largeM against packed reference first.
2. Consume already-resident expert INT8 copies in prefills, retaining originals
   and the same5GiB cap. Avoid decoding cached experts repeatedly; bounded route
   refresh remains unchanged. Build on1 only if accepted; otherwise preserve
   the incumbent path for misses and test a separate cache-aware prefill path.
3. Fuse dense GEMM epilogues: store final BF16 directly when no FP32 bias is
   pending, and fuse shared-UP ReLU2/GS128 INT8 output for full128-column groups.
   Profile shows~80k BF16 copy launches in decode and separate quant/reduce work;
   this has broader scope than the originally proposed nine-expert combine.
   Preserve FP32/unfused ordered group accumulation and BF16/INT8 rounding.
   Keep bias addition in FP32 before cast. Component tests compare direct BF16
   to FP32-plus-cast and fused quant outputs/scales to the current split reducer.
   Do not implement untilE2 completes; keep numerical and five-run phase gates.
4. Specialize all-groups-selected routing (checkpoint n_group=topk_group=1),
   removing group top2/mask work; consider the existing fused sigmoid/bias topK.
   Preserve selected experts, renormalization and routing weights.
5. One bounded dense-prefill GEMM tile/pipeline sweep using measured shapes;
   leave decode dispatch intact. Count related variants as ONE experiment.
No GPU micro/profile/benchmark overlap. User's prior no-new-baseline instruction
applied to the completed memory task; new workload requires its own incumbent.

Sources: root8a0e598d0e and AITER0d8d783ab; memory integration and E2 are published
(verify latest remote ref after push). Preserve unrelated work/staging. Before-snapshots
in agents/tmp/bolt-opt/static-drop/before; memory tests/results recorded in
ledger BOLT_STATIC_DENSE_EXPERT5_20261016, not a stable speedup claim.
The root security finding was resolved by explicit user direction: the offload
flag path now resolves relative to the checkout; no disclosure permission was
granted. Existing Git history remains unchanged. All three diagnostic flags
are absent, so the path correction preserves the measured serving behavior.

Protected unrelated root changes include AGENTS.md, logs/README.md deletion,
scripts/serve_recipe_qwen38.sh, docs/recipes/qwen38_flash_next_support.md;
service migration and startup-watch edits also predate this task. Root-owned
new test files should be chowned tocurved before handoff. Use Git mutations as
curved to avoid creating root-owned Git objects. No venv dependency changes.

Costly prior evidence: CPU24 full-region registrations exceeded AMD host-memory
accounting; rank-local pin/copy24GiB TP4 micro and full boots passed. The old
watchdog targets production PID files; never reuse it for isolated operations.
Profile at context256k/chatC8: routedIQ2 35.78%, denseGS12817.48%, customAR16.02%,
rocBLAS8.9% (FP32 routers), Mamba5.03%, topk2.72%; reprofile actual4k workload.
Dense preshuffle exact but slower; FP32 router split8 end-to-end327 vs346;
IQ2 padding66->68 UP35–40%slower/DOWN9%faster. Do not repeat without new evidence.

Recent rejected prototypes (all ignored agents/tmp/bolt-opt):
- dynamic32-expert cache:1531.17/350.60 vs1544.58/361.69; reverted, buffers
  released. Cached fullweight micro22-30% faster, but overhead/reuse lost E2E.
  Real C8chat top32 experts48.2%, top64 62.8%, top12880.9% selections.
  Layers have240-496 experts (NOT512); initial prototype crashed admitting
  nonexistent padded experts. Fixed version archived packed_expert_cache_fixed.py
  plus dynamic-cache-trial.diff. Real240-expert zero-ID/partial micros exact.
- singleton SDOT path:1548.53/354.50; reverted despite4-12% C8 micro gains.
- quant_table_probe caches3 INT8 magnitudes per32 weights:exact but20-60%
  slower; reject. signed256KiB codebook also slower; reject.
Older rejected CK/NVFP4/shared-UP/LDS variants and accepted history in ledger.

Never overlap profiles, GPU micros, and unprofiled benchmarks. Diagnostic
run-quick.py: one untimed8192-token prefill warmup, two measured cold C1
prefills; decode8requests/C8/32input/1000output/temp0/seed0/2warmups. Older
32request benchmarks are not comparable. No further production benchmarks
while interactive traffic is active. Saved launcher must be edited atomically;
a running shell can otherwise misread its remaining shutdown commands.

Next-batch structural lead: native packed_routed already templates MT; MT2 can
reuse decoded weights across32rows instead of16. Requires32-row route alignment,
MT-aware COUNT guard, explicit tile-size metadata/API, and large/partial tests.
Do not silently assume16-row plans work for MT2. Defer beyond current five scopes.

Further source-backed lead for next batch: packed_routed hoists IQ2 FP16 base
scales only for DOWN&&ALIGNED. UP and unaligned DOWN still call decode4 with
per-subtile scale loads. Investigate hoisting one/two base scales per GS128
window and selecting the correct half across original256-element boundaries.
Require varying per-block FP16 scales (current synthetic helper uses constant
0x3000) plus actual-weight and all-partial-offset exact checks before timing.
