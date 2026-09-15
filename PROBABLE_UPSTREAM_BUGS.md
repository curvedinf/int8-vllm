# Probable upstream bugs — numerics defects found on gfx908 (MI100) that exist in stock vLLM / affect stock configurations

This file documents numerics bugs discovered during the int8-vllm garble
campaign (Aug–Sep 2026) that are **not fork-specific**: each was verified
against the last upstream merge state of the affected code (commit
`7c6729b769`, the most recent upstream sync touching these files) or is
reproducible with stock components alone. All were proven on 4x AMD
Instinct MI100 (gfx908 / CDNA1, XGMI full mesh), ROCm 7.1, torch
2.11.0+rocm7.1. None of the reproductions require fork code unless noted.

For each: bug-report format ready to file against `vllm-project/vllm`
(or the appropriate component), with reproduction steps.

Fork-only defects (g128-KV attention path, DFlash speculator GDN-state
handoffs, PTQR export bugs, offload-executor stream semantics) are
intentionally **excluded** — they cannot be hit from stock.

No stock-AITER **numerics** bug was proven in this campaign (the
`gemm_a16w16_asm` JIT gap for GDN-shaped N/K combos is a coverage
limitation, not a correctness defect; the W8A8 tuned-CSV misses were
performance-only).

---

## BUG 1 — Custom all-reduce: device-scope flag acquires do not cover remote (XGMI peer) agents → torn cross-rank reads

**Component**: `csrc/custom_collective_common.cuh` (barriers),
`csrc/custom_all_reduce.cuh` (reduce kernels)
**Affects**: stock vLLM on multi-GPU ROCm where P2P goes over XGMI
(proven on gfx908; the memory-model argument applies to any arch where
peers are remote agents relative to device scope — likely gfx90a/gfx942
as well)
**Severity**: high — silent numeric corruption (torn reads → NaN /
huge-finite seeds), episodic, load-dependent
**Type**: memory-ordering / race

### Summary

Upstream `barrier_at_start`, `barrier_at_start_release`, and
`barrier_at_end` perform their flag **acquire loads** with
`__MEMORY_SCOPE_DEVICE` (and `barrier_at_start` is fully
`__ATOMIC_RELAXED`). In the custom all-reduce kernels, the data guarded
by these flags is written by **remote agents** (peer GPUs staging their
contribution into IPC-shared buffers over XGMI). On gfx908 a
device-scope acquire does not formally order visibility of writes made
by a remote agent, so a reduce kernel can pass its barrier and read a
peer's buffer **while the peer's writes are still in flight** — torn
values. A torn bf16 value is a NaN/huge-finite seed; downstream it
produces single-layer activations corruption that degrades output into
junk-token walls.

### Upstream state (verified)

At upstream merge `7c6729b769`, `csrc/custom_collective_common.cuh`
lines ~264–320: all three barriers acquire with
`__MEMORY_SCOPE_DEVICE`; `cross_device_reduce_1stage` calls the fully
relaxed `barrier_at_start`. The fix in this fork (commit `2ed8fcac5c`)
promoted every cross-rank acquire to `__MEMORY_SCOPE_SYSTEM` and switched
both reduce kernels to `barrier_at_start_release`.

### Reproduction

Reliable in-engine repro is load/timing dependent (that is the nature of
the race); the practical recipe that exposed it here:

1. Serve any TP≥2 bf16 model on 4x MI100 with custom all-reduce enabled
   (`--tensor-parallel-size 4`, custom AR not disabled).
2. Drive long decode with speculative decoding (thousands of steps) at
   high concurrency; alternate prefill chunks and decode so ARs of many
   sizes interleave back-to-back.
3. Capture logits or hidden states across ranks and compare against a
   single-rank/gather reference: episodic one-layer divergences appear
   (torn residuals); with enough steps, outputs collapse to token loops.

Deterministic smoke test (shows the *ordering* violation, not the
timing race): inspect the SASS/behavior of two adjacent ARs on different
streams — with DEVICE-scope acquires the second kernel may begin
reading peer staging buffers before the peer's writes are visible.

### Suggested fix

Promote all cross-rank flag loads/stores in the three barrier helpers
to `__MEMORY_SCOPE_SYSTEM`; use `barrier_at_start_release` (RELEASE
store + ACQUIRE load) in both reduce kernels. Measured cost on gfx908:
none (SYSTEM-scope atomics on already-coherent XGMI paths).

---

## BUG 2 — Custom all-reduce is not bitwise-reproducible: 2-stage kernel rotates accumulation order per rank, and 1-stage/2-stage selection is message-size dependent

**Component**: `csrc/custom_all_reduce.cuh`
(`cross_device_reduce_2stage`, dispatch in `allreduce()`)
**Affects**: stock vLLM custom all-reduce at world size ≥ 3 with
messages crossing the 1-stage/2-stage boundary (512 KB at WS=4 in the
current dispatch table)
**Severity**: medium — silent bitwise nondeterminism that becomes
output-visible at long context
**Type**: numerical determinism

### Summary

Two stacked issues in the upstream kernels:

1. `cross_device_reduce_2stage` iterates peers as
   `int target = (rank + i) % ngpus; ptrs[i] = _dp->ptrs[target];` —
   **each rank accumulates in a different order**, so ranks can disagree
   on the last-ulp of every element. (The 1-stage kernel does not
   rotate — upstream's own comment there notes fixed order for bitwise
   identical results; the 2-stage kernel breaks that property.)
2. `allreduce()` selects 1-stage vs 2-stage by **message size**
   (`bytes < 512 * 1024` for WS≤4). With chunked prefill, the same
   layer's all-reduce changes size between chunks (and between prefill
   and decode), flipping the kernel — and the accumulation order —
   chunk-to-chunk.

Consequence: the same logical computation yields different results
depending on chunk geometry. At 20k+ context these per-chunk ulp
differences amplify into systematic decode-vs-prefill divergence; in
this fork's campaign that presented as garbled long outputs. Measured
on this hardware: an in-vitro bf16 all-reduce of identical data at
1664 rows vs 2048 rows differs in **34% of elements** (either kernel
family vs the other); real-engine seam probes showed 2–17 nat logit
outliers at chunk boundaries before unification.

### Reproduction (standalone, stock vLLM kernels)

```python
# torchrun --nproc_per_node=4 repro.py   (4x MI100, fork checkout only
# because stock vllm does not expose the kernels as a python API; the
# kernels themselves are byte-identical to upstream at merge 7c6729b769)
import torch, torch.distributed as dist
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
dist.init_process_group("gloo")
ws = dist.get_world_size()
x_small = torch.randn(1664, 5120, device="cuda", dtype=torch.bfloat16)
x_big = torch.cat([x_small, torch.randn(384, 5120, device="cuda", dtype=torch.bfloat16)])
car = CustomAllreduce(dist.group.WORLD, device=0)   # default max_size
out_small = car.custom_all_reduce(x_small.clone())
out_big_prefix = car.custom_all_reduce(x_big.clone())[:1664]
# out_small != out_big_prefix on a large fraction of elements:
mismatch = (out_small != out_big_prefix).float().mean().item()
print(dist.get_rank(), "frac differing:", mismatch)   # ~0.3, all ranks agree on count
```

In-engine A/B (pure stock semantics): run the same 20k-token prompt with
`--max-num-batched-tokens 2048` vs `1664` and compare prompt logprobs at
matched positions — chunk-size-dependent differences at every chunk
boundary.

### Suggested fix

Remove the per-rank rotation in `cross_device_reduce_2stage` (iterate
`ptrs` in fixed rank order, as 1-stage does), and make the accumulation
dtype/order uniform across both kernels (e.g., fp32 accumulate +
single downcast in both). Alternatively document that custom AR is not
bitwise-stable and gate it off for workloads requiring determinism.

---

## BUG 3 — TP all-reduce results depend on tensor size (NCCL/RCCL algo-by-size changes reduction order) → chunked-prefill nondeterminism

**Component**: environment/RCML behavior surfaced by stock vLLM TP
(`tensor parallel` all-reduces under `--enable-chunked-prefill`)
**Affects**: any stock vLLM TP deployment on ROCm (and by the same
mechanism, NCCL on CUDA); proven on gfx908 + RCCL (ROCm 7.1)
**Severity**: medium — same class as BUG 2 through the default
(pynccl/RCCL) path when custom AR is disabled or exceeds its envelope
**Type**: numerical determinism (environment-level, exposed by vLLM)

### Summary

`dist.all_reduce` on bf16 picks its collective algorithm by message
size. Different algorithms reduce in different orders, so the same data
reduced at different shapes gives different results. With chunked
prefill, a layer's TP all-reduce size varies between chunks; logits
become chunk-geometry dependent even with custom AR disabled. Measured
on 4x MI100: a bf16 `dist.all_reduce` of identical data at (1664, 5120)
vs (2048, 5120) differs in ~34% of elements (torch-only repro below —
no vLLM involved at all).

### Reproduction (pure torch, no vLLM)

```python
# torchrun --nproc_per_node=4 repro.py
import torch, torch.distributed as dist
dist.init_process_group("nccl")          # maps to RCCL on ROCm
r = dist.get_rank()
g = torch.Generator(device="cuda").manual_seed(100 + r)
H = 5120
X = (torch.randn(2048, H, generator=g, device="cuda") * (0.1 + 0.05 * r)).to(torch.bfloat16)
ar = lambda t: (lambda o: (dist.all_reduce(o), torch.cuda.synchronize(), o)[2])(t.clone())
outB, outA = ar(X), ar(X[:1664].contiguous())
d = (outA.float() - outB[:1664].float()).abs()
print(f"{1664}-vs-{2048}: bitwise={torch.equal(outA, outB[:1664])} "
      f"frac_diff={(d > 0).float().mean().item():.5f} max={d.max():.3e}")
# -> frac_diff ≈ 0.34
```

vLLM-level demonstration: identical 20k-token prompt, two servers with
`--max-num-batched-tokens 2048` vs `1664`; `prompt_logprobs=1` on both;
matched positions differ by >1 nat at chunk-boundary-adjacent tokens.

### Suggested action

Document that TP all-reduce (custom or NCCL/RCCL) is not
chunk-geometry-invariant, or make vLLM pin a single algorithm/protocol
per layer shape class. For consumers needing determinism, provide a
supported "bitwise AR" mode (fixed-order fp32 accumulate + single
downcast — the approach this fork validated).

---

## BUG 4 — KV-offload `swap_blocks_batch` via `hipMemcpyBatchAsync`: D2H stores racing CUDA-graph replay corrupt device memory on gfx908

**Component**: KV-offload copy path
(`csrc/libtorch_stable/cache_kernels.cu` → `hipMemcpyBatchAsync`)
**Affects**: stock vLLM KV-offload connector (CPU tier) on ROCm 7.x +
gfx908 under CUDA graphs; earlier research found the batch API × graph
interplay still under construction upstream
(rocm-systems#10089) and no public report of this failure mode
**Severity**: critical — silent device-memory corruption during decode
**Type**: driver-level race (suspected) with an in-tree workaround

### Summary

GPU→CPU KV offload stores issued through the driver batch API
(`hipMemcpyBatchAsync`) corrupt device memory episodically on gfx908
when they overlap CUDA-graph replay on the compute stream. Symptoms
proven in-engine: single-layer activation corruption → junk-token
output walls; a hard `HSA_STATUS_ERROR_EXCEPTION` fault when a
mid-forward host sync perturbs the interleaving. Severity is
allocation-layout dependent (the same code is clean or corrupt
depending on which live tensors occupy the victim addresses), which is
why it presents intermittently across boots.

A detailed, upstream-ready bug report already exists in this repo:
**`docs/recipes/bug_rocm_batch_memcpy_graph_replay_race.md`** —
including the bisection knife table (FAKESTORES clean / batch path
corrupt), the A/B evidence, and the reproduction protocol. Two upstream
sub-findings from that investigation:

- the `attrIdxs` out-of-bounds read below (BUG 5) is inert on ROCm
  < 7.13 (attrs gated off) but the race reproduces even with
  `count == 1`, so the OOB is not the sole cause;
- replacing the batch API with **one plain `hipMemcpyAsync` per
  descriptor, enqueued on the legacy default stream** is clean on
  gfx908 at equal throughput (the fork's production fix).

### Reproduction

See the referenced doc; the engine-level protocol is: enable the CPU
KV-offload connector on 4x MI100 with CUDA graphs on, drive 20k-context
decode for thousands of steps, and read outputs in full — corrupted
runs show early token loops or late degradation; `VLLM_OFFLOAD_*`
knives in the fork isolate the store path without disabling the tier.

---

## BUG 5 — `swap_blocks_batch` passes a single stack `size_t` as `attrIdxs`: out-of-bounds driver read whenever `numAttrs > 0` and `count > 1`

**Component**: `csrc/libtorch_stable/cache_kernels.cu`
(`swap_blocks_batch` binding)
**Affects**: stock vLLM (tracked upstream as issue #53863; fix PRs
#53971 / #53860 were still open as of 2026-09-12); live on ROCm ≥ 7.13
where attrs are enabled, inert below (numAttrs = 0)
**Severity**: medium — driver reads past a stack object on every
batched copy with attributes
**Type**: memory safety (OOB read)

### Summary

The binding constructs `size_t attr_idx = 0;` on the stack and passes
`&attr_idx` as the per-descriptor attribute-index array. Whenever the
driver consumes more than one entry, it reads past the single stack
word — undefined behavior in the driver on the exact call path used by
KV offload. Verified in this fork's tree at merge `5ad03101d5`; the
fork applied the properly-sized-array fix (commit `f2dc8f1912`,
mirroring PR #53971) and confirmed the OOB was **not** the significant
contributor to the gfx908 corruption (see BUG 4) — but it is a real
defect on ≥ 7.13 where it is reachable.

### Reproduction

Call `swap_blocks_batch(src_addrs, dst_addrs, sizes, count=8)` on a
ROCm ≥ 7.13 build with the attrs path enabled; run under
compute-sanitizer / driver tracing — the driver performs 8 reads from a
1-element stack array.

---

## Filing notes

- Bugs 1–2 are independently sufficient to break bitwise reproducibility
  of TP inference under chunked prefill on stock vLLM; bug 3 shows the
  same symptom class through the default collective even with custom AR
  disabled. They compound.
- Bug 1's fix has been soak-validated here (24/24 long greedy legs vs
  ~5/8 boot failures before); bug 2's fix shape (fixed-order 2-stage +
  fp32 accumulate) is validated by this fork's `car1` mode, which
  passes full-read 4k-output gates and seam probes at 20k context.
- All evidence chains, knife switches, and ledger rows are in
  `docs/recipes/surface_experiments_ledger.jsonl` (experiments
  `G2_PHASEB_CAR_ENVELOPE_AND_OFFLOAD_ROOT_CAUSE`,
  `G2_ATTN_SASS_ROOT_CAUSE`, G1/G4 series) and
  `docs/recipes/bug_rocm_batch_memcpy_graph_replay_race.md`.
