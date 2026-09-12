# Bug report: GPU memory corruption when CPU-offloading stores overlap CUDA-graph replay (hipMemcpyBatchAsync, ROCm)

**Title:** Episodic device-memory corruption during decode when the CPU
offloading connector submits `hipMemcpyBatchAsync` D2H stores concurrently
with CUDA-graph replay on the compute stream (AMD gfx908 / ROCm 7.1)

**Severity:** High — silent output-quality corruption (text collapse) in any
configuration that combines the offloading connector with speculative
decoding and CUDA graphs on CDNA1 hardware. No error is raised; generation
continues from corrupted attention state.

---

## 1. Summary

On AMD Instinct MI100 (gfx908, CDNA1) under ROCm 7.1, a vLLM engine configured
with

* the **CPU offloading connector** (`OffloadingConnector`, primary CPU tier),
* **speculative decoding** (any depth; DFlash/MTP/Eagle-family), and
* **CUDA graphs** for the decode/verify forward

produces *episodically corrupted KV-cache contents*, which the model then
attends over, degrading generation quality until output collapses into
repetition walls or multilingual token soup. The corruption:

* occurs **only when the offloading tier performs decode-phase stores**
  (GPU→CPU copies),
* is **independent of the stored content** (content-level fixes did not
  eliminate it),
* is **not caused by restores** (zero loads occur during decode),
* is **not caused by GPU-block recycling** (descriptor buffers are
  event-gated and never reused while in flight),
* and its **probability scales with submission overlap**: every knob that
  reduces the temporal overlap between store submission and the next
  forward's CUDA-graph replay reduces the corruption severity; the only
  fully clean configuration is zero decode-phase stores.

The remaining mechanism consistent with all evidence is a **driver-level race
between `hipMemcpyBatchAsync` D2H transfers issued on a side stream and
CUDA-graph replay on the compute stream** on ROCm 7.1 / gfx908: the batched
driver copy path (submitted on the transfer stream, gated only by
`wait_stream(compute)` at submission time) corrupts device memory while a
captured graph is being replayed on the compute stream.

## 2. Environment

| Component | Version |
| --- | --- |
| GPU | 4× AMD Instinct MI100 (gfx908 / CDNA1), XGMI full mesh |
| ROCm / HIP runtime | 7.1 (torch reports `7.1.52802`) |
| PyTorch | 2.11.0+rocm7.1 |
| vLLM | fork of current main (commit `5ad03101d5`); affected code paths are upstream-shared |
| Kernel | Linux 5.0.0-31 (AlmaLinux 8.10 userland) |
| Model | Qwen3.8-27B GDN hybrid (48 linear-attention + full-attention layers), TP4 |
| KV cache | int8 block-g128 (grouped fp16 scales inline), block size 64 |
| Spec decode | DFlash drafter, NS=6 (7-row verify batches) |
| CUDA graphs | piecewise/full graphs capture the decode+verify forward |

## 3. The code path involved

Store flow (file references from the tree at commit `5ad03101d5`):

1. **Scheduler side** — `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`,
   `storable_chunks()` (~line 464): decides how many leading chunks are
   eligible for store each step.
2. **Deferral** — `vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py`,
   `prepare_store_kv()` (~line 353): store jobs are *deferred to the
   beginning of the next engine step* and submitted from
   `start_kv_transfers()` → `start_load_kv()` (invoked inside the next
   forward's forward-context). So the D2H copies are submitted **while the
   next step's model forward is starting / running** — and on the decode
   path that forward is a **CUDA-graph replay**.
3. **Transfer handler** — `vllm/v1/kv_offload/cpu/gpu_worker.py`,
   `SingleDirectionOffloadingHandler.submit_store()` (~line 630):
   * picks a side CUDA stream from a pool,
   * issues `stream.wait_stream(current_platform.current_stream())`
     (line 655) — a barrier that only orders against compute-stream work
     **submitted before this call**,
   * then calls `ops.swap_blocks_batch(...)`.
4. **C++ batched copy** — `csrc/libtorch_stable/cache_kernels.cu`,
   `swap_blocks_batch()` (~line 105): on ROCm ≥ 7.1 this calls
   `hipMemcpyBatchAsync` (descriptor arrays are the pinned-CPU int64 tensors,
   reinterpret-cast directly; `srcAccessOrder = STREAM` for GPU→CPU).
   The same file documents several known ROCm quirks of this API
   (7.2.1–7.2.3 reject `numAttrs > 0`; >8192 descriptors fault on 7.15).

Load flow (for contrast): loads are only issued at request admission on
prefix-cache hits; during steady decode **no loads occur** (verified with a
jsonl logger on `_maximal_prefix_lookup`, which stayed empty across full 4k
generation legs).

## 4. Observed failure and signature

* Long-context generation (20k-token prompts, ≥1–4k output tokens, sampling
  temperature 1.0) degrades nondeterministically: onset varies between ~50
  and ~2600 output tokens across seeds; the text collapses into token
  repetition walls ("the the the …"), quote-fragment storms, or multilingual
  token soup; sometimes the model later partially recovers when the corrupted
  region loses attention dominance — a *transient overlay* signature, not
  permanent state loss.
* Greedy decoding is largely immune (argmax robustness), spec-off decoding is
  immune, and **disabling the offloading tier is immune** — the fault was
  initially misattributed to the spec-decode state machinery for this reason.
* The corruption is *content-independent*: fixing every content-level hazard
  (storing volatile trailing chunks that rejected spec tokens will rewrite;
  storing the drafter's sliding-window KV that is re-projected every round)
  measurably reduced but did not eliminate it.

## 5. Evidence chain (all rows reproduced on the same engine config; text
quality judged by full manual reads of detokenized committed-token streams)

| Experiment | Overlap between stores and graph replay | Result |
| --- | --- | --- |
| Offloading tier fully ON (baseline) | maximal (deferred to next-step start, batched driver API) | ~2/3 of 4k legs collapse |
| Stored-content fixes (volatile-tail exclusion for all groups; drafter-group store ban) | unchanged timing, better content | reduced rate, still collapses (e.g. seed 301) |
| `VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS=1` (single-descriptor `hipMemcpyBatchAsync` calls) | same timing, 1 descriptor/call | seed 301: wall → structured word-loops (improved); seed 307: still soup |
| `VLLM_OFFLOAD_SUBMIT_AT_FINISH=1` (submit stores in `get_finished`, after the step's forward/sampling completed) | overlap window closed for submission, copy still concurrent with later steps | seed 301: clean-to-mild (best run); seed 307: moderate drift — still not clean |
| **Decode-phase stores disabled** (`num_chunks = 0` while decoding; prefill stores remain) | **zero** decode-phase D2H | **fully clean, 2/2 canonical failing seeds flip to coherent 4k legs**; −15% steady throughput cost at C6/20k as the practical price |
| Tier disabled entirely | zero | clean (historical baseline) |
| Restore-path logger (`_maximal_prefix_lookup` hits) | n/a | **zero loads during decode** — restores exonerated |
| Descriptor-buffer lifetime audit | n/a | buffers recycled only after `end_event.query()` — event-gated, safe on paper |

Interpretation: every knob that *reduces* the concurrency between the batched
driver copy and graph replay *reduces* corruption probability; only zero
overlap is fully clean. That dose-response pattern, with content and lifetime
causes excluded, points at the `hipMemcpyBatchAsync` ↔ graph-replay
interaction itself (driver level), not at vLLM logic.

## 6. Why the barrier does not save you

The transfer stream's `wait_stream(compute)` (gpu_worker.py:655) provides
ordering against compute-stream work *submitted before the store*. But stores
are deliberately deferred to the *next* step (`prepare_store_kv`, worker.py),
so at submission time the compute stream is about to (or already) replay a
captured graph writing the same KV pages the copy reads. A correct
implementation would either serialize the copy against the replayed graph or
be safe against concurrent writes to the *source* of a D2H copy; the observed
corruption indicates the batched driver path violates one of these on
gfx908/ROCm 7.1.

## 7. Reproduction

1. Boot vLLM on MI100 (TP4) with: CPU offloading connector enabled, any
   Eagle-family speculative decoding (e.g. NS=6), CUDA graphs on, int8
   block-quantized KV cache.
2. Send 20k-token prompts, `max_tokens=4096`, `temperature=1.0`,
   `top_p=0.95`, `top_k=20`, several seeds.
3. Detokenize the committed token ids and inspect the second half of each
   leg. Expect collapse in a majority of legs; onset varies per seed.
4. Control A: restart with decode-phase stores disabled
   (`storable_chunks() → 0` while `num_offloadable_tokens >
   num_prompt_tokens`). Same seeds now finish clean.
5. Control B: disable CUDA graphs (`--enforce-eager`) and re-run — expected
   discriminator between "batched driver copy races any concurrent compute"
   vs. "specifically graph replay" (not yet isolated on our side).

Minimal repro (driver-focused, engine-free) would issue, from two streams:
one stream replaying a captured graph that writes a large buffer in a loop,
the other repeatedly calling `hipMemcpyBatchAsync` D2H on overlapping source
ranges — then checksum the written buffer.

## 8. Workarounds (in-tree today)

* **`VLLM_OFFLOAD_NOSTORE_DECODE=1`** — zero decode-phase stores; clean
  output; ~−15% steady throughput at C6/20k (prefill-phase stores and
  prefix-cache reuse still work).
* `VLLM_OFFLOAD_SUBMIT_AT_FINISH=1` — submit stores after the step
  completes; reduces, does not eliminate.
* `VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS=1` — single-descriptor calls;
  reduces, does not eliminate.
* Disabling the offloading tier entirely — clean, larger capacity cost.

## 9. Suggested fixes

1. **Near-term (vLLM-side, full speed):** route GPU→CPU stores through a
   normal stream-ordered **Triton copy kernel** on the transfer stream (the
   tree already contains one: `vllm/v1/kv_offload/cpu/swap_blocks_triton.py`)
   instead of the driver batch API, behind a flag
   (`VLLM_OFFLOAD_TRITON_STORES=1` in our tree) or as the ROCm default.
   A regular kernel launch participates in stream ordering and does not use
   `hipMemcpyBatchAsync` at all.
2. **Upstream (AMD):** investigate `hipMemcpyBatchAsync` interaction with
   concurrent graph replay on gfx908/ROCm 7.1 — source-page reads racing a
   replayed graph's writes must not corrupt device memory.

## 10. Confidence statement

Proven (high confidence): decode-phase *stores* are the trigger; loads,
stored content, and descriptor recycling are exonerated; severity scales
monotonically with store/replay overlap; zero-overlap is fully clean.
Inferred (moderate-high confidence): the fault lies in the driver's batched
copy path rather than in vLLM bookkeeping — this is the remaining hypothesis
after the alternatives were excluded, but a driver-only minimal repro
(§7.5) has not yet been produced on our side.
