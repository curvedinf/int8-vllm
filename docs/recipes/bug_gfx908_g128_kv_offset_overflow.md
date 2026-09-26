# G128 attention corruption after deep-context traffic

The 2026-09-26 persistent-garble incident has a reproducible software defect:
the gfx908 Gluon attention cores multiply an **int32 physical block ID** by
the KV page stride **in int32**. The multiplication overflows before the
offset reaches the 64-bit pointer. A valid block table then causes a read of
unrelated device memory.

The fix casts the loaded block ID to `gl.int64` **before multiplication** in
all five G128 cores: target 64-row and 32-row decode, DFlash2 draft verify,
target prefill, and the optional packed prefill. KV data remain INT8 and
scales remain FP16. This is an address-width correction.

## Minimal proof

For the production target layout, a page contains 1664 tokens of 520 bytes:

```text
page stride                  = 865,280 bytes
block 2481 * page stride      = 2,146,759,680  (fits int32)
block 2482 * page stride      = 2,147,624,960  (overflows signed int32)
```

A guarded GPU allocation allows the wrapped address to read known foreign
bytes without causing an illegal-address fault. The logical K/V and scales
are identical at each tested block. V is 30 and its FP16 scale is approximately
0.01, so the output should be approximately 0.30.

| Block ID | Generic reference | Gluon before fix | Gluon after fix |
|---|---:|---:|---:|
| 2 | 0.30078125 | 0.30078125 | 0.30078125 |
| 2481 | 0.30078125 | 0.30078125 | 0.30078125 |
| 2482 | 0.30078125 | **0.05004883** | 0.30078125 |
| 2600 | 0.30078125 | **0.05004883** | 0.30078125 |

Before/after output is preserved in
`logs/garble/attn_offset_{before,after}_20260926.log`. The regular Triton
attention path already widens block IDs and does not show this error.

The committed regression suite uses random K/V and an FP32 SDPA oracle,
covering all five cores at block IDs 1, 2481, 2482, 4964, and 5000. The latter
two also cross the **4 GiB byte boundary**, where FP16 scale offsets overflow
their own int32 element addressing. It passed **25/25** on MI100.

## Why the previous investigation missed it

The trigger is the physical page offset, rather than an exact prompt length.
Several deep conversations leave cached blocks resident and drive subsequent
allocations to larger IDs. The incident captures include IDs above 4000.

Python readbacks and hashes use correct 64-bit indexing. They can establish
that the *intended* KV bytes are unchanged while the Gluon kernel reads an
entirely different address. Treating allocation-dependent block IDs as noise
removed the distinguishing input. Isolated tests using low block IDs also
never crossed the overflowing offset.

This explains boot-history dependence, corruption of fresh short prompts,
foreign-content contamination, lost draft acceptance, and the possible GPU
hang with another allocation layout. The earlier claim that the software
observable space was exhausted is withdrawn. Hardware or driver engagement
is not needed to reproduce this defect.

## Related regression findings

The broader oracle checks exposed two optional-core defects. Padding rows in
the 32-row decode and packed-prefill CTAs could write the next CTA's query
rows using a shorter key prefix. Their load, score, and store masks now
exclude those rows. The dispatcher also stops passing the 64-row core's MMA
arguments to the legacy 32-row core. Neither optional core is the deployed
64-row default.

The draft dispatch predicate previously evaluated grouped-scale attributes
even when scales were absent. Its guard now avoids that evaluation, restoring
the noncausal per-token-head INT8 micro test.

## Validation and deployment

Run the regression on an idle MI100 with at least 10 GiB free:

```bash
ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib \
PYTORCH_ROCM_ARCH=gfx908 GPU_ARCHS=gfx908 VLLM_TARGET_DEVICE=rocm \
HIP_VISIBLE_DEVICES=0 PYTHONPATH="$PWD:$PWD/../aiter" \
.venv/bin/python -m pytest \
  tests/kernels/attention/test_gfx908_g128_addressing.py -q
```

With the production server running and its API key in `VLLM_API_KEY`, the
repeatable serving gate is:

```bash
.venv/bin/python scripts/repro_garble_long_context.py \
  --target-tokens 210000 --conversations 3 \
  --out logs/garble/repro_long_context.jsonl
```

It fills documents with independent seeds 7000, 9000, and 11000 using the
original `--turn-tokens 16384` filler setting, continues the first document
after the other two, and checks 36 greedy
completion probes plus a fresh chat on the same boot. The incident detector
expects ` Paris` with logprob above -1.0; the healthy checkpoint is -0.54635.
The complete continuation text is logged for coherence review.

An interleaved low-offset timing comparison loads the old core from git into
the benchmark process. It uses identical C6/32k, seven-query inputs that are
correct on both versions, and checks bit-identical outputs before timing:

```bash
ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib \
PYTORCH_ROCM_ARCH=gfx908 GPU_ARCHS=gfx908 VLLM_TARGET_DEVICE=rocm \
PYTHONPATH="$PWD:$PWD/../aiter" .venv/bin/python \
  benchmarks/kernels/benchmark_gfx908_g128_address_width.py \
  --baseline-ref 7d36b7a2da --pairs 5 --iters 100
```

The INT8 micro test passes, including noncausal attention, and the gfx908
battery passes **4/4**. The production soak passed on one boot:

| Check | Result |
|---|---|
| Independent conversation A | 219,063 prompt tokens |
| Independent conversation B | 218,519 prompt tokens |
| Independent conversation C | 218,742 prompt tokens |
| Fixed greedy probes before/after fills and continuation | 36/36 identical: ` Paris`, logprob -0.5463467240333557 |
| Old-prefix A continuation | 219,094-token prompt; readable reasoning through the 256-token output limit |
| Fresh chat after all deep traffic | Correct capital and count: Paris; 1, 2, 3, 4, 5 |

The old-prefix request reached its reasoning-token limit before producing a
final summary; its logged output remained readable. Full traces are in
`logs/garble/repro_offsetfix_soak_20260926.jsonl`.

Five interleaved timing pairs measured **812.13 us before / 812.54 us after**
(+0.05%) with bit-identical low-offset outputs. These are isolated core-call
timings. Raw pairs are in `logs/garble/address_width_ab_20260926.log`.

The subsequent C6 serving smoke completed **32/32 requests**, generated
32,000 tokens in 111.79 seconds (**286.25 output tok/s**), and measured mean
draft acceptance length **4.575** (59.59% of draft tokens accepted). This is
a single patched-server measurement after the deep soak. Results are in
`logs/garble/bench_offsetfix_c6.json`. A short run of the committed repro
harness also passed afterward, including six unchanged logprob probes and
a correct fresh chat.

The current canonical README and launcher specify TP4/C6/NS6; this validation
preserves that deployed recipe, the INT8 target and draft KV caches, W8A8
GEMMs, CUSTOM all-reduce, and FP32 Mamba state. Graphs, prefix caching, and
CPU offload remain enabled. Diagnostic bypasses and tensor-dump instruments
are disabled.

The kernels are JIT compiled. Restart the server to use the changed source;
no C++ rebuild or sibling AITER source change is required.

The [audit of the prior investigation's retained changes](garble200k_prior_agent_audit.md)
removes its temporary tracing and unneeded block-pool assertion, retains
the diagnostic NO_LOADS miss-semantics correction, and corrects the earlier
Mamba-store conviction.
