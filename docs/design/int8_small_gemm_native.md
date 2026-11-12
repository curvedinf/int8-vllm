# int8 native small-kernel system (gfx908)

Status: stage 0 complete; stage 1 kernel VALIDATED. The serving failures
were a broken comparison harness (mutated kwargs) and the Hadamard rotation
collapsing DFlash2 acceptance, not the kernel: offline Triton vs native on a
dumped real call agree to 1.6e-2, and served C12 cold the native arm is
60.1-61.0 ms/step vs Triton 65.3-68.1 at acceptance 1.53 vs 1.71-2.09.
Dispatch stays opt-in (`AITER_INT8_NATIVE=ua_decode`); rotation now defaults
OFF (`VLLM_GFX908_KV_ROTATE=1` to enable) because it drops acceptance to 1.0
until the draft KV-injection path rotates consistently.
Owner: serving fork (`curvedinf/int8-vllm` + `curvedinf/int8-aiter`).
Scope: int8 only — there is no non-int8 arm in this design.

## Why this exists

Every int8 MMA needs both operands in MFMA fragment layout. Two ways to get
there:

* absorbed by the kernel's load path — CK/asm kernels read preshuffled weights
  and stage activations through swizzled LDS; conversion is paid once at load
  or hidden behind memory latency;
* emitted as separate ops around the dot — Triton. `tl.dot` allocates LDS,
  remaps the blocked layout, then feeds the MFMA, on every tile of every call.

The fork's decode band runs Triton: `aiter_w8a16.py` sends M ≤ 128 to
`gemm_a8w8_blockscale` (plain `[N,K]` int8 weights), and M > 128 to
`gemm_a8w8_CK`. So attention, the dense/shared-expert GEMMs, the lm_head, and
every draft GEMM pay the conversion; routed MoE experts and prefill GEMMs
(CK asm) do not.

Measured for UA decode at 16k context, per layer:

| component | time | rate |
|---|---|---|
| loads + softmax | 147 µs | 0.68–1.5 TB/s in isolation |
| QK int8 dot | 139 µs | 2.9 TFLOP/s |
| PV dot | 189 µs | 2.1 TFLOP/s (fp16) |
| full kernel | 482 µs | practical roofline 54.5 µs (937 GB/s) |

Per-K-tile conversion moves ~`(M_tile + N_tile)·K_tile` elements against
`M_tile·N_tile·K_tile` MACs, so overhead scales as `1/N_tile + 1/M_tile`:
~12.5% for the 16×16 attention tile (measured effect is much larger because
the moves are LDS/shuffle instructions), ~1.8% for a 96×128 dense GEMM, ~7%
for a 12-row draft GEMM, negligible for prefill. That ranking sets the work
order.

## Principle

**The fragment layout is the contract.** A producer writes data in the order
the consumer's fragments need, or the consumer's own load path does the
swizzle; no int8 dot is ever fed by a standalone layout conversion.

For weights this means preshuffle at load (one-time). For activations and the
KV cache — rewritten every step — it means the kernel loads directly into
fragments.

## Architecture

```
aiter/csrc/kernels/int8_mfma_lane.cuh            shared primitives (this stage)
aiter/csrc/kernels/int8_mfma_selftest.hip        bit-exact primitive self-test (this stage)
aiter/csrc/kernels/paged_attn_decode_int8_pthead_gfx908.hip   stage 1
aiter/csrc/kernels/gemm_a8w8_m16_gfx908.hip      stage 2
aiter/ops/int8_native.py                         loader + dispatch registry + gates
docs/design/int8_small_gemm_native.md            this file
```

Layer 0 — `int8_mfma_lane.cuh`

* `v_mfma_i32_16x16x16i8` lane maps, documented per operand and verified by
  the self-test: A `row = lane%16, k = 4·(lane/16)..+3`; B `col = lane%16,
  k = 4·(lane/16)..+3`; C `col = lane%16, row = 4·(lane/16)..+3`.
* i8×4 pack/unpack, 16-byte LDS fragment load (`ds_read_b128`) for one row's
  four k-steps, C-row store, per-group scale application in the C layout.
* Every primitive is covered by a host-side bit-exact check; a lane-map
  regression fails in under a second, not in a benchmark.

Layer 1 — producer adapters (compile-time traits)

* int8 per-token-head KV (`[nb, 64, 1, 128]` int8 + fp16 per-token scales);
* GPTQ GS128 weights (`uint8b128` payload + group scales) — for the GEMM
  kernels the weights are preshuffled once into fragment order;
* activation int8 from `ACT_QUANT=round` / fused epilogues, row-major with
  per-group scales.

Layer 2 — kernels (one file per call site, all sharing Layer 0)

* attention decode: partition kernel (QK MFMA, scale, online softmax in the
  C layout, P requantized in place, PV MFMA) + segment reduce;
* small-M A8W8 GEMM: `BLOCK_M` 16/32, `BLOCK_N` 128, K loop over GS128 groups
  so scales are applied per group; optional int8 payload epilogue (the
  `_gfx908_payload_out` contract);
* grouped variant for routed experts only if the CK measurement shows headroom.

Layer 3 — dispatch

`aiter.ops.int8_native` owns every gate in one place:

* `paged_attn_decode(...)`: gfx908, int8 per-token-head KV, hdim 128, GQA
  template (Flint target 8, draft 4), `max_seqlen_q ≤ 16`;
* `gemm_a8w8(...)`: gfx908, M ≤ 128, K % 128 == 0, group size 128;
* any predicate failure falls back to the existing Triton/CK op unchanged;
* one env switch, `AITER_INT8_NATIVE` (`1`/`0` or a comma list of sites), for
  A/B and rollback;
* each dispatch logs `site:impl` once so serving logs say what ran.

Layer 4 — gates

`scripts/bench_int8_native.py` holds one row per (site, shape) and reports
% of practical bandwidth and TFLOPS against the same roofline model for every
consumer. Accepted changes get an A/B ledger row; serving changes also need
the acceptance gate.

## Flint inventory (49 MoE layers, hidden 2048, topk 6, 256 experts)

| site | shapes (M × N × K) | current | target |
|---|---|---|---|
| UA decode QK/PV | 96 q × 32 h, 16k ctx | Triton UA | stage 1 native |
| qkv_proj | 96 × 5120 × 2048 | Triton blockscale | stage 2 native |
| o_proj (int8 payload out) | 96 × 2048 × 4096 | Triton blockscale | stage 2 native |
| shared gate_up (gated out) | 96 × 2048 × 2048 | Triton blockscale | stage 2 native |
| shared down (payload out) | 96 × 2048 × 1024 | Triton blockscale | stage 2 native |
| lm_head int8 | 96 × 150144 × 2048 | Triton blockscale | stage 3 native |
| routed experts | ~2.3 rows/expert × 1024/2048 × 2048/512 | CK asm fmoe | measure, likely keep |
| draft attention + GEMMs | M = 12, same N/K family | Triton UA / blockscale | inherits stages 1–2 |
| prefill GEMMs | M ≥ 2048 | CK asm | keep (roofline says fine) |

## Staging plan

Stage 0 (done): header + self-test green on gfx908.

  .venv/bin/python scripts/int8_native_selftest.py
  -> {'lane_maps': True, 'lds_row16': True, 'token_reduce': True}

`lane_maps` pins A/B/C; `lds_row16` pins the `ds_read_b128` fragment load
(a lane uses field `quad` of the 16-byte row load); `token_reduce` pins the
four-local-adds + xor(16,32) softmax reduction.


Stage 1: UA decode native kernel — first version landed.
`aiter/csrc/kernels/paged_attn_decode_int8_pthead_gfx908.hip` (partition +
reduce, int8 QK and PV, transposed orientation, 64-token LDS tile, 16
queries per wave, 4 waves sharing the tile).
Correctness: 0.0015 max abs vs an fp32 reference over the same quantized
cache; 1/2/8 segment splits agree.
Microbench, 12 seqs x 8 queries, int8 PTH, per layer:
  2k:  Triton 76.4 us -> native 66.9 us (1.14x)
  16k: Triton 440.6 us -> native 286.6 us (1.54x)
Fixed cost ~40 us/layer is three launches (pack + partition + reduce).
Next: fuse the Q pack into the partition kernel, replace the V byte-scatter
staging with a register transpose, then wire the dispatch behind
`AITER_INT8_NATIVE=ua_decode` and run the serving gate.

Stage 2: small-M A8W8 GEMM, first as an isolated GEMM A/B at M = 12 and 96
against Triton blockscale and against CK bpreshuffle, then wire the winner
per site. Gates: isolated ≥10% before any wiring; end-to-end TPOT; acceptance
unchanged.

Stage 3: int8 lm_head at M = 96/12 (N = 150144, weight-bound) — same kernel
with a larger N loop; preshuffled weights.

Stage 4: routed experts only if the CK fmoe measurement shows headroom;
grouped-GEMM traits reuse Layer 0.

Stage 5: prefill — only after a roofline measurement; no work before that.

## Non-goals and open questions

* No dtype other than int8. No bf16/fp16 arms, no "fallback dtype" rows. The
  recipe scripts now hard-fail if `KV_DTYPE`/`DRAFT_KV_DTYPE` select anything
  that is not int8; 16-bit KV cannot be reached by a lever.
* No KV cache format change in stage 1: the QK fragment is already
  token-major/contiguous; V transpose is handled in-kernel (byte permute) and
  revisited only if measurement demands an SoA sidecar.
* CK bpreshuffle is an existing lever for the GEMM band; if it wins at
  M = 96/12 the native GEMM kernel may only be needed where the payload
  epilogue or the draft's M = 12 shapes make CK unsuitable.

## Stage 2 v1 (2026-11-12): correct, not yet competitive

`aiter/csrc/kernels/gemm_a8w8_m16_gfx908.hip` implements the small-M A8W8
GEMM: one wave per CTA, BLOCK_M x 64 output tile (four 16x16 B/C fragments
share the lane's A fragment), weights preshuffled into B-fragment order
(`wp[n_tile][k16][lane] = W32[n_tile*16+lane%16, k16*4+lane/16]`), GS128
group scales applied every 128 K, bf16 out. Template on BLOCK_M; the
launcher defaults to 16.

Correctness: rel error <= 3.2e-3 vs a float32 dequantized reference at
M/N/K = 8/64/256, 12/80/256, 20/64/384, 32/128/256, 96/1280/2048.

Timing (CUDA-graph replay, kernel time; `scripts/bench_native_gemm.py`):

| site | M | native BM16 | Triton | gap |
|---|---|---|---|---|
| qkv 1280x2048 | 96 | 67 us | 28 us | 2.4x slower |
| o_proj 2048x1024 | 96 | 40 | 28 | 1.4x |
| shared gate_up 2048x2048 | 96 | 70 | 44 | 1.6x |
| draft gate_up 12288x2048 | 96 | 251 | 154 | 1.6x |
| lm_head 37536x2048 | 16 | 133 | 133 | tie |

The gate (>=10% over Triton) is NOT met, so no site is wired and
`AITER_INT8_NATIVE` still exposes only `ua_decode`. The 1.4-2.4x gap is
consistent with BM16 reading the weights `ceil(M/16)` times vs Triton's
BM32, plus per-k16 load latency. BLOCK_M 32/64 regress (136 us at qkv
M=96) from register pressure (32-64 fp32 + equal int32 accumulators plus
the load batch); a 4-step load batch did not fix it.

Next for stage 2:
1. BLOCK_N=32 with BLOCK_M=32 (2 n-tiles): matches Triton's 3x weight
   traffic at full 120-CTA occupancy and fits in registers (~70).
2. Coalesced A staging: the current per-lane A load is a 16-sector gather
   per k16; stage a row-major A tile through LDS with a padded transpose.
3. Split-K (4) for the narrow-N shapes, with an fp32 partial buffer and a
   convert kernel (the Triton configs use NUM_KSPLIT=4 there).
4. Only then re-run the per-site gate and wire winners behind
   `AITER_INT8_NATIVE=gemm_m16`.

## Stage 2 v1.1 (2026-11-12): BLOCK_N and split-K

Added a BLOCK_N template (16/32/64/128), KSPLIT (1/2/4) with an fp32 atomic
partial buffer plus a convert kernel, and a tile sweep in
`scripts/bench_native_gemm.py`. Split-K is numerically correct (rel
2.6e-3 vs the float reference for 32x64k4 and 32x128k4).

Best measured per site (CUDA-graph replay, vs the production Triton call):

| site | M | best native | Triton | gap |
|---|---|---|---|---|
| qkv 1280x2048 | 16 | 37.0 us (16x64k4) | 31.7 | -17% |
| qkv 1280x2048 | 96 | 46.1 us (16x64k4) | 29.2 | -58% |
| o_proj 2048x1024 | 16 | 27.9 (16x32k1) | 20.5 | -36% |
| o_proj 2048x1024 | 96 | 35.3 (16x32k1) | 26.3 | -34% |
| lm_head 37536x2048 | 96 | 888 (32x64k2) | 505 | -76% |

No configuration clears the gate. The 32x64k4 case matches Triton's weight
traffic (3x) and CTA count at M=96 yet runs 54 us vs 29 us, so the residual
is per-CTA efficiency, not tiling: the A operand is a 16-sector gather per
k16 and there is no cross-group software pipeline (Triton stages K). Next
iteration must add double-buffered staged loads and a pipelined K loop
before re-running the gate; the kernel stays unwired.
