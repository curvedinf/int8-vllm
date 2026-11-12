# Flint + Flint-DFlash2 Int8 PTQR Recipe (gfx908 / 4x MI100)

Peer recipe to the [Qwen3.8-27B baseline](README.md) and the
[Nemotron Bolt + LiLiCorr recipe](nemotron_bolt.md). Same int8-native
doctrine: every dense GEMM is AITER W8A8 INT8 GS128, routed experts run a
true W8A8 grouped kernel, and KV is `int8_block_g128` on **both** target and
draft (int8 data + one fp16 scale per 128-dim group, 16-byte-aligned rows
since 2026-11-12; `int8_per_token_head` remains the rollback lever). There
is no 16-bit fallback in this recipe.

- Upstream target: [`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint)
- Upstream draft: [`incoai/Flint-DFlash2`](https://huggingface.co/incoai/Flint-DFlash2)
- Serving script: `scripts/serve_recipe_flint.sh` (`start|stop|restart|status`;
  `EAGER=1` disables graphs, `PROFILE_DIR=<dir>` exposes the torch-profiler
  `/start_profile` + `/stop_profile` endpoints)
- systemd unit: [vllm-flint.service](../../scripts/vllm-flint.service)
  (see [service installation and selection](services.md))
- Ledger / experiment record: `agents/scratch/flint-int8-recipe.md`

## Model shape

49 layers (dense MLP layers 0-1, 47 MoE layers), 256 experts top-6,
hidden 2048, MoE intermediate 512, 32 heads / 4 KV heads (8 q / 1 kv per TP4 rank), head_dim 128,
vocab 150144, context 262144. The draft is a 5-layer DFlash2 block
(`block_size` 8, selector rank 256, top-k 16) consuming EAGLE3 auxiliary
states from target layers (6, 16, 27, 37, 47).

## Checkpoints (PTQR, GPTQ INT8 GS128)

Both checkpoints are PTQR retrained, not one-shot GPTQ: weights and
group scales were trained against the bf16 teacher on serving-shaped
4096-token windows, then exported to the standard GPTQ layout
(`qweight`/`qzeros`/`scales`, no `g_idx`). `config.json` carries the
`quantization_config` block — vLLM 0.33 detects the quant method from
there, not from `quantize_config.json`.

- Target (recipe default): `Flint-GPTQ-INT8-W8A8-GS128-PTQR-r1s40-512k` —
  rung 1 via `scripts/ptqr_flint_rungs.sh 1` (seq 2048, kd_cache 20,
  60 steps, lr 1e-5), step-40 save exported; trained on the extended
  corpus with eight 512k-token needle documents (7.07M tokens). 35
  shards, 36,439 tensors, 38.5 GiB. Same-seed A/B vs r1s40-v2:
  TPOT 71.75 vs 75.56 ms, acceptance 4.88 vs 5.10 (-0.22, within the
  +-1 gate). Export: `scripts/ptqr_export_flint.py`; trainer
  `scripts/ptqr_train_flint.py`.
- Draft: `flint-dflash2-ptqr-r1-int8` — **int8 PTQR, same structure as
  the deployed Qwen int8 draft**: all 35 attention/MLP linears are GPTQ
  int8 GS128 (W8A8) and the 12 dense/conv/selector surfaces keep the
  trained grid in bf16 (the identical exception set as Qwen's PTQR
  draft: 35 GPTQ linears + 34 float weights). 151 tensors, 518 MB
  (bf16 source: 805 MB). Export:
  `scripts/ptqr_export_flint_draft.py`; trainer
  `scripts/ptqr_train_draft_flint_ptqr.py`; KD cache
  `scripts/ptqr_flint_draft_cache.py`.

Dequant audit vs the trained masters: dense linears 0.66 % mean
(0.70 % worst), identical to the target export's class; bf16-grid
surfaces 0.16 %; untouched tensors byte-identical to the bf16 draft.

## Int8 engine surface

| Surface | Path |
|---|---|
| Dense linears | AITER CK W8A8 INT8 GS128 (`AiterW8A16LinearKernel` legacy name dispatches W8A8). Deployed decode band `65<M<=128` (C12 with NS=7) runs the tuned Triton `a8w8_blockscale` off the retained GS128 originals instead (`VLLM_GFX908_DECODE_BLOCKSCALE=1`, bm=32/bn=128, NUM_KSPLIT=4 for N<=1536 else 1, except mlp down_proj which stays CK); prefill (M>128) stays CK. Measured: qkv 95.3->34.7us end to end, shared-expert gate_up 68.3->26.8us. The shared expert's fused gate_up+SwiGLU GEMM (`gemm_silu_quant_out`) picks bm=16/warps 8/stages 2 for m<=128 (aiter cd53c11c6d): at the decode shape M=96, 2N=2048, K=2048 the old bm=64 launched 16 CTAs and measured 143-334 us; the new tile launches 96 and measures 53-68 us. |
| Routed experts | `vllm/model_executor/layers/quantization/gptq_moe_a8w8.py` — GPTQ int8-group → AITER grouped `a8w8_blockscale` (`[E,K,N]` int8 + per-column GS128 fp32 scales, `w_blockscale_n=1`). Decode-band tiles: gate_up bn=256/warps 8/stages 2 (186 us/layer, 60% of HBM peak — wave-quantization limited, 256 tiles over 120 CUs); down bn=256/warps 8/**stages 1** (96.9 us/layer, 715 GB/s, was 154.0 at bn=128/stages 2 — one stage keeps the 32 KB weight tile so two CTAs are resident per CU). The block is pure weight streaming: the int8 dot for the whole 2048-tile down grid is 16.6 us. Bench: `scripts/bench_flint_moe_kernel.py` (graph-timed, production kwargs; one config per process because a bad tile faults the device). |
| Attention | AITER unified attention. The 3D-decode launch heuristic (2026-11-12) picks warps/stages by geometry: nq/kv=8 (target, BLOCK_Q=2) -> 1 warp / 1 stage, nq/kv=4 (draft, BLOCK_Q=4) -> 2 warps / 1 stage, versus Triton's default 4 warps. Bit-identical (warps partition the output tile, no cross-warp reduction; fixed-prompt hash unchanged) and measured -19..29% of the attention family in-graph (25.51 -> 18.20 ms per 3 decode steps), C12 median 482 -> 515. `VLLM_GFX908_ATTN_WARPS`/`_STAGES` still override. Since 2026-11-12 the single-group block-g128 path (Flint's head-128 target+draft geometry) keeps K/V **int8 through both dots**: QK with Q RN-quantized per row (`G8_INT8_DOT`, int32 S accumulator, q row scale × k group scale folded post-dot) and PV with P requantized per tile against the tile's largest V group scale (`G8_INT8_PV`). No float matmul remains in the forward. PV costs +59 us/step net on the pair (draft −103, target +162) and lands acceptance back at the float-dot baseline. Levers: `VLLM_G8_INT8_DOT`, `VLLM_G8_INT8_PV`, `VLLM_G8_INT8_DOT_CAUSAL`. Prefill runs `VLLM_UA_PREFILL_BLOCKM=128` (BLOCK_Q=16): 2.85k -> 4.73k tok/s on a 2048-token chunk, bit-identical; the prefill-env gate is `max_seqlen_q > 64`, because `> 1` also matches the spec-decode verify batch (q=8) where BLOCK_M=128 collapses decode. |
| KV (target + draft) | `int8_block_g128` (int8 data + one fp16 scale per 128-dim group). Since 2026-11-12 each half's data row is 16-byte aligned (144 B, content 288 instead of the natural 130/260): the natural padding left alternate K/V rows misaligned and the 2D non-causal draft kernel de-vectorized. Measured after alignment (at the production draft batch, tokens=96): draft graph 7.95 -> 6.22 ms, C1 step 32.4 -> 29.5 ms, W1 265.2 / W2 279.1 agg, acceptance unchanged; KV pool 4.42M -> 3.99M tokens (12 x 512k was already above the pool). Rollback lever: `KV_DTYPE=int8_per_token_head`. |
| Prefix cache (hybrid) | Mamba prefix caching **on** (`MAMBA_MODE=align` default; the deprecated `--mamba-cache-mode` now reaches `CacheConfig`). Recurrent checkpoints land on 128-token block boundaries; measured warm-cache TTFT 205-232 ms (was 1165 ms without it) and 269-548 ms on a partial-prefix hit (was 3519 ms). Sub-block (partial) Mamba hits are not reachable: they need every prefix-cacheable group to support fine-grained lookup and the attention groups use 128-token blocks. |
| All-reduce | vLLM CUSTOM (`VLLM_ROCM_USE_AITER_CUSTOM_AR=0`); residual-stream collects use `residual_g128_all_reduce` on int8 codes+scales |
| Act quant | `VLLM_GFX908_ACT_QUANT=round` |
| Residual stream | int8 GS128 payloads between every decoder layer (`residual_int8`), fused residual+norm+quant seams |
| Attention/input norm prologue | `residual_norm(emit_pertoken=True)` stashes the CK per-token quant; the CK GEMM consumes `PREQUANT_ATTR` |
| MoE gate_up | fused SwiGLU + per-128 int8 epilogue in the grouped GEMM (`QUANT_OUT`, interleaved gate/up weights) |
| MoE down/combine | `reduce_grouped` quant-out: shared branch folded as an int8 residual, combined row quantized per 128; runner float AR disabled, one int8 collective | The shared expert runs on a side CUDA stream and overlaps the routed gate_up (`VLLM_GFX908_SHARED_OVERLAP=1`, -4.5ms/step).
| MoE routing | The router's top-k ids/weights are consumed directly (`VLLM_GFX908_ROUTING_REUSE_TOPK=1`, default): the synthetic-logits matrix and the second bitmatrix top-k that recovered them are gone (~70 us + fill/scatter per layer). Expert-id sets are identical; the slot order is a permutation and the weights differ by 1 ulp from `tl.softmax`'s reduction order, so it is a phase-0 numerics change (fixed-prompt hash differs) decided before the next PTQR. `=0` restores the reconstruction. |
| Dense/shared/draft MLP | `gemm_silu_quant_out`: SwiGLU + GS128 int8 output from resident blockscale weights |
| Router gate, lm_head | int8 GS128 at all batch sizes; norm epilogues hand both consumers their per-group stash. The gate GEMM uses the fork's tuned `a8w8_blockscale` config (bm=16/bn=128/ksplit=4 at M=96: 70.5 -> 20.4 us, 47 calls/step, same output to 2.4e-7); `VLLM_GFX908_GATE_UNTUNED=1` restores aiter's heuristic. |
| Embedding | int8 table, gather-dequant |

Activation doctrine (2026-11-09): every stored activation between
quantizable ops is int8; the only float math is attention
(softmax/QK^T/PV, identical to the Qwen and Bolt recipes) and the logits
interface. Selector codebooks, the draft's conv planes and `fc` are int8
at load. The draft's 35 attention/MLP linears are int8 W8A8 GS128.

The expert path is the production path. `VLLM_GFX908_MOE_A8W8=0`
(reverting to WNA16/A16W8) is a diagnostic lever only. The startup log
must show:

```
gfx908 routed experts: GPTQ int8-group -> AITER grouped a8w8-blockscale (W8A8)
```

Padding/dummy batches carry expert id `-1`; the routing reconstruction
zero-weights those slots. Without that fix the fused routing kernel
faults at CUDA-graph capture (fixed 2026-11-08).

## Serving

```bash
scripts/serve_recipe_flint.sh start    # TP4, C8, port 8020
scripts/serve_recipe_flint.sh status
scripts/serve_recipe_flint.sh stop
```

Recipe contract: TP4, `--max-num-seqs 12` (C12), bf16 compute dtype,
`--max-model-len 524288` (512k per-request cap), `--max-num-batched-tokens
2048`, `--gpu-memory-utilization 0.90`, `int8_per_token_head` KV, DFlash2
draft `NS=7` with its own `int8_per_token_head` KV, FULL_DECODE_ONLY graphs.

512k preparation (2026-11-08): the checkpoint documents 262,144 tokens,
while the release announcement states 512k support (recommended
<=256k). The recipe sets `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` and both the
target and draft extend their rope tables to the runtime length; only
the 513-token sliding layers rotate, so positions past the trained range
stay local. The 512k/C12 boot is live (2026-11-09): `--max-num-seqs 12`,
`--max-model-len 524288`, int8 KV on both models, DFlash2 NS=7. The
numbers below are the C8/262k legs; the C12/2048-token measurements are
in the int8-only section at the end.
`VLLM_FS_GPU=auto` (AIS direct-to-GPU) is the default; the first boot
writes a 41 GiB per-rank repack cache (~15 min), later boots load from
it (~4.5 min). `VLLM_FS_GPU=0` is the plain-safetensors fallback.

At `--gpu-memory-utilization 0.90` with a 262k cap the int8 KV pool held
**4,476,935 tokens (~4.48M shared GPU context)**; the 512k rope tables
cost ~0.27 GB/GPU, so expect ~4.4M tokens at the prepared cap (re-measure
at the first 512k boot). At C12 that is ~366k tokens per stream; twelve
full 512k streams would need 6.29M tokens and exceed the pool, so the
scheduler queues/preempts. There is no CPU KV tier in this recipe.

## Gates (2026-11-08, C8 / 262k configuration)

- Serving NLL (int8 export, v2): 4.6588 @64 tokens, 1.3898 @1024
  (bf16 reference 4.6590 / 1.39, teacher 4.6586).
- Serving-gate KLD (1024-token val window): int8 export **0.0095**
  median (bf16 floor 0.0055); int8 KV 0.0103; int8 KV + W8A8 experts +
  graphs 0.0110.
- Draft acceptance (TP4/C8, int8 KV, NS=7, random 32/128): int8 PTQR
  draft **2.25** vs bf16 draft 2.27 (Δ −0.02; rate 17.93 % vs
  18.21 %). TPOT 30.0 ms both.
- C8 throughput (random corpus): 32 in / 256 out, 64 reqs →
  **261.1 output tok/s**, TTFT 627 ms, TPOT 27.2 ms, acceptance 2.49
  (target-only same box: 171.7 tok/s). 2048 in / 64 out → TTFT 6.20 s
  (≈5.3k prefill tok/s), TPOT 44.1 ms.
- AIS byte gate: 38.50 GiB verified byte-identical across 109,620
  chunks; target + draft load through AIS.

## int8-only fusion state (2026-11-09)

The MoE block, dense/shared MLP and draft MLP are end-to-end int8:

- Gate_up: interleaved gate/up weights feed the grouped GEMM's fused
  SwiGLU epilogue with per-128 int8 output (bit-exact contract vs the old
  `silu_mul_quant_rn` chain: 2e-5 code mismatch from accumulation order,
  scales exact).
- Down/combine: the reduce epilogue folds the shared branch as an int8
  residual and quantizes the combined row; the runner's float all-reduce
  is disabled (`skip_final_all_reduce`) and the payload is TP-reduced in
  the int8 domain.
- Dense/shared gate_up: resident `[2N,K]` int8 weights + `gemm_silu_quant_out`
  (SwiGLU+GS128 in one kernel); the row-parallel down emits a payload.
- Measured on the retrained target: boot healthy, coherent; steady C12
  2048/256 TPOT 66.45 ms / 172.3 tok/s (same-seed A/B vs the previous
  target 75.56 ms); staggered 6+6 TPOT 56.9/62.3 ms preserved.
- **NS sweep (12x 2048/256, seed 91131)**: NS=5 69.44 ms / 161.7 tok/s
  (len 4.14), **NS=7 66.45 ms / 172.3 tok/s (len 4.95)**, NS=9 88.88 ms
  / 147.9 (len 6.86), NS=11 102.37 ms / 134.2 (len 6.74). NS=7 stays the
  recipe default. Ledger: FLINT_R1S40_512K_RETRAIN_20261109,
  FLINT_NS_SWEEP_C12_20261109.

## Known deltas

- Tool calling: the template's XML `<tool_call>{name}<arg_key>k</arg_key>
  <arg_value>v</arg_value></tool_call>` format is GLM-4.7's, so the recipe
  boots `--enable-auto-tool-choice --tool-call-parser glm47`
  (`TOOLPARSER=""` disables it). Verified end-to-end: an `auto` tool call
  parses to JSON arguments, and a streamed `<|observation|>` tool-result
  turn answers correctly.
- Reasoning: the template emits `<think>` blocks; the recipe uses
  `--reasoning-parser qwen3`.
- The draft export keeps the trained bf16-grid surfaces; re-quantizing
  them would need a new scaled kernel (same exception as Qwen).

## Current performance state (2026-11-12, aligned block-g128)

Shipped recipe: aligned `int8_block_g128` KV on both models (144-byte
rows), widened decode band (`16<M<=128`), g8 scale broadcast, AITER W8A8
everywhere, TP4 / C12 / NS=7. Harnesses: `scripts/bench_prod_flint.py`
(W1 agent-steady with a ~2.4k shared prefix + tools; W2 12 cold ~2.4k
prompts), the C1 step probe, and the g8 attention harnesses.

| measurement | value |
|---|---|
| W1 C12 aggregate / per-stream | **328.9-382.2 / 27.4-31.9 tok/s** across boots (float dots 284.1) |
| W1 acceptance / TTFT p50 | **2.415-2.466** / 0.46-1.78 s (float-dot paired probe 2.690) |
| W2 C12 aggregate / per-stream | **304.7-327.0 / 25.4-27.3 tok/s** |
| W2 TTFT p50 / p95 | 3.29-3.31 s / 8.5 s |
| C1 2.4k step (deterministic) | **26.4-27.3 ms** (float dots 28.8-30.0), acceptance 2.46-2.50 |
| Controlled C12 (12x256, shared prefix) | **532.3 / 534.9 tok/s** (interleaved A-arm medians; B arm with the gate+routing levers off 479.2 / 499.8) |
| 40k-token prompt | TTFT **8.05 s** (~5.15k tok/s prefill; 15.8 s with float dots), decode 86-106 tok/s |
| battery | 4/4 PASS |

The gate-config + routing-reuse pair is verified by an interleaved A,B,A,B boot
protocol (ledger GATE_ROUTING_INTERLEAVED_AB_20261112): +6.5% controlled C12 and
-1.5 ms C1 step, winning both pairs with no C1 overlap. Distant boots spread
+-10% (the cards sit at a manual DPM level set outside this repo) while adjacent
interleaved boots reproduce within ~1%; use A,B,A,B for sub-10% claims.

Block budget at C12 (all measured this session):

| block | ms | state |
|---|---|---|
| Routed MoE GEMM (9.46 GB/step) | 12.1 | 79% of HBM in the last profile; config verified optimal (CTA-bound, not traffic-bound) |
| Attention, target + draft (int8 QK+PV) | 6.0 | was 7.5-8.5 before the launch heuristic; 5% of its ~0.3 ms HBM floor, latency-bound at 96-192 CTAs |
| Residual GS128 all-reduce | 5.9 | ~98 calls/step at ~44-60 us; serial on the main stream, peer-rendezvous bound (block count is already at the 64 cap) |
| Blockscale dense GEMMs (qkv/o_proj/shared/dense/lm_head) | 5.6 | tuned configs, CTA-concurrency bound |
| Shared gate_up (`_gemm`, fused SiLU) | 3.8 | was 7.3 before aiter bm=16 |
| Routing (`topkGating` + `routing_experts_parallel`) | 2.4 | top-k reuse removed ~70 us/layer |
| Norm/quant glue (`_add_norm`, elementwise, materialize) | ~3.4 | 98 seams/step |
| Sampler / lm_head glue | ~1.5 | - |
| Engine<->worker RPC residual | ~27-40 at C1 | at C12 the GPUs are 95-97% busy, so this is GPU wait, not host; every hop individually <0.1 ms |

Kernel shares are from the last full decode-window profile (3 steps, rank 0);
that boot was in a ~10% slower machine state than the C12 medians above, so use
the ordering, not the absolute values. Prefill is a different mix: GS128
all-reduce 34.9%, MoE 17.5%, attention 13.6%, GEMMs 17.4% of a 2048-token chunk.

Optimization effect (2026-11-12, int8 dots + gate + routing + launch
heuristic + prefill tile): C1 step 28.8-30.0 -> 26.4-27.3 ms; 40k cold
TTFT 15.8 -> 8.05 s (prefill 2.5k -> 5.15k tok/s); draft attention 2x
at bf16 Q (481.5 -> 241.5 us/layer isolated); W1 284.1 -> 328.9-382.2
agg; controlled C12 428-439 -> 532/535 (interleaved A medians; B arm
479/500). Acceptance on the int8 arms spans 2.31-2.69
across boots on identical code (batch-composition-dependent sampling),
so no acceptance difference is established yet; the phase-2 PTQR rung
remains the clean way to anchor it. See ledger
G8_ATTN_ACCEPTANCE_CORRECTION_20261112.

Aligned-layout effect (natural 130 B rows -> aligned 144 B): draft graph 7.95 -> 6.22 ms at tokens=96; C1 step 32.4 -> 29.5-30.0 ms; W1
223-265 -> 265-284; W2 217-279 -> 279-308; KV pool 4.42M -> 3.99M
tokens. The engine step at C12 W1 is ~50-60 ms with the target graph
~7 ms: the step is dominated by the engine<->worker RPC response path,
not kernels. A/B ledger: `surface_experiments_ledger.jsonl`.

## Performance state (phase-1 cycle, 2026-11-10) — HISTORICAL



Measured under `optimization_procedure.md` with the W1 harness
(`scripts/bench_cycle.py`: agent-shaped system-prefix + short user turn,
tools present, served sampling temperature 1.0 / top-p 0.95 / top-k 20).
Three repeats per arm, median reported.

| Measurement (C12 W1, 3 reps) | median per-stream tok/s | acceptance | step ms |
|---|---|---|---|
| 2.4k ctx, `int8_per_token_head` (recipe default) | **41.3** (36.6-41.6) | 2.78 | 67-73 |
| 2.4k ctx, `int8_block_g128` (rollback) | 37.8 (32.7-38.1) | 2.72 | 74-79 |
| 18k ctx, `int8_per_token_head` | **34.5** (31.6-40.3) | 3.54 | 88-112 |
| 18k ctx, `int8_block_g128` | 26.1 (20.8-32.0) | 3.35 | 117-142 |
| W5 witness (single stream, ~3k prompt) | 63.1-67.0 | 2.6-2.8 | 41 |
| W2 cold, C12 12x2k | TTFT p50 7.4 s / p95 13.6 s; 22.4 | 1.82 | - |
| W3 long turn, C12 18k to 2048 | 27.1 | 3.17 | 117 (total p50 77.1 s) |

Phase-3 sweeps: NS 5/6/7/8 all land within ~10% of each other at served
sampling (NS=6 nominally best at 43.5/35.9, NS=7 fresh control 41.3/34.5),
below the boot-to-boot variance floor - NS stays 7. Draft-only KV layout
(`DRAFT_KV_DTYPE=int8_block_g128`) changes acceptance by <2%: both stay on
`int8_per_token_head`. `bench_cycle.py` reports the per-position acceptance
vector with every run.

Phase-1 isolation (forced rejection, acceptance pinned at 1):
C12 step 101.2/155.0 (`block_g128`), 80.0/117.8 (`per_token_head`) at
2.4k/18k. The `block_g128` vs `per_token_head` arms were same-session and
paired, so that comparison stands.
Single-stream C1 (forced rejection): 38.8 ms/step at ~3k context.

**CORRECTION (2026-11-11): the "bf16 KV is faster" claim is retracted**, and
the int8 KV decode path was then fixed. The claim came from comparing a
single `KV_DTYPE=auto` boot against int8 arms measured on other boots -
never paired (cross-boot variance 15-30%; auto selects the same backend).
Isolated, the int8 path was only 7-15% faster than bf16 because the gfx908
config forced warps=4 / stages=1 / heuristic segments and left the decode
grid latency-bound (~62 GB/s, 0.5% of peak). After the geometry fix
(warps=2 / stages=2 / 8 segments for the verify-block shape):
**int8 is 1.70x faster than bf16 at 2k (69.2 vs 117.6 us/layer) and 1.88x
at 16k (539.7 vs 1016.3)**. Serving: +20% per stream at 18k (34.5 -> 41.4),
flat at 2.4k, acceptance unchanged, cold TTFT 7.45 -> 6.70 s.

The retired "650-705 tok/s" figure was a 12-stream aggregate measured greedily
on random-token prompts with a warm prefix cache. It is void as a production
number; the procedure document records the audit.

Acceptance is corpus-dependent: docs-prose prompts reach ~2.7-3.5, while
agent/tool traffic measured 1.0-2.4 earlier. W1 is now agent-shaped, and any
acceptance claim must name the corpus. At served sampling, throughput is
acceptance / step time.

Warm-cache TTFT (single request, two-turn probe): **174 ms** for 2048 cached +
128 new tokens, i.e. ~25 ms of prefill + one ~40 ms decode step + ~110 ms of
fixed request/engine overhead. Cold single-request prefill is 396 ms for 2048
tokens (~5.2k tok/s) under the retired prompt path; re-measure under W2.

Warm-cache TTFT (single request, two-turn probe): **174 ms** for 2048 cached +
128 new tokens, i.e. ~25 ms of prefill + one ~40 ms decode step + ~110 ms of
fixed request/engine overhead. Cold single-request prefill is 396 ms for 2048
tokens (~5.2k tok/s) under the retired prompt path; re-measure under W2.

Step budget (per step, measured from the parts). SUPERSEDED by the block
budget above (2026-11-12); kept as the phase-1 record. The part-level
numbers were collected under the retired procedure.

| Block | ms | state |
|---|---|---|
| Attention kernels (target 14 x 121 + 35 x 61 + draft 5 x 263 µs) | 5.2 | aligned-layout pass done |
| Routed MoE (gate_up 186 µs + down 96.9 µs per layer) | 13.3 | at 75-80% of achievable |
| Dense linears (qkv / o_proj / shared MLP) | 5.5 | at kernel limits |
| Draft graph (5 layers, aligned g8) | 6.2 (attention module 3.49, MLP 1.11, other 1.61) | at kernel limits |
| Sampler | ~1 | — |

### Int8 roofline at C12 W1 (2.4k ctx, M=96 verify)

Ceilings used: INT8 **184.6 TOPS** per GPU (the only compute peak this
project uses), measured achievable HBM stream **921-953 GB/s** per GPU
(four-GPU concurrent, 2026-11-10), XGMI 92 GB/s. Bytes are the weights
actually read per step at C12 (all 64 local experts are hit, every layer's
weights stream once); FLOPs are 2*M*N*K over the same shapes.

| Block | int8 weight bytes/step | GFLOP/step | Floor (max bytes/BW, flops/TOPS) | Measured | % of floor |
|---|---|---|---|---|---|
| Routed MoE (47 layers x 64 experts) | 9.46 GB | 170 | 10.0-10.3 ms | 13.3 ms | 75-77% |
| Dense linears (qkv/o_proj/shared MLP + lm_head) | 0.40 GB | 78 | 0.44 ms | 5.5 ms | 8% |
| Attention (target 49 + draft 5) | 0.36 GB KV | 46 | 0.30 ms | 3.83 target + 1.32 draft kernel | 6% |
| Draft graph (5 layers, TP4-sharded) | 0.07 GB | 15 | 0.08 ms | 4.25 ms | 1.9% |
| Sampler | - | - | - | ~1 ms | - |

Reading: the MoE is the only block close to a bandwidth roofline. The
dense linears, attention and the draft are all **latency/tile-bound at
M=96**, one to two orders of magnitude below their roofline: small-M int8
GEMMs cannot fill the CUs from a single 96-row tile, and the attention
dots run on 16-row tiles. That is the structural target for the native
small-GEMM/int8-dot path (`docs/design/int8_small_gemm_native.md`), not a
bandwidth problem.

CORRECTION (2026-11-12): the first draft of this table used unsharded
`intermediate_size` shapes for the shared-expert and draft GEMMs. At TP4
the real per-rank shapes are shared gate_up 512x2048, shared down
2048x256, draft gate_up 3072x2048 and draft down 2048x1536 (confirmed by
the boot log's CK warmup shapes), so the weight bytes, floors and rows
above are corrected. The draft-graph row is now the *whole* graph
(measured 4.25 ms at C12 block-g128; 3.61 ms with per-token-head KV), and
it is the furthest from its roofline in relative terms - entirely
latency/launch/attention, with only 0.08 ms of weight traffic. The
collectives are *not* part of the residual: four served boots at 16 vs 64
reduce blocks moved the step by <0.3 ms (barrier latency is hidden by the
inter-layer compute), so the ~9 ms that used to be attributed to them is
the draft propose plus sampler/glue.


Attention cost scales at **~6.8 µs per context token per step** (period
25.85 / 27.40 / 32.79 / 38.89 ms at 128 / 512 / 1024 / 2048 input tokens) —
a retired-procedure measurement, directionally useful but not yet reproduced
under W1. Halving the sequence count removes half of it. It is compute-bound
(~200 GFLOP/step at ~2% of fp16 peak), not KV-bandwidth-bound.

Isolated with `scripts/bench_attn_micro.py` (CUDA-graph timing of the real
call at production geometry, per-token-head KV): one target-shaped layer is
107 µs at 2k context and one draft-shaped (sliding-window) layer is 230 µs,
so the attention *kernels* are ~6.3 ms of the step. The rest of the raw
6.8 µs/token slope is the draft attention module's projections/norms plus
other context-scaled work. Segment count barely matters (4 segs 102.9 µs,
8 107.7, 16 114.5, 32 144.2). The sliding-window path costs 2.1x the full
path at decode shapes (230 vs 108 µs) but only 1.03x at prefill shapes
(445 vs 427 µs), so the penalty is decode-specific and worth at most ~0.6 ms
for the five draft layers.

**Practical stream ceiling on this mesh is 877-953 GB/s**, not the 1.23
TB/s nominal. Re-verified 2026-11-10 with four GPUs streaming concurrently:
921-939 GB/s per GPU, 3.72 TB/s aggregate, so there is no cross-GPU bandwidth
collapse. The MoE reads 9.5 GB/step per rank, so its floor is ~10 ms and it
measures 13.3 — do not expect a 2× there.

### Accepted in the loop (step-time based)
These were accepted on kernel/step-time measurements under the retired
procedure; they reduce ms/step and remain sound, but re-validate under W1
before citing them as throughput gains.
- Decode band widened to 16<M<=128 (2026-11-12, `_get_aiter_w8a8_config`):
  the M<=64 narrow/no-split path is 37-65% slower at M=20-32 on every dense
  site, so the tuned 128-column/split-K tile now covers the C3-C8 tail
  (M=16 keeps the old path: the wide tile loses 18-36% there). Served C3
  2.4k/256: 151.2 -> 157.4 agg (+4.1%), step 16.55 -> 15.95 ms, acceptance
  2.500/2.502 in both arms. C12 (M=96) config is identical before/after.
- MoE down grouped GEMM: `bn=256, num_stages=1` (154.0 -> 96.9 µs/layer;
  one stage keeps the 32 KB tile so two CTAs are resident per CU).
- Dense decode band: `bm=16` whenever the grid stays <=120 CTAs
  (o_proj 30.8->26.6, shared gate_up 24.7->22.3, shared down 18.3->15.2 µs);
  qkv keeps bm=32.
- Draft decode tiles: gate_up `bn=256/8 warps` (252.8->214.3 µs), qkv
  `bm=128/bn=64/ks=4` (79.7->61.3 µs). Payload layers are pinned to
  bn=128 by the fused int8-scale epilogue and must keep 4 warps
  (bn=128 + 8 warps measures 7255 µs).
- Draft grouped-conv base planes dequantized once at load.
- `KV_DTYPE` passthrough added to the serve script (default
  `int8_block_g128`); `VLLM_MOE_CFG_LOG=1` reports the effective MoE tile
  at boot; `VLLM_DRAFTTRACE`/`VLLM_ATTNTRACE` give the draft graph split.

### Closed (measured, do not re-open)
- **gate_up**: two-K-loop half-tile 199-214 µs vs 186 stock; split-K 236 vs
  179; persistent kernels 2× slower; bn=64/32 far worse. Its N is one
  expert because the fused epilogue quantizes in 128-column groups, so
  2.13 waves (256 tiles / 120 CUs) is inherent.
- **Attention config surface**: grid (`BLOCK_M` 16->8) worse and flat at
  long context; forcing the 3D kernel worse and numerics moved; the 3D
  path already splits the key dimension 8-way; `int8_per_token_head` is
  slower *and* worse quality, so block dequant is not the cost; TILE_SIZE
  64->128 gives no gain and moves numerics.
- **NS is NOT closed.** It was selected under acceptance 5.3-5.9 (greedy,
  random-token prompts). At served sampling acceptance is ~1.1-1.8, so NS=7
  pays seven draft verifies for ~0.1-0.8 extra tokens; the optimum must be
  re-derived in phase 3 (procedure section 4). Keep in mind that `NS+1` fixes
  the captured graph shapes, the MoE prewarm counts and
  `max_num_scheduled_tokens`, so any NS change needs its own graph capture.

### Measurement traps on this stack
- CUDA event pairs recorded inside a hipGraph raise
  `hipErrorInvalidHandle` at drain — no in-graph kernel timing.
- Wall-clock around `run_fullgraph` + `synchronize` (`VLLM_GRAPH_SYNC`)
  drains only part of the queue (it can read *less* than the MoE alone).
- Ablating the target's attention collapses routing, so the MoE's weight
  stream collapses too — never ablate a quantity that feeds the router.
- SSE chunks are not tokens: count from `usage.completion_tokens` or the
  Prometheus counters, and never treat `tpot_ms` as a token interval. See
  `optimization_procedure.md` section 3 for the authoritative definitions.
