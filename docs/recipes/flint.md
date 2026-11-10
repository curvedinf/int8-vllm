# Flint + Flint-DFlash2 Int8 PTQR Recipe (gfx908 / 4x MI100)

Peer recipe to the [Qwen3.8-27B baseline](README.md) and the
[Nemotron Bolt + LiLiCorr recipe](nemotron_bolt.md). Same int8-native
doctrine: every dense GEMM is AITER W8A8 INT8 GS128, routed experts run a
true W8A8 grouped kernel, and KV is `int8_block_g128` on **both** target and
draft. There is no 16-bit fallback in this recipe.

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
hidden 2048, MoE intermediate 512, 32 heads / 8 KV heads, head_dim 128,
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
| Dense linears | AITER CK W8A8 INT8 GS128 (`AiterW8A16LinearKernel` legacy name dispatches W8A8). Deployed decode band `65<M<=128` (C12 with NS=7) runs the tuned Triton `a8w8_blockscale` off the retained GS128 originals instead (`VLLM_GFX908_DECODE_BLOCKSCALE=1`, bm=32/bn=128, NUM_KSPLIT=4 for N<=1536 else 1, except mlp down_proj which stays CK); prefill (M>128) stays CK. Measured: qkv 95.3->34.7us end to end, shared-expert gate_up 68.3->26.8us. |
| Routed experts | `vllm/model_executor/layers/quantization/gptq_moe_a8w8.py` — GPTQ int8-group → AITER grouped `a8w8_blockscale` (`[E,K,N]` int8 + per-column GS128 fp32 scales, `w_blockscale_n=1`). Decode-band tiles: gate_up bn=256/warps 8/stages 2 (186 us/layer, 60% of HBM peak — wave-quantization limited, 256 tiles over 120 CUs); down bn=256/warps 8/**stages 1** (96.9 us/layer, 715 GB/s, was 154.0 at bn=128/stages 2 — one stage keeps the 32 KB weight tile so two CTAs are resident per CU). The block is pure weight streaming: the int8 dot for the whole 2048-tile down grid is 16.6 us. Bench: `scripts/bench_flint_moe_kernel.py` (graph-timed, production kwargs; one config per process because a bad tile faults the device). |
| Attention | AITER unified attention |
| KV (target + draft) | `int8_block_g128` |
| Prefix cache (hybrid) | Mamba prefix caching **on** (`MAMBA_MODE=align` default; the deprecated `--mamba-cache-mode` now reaches `CacheConfig`). Recurrent checkpoints land on 128-token block boundaries; measured warm-cache TTFT 205-232 ms (was 1165 ms without it) and 269-548 ms on a partial-prefix hit (was 3519 ms). Sub-block (partial) Mamba hits are not reachable: they need every prefix-cacheable group to support fine-grained lookup and the attention groups use 128-token blocks. |
| All-reduce | vLLM CUSTOM (`VLLM_ROCM_USE_AITER_CUSTOM_AR=0`); residual-stream collects use `residual_g128_all_reduce` on int8 codes+scales |
| Act quant | `VLLM_GFX908_ACT_QUANT=round` |
| Residual stream | int8 GS128 payloads between every decoder layer (`residual_int8`), fused residual+norm+quant seams |
| Attention/input norm prologue | `residual_norm(emit_pertoken=True)` stashes the CK per-token quant; the CK GEMM consumes `PREQUANT_ATTR` |
| MoE gate_up | fused SwiGLU + per-128 int8 epilogue in the grouped GEMM (`QUANT_OUT`, interleaved gate/up weights) |
| MoE down/combine | `reduce_grouped` quant-out: shared branch folded as an int8 residual, combined row quantized per 128; runner float AR disabled, one int8 collective | The shared expert runs on a side CUDA stream and overlaps the routed gate_up (`VLLM_GFX908_SHARED_OVERLAP=1`, -4.5ms/step).
| Dense/shared/draft MLP | `gemm_silu_quant_out`: SwiGLU + GS128 int8 output from resident blockscale weights |
| Router gate, lm_head | int8 GS128 at all batch sizes; norm epilogues hand both consumers their per-group stash |
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
scripts/serve_recipe_flint.sh start    # TP4, C8, port 8030
scripts/serve_recipe_flint.sh status
scripts/serve_recipe_flint.sh stop
```

Recipe contract: TP4, `--max-num-seqs 12` (C12), bf16 compute dtype,
`--max-model-len 524288` (512k per-request cap), `--max-num-batched-tokens
2048`, `--gpu-memory-utilization 0.90`, `int8_block_g128` KV, DFlash2
draft `NS=7` with its own `int8_block_g128` KV, FULL_DECODE_ONLY graphs.

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

## Performance state after the tuning loop (2026-11-10)

Current, warm, C12 × 2048 in / 256 out, seed 62408: **650-705 output
tok/s, TPOT 57.5-59.1 ms, acceptance 5.3-5.9**. Six consecutive warm runs
measured 650.1 / 705.4 / 632.7 / 661.8 / 657.3 / 656.8, so the steady state
is stable within ±5%; the wider 535-707 spread seen earlier is a boot
transient - always take one warm-up run before an A/B. Loop start was
71.5 ms / ~520 tok/s.
Warm-cache TTFT (single request, two-turn probe): **174 ms** for 2048
cached + 128 new tokens, i.e. ~25 ms of prefill + one ~40 ms decode step +
~110 ms of fixed request/engine overhead. Cold single-request prefill is
396 ms for 2048 tokens (~5.2k tok/s); a 12-request cold batch is ~5.7 s.

Step budget (per step, measured from the parts):

| Block | ms | state |
|---|---|---|
| Attention kernels (target 47 x 108 µs + draft 5 x 230 µs) | ~6.3 | kernel-rewrite only |
| Routed MoE (gate_up 186 µs + down 96.9 µs per layer) | 13.3 | at 75-80% of achievable |
| Dense linears (qkv / o_proj / shared MLP) | 5.5 | at kernel limits |
| Draft graph minus attention | 2.6 | at kernel limits |
| Sampler | ~1 | — |

Attention cost scales at **~6.8 µs per context token per step** (period
25.85 / 27.40 / 32.79 / 38.89 ms at 128 / 512 / 1024 / 2048 input tokens);
halving the sequence count removes exactly half of it. It is compute-bound
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
TB/s nominal. The MoE reads 9.5 GB/step per rank, so its floor is ~10 ms
and it measures 13.3 — do not expect a 2× there.

### Accepted in the loop
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
- **NS** is not a free knob: `NS+1` fixes the captured graph shapes, the
  MoE prewarm counts and `max_num_scheduled_tokens`. NS=6 and NS=8 fall
  ~3× onto uncaptured shapes (226-358 tok/s). Keep NS=7.

### Measurement traps on this stack
- CUDA event pairs recorded inside a hipGraph raise
  `hipErrorInvalidHandle` at drain — no in-graph kernel timing.
- Wall-clock around `run_fullgraph` + `synchronize` (`VLLM_GRAPH_SYNC`)
  drains only part of the queue (it can read *less* than the MoE alone).
- Ablating the target's attention collapses routing, so the MoE's weight
  stream collapses too — never ablate a quantity that feeds the router.
- Bench `tpot_ms` is the per-SSE-chunk interval, not per-token; use
  `output_tok_s` plus `VLLM_STEPWALL`'s wall-clock `period`.
