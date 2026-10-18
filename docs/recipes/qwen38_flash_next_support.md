# Qwen3.8-Flash-Next on the 4x MI100 int8 stack — research handoff

**Status: RESEARCH ONLY (2026-10-06). Nothing here is implemented, gated, or part of
the production recipe.** This document records the investigation into what it would
take to serve Qwen3.8-Flash-Next on this fork under the int8-native doctrine, so a
future agent can get up to speed without redoing the research. Read `AGENTS.md` and
`docs/recipes/README.md` first — they are the baseline this must not quietly violate.

Provenance: all external facts were fetched directly (HuggingFace/GitHub raw files,
PR pages, git-trees API) on 2026-10-06. Web search was unavailable that session, so
there is no blog-corpus corroboration. Line numbers cited for Strata/gufo are from
that day's `main` snapshots. **Nothing has been validated on our hardware.** The fit
arithmetic below is derived from the published config, not measured.

## 1. What the model is

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (HF
created 2026-08-24; announced 2026-08-26). A **125B-total / 6B-active multimodal
MoE**, an experimental preview of the Qwen4 architecture ("plays the role Qwen3-Next
played for Qwen3.5"). License `qwen-community-1.0` — **not Apache**; check terms
before any redistribution of derived checkpoints (our PTQR exports would be derived).

Config facts (arch `Qwen4ExpForConditionalGeneration`, model_type `qwen4_exp`):

- ~125B core + **51B n-gram "PLE" embedding table** + **4B MTP module** ≈ 180B total.
  Hidden 2560, 48 layers.
- Layer layout: 12 × (3 × GDN→MoE, then 1 × QSA→MoE) — same 3:1 interval-4 pattern
  as Qwen3.8-27B (`layer_types`, `full_attention_interval: 4`).
- **GDN identical head geometry to Qwen3.8-27B**: 48 V heads / 16 QK heads, head dim
  128, conv kernel 4, `mamba_ssm_dtype: float32`. → our whole GDN stack (fp32 state,
  WY-fp32 chunked prefill, fused_sigmoid_gating, TINY_BA) is reusable as-is.
- **QSA** (the full-attention layers): 24 Q / 2 KV heads (GQA 12:1), head dim 256,
  partial RoPE factor 0.25 (64 rotary dims), rope_theta 1e7, mrope_interleaved
  [11,11,10]. Plus a **sparse indexer**: MQA with 4 Q heads + 1 shared KV head,
  indexer head dim 128, `indexer_budget: 2048` tokens (512 blocks of 4), compress
  ratio 4 — it scores 4-token blocks and each query attends only the top-512 blocks.
  Qwen-reported attention speedups (quoted in vLLM's recipe): up to 10.2× prefill /
  6.6× decode at 1M tokens.
- **MoE**: 512 experts/layer, 10 routed + 1 shared, expert intermediate 640.
  Routed experts = 48 × 512 × 3 × 2560 × 640 ≈ **120.8B params** (95% of mass).
- **Hyper-connections** ([arXiv 2409.19606](https://arxiv.org/abs/2409.19606)) replace
  the plain residual: 4 residual streams (`hc_count: 4`), low-rank 320 mixers,
  per-branch `GroupedGemmaRMSNorm`, checkpoint residual layout `[..., HC*HS]`.
- **PLE n-gram embedding**: 20M-entry table (n-gram size 3) injected at layer 2
  (`ple_layer_ids: [2]`), `split_ngram_parts: 128`, `heads_per_ngram: 8`,
  ple_embed_dim 2560, ple conv kernel 4. Designed for CPU offload + async prefetch.
- **MTP**: 1 hybrid full-attention layer, trained multi-step. vLLM exposes it as
  `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.
- Context 262,144 native, 1M via YaRN. Vision tower: SigLIP-style ViT (depth 27,
  hidden 1152, patch 16) — skippable for a text-only recipe.
- Tokenizer/vocab unchanged from Qwen3.8-27B (vocab 248320, eos 248044).

## 2. Delta vs. what we run today (Qwen3.8-27B dense + DFlash2)

| Surface | Today's recipe | Flash-Next |
|---|---|---|
| FFN | dense 17408, 64 layers | 512-expert MoE (~121B), 10+1 active |
| Full attention | gated SWA-style, 24Q/4KV | QSA + sparse indexer (budget 2048) |
| Residual | plain + Gemma norm | hyper-connections (4 streams) |
| Extra params | — | PLE table (51B), MTP (4B), vision tower |
| Draft | external DFlash2 (no Flash-Next equivalent exists) | shipped MTP module |
| GDN | 3:1 interval | identical geometry |

Fork state: **zero** `qwen4_exp` references in-tree. Qwen3.8-27B lives in
`vllm/model_executor/models/qwen3_5.py` (+ `qwen3_5_mtp.py`, `qwen3_dflash2.py`).
Fork base is the old `model_executor/models` layout (version reads "dev").

## 3. The fit problem (this shapes everything)

4× MI100 = 128 GB VRAM at 0.92 util; host 61 GB (12 GiB already the CPU KV tier).
Weights at ~1 byte+scale (int8) vs ~0.52 byte+scale (int4 g128):

| | int8 everywhere | int4 experts + int8 rest |
|---|---|---|
| Routed experts (120.8B) | ~124 GB | **~63 GB** |
| Non-expert core + MTP (~8B) | ~8.3 GB | ~8.3 GB |
| Total GPU weights | ~133 GB → 33+ GB/GPU | **~71 GB → ~18 GB/GPU** |
| Verdict | **over budget before any KV** | fits, ~11 GB/GPU left for KV/states/graphs |

- **Int8-everything does not fit GPU-resident**; it forces Strata-style tiered
  residency (hot experts in VRAM, cold pinned on host, PLE SSD-paged) — the single
  biggest scope item.
- **Int4 experts fit**: all experts GPU-resident, expert cache machinery becomes
  optional insurance. PLE still never fits VRAM: int8 51 GB / int4 ~27 GB — at int4
  it can be host-resident (tight: 27 GB + 12 GiB tier + overhead on 61 GB) or
  SSD-paged through the page cache (the lookup is locality-heavy; Strata and halogen
  both page it).
- KV is a non-problem: 12 QSA layers × 2 KV × 256 ≈ 6–12 KB/token/rank → millions of
  tokens in the leftover VRAM. GDN fp32 state ≈ 113 MB/request (fine at C6–C8).
- Bandwidth bonus: decode is weight-traffic-bound here; int4 halves expert traffic.
  Bounded overall (non-expert int8 reads persist, MTP adds passes) — treat 15–30%
  TPOT as a hypothesis for the ledger, not a promise.

## 4. Doctrine considerations

The AGENTS.md tenet is "full int8" as the **compute** dtype (W8A8 GEMMs, int8 KV,
PTQR gates, minimum weight bytes). Two coherent readings for Flash-Next:

1. **Int8-only** (literal): requires the tiered-residency subsystem because the
   weights don't fit. Int8 compute stays pure.
2. **Int4-storage / int8-compute** (recommended): experts W4 with in-kernel unpack
   feeding int8-class math; activations, dense GEMMs, KV, and every existing surface
   stay int8. Anchors: AMD's own [Quark MXFP4 build](https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8)
   quantizes exactly the experts (routed **and** shared) to 4-bit — GSM8K recovery
   99.5% — and keeps QSA/GDN projections/routers/HC/MTP/lm_head/vision bf16;
   [ISTA-DASLab GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
   holds PLE fixed at IQ4_NL ~4.5 bpw "a lookup, not a matmul weight" and searches
   only experts (~95% of mass). This is still a **doctrine amendment that needs
   explicit user sign-off + acceptance gate + ledger row** before becoming recipe.

Unchanged regardless: TP4 + vLLM CUSTOM AR, `VLLM_GFX908_ACT_QUANT=round`, AITER CK
W8A8 + tuned-CSV discipline, fp32 GDN state, bf16 residual stream, the gfx908
torch.compile ban (all upstream qwen4_exp AMD kernels are eager Triton —
compatible), the ROCm 7.2 runtime override, the testing ladder, KLD gate method,
ledger discipline.

Hardware constraints from our own GOALOPT closures that bound the kernel design:
int8 MMA lowering is broken in Triton/AMDGPU MLIR on this build; the i8 16×16×16
MFMA measured slower than fp16 MFMA → prefer fp16 dots where operands allow; there
is no MXFP4 hardware on gfx908 → **int4 is storage-only, dequant in-kernel**.

## 5. Quantization program

- **No official GPTQ/AWQ int8 exists.** Official checkpoints: BF16 (335.28 GiB) and
  FP8 (172.78 GiB, 128-wide blocks → plain TP8 incompatible; also currently fails to
  load on ROCm, [issue #58688](https://github.com/vllm-project/vllm/issues/58688)).
- Baseline plan: one-shot **GPTQ int4 GS128 on experts** (least sensitive, biggest
  mass), **PTQR-retrain the sensitive surfaces** (QSA/GDN projections, HC mixers,
  MTP, lm_head — per AMD's keep-list), mixed-precision promotion as the quality
  lever (halogen's sidecar finding: promoting a handful of measured-sensitive
  tensors recovered nearly all quant loss; the analogous move is ~5–10% of experts
  at int8, ~1.5 GB/GPU cost).
- Keep-list per fork doctrine: routers bf16/fp32, GDN state fp32, codebook-likes
  bf16, PLE paged lookup (int4 host / int8-or-fp8 SSD; gate separately — AMD chose
  FP8 for PLE, ISTA chose fixed 4.5 bpw; mixed signals).
- Logistics must change: the single-GPU `offload_to_disk=False` recipe cannot hold
  180B; run layer-wise with host streaming and experts sharded across the 4 GPUs
  (avoid gptqmodel AFDP — known-bad here). PTQR per-expert is embarrassingly
  parallel.
- Gate everything: KLD probe, greedy agreement, 40k greedy acceptance leg, rank-20
  oracle — the usual battery. 4-bit raises the stakes on the long-context leg (our
  garble history: small verify-decision noise compounds).

## 6. Speculative decoding

DFlash2 is Qwen3.8-27B-specific; no Flash-Next external draft exists. Use the
**shipped MTP module**: upstream `Qwen4ExpMTP` (1 layer reusing the backbone, reads
the 4 residual streams). Fork reference: `qwen3_5_mtp.py`. Adopt gufo's
exact-fp32 GDN snapshot rollback for spec verify (vs our bounded-lookup mechanism)
and halogen's byte-identical-to-serial-greedy release gate as the bar. Re-sweep NS
with the canonical protocol (upstream recipes use MTP3; Strata sees 2.4–3.2
tokens/pass). Prompt-lookup drafting likely stays closed — GOALOPT already convicted
it on this workload class.

## 7. Kernel baselines at current heads (2026-10-06)

Licenses: vLLM/SGLang Apache-2.0; aiter/Strata/gufo MIT. halogen is closed-source —
design reference only.

| Family | Primary baseline | Alternate | gfx908/int8 port note |
|---|---|---|---|
| QSA sparse attn (decode+verify) | vLLM `vllm/models/qwen4_exp/amd/ops/qsa.py`: `_qsa_sparse_paged_gqa_splitk_kernel`, `_qsa_mqa_paged_kernel`, `_expand_qsa_indices_kernel`, `_qsa_merge_splitk_kernel`; backend `Qwen4ExpQSAFlashAttentionBackend` (`QWEN4_EXP_QSA_TRITON`, UNIFORM_BATCH graphs) | Strata `src/kernels/cuda/qsa_decode_attn.cu` + `kv_q8.cu` (int8-KV variant!) | Moderate. Eager Triton, bf16 `tl.dot` → MFMA on CDNA1 (no i8 dot needed). Retune BLOCK_M/N (BLOCK_D=256 register pressure, 64-CU MI100). Int8-KV is our addition — same seam as the Gluon G128 cores. 2048-token budget ≈ our DFlash2 windowed draft core, block-gathered instead of contiguous |
| QSA prefill attention | Strata `qsa_prompt_attn.cu:107` `prompt_attn_kernel` (FP32 VALU — our hardware class), `:464 prompt_attn_i8_kernel` | gufo `WmmaCausalAttentionKernel` | Strata ports as-is; fp16-MFMA upgrade later per our fp16-dot doctrine. gufo needs WMMA→MFMA + de-hardcoding 24×256/2-KV |
| Indexer (pool/score/select) | vLLM `amd/ops/qsa.py` `_compress_qsa_groups_kernel`, top-k in `qsa_select_paged_tokens`; fused prep `amd/ops/qsa_pre_indexer.py` `_qsa_pre_indexer_kernel` (gated hd==128 — matches) | gufo `PoolBlocksKernel`/`SelectScoreKernel`/`SelectMarkKernel` (uint32 visibility bitmap — cleanest contract) | Easy. All VALU. Adopt the bitmap contract if rewritten |
| Grouped expert GEMM (tiny-M, 10+1 of 512) | aiter main (MIT): `aiter/ops/moe_op.py`, `moe_sorting.py`, `gemm_op_a8w8.py` (+a4w4/a8w4/mixed_mxfp siblings; merged PR #5974 added Flash-Next gfx942 a8w8 tunings) | Strata `iq_kernels.cu:3102` `native_gu_lds_kernel` mode-7 (dp4a, wave64-native; `native_gu_fused_kernel` = gate/up+SwiGLU+act-quant fused); gufo `RoutedF16GEMMKernel` (W4→F16 LDS dequant, whole-K fp32 acc) + `RoutedCompact` | **Main fork lift.** Local `../aiter` already has `fused_moe.py`/`fhmoe.py`/`int4_utils.py` + dense `module_gemm_a8w8.so` + CSV tuning — extend to grouped 512-expert dispatch, emit gfx908-tuned grouped a8w8, add M∈{1..4} rows. Strata dp4a is a correct wave64 floor (MI100 MFMA-i8 strictly beats gfx906 dp4a) |
| HC mix + GroupedGemma norm | vLLM `amd/ops/hc.py`: `_hc_gate_mix_kernel`, `_hc_combine_norm_kernel`, `_grouped_gemma_rmsnorm_kernel` (pure Triton, eager, no MMA) | aiter PR [#5913](https://github.com/ROCm/aiter/pull/5913) two-stage (a16w16, gfx942); gufo `HcCombineMoeF16Kernel` — residual+MoE-combine+norm+**tiled-Q8 quant-out** fused | Base kernels trivial. The quant-out fusion is fork-new; gufo proves it works without torch.compile but hardcodes 4×2560 + wave32 WMMA — adapt the contract onto our `round` act-quant / SMALLM_NORM epilogues |
| Router | Strata `router_top10.cu:198` `router_top10_wave_kernel` (`__launch_bounds__(64)`, register argmax, bitwise-checked) | gufo `RouterTopKKernel` | Near-zero (Strata) — exact top-10-of-512 shape |
| MTP head | vLLM `amd/mtp.py` `Qwen4ExpMTP` (orchestration; eager-capable) + fork `qwen3_5_mtp.py` | gufo `MtpHidden/AddEmbedding/CandidateTile` + FP64 `PenalizedArgmax` (graph-safe); Strata async `chain_launch/poll` + `kv_restore` | Mostly wiring; kernels are the families above |
| PLE (hash/gather/conv/inject) | vLLM `amd/ple_layer.py` (eager `F.conv1d`+`index_select` — slow but correct bring-up) | SGLang `kernels/ops/embeddings/qwen4_ngram.py` `_qwen4_ngram_hash_kernel` (portable Triton); gufo `PleConvKernel` w/ prefix snapshots; vLLM NVIDIA `ops/ple.py` (strip CUDA PDL intrinsics) | Easy. VALU. Fuse after bring-up; table stays paged/host |
| GDN decode + spec rollback | **Fork's own stack** (identical geometry) | gufo `GatedDeltaNetBatch` (1–8 reqs × rows) + exact-fp32 `RollbackRows`/`RestoreGdnStateKernel`; Strata `gdn_state_replay_kernel`; aiter main `fused_rearrange_sigmoid_gated_delta_rule` | Zero for the base. gufo snapshot-rollback is the MTP-verify upgrade |

**Genuinely fork-new kernel work** (no baseline exists):
1. `int8_block_g128` KV inside the QSA gather (upstream AMD is bf16-KV-only; fp8 KV
   is NVIDIA-side in-kernel dequant — pattern reference only).
2. A gfx908 grouped W8A8 (+W4 in-kernel unpack) expert GEMM artifact — aiter main
   has the API shape but only gfx942/950 artifacts.
3. HC + norm + int8 quant-out epilogue fusion at the HC seam (same class as our
   built-but-off AR epilogue).

**Watch, don't port** (gfx942/950-bound): aiter FlyDSL QSA K1/K2
([#5996](https://github.com/ROCm/aiter/pull/5996)/[#6083](https://github.com/ROCm/aiter/pull/6083),
wired by vLLM [#59437](https://github.com/vllm-project/vllm/pull/59437) behind
`VLLM_ROCM_QSA_FLYDSL=1`), two-stage HC via vLLM
[#59442](https://github.com/vllm-project/vllm/pull/59442)
(`VLLM_ROCM_HC_FLYDSL_MAX_M`, default M≤8).

## 8. Upstream integration facts

- vLLM support: PR [#53896](https://github.com/vllm-project/vllm/pull/53896)
  merged 2026-08-31; first stable release **v0.29.0** (2026-09-09). **There is no
  old-layout release** — it landed directly as the new package
  `vllm/models/qwen4_exp/{amd,common,nvidia}/`, self-contained Python+Triton over
  shared `model_executor/layers` the fork already has → drop in + registry adapter,
  not a layout rewrite. Main adds `amd/ops/qsa_pre_indexer.py` and grows
  indexer/PLE files vs v0.29.0.
- AMD path constraints (why ours is fork work): **bf16 weights + auto/bf16 KV
  only**, `supports_kv_connector()=False`, FP8 checkpoints fail on ROCm (#58688).
  NVIDIA path additionally has fp8/fp8_e4m3 KV dequant in-kernel.
- Everything AMD-side is eager Triton via `direct_register_custom_op` —
  compile-*compatible*, never compile-*requiring* → matches our compile ban.
- MI355X recipe: `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=0`, TP4, MTP3 →
  even upstream doesn't use aiter MoE in production. Official recipe:
  [recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next),
  image `vllm/vllm-openai:qwen38-flash-next`.
- Only MI100 datapoint anywhere: [issue #57138](https://github.com/vllm-project/vllm/issues/57138)
  — tiny checkpoints boot on the AMD path; they fail on layer-type naming
  (`qwen_sparse_attention` vs `full_attention`; workaround: edit config.json) and
  select the Triton unquantized MoE backend.
- SGLang: qwen4_exp live (cookbook #36496); useful extras — fp8-e4m3 indexer cache
  (#39610/#39614), **closed-unmerged** ROCm MFMA head-alignment fix for QSA MQA
  decode ([#38910](https://github.com/sgl-project/sglang/pull/38910) — read before
  MQA scorer work), `qwen4_exp_mtp.py` pinned against vLLM's, verify-conv kernels
  (`qwen4_short_conv.py`), PLE hash kernel.
- llama.cpp: sparse-FA kernel [#28770](https://github.com/ggml-org/llama.cpp/pull/28770);
  two-tier expert-cache FR #20757 (residency reference).
- Bring-up accelerators to port: `tests/models/qwen4_exp/test_qsa_amd.py`,
  `test_qsa_pre_indexer_amd.py`, `test_qsa_reference.py` (51 KB eager golden),
  `test_ple.py`, `test_hc_ops.py`, `test_qsa_flydsl.py`,
  `vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py`. Tiny checkpoints:
  [inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B-MTP](https://huggingface.co/inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B-MTP),
  `qikp/tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next`.

## 9. Learnings from the three reference projects

- **[Strata](https://github.com/Niko1221/Strata)** (15.3k★, MIT, C++/CUDA+HIP +
  ggml parts): consumer engine for this model. Tiered residency done right —
  adaptive expert cache (~700 experts/GB, profile-driven), RAM-pinned experts with
  **CPU computing uncached experts concurrently with the GPU**, SSD-paged PLE,
  next-layer expert prefetch overlapped with attention, MTP(≤3) + prompt-lookup
  drafting. **Its gfx906 build is the only CDNA1 work anywhere**: `hip_compat/`
  CUDA→HIP warp-mapping (half-wave shuffles), dp4a sign-trick expert kernels,
  64-lane MMVQ butterfly, register top-10 router, FP32 QSA prompt attention, and
  per-arch hipBLASLt solution-id tables (`tools/hip/*.txt`) — the same tuning
  discipline as our aiter CSV. 2× MI50 16 GB: 57–59 tok/s decode @131k. gfx906 ≠
  gfx908 but both CDNA1 wave64 — closest porting reference for layout.
- **[gufo](https://github.com/gufo-org/gufo)** (571★, MIT, C++20/HIP, gfx1151-only,
  no Triton/CK/MIOpen): best *contracts*. Tiled-Q8 W8A8 layout (`QuantizeQ8Tiled` →
  `W8A8Gemm`, wave64 variant), fused `HcCombine*` residual-mix+norm+MoE-epilogue+
  int8-quant-out, indexer ring→pool→bitmap select, exact-fp32 GDN snapshot rollback
  for speculation, `ContinuationCache` hybrid-state prefix reuse, HIP-graph decode
  with device-side position control. All matrix-heavy kernels are WMMA-wave32 with
  hardcoded 24×256/4×2560 geometry → need MFMA fragment rework, not drop-in.
- **[halogen](https://github.com/peonist-ai/halogen-flash-server)** (862★,
  closed-source engine, open deploy docs): design reference only. Calibrated
  sidecar (find the few tensors carrying quant loss, promote just those —
  12 `o_proj` tensors at 8-bit tied with the bf16 ceiling), 47.7 GiB FP8 PLE paged
  through page cache never resident, byte-identical-greedy speculation gate, KV
  region allocator with no-loss moves, baked GEMM solution tables, ppl/KL/niah
  harness as image modes.

## 10. Recommended plan (if we proceed)

0. **Decision gate**: int4-experts is a doctrine amendment — explicit sign-off
   first. The legitimate exit: wait for the smaller/dense Qwen4 release this
   preview foreshadows (Qwen3-Next→Qwen3.5 precedent); phase-1 plumbing transfers.
1. **Plumbing**: port the `qwen4_exp` package (text-only `ForCausalLM` first) +
   registry adapter; boot the 0.2B tiny checkpoint on gfx908 (fix layer-type
   naming); port the upstream test files.
2. **bf16 bring-up**: QSA Triton kernels as-is (bf16 KV interim), eager PLE, MTP
   depth sweep on tiny + FP8/BF16 official shards quantized later — never the FP8
   checkpoint on ROCm (#58688).
3. **int8-KV seam**: `int8_block_g128` into the QSA gather (Gluon-G128 pattern);
   indexer cache int8. Gate: micro test + battery extensions.
4. **Quant campaign**: GPTQ int4 GS128 experts one-shot; PTQR the sensitive
   surfaces; mixed 4/8 promotion as the lever. Full KLD/greedy/40k/oracle battery.
5. **Kernels**: grouped W8A8 expert GEMM artifact + tuned rows M∈{1..4}; W4 in-kernel
   unpack variant; HC quant-out epilogue fusion; router wave kernel.
6. **Productionize**: serve script + systemd unit (runtime override preserved),
   battery/soak assertions for MTP + new caches, NS re-sweep, ledger rows, C re-gate.

## 11. Risks / open questions

- **Scope**: still the biggest program this fork has taken on — model port + three
  kernel families + quant campaign, even with baselines for everything.
- **4-bit under OUR gates**: AMD/ISTA evidence is GSM8K/perplexity-class; our bar
  (KLD 0.0069-class, 40k greedy, rank-20 oracle) is stricter, and our garble
  history says verify-decision noise compounds at long context. PTQR + mixed
  promotion is the mitigation, not a guarantee.
- **No MI100 prior art at scale** anywhere (only the tiny-checkpoint boot).
- **aiter MoE immaturity** (upstream MI355X disables it); grouped gfx908 a8w8 is
  ours to build and tune.
- **Quantizer logistics**: 180B on 32 GB GPUs needs the sharded/streaming rebuild;
  AFDP known-bad.
- **PLE quality** at int4 vs fp8 — mixed signals (AMD vs ISTA); gate separately.
- **License**: `qwen-community-1.0` on the base model constrains derived-checkpoint
  redistribution.
- Unverified: every perf number above is upstream's or the projects' on other
  hardware. Run the interleaved-boot harness before believing any of it here.

## Reference index

Model & checkpoints:
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next (config.json, model card)
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8
- https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8
- https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
- https://huggingface.co/inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B-MTP
- https://github.com/QwenLM/Qwen3.8-Flash-Next (news, tech_report.pdf)
- https://qwen.ai/blog?id=qwen3.8-flash-next (CSR page — text not extractable that session)
- https://arxiv.org/abs/2409.19606 (hyper-connections)

vLLM:
- https://github.com/vllm-project/vllm/pull/53896 (model support; formats: BF16/FP8/NVFP4 NVIDIA, BF16 MI355X)
- https://github.com/vllm-project/vllm/pull/59437 (AITER FlyDSL QSA, env-gated)
- https://github.com/vllm-project/vllm/pull/59442 (AITER two-stage HC)
- https://github.com/vllm-project/vllm/pull/54517 (fused PLE kernels, NVIDIA/PDL)
- https://github.com/vllm-project/vllm/pull/58856 (PLE fp8 embedding method)
- https://github.com/vllm-project/vllm/pull/59732 (fused pre-indexer, main)
- https://github.com/vllm-project/vllm/issues/58688 (FP8 load fails on ROCm)
- https://github.com/vllm-project/vllm/issues/57138 (MI100 tiny-checkpoint boot)
- https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next

aiter:
- https://github.com/ROCm/aiter/pull/5913 (two-stage HC, open)
- https://github.com/ROCm/aiter/pull/5996, /pull/6083 (FlyDSL QSA, open)
- https://github.com/ROCm/aiter/pull/5974 (Flash-Next gfx942 a8w8 tunings, merged)

SGLang / llama.cpp:
- https://github.com/sgl-project/sglang/pull/36496 (cookbook), /pull/38116, /pull/39614
- https://github.com/sgl-project/sglang/pull/38910 (closed-unmerged ROCm QSA MQA fix)
- https://github.com/ggml-org/llama.cpp/pull/28770 (sparse-FA kernel)

Engines studied:
- https://github.com/Niko1221/Strata (+ docs/HOW_IT_WORKS.md, docs/MODELS.md, docs/AMD_HIP.md)
- https://github.com/gufo-org/gufo (+ docs/models/qwen3.8-flash-next/README.md, docs/KV-CACHE.md)
- https://github.com/peonist-ai/halogen-flash-server (+ docs/QUANT.md, docs/FLAGS.md)
