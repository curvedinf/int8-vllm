<!-- markdownlint-disable MD001 MD041 -->
# int8-vllm — vLLM with a complete INT8 serving path, tuned for 4x MI100

A fork of [vLLM](https://github.com/vllm-project/vllm) whose goal is to add
**complete INT8 support that upstream is missing**: W8A8 INT8 GEMMs at
every decode and prefill shape, an INT8 per-token-head KV cache, INT8
lm_head, an INT8 speculative-decoding draft path, and the glue (quantizers,
cache writers, dispatch) to keep activations INT8 end-to-end instead of
falling back to fp16 between ops.

The stack is tested and tuned on **4x AMD Instinct MI100 (gfx908 / CDNA1,
32GB, XGMI full mesh)**. MI100 gets 2x int8 rate vs any dtype other than
FP16 (equal), at half the memory bandwidth — so this branch treats **int8 as
the native dtype** across the whole model surface and exploits it everywhere
it wins, with measured float exceptions where precision actually matters.

> [!NOTE]
> **Optimized configuration:**
>
> - Models: [Qwen3.8-27B GPTQ INT8 W8A8 GS128](https://huggingface.co/curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128)
>   + its matching [DFlash2 GPTQ INT8 W8A8 GS128 draft](https://huggingface.co/curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128)
>   (speculative decoding always on, NS=15)
> - Compute: AITER CK W8A8 INT8 GEMMs at every decode/prefill shape, INT8
>   lm_head
> - Attention/KV: AITER unified attention; INT8 per-token-head KV on both
>   target and draft
> - Topology: TP4 over XGMI, 8 concurrent sequences (C8)
>
> Other models and settings generally work, but only this combination is
> tuned and gated.

## Performance

This branch was tuned the slow way: every performance-affecting surface of
the model was measured, converted to INT8, and re-gated for quality before
being kept. The order of attack was (1) replace every GEMM with AITER CK
W8A8 INT8 kernels at all decode and prefill shapes, (2) move both the target
and draft KV caches to int8 per-token-head with inline scales, (3) serve all
attention through AITER's unified kernel with gfx908-specific tuning,
(4) replace stock collectives with vLLM's custom XGMI all-reduce, and
(5) put a DFlash2 INT8 speculative drafter (NS=15) in front of decode.
Surfaces where INT8 cost measurable quality — the GDN recurrent state,
selector codebooks, the draft ctx-KV projection — stay in higher dtypes,
decided by KLD gating against a BF16 reference rather than guesswork. The
full component-level dtype table lives in `docs/recipes/README.md` and the
measurement program in `INT8_AUDIT_RESULTS.md`.

All numbers below come from C8 serving benchmarks — 8 concurrent requests,
TP4, 32-token inputs / 1000-token outputs, greedy, fresh-boot
`vllm bench serve` legs. The primary metric is steady-state **TG**
(concurrency ÷ mean TPOT: pure decode throughput, prefill excluded);
**wall-clock output tok/s** and **PP tok/s** are reported alongside and are
only comparable within their own column. Current-stack numbers live in
`docs/recipes/README.md` "Current production status" — the one source of
truth — and are not restated here.

Highlights of the current stack:

- Int8-native: W8A8 GEMMs (CK), int8 lm_head, int8-PTH KV (target + draft),
  int8 embedding, round-to-nearest per-token act quant.
- Measured float exceptions (never quantize without re-gating): GDN state
  stays fp32 (int8 corrupts with int8 KV; fp16 blew states to the fp16
  ceiling), selector codebooks bf16, draft ctx-KV projection bf16.
- TP4 / C8, DFlash2 NS=15, vLLM CUSTOM all-reduce.

### History (abbreviated, 27B C8 serving)

Metrics are split by definition — never compare a TG cell against a
wall-clock cell. Rows 1–2 predate the INT8 conversion and ran fp16 KV.
`—` = not measured under that definition.

> [!IMPORTANT]
> **All numbers below were captured with the MI100s on a 105W low-power
> profile, not the 290W stock power limit.** Expect roughly 2x performance on
> power-unconstrained cards; do not compare these figures against
> stock-power MI100 benchmarks.

| Date | Stack | Config | Wall Clock Output tok/s | TG tok/s | PP tok/s | Notes |
|---|---|---|---:|---:|---:|---|
| 2026-06 early | Qwen3.6-27B GPTQ-8, TRITON_ATTN, no spec | TP4, fp16 KV | 129 (c=8 mixed) | 151 | 1,490 | TPOT 20.1 ms at c=1 |
| 2026-06-11 | Qwen3.6-27B + AITER UA + MTP n=3 | TP4, fp16 KV | 212 (c=8) | — | — | +29% at 16K ctx; +15% on 4k-in/6k-out batch; UA ≈ TRITON without MTP |
| 2026-08-25 | Qwen3.8-27B INT8 GS128 + DFlash2 | current recipe: W8A8 everywhere, int8-PTH KV both models, UA, fp32 GDN state, round act quant, TP4/C8 | 348 | — | 2,175 | INT8 conversion + accuracy program; NS=15 selected (NS=17 acceptance collapses to ~30%); TPOT 12.52 ms; ~71% draft acceptance; KLD median 0.0153 vs BF16; PP from the 8×8192 prefill leg |

> [!NOTE]
> Higher numbers than the ones above may appear in historical logs. Those
> were captured from configurations later shown to degrade output quality
> (pre-corruption-bisect, pre-accuracy-program) and are superseded — see the
> audit doc's history section for why they don't count.

## Other hardware this is a good baseline for

The kernels here are gfx908-tuned (CK instances, Triton autotunes), so a port
means regenerating/retuning kernels — but the **architecture transfers**:
INT8-native surfaces, per-surface dtype ladders, and the KLD-gated audit
method apply to any accelerator where INT8 is among the best native dtypes:

- **AMD CDNA family** — MI50/MI60 (gfx906, INT8 dot4), MI210/MI250 (gfx90a).
  Closest relatives; much of the ROCm/AITER path carries over directly.
- **AMD RDNA (consumer)** — RDNA2 (gfx103x) runs INT8 via DP4A
  (`v_dot4_i32_i8`) on the vector ALUs; RDNA3/3.5 (gfx11xx) add WMMA matrix
  cores with native INT8 at 2x FP16. Single-GPU small-model serving is the
  fit; the Triton paths carry over, CK GEMMs need WMMA replacements. RDNA4
  (gfx12xx) adds FP8, which usually beats INT8 there — same caveat as Hopper.
- **NVIDIA Turing/Ampere/Ada** — T4, A10/A30, A100, L4/L40S: INT8 tensor-core
  rate is 2x FP16 at half the bandwidth, the same trade MI100 makes. The
  Triton paths are largely portable; the CK GEMMs need a cuBLASLt/cutlass
  substitute.
- **Hopper and newer** have FP8, which usually beats INT8 there — this stack
  is most valuable on hardware whose sweet spot is INT8.

If your GPU's INT8:FP16 rate ratio is 2:1 (or INT8 is its peak dtype), this
repo's recipe and measurement harness are a better starting point than
stock vLLM.

## Dependencies

Two repos are maintained as sibling forks, synced to upstream:

| Repo | Branch | Base |
|---|---|---|
| [`curvedinf/int8-vllm`](https://github.com/curvedinf/int8-vllm) | `main` (authoritative) | vllm-project/vllm main @ 2026-08 |
| [`curvedinf/int8-aiter`](https://github.com/curvedinf/int8-aiter) | `main` (`3a4f0aaf4`) | ROCm/aiter main @ 2026-08; carries int8 unified-attention + gfx908 tunings |

The aiter checkout is consumed at runtime via `PYTHONPATH=../aiter` (a
sibling checkout) and
provides every attention and GEMM kernel in the recipe; no separate attention
package is used.

Fork-specific code that matters: the compatibility-named
`AiterW8A16LinearKernel` selects AITER A8W8 for every GS128 shape
(`vllm/model_executor/kernels/linear/mixed_precision/aiter_w8a16.py`), gfx908
attention tuning
(TILE_SIZE=32 decode, adaptive flash-decoding split-K) in
`vllm/v1/attention/ops/triton_unified_attention.py`, gfx908 env defaults in
`vllm/platforms/rocm.py`, and the int8 per-token-head port in aiter's
`unified_attention`.

## Models

- **[Qwen3.8-27B-GPTQ-INT8-W8A8-GS128](https://huggingface.co/curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128)**
  (matching target model; local deployment: `<models>/Qwen3.8-27B-GPTQ-8bit-gs128`)
- **[Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128](https://huggingface.co/curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128)**
  (required speculative-decoding companion; not standalone; local deployment:
  `<models>/dflash2-int8/Qwen3.8-27B-DFlash2-GPTQ-8bit`)
- Quantization recipe: `<models>/quantize_qwen38_27b_gptq8.py` (outside the repo)
  (GPTQModel 7.3.4; bits=8, group_size=128, sym, true-sequential; 512 mixed
  code+C4 calibration samples binned 256–2048)

## Serving

```bash
# The production script is fixed to the complete AITER INT8 TP4/C8 contract.
scripts/serve_recipe_qwen38.sh {start|stop|restart|status}

# systemd
sudo cp scripts/vllm-openai-gfx908-qwen38.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now vllm-openai-gfx908-qwen38
```

The complete flag contract (env pins, dtypes, spec config, NS, levers) is
maintained in exactly one place: `docs/recipes/README.md`. The launcher
`scripts/serve_recipe_qwen38.sh` is its executable form.

## Hardware / software

| Component | Value |
|---|---|
| GPU | 4x AMD Instinct MI100 32GB (gfx908), XGMI full mesh |
| ROCm | 7.x (`/opt/rocm`) |
| PyTorch | 2.11.0+rocm7.1 (serving venv) / 2.13.0+rocm7.2 (quantization venv) |
| Triton | 3.6.0 |
| Build | `PYTORCH_ROCM_ARCH=gfx908 VLLM_TARGET_DEVICE=rocm`, see `docs/contributing/incremental_build.md` |

## Testing

```bash
# int8 KV micro-correctness (AITER unified-attention path)
PYTHONPATH="$PWD:$PWD/../aiter" .venv/bin/python scripts/test_int8_kv_micro.py

# post-sync battery: int8 KV at production shapes (hdim 256, GQA 6:1),
# varlen attention interface checks, boot-import chain, GEMM whitelist
HIP_VISIBLE_DEVICES=1 .venv/bin/python scripts/battery_gfx908.py
```

Optimization history with full A/B data lives in
`docs/recipes/surface_experiments_ledger.jsonl` (committed under
`docs/recipes/` because `logs/` is gitignored); the recipe doc
`docs/recipes/README.md` is the current-status source of truth.
