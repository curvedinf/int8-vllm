# Flint + Flint-DFlash2 Int8 PTQR Recipe (gfx908 / 4x MI100)

Peer recipe to the [Qwen3.8-27B baseline](README.md) and the
[Nemotron Bolt + LiLiCorr recipe](nemotron_bolt.md). Same int8-native
doctrine: every dense GEMM is AITER W8A8 INT8 GS128, routed experts run a
true W8A8 grouped kernel, and KV is `int8_block_g128` on **both** target and
draft. There is no 16-bit fallback in this recipe.

- Upstream target: [`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint)
- Upstream draft: [`incoai/Flint-DFlash2`](https://huggingface.co/incoai/Flint-DFlash2)
- Serving script: `scripts/serve_recipe_flint.sh` (`start|stop|restart|status`)
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

- Target: `Flint-GPTQ-INT8-W8A8-GS128-PTQR-r1s40` (rung 1, step 40 kept;
  35 shards, 36,439 tensors, 38.5 GiB). Export:
  `scripts/ptqr_export_flint.py`; trainer `scripts/ptqr_train_flint.py`.
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
| Dense linears | AITER CK W8A8 INT8 GS128 (`AiterW8A16LinearKernel` legacy name dispatches W8A8) |
| Routed experts | `vllm/model_executor/layers/quantization/gptq_moe_a8w8.py` — GPTQ int8-group → AITER grouped `a8w8_blockscale` (`[E,K,N]` int8 + per-column GS128 fp32 scales, `w_blockscale_n=1`) |
| Attention | AITER unified attention |
| KV (target + draft) | `int8_block_g128` |
| All-reduce | vLLM CUSTOM (`VLLM_ROCM_USE_AITER_CUSTOM_AR=0`) |
| Act quant | `VLLM_GFX908_ACT_QUANT=round` |

Deliberate float exceptions (same class as the Qwen findings): selector
codebooks, the draft's conv `kernel_projection` ×10, `fc` and
`candidate_selector.hidden_projection` stay bf16 on their trained grid;
embeddings/`lm_head` and the router ship bf16. Everything else is int8.
The draft's 35 attention/MLP linears are **not** an exception — they are
int8 W8A8 GS128, exactly as in the Qwen PTQR draft.

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
stay local. **The first boot at 512k/C12 and a long-context validation
are still pending** — the numbers below are the measured C8/262k legs.
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

## Known deltas

- Skill/tool calling: the checkpoint template defines an XML
  `<tool_call>` format with no vLLM parser yet; the recipe does not
  enable tool calling.
- Reasoning: the template emits `<think>` blocks; the recipe uses
  `--reasoning-parser qwen3`.
- The draft export keeps the trained bf16-grid surfaces; re-quantizing
  them would need a new scaled kernel (same exception as Qwen).
