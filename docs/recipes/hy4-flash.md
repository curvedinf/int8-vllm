# HY4-Flash on four MI100s

This recipe serves Tencent's 42-layer HY4-Flash vision-language model with tensor parallelism over four 32 GiB MI100s. The language weights use training-free NeUQI INT4. The vision encoder and five-layer DFly2 companion use PTQR INT8. All experts, the vision encoder and the drafter stay on the GPUs. The four n-gram embedding tables have their own storage setting. [Original model](https://huggingface.co/tencent/Hy4-flash) · [NeUQI](https://arxiv.org/abs/2505.17595)

## Docker Compose

Both supported layouts use TP4 on four MI100s. From the repository root, start the target and its DFly2 companion on an XGMI mesh.

```bash
docker compose -f docker/mi100/compose.hy4.yaml up --build -d
docker compose -f docker/mi100/compose.hy4.yaml logs -f serve
```

For four cards connected over PCIe, start the same image with the RCCL profile.

```bash
MI100_XGMI=off docker compose -f docker/mi100/compose.hy4.yaml up --build -d
```

Compose uses the same launcher and model defaults as the native recipe. It downloads the two checkpoints into the persistent cache on first use. To reuse local checkpoints and keep Engram on an SSD, point `MODELS_DIR` at an SSD directory containing `Hy4-flash-NeUQI-W4A8` and `Hy4-flash-DFly2-PTQR-INT8`. The target's `engram/` subdirectory holds the four tables. For tables elsewhere under the mount, set `ENGRAM_DIR=/models/path/to/tables` in `docker/mi100/recipe.env`. The optional TP-local startup cache needs about 82 GiB of additional disk space. Set `ADDITIONAL_CONFIG={"hy4_rank_cache":false}` in the same file to disable it.

The ROCm 10.2 image defaults to four RCCL channels and a 1 MiB buffer. In a four-rank startup probe this left at least 30.63 GiB free per card before weight loading, up from 27.49 GiB with the runtime defaults. Set `NCCL_MAX_NCHANNELS` and `NCCL_BUFFSIZE` in `docker/mi100/recipe.env` to override them.

The HY4 router uses a fixed order for its four FP32 split-K partial sums. The packed expert kernel also fixes the rounding of its FP16 SiLU before activation quantization. These changes reproduced native greedy IDs on all 64 saved teacher prefixes under ROCm 10.2. The largest difference in per-prefix top-20-plus-tail KL was below 0.00000008.

Check readiness with `curl -fsS http://127.0.0.1:8020/health`. Retrieve the generated API key with `docker compose -f docker/mi100/compose.hy4.yaml exec serve cat /cache/mi100/api.key`. The server exposes `hy4-flash`. To stop it, run `docker compose -f docker/mi100/compose.hy4.yaml down`. [MI100 installation and overrides](mi100-install.md)

## Native serving

Use the fork’s MI100 serving environment, the sibling `int8-aiter` checkout and FlyDSL 0.5.0. From the repository root, install the compiler, download both checkpoints and start the server.

```bash
.venv/bin/python -m pip install --dry-run 'flydsl==0.5.0'
.venv/bin/python -m pip install 'flydsl==0.5.0'
export MODEL_DIR="${MODEL_ROOT:-$HOME/models}/Hy4-flash-NeUQI-W4A8"
export DRAFT_MODEL_DIR="${MODEL_ROOT:-$HOME/models}/Hy4-flash-DFly2-PTQR-INT8"
hf download voodooquant/Hy4-flash-NeUQI-INT4-W4A8-GS128 --revision 6bcd3e95f31ff6a23201e99273ddf5ab134faec0 --local-dir "$MODEL_DIR"
hf download voodooquant/Hy4-flash-DFly2-PTQR-INT8-W8A8-GS128 --revision 026c6c08dc582c7dca03b4db4c14dcf85a9d006a --local-dir "$DRAFT_MODEL_DIR"
scripts/serve_recipe_hy4.sh start
```

On a PCIe-only host, use `MI100_XGMI=off scripts/serve_recipe_hy4.sh start`. This selects RCCL and enables fine-grained PCIe peer memory. On an XGMI mesh, the default selects vLLM custom all-reduce.

The native and Docker defaults allow eight concurrent requests and 262,144 total input and output tokens per request. Active requests share the GPU KV pool. The host RAM reserve defaults to 8 GiB (`HY4_HOST_RAM_RESERVE_GIB`). The launcher checks Linux `MemAvailable` and any tighter cgroup v2 memory limit. SSD reads can use reclaimable Linux file cache. Selecting `ENGRAM_MODE=ram` additionally requires room for all four pinned tables. The model is served as `hy4-flash` on port 8020. Startup writes to `logs/serve_recipe_hy4/server.log`. Use `scripts/serve_recipe_hy4.sh status` to check the launcher, `curl -fsS http://127.0.0.1:8020/health` to check readiness, and `scripts/serve_recipe_hy4.sh stop` to stop it. `MODEL_DIR`, `DRAFT_MODEL_DIR`, `MI100_DEVICES`, `C`, `NS`, `MAXLEN`, `MNBT`, `KV_DTYPE`, `HOST` and `PORT` can be supplied in the launch environment. The engine applies Tencent's chat template, reasoning parser, tool parser and vision processor.

### N-gram storage

| `ENGRAM_MODE` | How rows reach the GPU |
|---|---|
| `bcache` | Reads missing 54-byte rows from the model filesystem into a bounded GPU row cache. Large batches issue filesystem readahead. Linux can retain recently read pages. |
| `direct` | Reads packed pages into registered VRAM through physical AMD Infinity Storage. Place `ENGRAM_DIR` on a filesystem that passes a strict P2P read. |
| `ram` | Pins each TP rank's slice of each packed table in host memory. GPU threads gather and decode selected rows. The four ranks together use about 33.57 GiB of pinned RAM. |

`ENGRAM_DIR` defaults to the checkpoint's `engram` directory. `ENGRAM_CACHE_MIB` sets the GPU cache budget per table and rank in both disk modes. With `bcache`, `ADDITIONAL_CONFIG={"hy4_engram_cache_rows":false}` selects the page-cache implementation for small lookups. All modes use the same 54-byte Q4_0 row and produce the same decoded row. A 4 KiB page holds 75 rows. All four disk lookups share one transfer of hash IDs before a graph replay. The GPU decodes the selected rows in every mode. The engine computes hashes from the frozen tokenizer map and the preceding two tokens, including across speculative or chunked-prefill boundaries.

On a mirror backed by an SSD read cache, check cold random-read latency before benchmarking. A cold 1,997-token prefill on the test system spent five minutes waiting on storage and hit the worker RPC timeout. The server had already answered a short chat correctly. The launcher defaults to `ENGRAM_PRIME=auto`. Before model loading it probes random 4 KiB reads from each table on bcache. If a table’s 90th percentile exceeds 2 ms, it issues bounded 64 KiB direct reads over the file to populate the SSD. Set `ENGRAM_PRIME=off` to skip this step or `on` to probe any filesystem supporting direct reads. The warmer can also run on its own. Linux bcache tracks sequential streams and can bypass the SSD on long scans. [Linux bcache documentation](https://docs.kernel.org/admin-guide/bcache.html)

```bash
python3 scripts/warm_hy4_engram.py "$MODEL_DIR/engram" --probe-only
python3 scripts/warm_hy4_engram.py "$MODEL_DIR/engram" --auto
```

### Startup weight cache

The first native or container boot can create four rank-local files totaling about 82.2 GiB. Later boots validate the source fingerprint and tensor extents before restoring the cache through hipFile. Set `VLLM_AIS_REPACK_DIR` to place these files on another writable volume. The Docker default keeps them in the persistent cache volume.

On the validation host, four simultaneous reads of a new 64 MiB region per rank delivered a median of 142 MiB/s aggregate with both 2 MiB and 16 MiB transfers. Reading each region a second time reached 670 MiB/s with 2 MiB transfers and 1,940 MiB/s with 16 MiB transfers. The recipe defaults to 16 MiB. `VLLM_AIS_REPACK_WINDOW_MB` can lower the transfer size independently of GPU tensor coalescing.

### Weight and attention formats

| Surface | Representation and compute |
|---|---|
| Routed experts and calibrated dense language projections | Two unsigned INT4 codes per byte, with an FP16 step and a floating zero per 128 input features. Activations use signed INT8 with round-to-nearest quantization. |
| Vision and DFly2 projections | Trained, signed INT8 weights with one scale per 128 input channels. DFly2 shares the target's embedding, vocabulary head and five target MLA caches. |
| MLA keys, values and queries | A normalized 512-channel latent and 64 rotary channels. The default latent scale group is 128 and the rotary group is 64. QK uses signed INT8 matrix products; the default probability/value product uses FP16. |
| IndexPool | Signed INT8 128-channel pooled keys and queries. Each sparse layer selects historical four-token groups and appends the live tail. |
| Engram | Four Q4_0 tables. The decoded 96-component vectors and their projections feed the four residual streams. |
| Token embedding and vocabulary head | The checkpoint retains the source BF16 matrices. On MI100 the loader prepares an INT8 embedding lookup and the W8A8 vocabulary projection shared with DFly2. |
| Other floating operations | Model and vision norms use FP16. The router preserves its FP32 weights. Reductions, softmax and hyper-connection gates can accumulate in FP32. |

The sparse path computes selected QK scores once and reuses them across value tiles. Sliding attention can generate its own window positions, and chronological prefill attends directly over the block table. The W4 and W8 paths share activation quantization, packed GEMM dispatch, routed tiling and expert combination in the AITER fork. The narrow unsigned dot path can consume packed nibbles on gfx908. Matrix tiles expand weights in registers for signed INT8 MFMA. KV-absorption matrices needed by MLA are materialized in FP16 on each rank.

## MI100 kernel measurements

The four cards were capped at 105 W each. The dense projection sweep covered 169 combinations of precision, projection dimensions and token count from the TP4 language model, encoder and drafter. Each card ran five paired measurements on the same inputs. The GEMM schedules share one implementation for W4A8 and W8A8. While the matrix unit computes the next K128 group, the vector ALU applies activation and weight scales from the completed group. Packed INT4 weights stay packed until the instruction operands enter registers.

For each new fusion on gfx908, start with the deployed operation. Time a fused kernel that finishes vector work immediately, then time the same tile with independent work deferred into the next matrix instruction sequence. Use the same inputs and include all consumers through the operator output. Candidates include scaling completed accumulators, applying SwiGLU and quantizing completed gates, preparing sparse KV addresses and reducing completed attention scores. Preserve accumulation order, masks, signed extrema and tail dimensions in the comparison. Check the generated ISA for the intended overlap, register use and spills. Measure all four ranks together, then verify any operator gain on real prompts at the context and concurrency where the kernel runs.

A separate experiment used DOT4 on the vector ALU to compute four to nine additional token rows per wave. Disassembly confirmed that DOT4 could use scalar activation operands. Extra rows generally took longer than the deferred-scaling schedule for the same complete GEMM. The routed experts, vocabulary projection, IndexPool and vision attention were also tested at their deployed dimensions. The release uses the existing path wherever the complete operation measured faster.

| Paired operator measurement | Existing path | Candidate |
|---|---:|---:|
| First dense W4 feed-forward layer, gate/up through down projection, 96 tokens | 428 µs | 323 µs |
| Shared W4 expert, gate/up through down projection, 2,048 tokens | 1,017 µs | 907 µs |
| Selected MLA scores, 96 tokens, 8 heads, 2,048 selected keys from a 16k context | 1,042 µs | 300 µs |
| Selected MLA with the fast score path in both arms and pipelined FP16 probability/value products in the candidate | 1,490 µs | 1,225 µs |
| Complete TP4 Engram layer with 12 prepared tokens, including the collective and residual | 481 µs | 81 µs |
| The same TP4 Engram layer with 96 prepared tokens | 765 µs | 290 µs |

In the Engram measurement each rank read its four trained 96-channel heads. FlyDSL overlapped the next head’s normalization with the current head’s FP16 matrix product. A separate kernel combined the post-collective normalization, gates and residual. The timing starts after the selected Q4 rows have arrived. The serving measurements include storage I/O.

Large prefill batches cross the custom all-reduce size limit. Gathering the 1,536 normalized input features and replicating the small FP16 projection weights avoids reducing 15,360 output features per token. In a separate TP4 run this lowered the complete Engram layer from 1,189 to 553 µs at 137 tokens and from 13.40 to 3.53 ms at 2,048 tokens. The local-shard path was still faster at 136 tokens (473 versus 611 µs). The replicated projections require an additional 180 MiB per card across four layers. A second design sharded the output stream across GPUs and used two gathers. It was slower on this machine. The replicated path changes the order of FP16 rounding, so its serving quality is checked independently.

The FlyDSL operations cache their compiled host launchers. Sampling an eager worker during graph capture showed that direct calls to the JIT function traced the entire MLA kernel again at each layer. On all four ranks, a synchronized 12-row, eight-head, 64-key score call took 176–178 ms with repeated host tracing and 0.189–0.192 ms after reusing the compiled launcher. Both methods produced identical scores and physical addresses. This measurement includes Python dispatch and a GPU synchronization.

The selected MLA score and probability/value paths matched the production outputs exactly in the group-size, padding and page-boundary checks. Floating-point accumulation order mattered. Each probability tile had to seed its first matrix instruction with the rescaled running output to reproduce the existing kernel’s rounding. In the vision encoder, an isolated short-image attention win disappeared in the complete transformer block. The released checkpoint uses learned visual positions and the existing vision attention backend.

The iHC gates use the eight columns that would otherwise be idle in a 16-column FP16 matrix instruction to accumulate the residual of the FP32 gate weights. The vector units sum the squared residual activations at the same time. Over four cards, the 96-token pre gate took about 24 µs versus 32 µs with the original Triton implementation. At 2,048 tokens the complete final hyper-connection head and norm took 315 µs versus 2.57 ms on the original path. The middle-of-layer residual update benefited from fusion at 137 tokens. At 2,048 tokens, updating the residual separately and then running the overlapped gate took 466 µs, compared with 788 µs for the fused operation.

The replicated 512-expert router keeps its group scales in FP32. Its single group contains every expert. A native ROCm sigmoid/top-16 operation therefore applies the bias, selects experts and normalizes probabilities in one pass. Combined with shape-specific FlyDSL weight projection, the entire routing operation fell from 123 to 56 µs at 96 tokens and from 280 to 61 µs at 137 tokens. The selected expert sets matched on all four GPUs at every measured size. The largest routing-weight difference was below 1e-7.

The fixed-order top-16 selector also prevents greedy drift. On a repeated 1,994-token input, the grouped HIP selector returned the same expert set but changed its order for 1,660 tokens at the first routed layer. Normalized weights changed for 477 tokens, ten MoE outputs changed, and the following attention layer propagated those differences to 1,295 tokens. The HY4 launcher uses the fixed-order selector with either router projection. Repeating the trace produced identical routing weights and attention outputs.

DFly2’s anchor fusion and serial hidden correction can both consume a pair of normalized, quantized inputs directly. The FlyDSL prologue reads each strided row once per output stripe, computes its RMS and emits the G128 codes and scales expected by the existing W8 projection. At 12 requests, complete TP-local anchor fusion fell from 75 to 60 µs and hidden correction from 88 to 79 µs. The largest code difference across 128 rank and shape cases was one INT8 unit. Projecting every hidden half in advance took 552 µs for the seven-step correction chain at C12 versus 544 µs with the joint projection.

### Packed INT4 vector dots

The FlyDSL hybrid wave keeps the eight W4 weights in each 32-bit word packed for its vector rows. Two signed INT4 dot products consume the high and biased low nibbles of eight INT8 activations. A per-group correction restores the original integer sum. Matrix rows continue to use INT8 MFMA. The existing single-row HIP GEMV also uses packed INT4 dots.

Four MI100s ran at 105 W each. The table gives median complete GEMM latency for N=1,024, K=3,072 and GS128. Each entry includes any activation packing and split-K reduction used by that arm. Each GPU ran seven randomized paired repetitions.

| Activation rows | Tuned reference | Extra rows with INT8 DOT4 | Extra rows with native INT4 DOT8 |
|---:|---:|---:|---:|
| 96 | 32.3 µs | 37.9 µs | 38.7 µs |
| 320 | 85.6 µs | 103.6 µs | 103.1 µs |
| 640 | 138.2 µs | 169.7 µs | 172.5 µs |

With four extra rows, an instruction-ordering trial placed native INT4 dots in all 31 gaps between 32 MFMAs. Kernel-time measurements are in the experiment ledger. The integer-oracle gate covered partial tiles, signed extrema and different activation and weight scales in consecutive K groups. All 239,808 compared outputs matched.

The 576-channel MLA projection was also timed through FP16 output on three prefill shapes. With the shared quantized input used by the recipe, its measured latencies were 293.3 / 289.3 µs at 1,994 rows, 295.5 / 287.9 µs at 2,036 rows, 297.1 / 286.3 µs at 2,048 rows for the production and best packed-DOT8 paths respectively. Each GPU ran 13 paired repetitions.

The full-model C8 comparison used eight distinct cold prompts of about 2,000 tokens each. Median aggregate prefill was 1,517 input tok/s on the production recipe, 1,518 with the extra INT8-dot rows, and 1,508 with packed INT4-dot rows. Compared within each repetition, packed INT4 delivered 1.001× the production throughput. All three used the same four workers and 12 counterbalanced repetitions.

### Fusion and expert locality at C8

On four MI100s at 105 W each, the 64-token verification graph contained 4,855 target kernel nodes per GPU. Moving FP16 conversion into the final store of 222 dense GEMMs removed all 222 separate launches. The fused store preserved every bit across 3,803,904 checked outputs from W4A8 and W8A8 projections. Three fixed greedy requests also returned identical token IDs. Alternating the configurations with every draft forced to reject reduced median step time from 207.6 to 205.9 ms. The sampled real-prompt runs had enough baseline drift to leave the decode throughput gain unresolved.

Deferring the sparse KV address writes and lane shuffles until the first QK matrix instructions improved the complete attention operation by 0.5–1.4% on 512-token appends. The paired median full-target step improvement was 0.10% at 16,384 cached tokens, 0.28% at 65,536 and 0.04% at 131,072. Each depth used eight comparisons, with the prefix-cache hits checked for every append. Pairing two output tiles to defer SwiGLU or FP16 stores also left the best existing single-tile implementation ahead on every tested shape.

A 2,036-token cold prefill touched a median of 492 of the 512 experts in a routed layer. Visiting four nearby row tiles together raised measured L2 hits in the layer-20 down projection from 44.1% to 59.9%. Those counters came from a single-card replay with recorded routes and the production packed weight layout. Across six paired whole-model runs with eight independent cold prompts, the median aggregate input rates were 1,419 tokens/s with the existing order and 1,430 with grouped tiles. Grouping won four pairs. These schedules remain experimental. The experiment ledger records the run conditions and spreads. [Experiment ledger](surface_experiments_ledger.jsonl)

### Causal prefill, IndexPool and tensor-parallel reductions

The four cards ran at 105 W each. Operator comparisons used frozen serving inputs on all four ranks. Full-model trials used eight real requests with seven DFly proposals. The long-context comparison appended 512 tokens to a cached prefix.

| Causal attention shape | Baseline | Fastest measured tile | Maximum output difference |
|---|---:|---:|---:|
| 512 tokens, 16 heads, 131072-token prefix | 1.27 ms | 1.22 ms | 0 |
| 512 tokens, 16 heads, 16384-token prefix | 1.26 ms | 1.21 ms | 0 |
| 512 tokens, 16 heads, 65536-token prefix | 1.27 ms | 1.21 ms | 0 |
| 2,036 tokens, 16 heads | 4.17 ms | 3.47 ms | 0 |
| 2,036 tokens, 8 heads | 4.25 ms | 3.71 ms | 0 |

The single-row vector-dot path reproduced the routed MFMA output on all three sampled layers. Its best inline variant was 11–24% slower with warm inputs. The separate-dispatch version also increased latency.

Widening each causal value tile to 256 channels and sharing page addresses increased aggregate cold prefill by 5.3% in paired runs.
On XGMI the recipe divides long IndexPool searches by query row and gathers the selected IDs. The 512-token full-model append took 2.8% less time at 65,536 tokens, 7.8% less time at 131,072 tokens. Greedy output also matched on appends of 513, 777 and 1,023 tokens.
The selected XGMI collective configuration increased aggregate cold prefill by 13.4% across eight interleaved comparisons. It raises the custom all-reduce message limit from 8 to 64 MiB.
Combining routed and shared expert reductions removed 41 calls from the captured target graph. The serving comparison retained separate reductions.
Layer-specific expert fusion was tested across all 41 routed layers and on two independent verification inputs. 3 layers passed the operator gate. The end-to-end gate retained the existing verify schedule.

The `hy4_index_tp_min_width` setting counts compressed groups of four tokens each. The default starts sharing at 16,384 groups on XGMI. Set it to zero in `ADDITIONAL_CONFIG` to keep each search local to its rank. Short batches and the PCIe profile use the replicated search.

For XGMI, `VLLM_TP_CAR_MAX_BYTES` sets the custom all-reduce buffer size in bytes. The recipe uses 67,108,864.

Causal prefill supports `hy4_causal_long_chunk_value_tile` and `hy4_causal_long_chunk_page_pair` for batches of at least 1,024 tokens. `hy4_causal_page_pair` controls the shorter sliding-window batches. These G128 optimizations apply to requests prefilling at least 128 query tokens.

The experiment ledger contains the paired serving ratios, rank-local graph timings, clock measurements and exact settings.

### C8 verification at long context

With seven draft positions, eight requests give the target 64 rows to verify. For a compressed IndexPool search width of at least 32,768, each MI100 on XGMI searches 16 rows. Each GPU contributes the first token ID of each four-token group through the custom all-reduce. Each rank reconstructs its chronological indices and masks the last group at the query position. In the live comparison, all four ranks reproduced the replicated search across 242,221,056 selected IDs per rank.

At 105 W per card, paired native C8 measurements increased aggregate decode by 4.9% on the ordinary W1 workload, 10.1% with a 131,072-token shared prefix, and 11.3% with a 260,096-token shared prefix. Each long-context request added its own continuation and generated 160 tokens at temperature 1. The long runs share KV pages across requests.

The merged Q-A and KV-A prefill projection reproduced both outputs at all 42 layers on every GPU. It won 4/8 cold-prefill pairs with a median change of +0.3%. Multi-expert wave scheduling also matched the routed outputs. The existing expert kernel was faster on all three sampled layers, including tests that assigned only light experts to the new kernel.

Vocabulary-sharded DFly sampling produced the same draws and committed tokens on identical live inputs. With a single distribution exchange on a side stream, the paired C8 decode change was -0.6% (3/8 pairs faster). The recipe retains the original sampling path. Redistributing verification by request reduced forced-rejection step time by 3.0%, and won 5/8 ordinary serving pairs. [Experiment ledger](surface_experiments_ledger.jsonl)

## Serving performance

Four MI100s ran at 105 W per card with TP4, C8, a 262,144-token maximum context, DFly2 NS=7, latent group 128 and rotary group 64. Native used PyTorch 2.13.0+rocm7.14.0. Docker used PyTorch 2.15.0+rocm10.2.0. Both used FlyDSL 0.5.0. The XGMI runs used the installed full mesh. For the PCIe runs, RCCL staged transfers through host memory with GPU peer transport disabled. This measures the PCIe fallback on the same machine. Direct GPU-to-GPU PCIe performance on an unbridged host is unmeasured.

The table gives the median and range of three measured runs after warm-up. The clients used real code excerpts, temperature 1.0, top-p 0.95 and top-k 20. The packed Engram tables stayed on the SSD-backed filesystem.

| Workload | Runtime | Aggregate input tok/s | Aggregate output tok/s | Emitted tokens/round | Decode ms/round |
|---|---|---:|---:|---:|---:|
| 8 distinct 2k prompts, 256 output | Native XGMI | 1,389 (1,389–1,392) | 96.9 (95.1–96.9) | 2.50 | 206.6 |
| 8 distinct 2k prompts, 256 output | Docker XGMI | 1,497 (1,494–1,499) | 103.4 (98.6–104.4) | 2.50 | 194.6 |
| 8 distinct 2k prompts, 256 output | Native PCIe | 1,221 (1,220–1,221) | 99.0 (95.9–99.1) | 2.49 | 201.4 |
| 8 distinct 2k prompts, 256 output | Docker PCIe | 1,248 (1,246–1,248) | 98.9 (92.9–100.0) | 2.50 | 202.7 |
| 8 streams, warm 16k prefix, tools, 256 output | Native XGMI | — | 109.1 (107.5–109.6) | 2.57 | 188.2 |
| 8 streams, warm 16k prefix, tools, 256 output | Docker XGMI | — | 116.9 (114.1–117.1) | 2.73 | 187.7 |
| 8 streams, warm 16k prefix, tools, 256 output | Native PCIe | — | 104.4 (96.1–104.4) | 2.57 | 197.0 |
| 8 streams, warm 16k prefix, tools, 256 output | Docker PCIe | — | 109.3 (109.3–109.4) | 2.75 | 201.1 |
| 8 streams, warm 16k prefix, tools, 2,048 output | Native XGMI | — | 103.3 (102.4–104.7) | 2.70 | 209.0 |
| 8 streams, warm 16k prefix, tools, 2,048 output | Docker XGMI | — | 105.3 (104.1–107.4) | 2.84 | 215.0 |
| 8 streams, warm 16k prefix, tools, 2,048 output | Native PCIe | — | 101.1 (100.2–102.0) | 2.60 | 206.0 |
| 8 streams, warm 16k prefix, tools, 2,048 output | Docker PCIe | — | 104.6 (104.1–104.8) | 2.82 | 215.9 |
| 8 streams, warm 32k prefix, tools, 256 output | Native XGMI | — | 104.6 (94.9–104.8) | 2.65 | 202.7 |
| 8 streams, warm 32k prefix, tools, 256 output | Docker XGMI | — | 91.6 (84.2–97.9) | 2.48 | 216.8 |
| 8 streams, warm 32k prefix, tools, 256 output | Native PCIe | — | 100.2 (90.1–100.2) | 2.55 | 204.0 |
| 8 streams, warm 32k prefix, tools, 256 output | Docker PCIe | — | 94.0 (94.0–94.2) | 2.48 | 211.0 |
| 4 requests already decoding, then 4 cold 2k prompts | Native XGMI | 1,236 (1,236–1,236) | 19.7 (17.3–19.7) | — | — |
| 4 requests already decoding, then 4 cold 2k prompts | Docker XGMI | 1,254 (1,249–1,254) | 17.6 (17.5–17.6) | — | — |
| 4 requests already decoding, then 4 cold 2k prompts | Native PCIe | 1,105 (1,103–1,108) | 16.2 (15.5–17.2) | — | — |
| 4 requests already decoding, then 4 cold 2k prompts | Docker PCIe | 1,090 (654–1,097) | 15.4 (10.8–18.0) | — | — |
| 1 tool-enabled 3k prompt, 256 output | Native XGMI | — | 37.8 (37.4–37.8) | 2.98 | 78.8 |
| 1 tool-enabled 3k prompt, 256 output | Docker XGMI | — | 43.5 (42.3–44.8) | 3.32 | 76.5 |
| 1 tool-enabled 3k prompt, 256 output | Native PCIe | — | 26.4 (26.4–26.4) | 2.93 | 110.9 |
| 1 tool-enabled 3k prompt, 256 output | Docker PCIe | — | 29.9 (29.0–30.0) | 3.33 | 112.6 |

The 2k input rate comes from matched one-output-token requests. For the decode rates, every stream had emitted its first token and remained active throughout the measured interval. The mixed row counts both inputs and outputs over the same interval, from the second cohort’s arrival until its last request emitted a first token. A dash indicates an unmeasured phase. The ledger includes per-stream decode, latency distributions, acceptance by draft position, GPU clocks and batch shapes. [Experiment ledger](surface_experiments_ledger.jsonl)

### Matched native operator comparison

An earlier C12, 65,536-context experiment on the same four native workers compared conventional packed operators with the selected matrix, attention and auxiliary kernels. Both arms kept the checkpoint, fixed-order router selection, SSD row cache, seven draft tokens and G128/G64 attention scales. Five measured pairs alternated order after warm-up at 105 W per card. The table gives medians and ranges.

| Runtime operators | 2k aggregate input tok/s | Warm 16k aggregate output tok/s | Emitted tokens/round | Decode ms/round |
|---|---:|---:|---:|---:|
| Conventional packed | 1,144 (1,036–1,191) | 99.7 (95.4–103.4) | 2.67 | 320.7 |
| Selected | 1,241 (1,144–1,270) | 128.4 (127.0–139.9) | 2.67 | 242.7 |

The final C12 storage check used the SSD and reached 138.9 (137.1–143.1) aggregate output tok/s on the warm 16k workload. An earlier native configuration measured 160.9 (158.3–171.4) tok/s with the tables pinned in RAM. At the final comparison, pinning 33.57 GiB from 35.88 GiB available would have exceeded the eight-GiB reserve, so the fixed-order configuration stayed on SSD. These measurements came from different software configurations.

### Kernel dispatch

These are the defaults in the shared native and Compose launcher. The router always uses fixed-order expert selection. The dense and attention paths keep the existing implementation for shapes outside their tuned sets.

| Surface | Default |
|---|---|
| Dense W4A8/W8A8 projections | FlyDSL deferred group scaling at the selected shapes |
| Selected MLA scores | FlyDSL QK |
| Absorbed MLA verification query | Packed verification layout |
| Selected MLA values | FlyDSL pipelined FP16 probability/value product |
| Engram | FlyDSL head projection |
| Engram post-collective | Fused normalization, gates and residual |
| Identity hyper-connections | FlyDSL gate at the selected shapes |
| MoE router projection | AITER router projection |
| DFly2 anchor and hidden correction | Paired FlyDSL RMS and INT8 quantization |
| Vision MLP | Fused W8, bias, GELU and G128 quantization at measured patch counts |
| Large Engram prefills | Replicated projection from 137 tokens |

### Attention precision and draft width

These sampled native comparisons use the same selected projection kernels and SSD tables. Latent and rotary group sizes give the number of INT8 channels per scale. Each throughput is the median and range of three C8 full-overlap decode intervals on the warmed 16k tool workload. The independent 114-question MMLU check and the 64 saved calibration prefixes were evaluated on each arm. Each teacher prefix ran alone with a fresh cache salt to keep its prefill geometry fixed.

| Latent / rotary group | Draft tokens | MMLU | Teacher top-1 | Median grouped KL | Aggregate output tok/s | Emitted tokens/round | Decode ms/round |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 / 64 | 7 | 101/114 | 49/64 | 0.098 | 109.1 (107.5–109.6) | 2.57 | 188.2 |
| 128 / 64 | 3 | 101/114 | 49/64 | 0.098 | 146.1 (142.8–156.8) | 2.35 | 129.2 |
| 128 / 64 | 5 | 101/114 | 49/64 | 0.098 | 136.5 (133.7–139.8) | 2.45 | 143.5 |
| 32 / 16 | 7 | 102/114 | 50/64 | 0.121 | 95.1 (91.6–96.9) | 2.71 | 227.7 |
| 64 / 32 | 7 | 104/114 | 49/64 | 0.147 | 112.7 (110.4–115.4) | 2.63 | 187.0 |

The first row is the released configuration. The experiment ledger also records the 2k, 32k and 2,048-output-token comparisons for each arm.

To reproduce the 2k measurements, point `MODEL_DIR` at the target checkpoint and set `VLLM_API_KEY` when the server requires it. Run these commands from the checkout.

```bash
mkdir -p logs/hy4-bench
.venv/bin/python -m scripts.bench_hy4_workloads --prepare --revision 7fec42ad6f398ca3d5c09b28471b68de919e0815 --tokenizer "$MODEL_DIR" --manifest logs/hy4-bench/prompts.json
.venv/bin/python -m scripts.bench_hy4_workloads --manifest logs/hy4-bench/prompts.json --workload W2 --prefill-only --output logs/hy4-bench/prefill
.venv/bin/python -m scripts.bench_hy4_workloads --manifest logs/hy4-bench/prompts.json --workload W2 --output logs/hy4-bench/decode
```

## Quality and memory

Each of the four runtime and transport combinations passed all 52 chat checks, including ten strict-format prompts, and handled default and disabled thinking correctly. Each identified red and blue images, two cats in a COCO photograph and the color of a four-frame video. All returned the expected tool name and arguments and repeated their recall answers with identical token IDs at 16,530 and 32,995 input tokens.

| Runtime | MMLU correct | Unparseable answers | Teacher top-1 agreement | Grouped KL median / mean / p95 | Model GiB/card | KV GiB/card | KV token capacity |
|---|---:|---:|---:|---|---:|---:|---:|
| Native XGMI | 101/114 | 1 | 49/64 | 0.124 / 0.558 / 2.911 | 23.89 | 3.20 | 372,870 |
| Docker XGMI | 101/114 | 1 | 48/64 | 0.111 / 0.569 / 2.710 | 23.89 | 3.34 | 389,393 |
| Native PCIe | 102/114 | 0 | 49/64 | 0.124 / 0.558 / 2.911 | 23.89 | 3.50 | 407,395 |
| Docker PCIe | 101/114 | 1 | 46/64 | 0.121 / 0.604 / 2.537 | 23.89 | 3.34 | 388,900 |

MMLU used two questions from each of 57 subjects and requested one letter with thinking disabled. Teacher agreement compares the served model with the original target’s saved next-token distributions at 64 calibration prefixes. KL uses the 20 leading tokens and the combined tail. This calibration diagnostic and the independent multiple-choice evaluation measure different aspects of the conversion. Model memory is the engine’s per-rank model allocation. KV memory and capacity come from its startup report. [Multiple-choice corpus](https://github.com/Wallawalla47/Infernix/blob/0155ec1b94498d18dc38a5251a23accda9fbfa54/eval/corpora/iqs-1/tasks/mmlu-500.jsonl)

The concurrent teacher probe can schedule different prefill sizes on separate runs. Replaying the same 64 cold prefixes one at a time gave native xgmi 49/64 top-1 and median KL 0.098, docker xgmi 49/64 top-1 and median KL 0.098, native pcie 49/64 top-1 and median KL 0.098, docker pcie 49/64 top-1 and median KL 0.098. In a separate check at latent G128, rotary G64 and seven draft tokens, changing companion prompts at a fixed batch size preserved all six greedy choices on prefixes selected for their sensitivity. Increasing the batch from one to eight changed two of those choices. The attention-precision comparison uses the single-request probe.

### Long context

The launcher defaults to eight concurrent requests and 262,144 tokens per request. Tencent’s source config supports 1,048,576 tokens. For each length below, one request submitted the exact number of input token IDs and reserved 32 output tokens. A random-order archive of real code contained a single release codename about one fifth of the way through. The client checked the complete submitted token sequence and exact greedy answer, then repeated the request with the same cache salt. Each cold run used a new salt.

| Runtime | Offered tokens/request | Actual input tokens | Cold time | Repeat time | Exact answer and repeated output IDs |
|---|---:|---:|---:|---:|---|
| Native XGMI | 262,144 | 262,112 | 196.2 s | 0.9 s | Yes |
| Docker XGMI | 262,144 | 262,112 | 218.0 s | 0.9 s | Yes |
| Native PCIe | 262,144 | 262,112 | 271.0 s | 0.9 s | Yes |
| Docker PCIe | 262,144 | 262,112 | 275.9 s | 0.9 s | Yes |

A third request generated all 32 reserved tokens after the 262,112-token input, reaching 262,144 total tokens. A second test primed eight independent, equally long code archives, each with its own cache salt and codename. A synchronized burst decoded 64 tokens per client. All eight were active together and recovered their own codenames.

| Runtime | Input per independent request | Eight inputs combined | Peak simultaneous requests | Preemptions | Minimum host RAM available during the context tests |
|---|---:|---:|---:|---:|---:|
| Native XGMI | 36,864 | 294,912 | 8 | 0 | 41.1 GiB |
| Docker XGMI | 38,912 | 311,296 | 8 | 0 | 36.0 GiB |
| Native PCIe | 40,960 | 327,680 | 8 | 0 | 37.2 GiB |
| Docker PCIe | 38,912 | 311,296 | 8 | 0 | 35.3 GiB |

To repeat the 256k check from the checkout, set `MODEL_DIR` to the target checkpoint and supply `VLLM_API_KEY` if authentication is enabled.

```bash
.venv/bin/python -m scripts.bench_hy4_workloads --prepare --tokenizer "$MODEL_DIR" --manifest hy4-workloads.json
.venv/bin/python -m scripts.eval_hy4_flash --only context --context-manifest hy4-workloads.json --context-tokens 262112 --timeout 14400 --output hy4-context
```

The weights and active KV stay in GPU memory throughout. The reported KV capacity applies across active requests and varies with request lengths because sliding-window and full-history groups have different block requirements. The SSD-backed Engram table and the operating system’s file cache are accounted separately.

The target and vision safetensors occupy 84.55 GB, and the drafter occupies 0.86 GB. The four SSD-backed Engram tables are separate. After the first chat, whole-device VRAM usage including graph and KV allocations was 29.5–29.7 GiB/card native xgmi, 29.4–29.7 GiB/card docker xgmi, 28.8–28.9 GiB/card native pcie, 29.1–29.3 GiB/card docker pcie.

An independent inventory of the loaded target, vision encoder and DFly2 found every model parameter on its rank’s GPU. Deduplicating shared parameter storage gave 20.96 GiB per GPU, before additional kernel buffers, layout copies, KV and graphs.

From process start to the first verified chat response, the native XGMI cache-restored boot took 15.0 minutes. The XGMI Compose service took 42.0 minutes on its first model boot and 15.0 minutes after recreating the service with the same cache volume. The first container started with an empty model mount and fetched both complete public checkpoints into a new private cache volume. Both revisions were available after 0.3 minutes and the first model initialization and chat took another 41.8 minutes. Every downloaded checkpoint file was independently SHA-256 verified. The recreated service reused that cache. These times include storage and graph initialization.

System-wide monitoring every two seconds covered model startup, validation and the full performance workloads. The minimum Linux `MemAvailable` was 40.9 GiB Native XGMI, 35.3 GiB Docker XGMI, 36.9 GiB Native PCIe, 34.9 GiB Docker PCIe. The recipe’s target is 8 GiB available. Linux can reclaim file-cache pages used for the SSD table.

## Implementation references

Basalt’s HY4 implementation provides a comparison for the sparse attention index, DFly2 target-cache reuse and selective Engram reads. On MI100 the index is scored with signed INT8 matrix instructions, with equal scores ordered by their logical positions. [Basalt HY4](https://github.com/jesdga95/basalt/tree/caa7c7a08bf8307e7f6226e59782c3987cbf154e/src/models/hy4flash)

The routed kernels group token rows on the GPU and reuse each packed expert tile across those rows. The paired gate/up projection can write the quantized SwiGLU output directly for the down projection. These choices draw on Strata’s grouped prefill, Infernix’s resident expert implementation and NInfer’s quantized SwiGLU product. All 16 selected experts participate in inference. [Strata grouped prefill](https://github.com/Niko1221/Strata/blob/58ee3227720629ff2e41d640165ebe89c880dfef/engine/include/strata/prefill/moe_fused.hpp) · [Infernix resident MoE](https://github.com/Wallawalla47/Infernix/blob/0155ec1b94498d18dc38a5251a23accda9fbfa54/include/infernix/ops/resident_moe.h) · [NInfer quantized SwiGLU](https://github.com/Neroued/ninfer/blob/ebdc67e584bcd9ebe16ee63af9abeb2118bac6cd/src/ops/moe/swiglu/quantized_product.cuh)

## Building the checkpoints

The source release is pinned to `tencent/Hy4-flash@b2498eba6a1a3f9bbcab4190bc1313a3c7010c2f`. Conversion uses Tencent's reference model implementation from Transformers PR 50903 at `b865850d05326cf20e306b31f14b0027de802da3` in a separate environment. The serving implementation backports vLLM commit `b067b6f6fa509a8c98c5a79d75b2aa3742d539c6`. [Transformers reference](https://github.com/huggingface/transformers/pull/50903) · [vLLM implementation](https://github.com/vllm-project/vllm/commit/b067b6f6fa509a8c98c5a79d75b2aa3742d539c6)

1. Supply the original configuration, tokenizer, license and safetensors index. Prepare a JSONL calibration corpus with a `text` field and a separate image directory for PTQR. Choose a private scratch directory on a physical GPU-direct storage path. The quantizers require both `HIPFILE_ALLOW_COMPAT_MODE=false` and `HIPFILE_FORCE_COMPAT_MODE=false` before process startup.
2. For a complete source checkpoint on direct storage, `scripts/quantize_hy4_engram.py` packs the four tables. For immutable Hub range acquisition, `scripts/hy4_engram_source.py` packs the tables and saves the original rows requested by the calibration corpus. Use the same corpus, sequence count, length limit, workspace and source revision in language conversion.
3. Run `scripts/quantize_hy4_flash.py` with `--corpus`, `--draft-cache` and the source/output directories. Its per-layer frozen forward pass measures activation second moments, including separate diagonals for routed experts. NeUQI searches the scale and floating offset for each G128 group. The language model has no optimizer or training steps. With a small direct-storage workspace, `--storage-gpus` holds finished shards on peer GPUs through export and `--hub-repo` with `--revision` acquires bounded source ranges. The GPU stores target hidden states and the five selected KV projections for DFly2 training.
4. Run `scripts/ptqr_hy4_vision.py` on the images and `scripts/ptqr_hy4_dfly.py` on the captured target states. These procedures train the small models under their deployed INT8 grids. DFly2 compares corrected, sampled draft trajectories with the original drafter. The vision objective compares visual features against a frozen encoder.
5. Switch back to the MI100 serving environment and its normal `PYTHONPATH` before assembling. Mixing the reference Transformers environment with the serving extensions can load conflicting LLVM libraries. Run `scripts/assemble_hy4_flash.py SOURCE --target TARGET --draft DRAFT`. It verifies checkpoint indices, all expert payloads, safetensor extents and table sizes, and installs the tokenizer, chat template and license.

For the bounded Hub conversion, set `SOURCE` to the original config, tokenizer, processor and safetensors index directory. Set `DIRECT` to a writable native NVMe directory with verified physical hipFile reads and writes, `TARGET` and `DRAFT` to the two destination directories, `CORPUS` to the calibration JSONL, and `IMAGES` to the calibration image directory. The published corpus alternated 32 C4 and 32 Evol-CodeAlpaca records. Within each source it used 14 records of 200–511 tokens, 10 of 512–1,023, six of 1,024–2,047 and two of 2,048 or more. Input sequences were capped at 2,304 tokens. Sampling used the first 3,530 C4 and 3,940 Evol-CodeAlpaca training records longer than 500 characters, shuffled separately with Python seeds 418 and 419. The code records join `instruction`, `input` and `output` with two newlines. The SHA-256 of the UTF-8 calibration JSONL was `628904c08294bc018386ad33a6b8f3fe5bb9cddfe9ffd9cbedb8457df1c20368`. The vision run used 60 training and 12 held-out COCO images.

In the reference Transformers environment, run the following from the fork root. GPU 0 handles the numerical language conversion and GPUs 1–3 hold completed language shards. Table conversion can finish before language conversion starts.

```bash
export HIPFILE_ALLOW_COMPAT_MODE=false HIPFILE_FORCE_COMPAT_MODE=false
revision=b2498eba6a1a3f9bbcab4190bc1313a3c7010c2f
source_args=(--hub-repo tencent/Hy4-flash --revision "$revision")
corpus_args=(--corpus "$CORPUS" --samples 64 --length 2304)

python -m scripts.hy4_engram_source "$SOURCE" "$TARGET/engram" \
  "${source_args[@]}" "${corpus_args[@]}" --workspace "$DIRECT" --device 2
python -m scripts.quantize_hy4_flash "$SOURCE" "$TARGET" \
  "${source_args[@]}" "${corpus_args[@]}" --workspace "$DIRECT" --device 0 \
  --storage-gpus 1,2,3 --draft-cache "$DIRECT/draft-examples"
python -m scripts.ptqr_hy4_vision "$SOURCE" "$TARGET" "$IMAGES" \
  "${source_args[@]}" --download-workspace "$DIRECT" --constant-dither
python -m scripts.ptqr_hy4_dfly "$SOURCE" "$DRAFT" "$DIRECT/draft-examples" \
  "${source_args[@]}" --workspace "$DIRECT"
```

With the serving environment restored, assemble both exports.

```bash
.venv/bin/python -m scripts.assemble_hy4_flash "$SOURCE" --target "$TARGET" --draft "$DRAFT"
```

The bounded acquisition path uses an HTTP client to write source byte ranges to native NVMe. HipFile reads those ranges into registered VRAM. After numerical conversion, hipFile writes completed GPU artifacts to native NVMe and filesystem copying installs them on the destination volume. Network and filesystem transport can use host buffers. The quantizers never create CPU weight tensors. On this system strict physical P2P works on native NVMe; the layered bcache filesystem uses hipFile compatibility mode for serving.

Tencent Hy License, Version 1.0 applies to the checkpoints. [License](https://huggingface.co/tencent/Hy4-flash/blob/main/LICENSE)
