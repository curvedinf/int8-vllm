# Serve a recipe on MI100

The three settings below select a target and draft, a group of cards on one Linux host, and its collective backend. A card is one 32 GiB MI100. Use `rocm-smi --showtopotype` on the host to see how the cards connect.

## Docker

Install the host AMD GPU driver, Git, Docker Engine and Compose v2.24 or later. The image builds its own Python, Rust and ROCm user-space environment. From a fresh checkout:

```bash
git clone --branch main https://github.com/curvedinf/int8-vllm.git
cd int8-vllm
MI100_CARDS=2 MI100_XGMI=off MI100_RECIPE=qwen38 \
  docker compose -f docker/mi100/compose.yaml up --build -d
docker compose -f docker/mi100/compose.yaml logs -f serve
```

Compose fetches the fork's pinned AITER source. The existing `docker/Dockerfile.mi100` path now builds the same image directly with `docker build -f docker/Dockerfile.mi100 -t int8-vllm:mi100 .`.

| Setting | Values | Default |
| --- | --- | --- |
| `MI100_CARDS` | `2` and `4` use tensor parallelism. `3` uses three pipeline stages with the complete DFlash2 draft on the final stage. Its 32 query heads cannot be divided evenly across three tensor ranks. | `4` |
| `MI100_XGMI` | `on` for a full XGMI mesh. `off` selects RCCL and enables fine-grained PCIe peer memory. The kernel driver and hardware determine the physical path. | `on` |
| `MI100_RECIPE` | `qwen38`, `flint`, `bolt` | `qwen38` |

Compose uses AMD’s ROCm 10.2.0 user space, includes hipFile 0.6.0 for AMD Infinity Storage, and builds both forks’ native extensions. The host supplies the AMD kernel driver and the device nodes. The first start downloads the matching checkpoints into a named volume, or uses existing checkpoints under `$HOME/models`. Set `MODELS_DIR` on the host to mount a different directory. Downloads, AITER JIT modules, Triton binaries and compiler caches persist across container recreation. After changing AITER or its build ABI use a fresh AITER cache directory. With the earlier ROCm 7.2.1 image, the first two-card Qwen container reached health in 123 minutes with four compiler jobs and checkpoints already downloaded. The worker attributed 104 minutes to post-load processing, which included compiling CK A8W8. Recreating the container with the same volume took 5 minutes 59 seconds to health, including 69 seconds for target weights and 1.3 seconds for target post-load processing. These times exclude building the image and downloading the checkpoints.

Allow approximately 32 GiB for the Qwen checkpoint pair, 39 GiB for Flint or 24 GiB for Bolt, plus separate space for the image (about 57 GB unpacked on the validation host), build cache and any AIS repacks. Two-card and four-card repacks contain different tensor shards.

The API listens on `127.0.0.1:8020` on the host. A successful `curl -f http://127.0.0.1:8020/health` means the engine is ready. The launcher generates a private API key on first use. Retrieve it locally with `docker compose -f docker/mi100/compose.yaml exec serve cat /cache/mi100/api.key`, or supply your own through a private environment file. Stop with `docker compose -f docker/mi100/compose.yaml down`. Keep the named volume to reuse the downloads.

### Overrides

Set `MI100_ENV_FILE` to the absolute path of a private env file to pass settings to the recipe. Compose also reads `docker/mi100/recipe.env` if present. Shell variables `PORT`, `MI100_BIND`, `MAX_JOBS` and `MI100_SHM_SIZE` control the published port, host bind address, build parallelism and shared-memory capacity. The health check reports ready as soon as `/health` succeeds. Set `MI100_START_PERIOD` to change the default three-hour grace for first-time kernel compilation. Each recipe reads its own runtime environment:

| Setting | Applies to | Controls |
| --- | --- | --- |
| `MI100_DEVICES` | All | Ordered HIP device list. The launcher takes the first `MI100_CARDS` entries, honors an inherited HIP/CUDA visibility list, and checks XGMI by PCI address. |
| `CU_NUM` | AITER | Defaults to the CU count reported by HIP when all selected GPUs agree (120 on the validation cards). Passing that value to AITER avoids repeated `rocminfo` subprocesses during graph compilation. An explicit value overrides detection. |
| `MODEL_DIR`, `DRAFT_MODEL_DIR` | All | Explicit local checkpoint paths (use container paths under `/models` in Docker). A missing explicit path stops the launch. |
| `MODEL_REVISION`, `DRAFT_MODEL_REVISION` | Hub downloads | Override the pinned target or draft revision. |
| `VLLM_API_KEY`, `API_KEY_FILE` | All | Existing API credential or mounted key file. |
| `C`, `NS` | All | Concurrent sequences and speculative tokens. The portable Qwen launcher uses C8/NS13, Flint C12/NS7 and Bolt C8/NS7. |
| `MAXLEN`, `MNBT` | Qwen, Flint | Maximum context length and batched tokens. |
| `MAX_MODEL_LEN`, `MAX_BATCHED_TOKENS` | Bolt | Maximum context length and batched tokens. |
| `GPU_MEMORY_UTILIZATION` | All | Fraction available when vLLM sizes the KV pool automatically. A fixed KV arena takes precedence. |
| `KVMEM` | Qwen, Bolt | Per-GPU KV bytes. Qwen accepts `auto` for memory profiling. |
| `CPUTIER`, `DISKTIER` | Qwen | CPU and disk prefix-cache budgets in bytes. The portable launcher disables the disk tier by default. |
| `KV_OFFLOAD_BYTES` | Bolt | Host-memory prefix-cache budget in bytes, default 24 GiB. Leave RAM for the serving processes, weight loading and compilation as well. |
| `VLLM_BOLT_EXPERT_CACHE_GIB` | Bolt | GPU budget for decoded experts, in GiB per card. The portable profile uses 3 on two or three cards and the recipe default of 11 on four. |
| `OFFLOAD`, `PREFIXCACHE` | Qwen | Both default to `1`. For diagnosis, `OFFLOAD=0` disables the CPU connector and `PREFIXCACHE=0` disables prefix reuse. Change one setting per run. |
| `VLLM_AIS_REPACK_CPU_WRITE` | Portable launcher | Defaults to `1`. New rank caches use buffered host writes and a final sync. Cache hits use the normal AIS read path. |
| `VLLM_FS_GPU` | All | The default `auto` uses AMD Infinity Storage when available. `0` uses the ordinary weight loader. |

## Native

Use Python 3.12 with development headers, the ROCm development toolkit, a C/C++ compiler, Git, pkg-config, libnuma and OpenSSL development headers, and Rust via rustup. From the checkout, run `cargo --version` so rustup provisions the version selected in `rust-toolchain.toml`. Clone the AITER revision recorded in the Dockerfile with its composable-kernel submodule.

```bash
git clone --recurse-submodules https://github.com/curvedinf/int8-aiter.git ../int8-aiter
git -C ../int8-aiter checkout 4bd8f95f3a5cd3016592fd34ed5fb2cefc9670f0
git -C ../int8-aiter submodule update --init --recursive
python3.12 -m venv .venv
export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export PATH="$PWD/.venv/bin:/opt/rocm/bin:$PATH"
export PYTORCH_ROCM_ARCH=gfx908 GPU_ARCHS=gfx908 VLLM_TARGET_DEVICE=rocm
export PYTHONPATH="$PWD:$PWD/../int8-aiter"
docker/mi100/install-deps.sh
export MAX_JOBS=4 CARGO_BUILD_JOBS=4
.venv/bin/python -m pip install --dry-run --no-deps --no-build-isolation -e .
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
scripts/build_residual_g128.sh
.venv/bin/python scripts/test_int8_kv_micro.py
HIP_VISIBLE_DEVICES=0 .venv/bin/python scripts/battery_gfx908.py
MI100_CARDS=2 MI100_XGMI=off MI100_RECIPE=qwen38 scripts/mi100-serve.sh
```

Run the build steps in a fresh venv. The installer constrains the three PyTorch packages to ROCm builds, installs the fork's ROCm requirements, restores the ROCm Triton package if another dependency installed upstream Triton, then patches torch's bundled runtime. The native and Docker launchers use the same recipe scripts. Set runtime overrides directly in the native launch environment. The existing `scripts/serve_recipe_qwen38_tp2.sh` service commands use the same two-card PCIe profile and retain port 8021.

## Check a running recipe

The model IDs exposed by the API are `qwen3.8-27b-gptq8`, `flint` and `nemotron-bolt`. To exercise a prefix that spans multiple cache pages, run this probe from the native checkout or inside the serving container. Use the target checkpoint's tokenizer. The probe sends the same chat request twice, checks that both answers contain a verification phrase from the middle of the input and compares their final text.

```bash
.venv/bin/python scripts/probe_bolt_prefix.py \
  --base-url http://127.0.0.1:8020 --model flint \
  --api-key-file "${XDG_CACHE_HOME:-$HOME/.cache}/mi100/api.key" \
  --tokenizer "$MODEL_DIR/tokenizer.json" --background-tokens 5000 \
  --max-tokens 512 --output-dir /tmp/mi100-retrieval
```

Flint can spend its first tokens on a reasoning preamble. The larger output budget lets it finish the answer. For Qwen and Bolt the probe defaults to 24 output tokens. On an otherwise idle server, compare `vllm:prefix_cache_hits_total`, `vllm:spec_decode_num_drafts_total` and `vllm:spec_decode_num_accepted_tokens_total` at `/metrics` before and after the pair. GPU prefix reuse and reuse from the CPU tier have separate counters. On an isolated diagnostic server launched with `VLLM_SERVER_DEV_MODE=1`, add `--reset-gpu` to clear the GPU cache between requests. The probe requires an identical answer, a positive external prefix-hit delta and bytes copied from CPU to GPU. Use a fresh output directory for each run.

## Verified deployments

The following results were measured with the previous ROCm 7.2.1 Docker base and the native serving environment on one host with 32 GiB MI100 cards. The PCIe runs used the host driver's PCIe-only peer routing. The XGMI runs used all six physical links.

| Cards and fabric | Model parallelism | Native | Docker Compose |
| --- | --- | --- | --- |
| 2 PCIe | TP2 | Qwen, Flint, Bolt | Qwen, Flint, Bolt |
| 3 PCIe | PP3, TP1 | Qwen, Flint, Bolt | Qwen, Flint, Bolt |
| 4 PCIe | TP4 | Qwen, Flint, Bolt | Qwen, Flint, Bolt |
| 4 XGMI | TP4 | Qwen, Flint, Bolt | Qwen, Flint, Bolt |

Each of the 24 servers generated a correct short answer, returned the embedded verification phrase on both a fresh and a repeated roughly 5,000-token chat, and recorded prefix hits and accepted speculative tokens. Compose used the six pinned public checkpoints with an empty model mount. On two PCIe cards, Qwen and Bolt also returned the same phrase after the GPU cache was cleared and the CPU tier restored the prefix. The [experiment ledger](surface_experiments_ledger.jsonl) records the per-deployment counters under `MI100_ISSUE1_DEPLOYMENT_MATRIX`.

For the two-card Qwen slowdown in issue #1, a separate single-stream W5 diagnostic used a 3,219-token chat and 256 generated tokens per request. Across three prefix-warm repeats, median decode was **70.4 tokens/s native** (68.9–70.7) and **48.3 tokens/s in Compose** (45.1–50.1). Median warm time to first token was 0.115 s and 0.119 s respectively. The 2,405 target tensors and 186 draft tensors used by the two installations had identical contents. Both used torch 2.11.0+rocm7.1 and Triton 3.6.0. The native environment had Transformers 5.9.0 and the image had 5.20.2. Speculative acceptance differed between runs. Compare runs with the same prompt, concurrency, cache state and sampling settings. The ledger entry `MI100_ISSUE1_QWEN_TP2_PCIE_W5` includes each run and its accepted-token counts.

## Common issues

To check the cards before downloading checkpoints, run the launcher in `check` mode on idle GPUs. It reports each card's compute units and verifies a 4 MiB round-trip copy for every ordered GPU pair.

```bash
MI100_CARDS=2 MI100_XGMI=off docker compose -f docker/mi100/compose.yaml run --build --rm serve check
# Native
MI100_CARDS=2 MI100_XGMI=off scripts/mi100-serve.sh check
```

A correct copy can also travel through host memory. Use the topology and bandwidth checks below to distinguish transfer routes.

| Symptom | What to check or change |
| --- | --- |
| Torch sees no MI100 in Docker | Check `rocminfo` and `ls -l /dev/kfd /dev/dri/render*` on the host first. Pass both KFD and the render devices. For a non-root process grant the actual host device group IDs. Check `torch.cuda.device_count()` inside the image. [AMD container guide](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.1/how-to/docker.html) |
| Warm startup still takes tens of minutes, or tokens arrive seconds apart | Record the timestamps for model load, AITER builds, Inductor compilation, graph capture and ready. Check `rocm-smi --showtopotype`, `amd-smi topology -a`, `lspci -tv`, and the PCIe bridges' `ACSCtl` fields in `sudo lspci -vv`. Check Above 4G decoding, large BAR and negotiated link speed/width on every card. Measure **each pair** with ROCm Bandwidth Test or TransferBench. The reporter traced host routing to IOMMU and ACS. On a dedicated AMD bare-metal system, AMD's MI100 guide recommends enabling IOMMU in firmware and using `amd_iommu=on iommu=pt` at boot. Check `/proc/cmdline` and kernel IOMMU messages after reboot. If transfers still route upstream, review the GPU bridges' ACS redirect policy with the platform administrator and repeat the pair test after any change. ACS and IOMMU settings affect DMA and VM isolation. [Issue #1](https://github.com/curvedinf/int8-vllm/issues/1) · [AMD MI100 system guide](https://rocm.docs.amd.com/en/docs-6.2.0/how-to/system-optimization/mi100.html) |
| AITER or Inductor repeatedly compiles the same shapes | Keep the cache volume, pin the source and compiler versions, and check free space and inodes in the cache and `/tmp`. Look for JIT build messages on the second boot. A cached kernel can still take time to open. On the validation host we saw all four Bolt workers reading Triton's cached compiler artifacts while the engine logged a shared-memory broadcast wait. Check disk I/O and swap during that phase. After changing kernel sources or the tuned CSV, rebuild the affected JIT module once. |
| Flint reports `quant_out requires the 256-column gated tile` | Check the sibling AITER revision against the Dockerfile. The small-grid heuristic could shrink the two-card expert tile to 32 columns. The fused SwiGLU needs 256 columns to produce one 128-value INT8 scale group. |
| Three-card Qwen runs out of memory in `_quantize_embedding_int8_` | The last pipeline stage holds the full draft. The portable profile uses one copy of the CK weights for both prefill and decode. This fork converts vocabulary embeddings and LM-head weights in row chunks to bound temporary memory. |
| AIS is available on the host but the container uses safetensors | The loader checks for hipFile in the serving process. The host kernel driver alone does not provide the user-space library inside the container. `VLLM_FS_GPU=auto` falls back to safetensors. To check actual I/O, provide a writable scratch directory on the model filesystem and run `.venv/bin/python scripts/fs_gpu_probe.py --scratch /path/on/model/filesystem` from the checkout on an idle GPU. This checks reads, writes and that torch and hipFile load one HIP runtime. |
| First AIS repack stalls while the cache files barely grow | Check the worker's kernel wait and whether the rank cache is a miss. Repeated filesystem journal waits during `fdatasync` can serialize small hipFile writes. The portable launcher uses buffered writes for this one-time cache build. Use `VLLM_FS_GPU=0` to compare with the ordinary safetensors loader on the same filesystem. |
| Two-card startup runs out of VRAM | The portable launcher profiles Qwen's KV arena after loading the target and draft. On two cards it also releases the extra GS128 decode copy where the CK W8A8 weights can serve both prefill and decode. A fixed four-card KV allocation can exceed available VRAM even on a four-card host with different graph or draft settings. Check actual free memory on both devices, lower the context ceiling if needed, and check the model and draft both fit. |
| `Bus error` during offload initialization | Compare `df -h /dev/shm` inside the container to the configured CPU tier plus worker IPC. Compose allocates a private 32 GiB shared-memory filesystem. Increase `MI100_SHM_SIZE` or reduce the CPU tier within available host RAM. |
| Missing Rust, Python modules or a GPU extension at import | Install Rust and the pinned ROCm dependencies **before** the editable build. Verify with `.venv/bin/python -c 'import torch, triton, vllm._rocm_C; print(torch.__version__, torch.version.hip, triton.__version__)'`. Keep the sibling AITER checkout and submodule available at runtime. |
| Startup logs an import error for `is_hip_gfx1250` | The MXFP4 backend probes newer Triton target helpers. We observed this message with the pinned gfx908 stack while its INT8 microtests, four-part battery and inference passed. Verify the AITER backend in the startup log and run the battery before replacing ROCm Triton. |
| `setuptools_scm` rejects a tag or cannot find Git metadata | Use a full Git checkout with tags. For an exported source tree set a valid PEP 440 version, e.g. `export VLLM_VERSION_OVERRIDE=0.33.0+mi100` before building. The Dockerfile sets the override because its context excludes Git metadata. |
| Host memory grows during graph replay | The dependency installer reads only `torch/lib/libamdhip64.so` and `torch/lib/libhsa-runtime64.so` from the torch 2.13.0+rocm7.2 donor wheel, checks both production hashes, and preserves the original files beside them. Reapply after replacing the serving torch installation. `torch.version.hip` identifies the wheel's build version. |
| A8W8 has an untuned shape after changing TP | A CSV miss selects CK's default split. Collect the local M/N/K from the kernel log and match the serving output and scale dtypes. In the two-card Qwen test, the default split passed an independent dot-product check for all 185 reported missing shapes. A sweep of splits 0, 1, 2, 3, 4, 6 and 8 kept the default for every shape after requiring identical output and a 3% timing improvement to switch. The MI100 BF16 kernel rejected nonzero splits for the 14 × 17,408 × 5,120 example. Check those rows in the sibling fork's `aiter/configs/a8w8_tuned_gemm.csv`. Use its CK A8W8 tuner with the actual shapes and the serving venv, rebuild the affected module, then rerun the battery and serving check. [AITER tuner](https://github.com/curvedinf/int8-aiter/tree/main/csrc/ck_gemm_a8w8) |
| Four-card Qwen or Bolt starts and gives a plausible first token, then repeats fragments or loses a phrase from the prompt | Run the retrieval probe above with both a fresh and a cached prefix. We reproduced this with four PCIe cards. The first speculative step could seed Mamba state from another request slot. The fix selects the live request's block table. A separate fix restores symmetric rounding for negative queries in grouped INT8 attention. With both corrections, the same four-card deployments returned the exact verification phrase on both requests. |
| Earlier tokens change when a later value in the same attention tile changes | The grouped INT8 attention kernel used the largest value scale in the tile, including masked future tokens, to quantize every query’s attention probabilities. The fix scales the visible weighted probabilities separately for each query. A GPU regression changes a future value scale and requires all earlier causal outputs to remain bit-identical. |
| Flint produces incoherent text after disabling its INT8 output projection on PCIe | Keep the default GS128 output projection. The launcher reduces the decoded rank-local partials in FP32 on RCCL and requantizes the sum. Routed expert outputs are already reduced. The three-stage pipeline carries both codes and scales across stage boundaries. |
| First request works, repeated prefix stalls | Send the same multi-page prompt twice, save both outputs and check prefix-hit and accepted-token counters. Check shared-memory capacity and run the peer-transfer test before changing prefix caching, CPU offload or DFlash2. Each changes a different part of the recipe. |
| A repeated Qwen prompt hits in VRAM but recomputes after a GPU cache reset | Compare `vllm:external_prefix_cache_hits_total` and `vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"}` before and after the reset. An earlier connector discarded a complete DFlash prefill block as if it needed an EAGLE lookahead token. This prevented it from pairing the draft KV with the retained Mamba checkpoint. The connector now uses the same DFlash boundary as the GPU cache and still excludes volatile draft KV during decode. On two PCIe cards, the corrected native server restored 4,800 of 5,063 prompt tokens and 27,200 of 28,065, with the same answer each time. |
| Bolt matches a fine-grained prefix in VRAM but fetches nothing from RAM | Bolt keeps recurrent checkpoints inside a physical page. The offloader now pairs each partial recurrent state with the matching attention and draft pages. When the draft window crosses a page boundary, it retains and checks the preceding attention page too. The earlier store policy mistook a one-page sliding window for a rolling Mamba state. After clearing the GPU cache on two PCIe cards, Bolt restored 5,064 of 5,066 prompt tokens, 7,912 of 7,916 across a page boundary, and 28,064 of 28,067. All three answers matched the originals. |
| Comparing a single stream to the advertised aggregate | Record prompt length, sampling, concurrency, prefix reuse and accepted speculative tokens. Historical greedy aggregate throughput does not predict one production stream on two PCIe cards. [Measurement procedure](optimization_procedure.md) |
