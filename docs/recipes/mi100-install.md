# Install and serve a model on MI100

These recipes run an OpenAI-compatible API on a single Linux host with 32 GiB AMD Instinct MI100 cards. Choose a model and the connection between your GPUs. The launcher downloads the matching target and draft together.

| Model | `MI100_RECIPE` | Cards | API model ID | Disk for checkpoints |
| --- | --- | --- | --- | --- |
| [Qwen3.8-27B](README.md) | `qwen38` | 2, 3 or 4 | `qwen3.8-27b-gptq8` | About 32 GiB |
| [Flint](flint.md) | `flint` | 2, 3 or 4 | `flint` | About 39 GiB |
| [Nemotron Bolt](nemotron_bolt.md) | `bolt` | 2, 3 or 4 | `nemotron-bolt` | About 24 GiB |
| [HY4-Flash](hy4-flash.md) | `hy4` | 4 | `hy4-flash` | About 113 GiB including the n-gram tables |
| Inkling 2 Mini | `inkling2mini` | 2 or 4 | `inkling-2-mini` | About 15 GiB |

Run `rocm-smi --showtopotype` on the host. Use `MI100_XGMI=on` for a full XGMI mesh among the selected cards and `off` for PCIe. With two or four cards the launcher splits the model using tensor parallelism. With three it uses three pipeline stages. Inkling 2 Mini takes two or four cards only: its 8 KV heads cannot be split three ways.

## Docker

Install the AMD GPU driver, Git, Docker Engine and Docker Compose v2.24 or later. The host must expose `/dev/kfd` and the render devices under `/dev/dri`. Allow disk space for the image and build cache in addition to the checkpoints. HY4 can also create an 82 GiB startup weight cache.

Clone the fork once and run the serving commands from its root.

```bash
git clone --branch main https://github.com/curvedinf/int8-vllm.git
cd int8-vllm
```

Start the model you want. This example selects Qwen on four XGMI-connected cards. Change the three exported settings to match your model and host.

```bash
export MI100_RECIPE=qwen38 MI100_CARDS=4 MI100_XGMI=on
docker compose -f docker/mi100/compose.yaml up --build --wait
```

The first start builds the image, downloads any missing checkpoints and compiles GPU kernels. Compose returns when the service is healthy. Follow startup from another terminal with `docker compose -f docker/mi100/compose.yaml logs -f serve`. Stop it with `docker compose -f docker/mi100/compose.yaml down`. The named `cache` volume preserves downloads and compiled kernels across starts.

### Overrides

Set Compose variables in your shell before running `up`. Use a container env file for settings read by the launcher. Compose reads `docker/mi100/recipe.env` if present. To use a different file, set `MI100_ENV_FILE` to its absolute path.

| Where to set it | Setting | Purpose |
| --- | --- | --- |
| Host shell | `MODELS_DIR` | Mount a local checkpoint directory read-only at `/models`. Defaults to `$HOME/models`. |
| Host shell | `PORT`, `MI100_BIND` | Change the published port and address. Defaults to `8020` on `127.0.0.1`. |
| Host shell | `MAX_JOBS`, `MI100_SHM_SIZE` | Set build parallelism (default `4`) and container shared memory (default `32gb`). |
| Container env file | `MODEL_DIR`, `DRAFT_MODEL_DIR` | Use existing target and draft directories. Supply full paths as seen **inside** the container, such as `/models/my-target` and `/models/my-draft`. |
| Container env file | `MI100_DEVICES` | Select the ordered HIP device IDs, for example `1,3` with `MI100_CARDS=2`. |
| Host shell or container env file | `MI100_CPU_MAP` | Optional. Set one space-separated CPU list per GPU, in `MI100_DEVICES` order. |
| Container env file | `VLLM_API_KEY` | Use your own API key. Keep the env file private. |
| Container env file | `C`, `MAXLEN` (Qwen, Flint, HY4, Inkling 2 Mini), `MAX_MODEL_LEN` (Bolt) | Set concurrency and the per-request context limit. |

With automatic downloads the checkpoints live in the cache volume. Mounting `MODELS_DIR` supplies an alternative source for existing files. For HY4, put the actual n-gram files on SSD. [HY4 storage options](hy4-flash.md#n-gram-storage)

### Pin workers to CPU cores

For example, with four GPUs and CPUs 0–15 available, set
`MI100_CPU_MAP="0-3 4-7 8-11 12-15"`. Use commas to include SMT siblings in
the same group, such as `0-3,48-51`. Each group must belong to one NUMA
node. To keep a worker within one L3 cache, choose cores that share the
same L3 on your host. Both MI100 Compose files accept the map in the host
shell or in `recipe.env`. A nonempty shell value takes precedence. For native
starts, export the map in the serving shell. The native launcher requires
`numactl` when a map is set.

Run `scripts/mi100-serve.sh check` with your recipe settings to see the GPU,
CPU list, NUMA node and shared L3 CPU lists for each rank. For Docker use
`docker compose -f docker/mi100/compose.yaml run --rm serve check` with the
same Compose file and environment as your server. Omit `MI100_CPU_MAP` to
use the recipe's normal CPU scheduling.

The Inkling 2 Mini launcher also moves the API server and the engine core
onto the next free cores once the workers are bound (eight for the API server,
twelve for the engine core), so tokenization and scheduling never share a
rank's cores. Set
`MI100_PARENT_CPUS=none` to leave them to the normal CPU scheduling.

## Native

Start in the `int8-vllm` checkout. Use Python 3.12 with venv and development headers, the ROCm development toolkit with hipFile, a C/C++ compiler, libclang, libnuma and OpenSSL development headers, pkg-config, Git and Rust via rustup. The serving account needs access to the GPU devices. Qwen and Bolt also need sufficient locked memory and `/dev/shm` for their CPU prefix caches.

Clone AITER beside `int8-vllm` and use the revision pinned by this checkout. If the sibling checkout already exists, use the checkout and submodule commands there.

```bash
aiter_revision=$(sed -n 's/^ARG AITER_REV=//p' docker/mi100/Dockerfile)
git clone https://github.com/curvedinf/int8-aiter.git ../int8-aiter
git -C ../int8-aiter checkout --detach "$aiter_revision"
git -C ../int8-aiter submodule update --init --recursive
```

Build the serving environment in a fresh venv. These commands use the ROCm 10.2 stack shipped in Compose. For the fork’s original ROCm 7.1 stack, invoke `docker/mi100/install-deps.sh` without the `--rocm10.2` argument.

```bash
cargo --version
python3.12 -m venv .venv
export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PWD/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/python_startup:$PWD:$PWD/../int8-aiter"
export PYTORCH_ROCM_ARCH=gfx908 GPU_ARCHS=gfx908 VLLM_TARGET_DEVICE=rocm
export MAX_JOBS=4 CARGO_BUILD_JOBS=4
docker/mi100/install-deps.sh --rocm10.2
.venv/bin/python -m pip install --dry-run --no-deps --no-build-isolation -e .
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
scripts/build_residual_g128.sh
```

Use the native start command on your [Qwen](README.md#native-serving), [Flint](flint.md#native-serving), [Bolt](nemotron_bolt.md#native-serving) or [HY4](hy4-flash.md#native-serving) page. HY4 includes one additional package, FlyDSL. Set launcher overrides in your shell. For local checkpoints, `MODEL_DIR` and `DRAFT_MODEL_DIR` take host paths. To start a native recipe at boot, see the [systemd instructions](services.md).

## API access

Set your client’s base URL to `http://127.0.0.1:8020/v1` and use the model ID from the table above. If you supplied `VLLM_API_KEY`, use that value in your client. Otherwise the launcher uses the configured key file or generates a key on first start.

To serve without authentication, for example behind a reverse proxy that checks the bearer token itself, set the key to an empty string (`VLLM_API_KEY=""`). An empty key is used as is: the launcher neither reads nor generates a key file, and the server accepts requests without an `Authorization` header.

For Docker, read the generated key from the running container. Use the same Compose file you used to start it.

```bash
api_key="$(docker compose -f docker/mi100/compose.yaml exec -T serve cat /cache/mi100/api.key)"
```

For native serving, the launcher prints the generated key’s path. The default command is below. If you configured `API_KEY_FILE` or already use `/etc/llama/llama-api.key`, read that file instead.

```bash
api_key="$(cat "${XDG_CACHE_HOME:-$HOME/.cache}/mi100/api.key")"
```

List the served model with the same credentials your client will use. Each model guide includes a first chat request.

```bash
curl -fsS -H "Authorization: Bearer $api_key" http://127.0.0.1:8020/v1/models
```

## Common issues

| Symptom | What to do |
| --- | --- |
| The launcher cannot see your cards | Check `/dev/kfd` and `/dev/dri/render*` on the host. Give the serving account access to the render devices. Check that `MI100_DEVICES` includes the card count you selected. |
| The launcher reports an incomplete XGMI mesh | Inspect `rocm-smi --showtopotype` for the selected cards. Set `MI100_XGMI=off` for a PCIe topology. |
| Startup is still in progress | Read the Compose service logs. For a native background start, look in `logs/mi100/<recipe>-<cards>-<on-or-off>/server.log`. Check free disk space in the model, cache and temporary directories. |
| A local checkpoint cannot be found | Check that both target and draft directories contain `config.json`. In Docker, the paths in `recipe.env` must be visible inside the `/models` mount. To use the automatic download, remove the explicit path for that checkpoint. |
| A native Qwen or Bolt launch fails to lock host memory | Check the account’s `ulimit -l`, available RAM and `/dev/shm`. Qwen defaults to an 8 GiB CPU prefix cache and Bolt to 24 GiB. The supplied systemd units set `LimitMEMLOCK=infinity`. |
| The endpoint refuses a connection | Check `curl -fsS http://127.0.0.1:8020/health` and the startup log. If you changed the published port, use that port in the client URL. |
