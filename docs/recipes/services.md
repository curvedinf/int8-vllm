# Run a native MI100 recipe with systemd

Use these units to start Qwen, Flint or Bolt at boot on a four-card host. The services use the same GPUs and localhost port `8020`. Select one service for the host.

| Model | Unit template |
| --- | --- |
| Qwen3.8-27B | [vllm-qwen38.service](../../scripts/vllm-qwen38.service) |
| Flint | [vllm-flint.service](../../scripts/vllm-flint.service) |
| Nemotron Bolt | [vllm-bolt.service](../../scripts/vllm-bolt.service) |

## Install a service

1. [Build the native environment](mi100-install.md#native) and download the target and draft for your model. The units call the model-specific scripts directly and expect local checkpoints.
2. Create `/etc/llama/llama-api.env` with mode `0600`, owned by root. Set `VLLM_API_KEY` to your key. If the checkpoints are outside the serving account’s default model directories, set `MODEL_DIR` and `DRAFT_MODEL_DIR` to their absolute paths in the same file.
3. From the checkout root, install the selected unit. This example selects Flint. Use the filename from the table for another model.

```bash
unit=vllm-flint.service
sudo install -m 0644 "scripts/$unit" "/etc/systemd/system/$unit"
sudoedit "/etc/systemd/system/$unit"
```

In the installed unit, set `User`, `Group`, `WorkingDirectory` and the absolute paths in the `Exec` directives to match your serving account and checkout. The account needs access to the GPUs, venv, sibling AITER checkout and checkpoints. The templates grant the `video` and `render` supplementary groups and set the locked-memory limit for CPU caches.

Each template sets `MI100_CPU_MAP` for the reference host’s four L3 domains. Edit that line to match the CPU lists on your host, in GPU order. Remove it to use normal CPU scheduling. The recipe script applies the map when it starts each worker. [Choosing CPU lists](mi100-install.md#pin-workers-to-cpu-cores)

Set any launcher overrides you need in the service environment file. For example, the portable Qwen launcher sets `C=8`, `KVMEM=auto` and `DISKTIER=0` on four cards. Set those values there to use the same concurrency and cache settings under systemd.

## Start and manage the service

Stop any manually launched recipe using the same cards. To switch from another enabled unit, run `sudo systemctl disable --now` with that unit’s name first. Then enable the unit selected above.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now "$unit"
systemctl status "$unit"
```

Follow startup with `journalctl -u "$unit" -f`. When `http://127.0.0.1:8020/health` is ready, connect your client with the model ID and key from your service configuration. [API access](mi100-install.md#api-access)

Use `sudo systemctl restart "$unit"` after changing the environment file. To stop serving and remove the unit from boot, run `sudo systemctl disable --now "$unit"`.
