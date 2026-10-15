# Installing the recipe services

The two recipes have separate systemd templates:

| Recipe | Unit file | Installed instance |
| --- | --- | --- |
| Qwen3.8 + DFlash2 | [vllm-qwen38@.service](../../scripts/vllm-qwen38@.service) | `vllm-qwen38@USERNAME.service` |
| Nemotron Bolt + LiLiCorr | [vllm-bolt@.service](../../scripts/vllm-bolt@.service) | `vllm-bolt@USERNAME.service` |

The instance name is the serving account, with a matching primary group.
Templates expect `/home/USERNAME/int8-vllm`, its `.venv`, the sibling `aiter`
checkout, model assets, and access to the `video` and `render` groups. Prepare
the environment and checkpoints using the [Qwen recipe](README.md) or
[Bolt recipe](nemotron_bolt.md) first. For other checkout locations, adapt the
unit's working directory and executable paths before installation.

Both units require `/etc/llama/llama-api.env`, provisioned by the administrator
with the production `VLLM_API_KEY` or `LLAMA_API_KEY`. Keep it root-owned and
mode `0600`. Recipe environment overrides can also be placed there. Each unit
runs its recipe's `supervise` command and stops an existing manually launched
instance of that same recipe before starting.

From the checkout, install both definitions without starting either:

```bash
sudo install -m0644 scripts/vllm-qwen38@.service scripts/vllm-bolt@.service /etc/systemd/system/
sudo systemctl daemon-reload
```

When ready to activate, run the following from the serving account, selecting
one recipe. These commands stop the previous recipe and change boot selection:

```bash
# Select Bolt.
sudo systemctl disable --now vllm.service "vllm-qwen38@$USER.service"
sudo systemctl enable --now "vllm-bolt@$USER.service"

# Alternatively, select Qwen3.8.
sudo systemctl disable --now vllm.service "vllm-bolt@$USER.service"
sudo systemctl enable --now "vllm-qwen38@$USER.service"
```

If the legacy `vllm.service` was never installed, omit it from these commands.
The named instances conflict with each other for the same account and with
the legacy unit. Use one serving account for this four-GPU host; do not enable
both recipes at boot. Stop any manually launched *other* recipe before
activation, since systemd cannot manage a process outside its units.

Check the selected instance with `systemctl status` and `journalctl -u`, then
verify authenticated generation on its configured endpoint. The legacy
`scripts/vllm.service` remains available for existing installations; new
installations should use the named templates above.
