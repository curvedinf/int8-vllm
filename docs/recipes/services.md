# Installing the recipe services

The two recipes have separate systemd units:

| Recipe | Unit file | Service name |
| --- | --- | --- |
| Qwen3.8 + DFlash2 | [vllm-qwen38.service](../../scripts/vllm-qwen38.service) | `vllm-qwen38.service` |
| Nemotron Bolt + LiLiCorr | [vllm-bolt.service](../../scripts/vllm-bolt.service) | `vllm-bolt.service` |

Both services run as `curved`, with primary group `curved`.
The units expect `/home/curved/int8-vllm`, its `.venv`, the sibling `aiter`
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
sudo install -m0644 scripts/vllm-qwen38.service scripts/vllm-bolt.service /etc/systemd/system/
sudo systemctl daemon-reload
```

When ready to activate, run the following from the serving account, selecting
one recipe. These commands stop the previous recipe and change boot selection:

```bash
# Select Bolt.
sudo systemctl disable --now vllm.service "vllm-qwen38.service"
sudo systemctl enable --now "vllm-bolt.service"

# Alternatively, select Qwen3.8.
sudo systemctl disable --now vllm.service "vllm-bolt.service"
sudo systemctl enable --now "vllm-qwen38.service"
```

If the legacy `vllm.service` was never installed, omit it from these commands.
The services conflict with each other and with the legacy unit. Use one serving account for this four-GPU host; do not enable
both recipes at boot. Stop any manually launched *other* recipe before
activation, since systemd cannot manage a process outside its units.

Check the selected service with `systemctl status` and `journalctl -u`, then
verify authenticated generation on its configured endpoint. The legacy
`scripts/vllm.service` remains available for existing installations; new
installations should use the named units above.

## CPU cache residency and crash recovery

On ROCm, CPU KV pages are explicitly locked with `mlock` before HIP host
registration. HIP registration alone can leave pages pageable through HMM.
Each worker locks only its own slots; the 24 GiB Bolt CPU cache remains one
shared physical allocation. Lock failure aborts startup with a memlock/RAM
diagnostic instead of silently accepting a pageable cache. The units already
set `LimitMEMLOCK=infinity`; manual launchers need an adequate memlock limit.

After every worker has registered its KV cache and the scheduler has opened
any tier mappings, the connector handshake verifies the shared inode and
unlinks its name. Mappings remain usable until their last holder exits, at
which point the kernel reclaims the storage, including after SIGKILL.
The launcher's guarded cleanup remains necessary for failures before that
handshake. Distributed-host and PCP topologies retain creator cleanup.

Verify a running Bolt instance through `/proc/<worker-pid>/smaps`: its
`vllm_offload_*.mmap (deleted)` mappings should have `lo` in `VmFlags` for the
worker's slots, and `VmLck` in `/proc/<worker-pid>/status` should account for
roughly 6 GiB per rank. There should be no named offload file for that engine.

## Background kernel upgrades

An unattended AMD DKMS rebuild can consume enough RAM to disrupt serving.
Its configure probes and nested make use `nproc` themselves, bypassing the
ordinary DKMS parallel-job setting. On this 96-CPU host, constrain the upgrade
service to eight CPUs and a 4 GiB soft memory limit:

```bash
sudo install -d /etc/systemd/system/apt-daily-upgrade.service.d
sudo install -m0644 scripts/apt-daily-upgrade-vllm.conf \
  /etc/systemd/system/apt-daily-upgrade.service.d/vllm.conf
sudo systemctl daemon-reload
```

This applies to future runs, keeps automatic updates enabled, and does not
change the serving cache budget. `MemoryHigh` throttles/reclaims the build's
own cgroup rather than imposing a hard OOM cutoff. Adapt the CPU affinity to
other hosts; verify `nproc` under the same affinity before deploying there.
