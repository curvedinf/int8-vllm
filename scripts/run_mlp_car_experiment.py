#!/usr/bin/env python3
"""Root-run isolated A/B harness; never starts Nginx during an experiment.

Actions: boot, client, stop. The production service must already be stopped.
Credentials are inherited from its EnvironmentFile, never placed in argv/logs.
"""
import argparse
import json
import os
import pwd
import shlex
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = Path(os.environ.get(
    "MLP_CAR_CAMPAIGN_DIR", ROOT / "logs/agent_mixed/mlp_car_gs32_20261011"
)).resolve()
UNIT = "vllm-mlp-car-experiment"


def client_env():
    env = os.environ.copy()
    for raw in Path("/etc/llama/llama-api.env").read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.removeprefix("export ").split("=", 1)
            values = shlex.split(value)
            env[key] = values[0] if values else ""
    env["VLLM_API_KEY"] = (
        env.get("VLLM_API_KEY") or env.get("LLAMA_API_KEY")
        or Path("/etc/llama/llama-api.key").read_text().strip()
    )
    env.update(HOME="/home/curved", PYTHONPATH=f"{ROOT}:{ROOT.parent / 'aiter'}",
               ROCM_PATH="/opt/rocm", LD_LIBRARY_PATH="/opt/rocm/lib",
               PYTORCH_ROCM_ARCH="gfx908", GPU_ARCHS="gfx908",
               VLLM_TARGET_DEVICE="rocm", PYTHONUNBUFFERED="1")
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("boot", "client", "stop"))
    ap.add_argument("--arm", choices=("baseline", "candidate"), default="baseline")
    ap.add_argument("--tag")
    args, remainder = ap.parse_known_args()
    if args.action != "client" and remainder:
        ap.error(f"unexpected arguments: {remainder}")
    if subprocess.run(["pgrep", "-x", "nginx"], capture_output=True).returncode != 1:
        raise RuntimeError("Nginx must remain stopped")
    if args.action == "stop":
        pid = int(subprocess.check_output(
            ["systemctl", "show", UNIT, "--property=MainPID", "--value"], text=True
        ).strip() or "0")
        directory = None
        if pid:
            entries = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            values = dict(item.split(b"=", 1) for item in entries if b"=" in item)
            if b"VLLM_LIFECYCLE_DIR" in values:
                directory = Path(os.fsdecode(values[b"VLLM_LIFECYCLE_DIR"])).parent
        subprocess.run(["systemctl", "stop", UNIT], check=True)
        if directory is not None:
            allowed = {"baseline_boot", "candidate_boot", "baseline_profile_boot",
                       "candidate_profile_boot"}
            if (directory.resolve().parent != CAMPAIGN
                    or directory.name not in allowed):
                raise RuntimeError("Refusing to clean a cache outside this campaign")
            cache = directory / "kv"
            if cache.exists():
                if cache.is_symlink():
                    raise RuntimeError("Refusing to clean a symlinked test cache")
                size = int(subprocess.check_output(
                    ["du", "-sb", str(cache)]
                ).split()[0])
                shutil.rmtree(cache)
                (directory / "cache_cleanup.json").write_text(json.dumps({
                    "path": str(cache), "removed_bytes": size,
                    "reason": "Stopped experimental server; regenerable KV only",
                }))
                print(f"Removed stopped test cache: {cache} ({size} bytes)", flush=True)
        return
    if args.action == "client":
        command = remainder
        if command and command[0] == "--":
            command = command[1:]
        account = pwd.getpwnam("curved")
        result = subprocess.run([str(ROOT / ".venv/bin/python"), *command],
                                cwd=ROOT, env=client_env(), user=account.pw_uid,
                                group=account.pw_gid, extra_groups=["video", "render"])
        raise SystemExit(result.returncode)
    if subprocess.run(["systemctl", "is-active", "--quiet", "vllm"]).returncode == 0:
        raise RuntimeError("Stop production before booting an experiment")
    if args.tag is None:
        args.tag = f"{args.arm}_boot"
    if args.tag not in (f"{args.arm}_boot", f"{args.arm}_profile_boot"):
        ap.error("Use the arm's boot/profile_boot tag and a fresh campaign directory")
    account = pwd.getpwnam("curved")
    if not CAMPAIGN.exists():
        CAMPAIGN.mkdir(parents=True)
        os.chown(CAMPAIGN, account.pw_uid, account.pw_gid)
    if CAMPAIGN.stat().st_uid != account.pw_uid:
        raise RuntimeError("The campaign directory must belong to curved")
    directory = CAMPAIGN / args.tag
    directory.mkdir(exist_ok=False)
    os.chown(directory, account.pw_uid, account.pw_gid)
    subprocess.run(["systemctl", "reset-failed", UNIT], capture_output=True)
    command = ["systemd-run", "--unit", UNIT, "--collect",
               "--property=User=curved", "--property=Group=curved",
               "--property=SupplementaryGroups=video render",
               f"--property=WorkingDirectory={ROOT}",
               "--property=EnvironmentFile=/etc/llama/llama-api.env",
               "--property=LimitMEMLOCK=infinity", "--property=LimitNOFILE=1048576",
               "--property=MemoryMax=50G", "--property=MemorySwapMax=8G",
               "--property=OOMPolicy=stop", "--property=RuntimeMaxSec=5400",
               "--property=TimeoutStopSec=60", "--property=KillMode=control-group",
               "--setenv=HOME=/home/curved", "--setenv=PYTHONUNBUFFERED=1",
               "--setenv=LIFECYCLE_METRICS=1", "--setenv=VLLM_SERVER_DEV_MODE=1",
               "--setenv=VLLM_LIFECYCLE_KINETO=1",
               "--setenv=VLLM_LIFECYCLE_GPU_CHUNK_STEPS=64",
               f"--setenv=VLLM_LIFECYCLE_DIR={directory / 'host'}",
               f"--setenv=VLLM_LIFECYCLE_GATE={directory / 'active'}",
               f"--setenv=DISKROOT={directory / 'kv'}",
               f"--setenv=VLLM_MLP_CAR_INT8_GS32={int(args.arm == 'candidate')}",
               "--setenv=VLLM_MLP_CAR_INT8_LIBRARY="
               f"{ROOT / 'build/mlp_car_int8_gs32.so'}",
               str(ROOT / "scripts/serve_recipe_qwen38.sh"), "supervise"]
    subprocess.run(command, check=True)
    for _ in range(900):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8020/health", timeout=2):
                print(f"READY {args.arm} {directory}", flush=True)
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Experiment did not become healthy")


if __name__ == "__main__":
    main()
