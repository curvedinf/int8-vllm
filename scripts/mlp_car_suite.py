#!/usr/bin/env python3
"""Bounded live A/B accuracy + mixed-load suite for the MLP CAR experiment."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run_mlp_car_experiment import CAMPAIGN as BASE, ROOT


def run(name, *args):
    print(f"START {name}", flush=True)
    with (BASE / f"{name}.log").open("w") as log:
        p = subprocess.run([sys.executable, "scripts/run_mlp_car_experiment.py",
                            "client", "--", *map(str, args)], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT)
    print(f"END {name}: {p.returncode}", flush=True)
    if p.returncode:
        raise RuntimeError(f"{name} failed; inspect retained log")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=("baseline", "candidate"))
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    args = ap.parse_args()
    arm = args.arm
    manifest_path = BASE / "manifest.json"
    gate_path = BASE / "greedy_gate.json"
    sources = (
        (Path(os.environ.get("MLP_CAR_MANIFEST",
                             ROOT / "logs/agent_mixed/manifest.json")), manifest_path),
        (Path(os.environ.get("MLP_CAR_GREEDY_GATE",
                             ROOT / "logs/agent_mixed/greedy_gate.json")), gate_path),
    )
    for source, destination in sources:
        if not destination.exists():
            destination.write_bytes(source.read_bytes())
    reference = BASE / "reference52.json"
    if args.oracle:
        last_turn = len(json.loads(manifest_path.read_text())["output_budgets"]) - 1
        for source in ("baseline", "candidate"):
            for kind, name, limit in (
                ("cold", "c0-t0", 1536),
                ("cold", "c1-t0", 1536),
                ("cold", f"c0-t{last_turn}", 512),
                ("cold", f"c2-t{last_turn}", 512),
                ("warm", f"c6-t{last_turn}", 512),
                ("greedy_warm", "c0-t0", 512),
            ):
                folder = f"{source}_{kind}"
                run(f"oracle_{folder}_{name}", "scripts/trace_prefill_oracle.py",
                    BASE / folder / f"{name}.result.json", "--max-tokens", limit,
                    "--onset-token", 1024,
                    "--out", BASE / f"oracle_{folder}_{name}.json")
        return
    if args.profile:
        for state in ("cold", "warm"):
            name = f"{arm}_profile_{state}"
            run(f"{name}_start", "scripts/agent_profile_control.py", "start",
                "--tag", name, "--ranks", "0")
            try:
                run(name, "scripts/bench_agent_mixed.py",
                    "--manifest", manifest_path, "--replay-from",
                    BASE / f"baseline_{state}", "--output", BASE / name,
                    "--duration", 1200, "--request-timeout", 900, "--trace-gate",
                    BASE / f"{arm}_profile_boot/active")
            finally:
                run(f"{name}_stop", "scripts/agent_profile_control.py", "stop",
                    "--output", BASE / f"{arm}_{state}_capture.json")
        return
    if arm == "baseline":
        run("reference52", "scripts/eval_mlp_car.py", "--capture", "--reference",
            reference, "--output", BASE / "reference52_summary.json")
    else:
        run("candidate52", "scripts/eval_mlp_car.py", "--reference", reference,
            "--output", BASE / "candidate52.json")
    if os.environ.get("MLP_CAR_EXPERIMENT") == "decode_int8":
        run(f"{arm}_decode_accuracy", "scripts/eval_decode_int8.py",
            "--generations", reference, "--reference", BASE / "decode_reference.json",
            "--output", BASE / f"{arm}_decode_accuracy.json",
            *(["--capture"] if arm == "baseline" else []))
    for state in ("cold", "warm"):
        run(f"{arm}_greedy_{state}", "scripts/bench_agent_mixed.py", "--manifest",
            gate_path, "--output",
            BASE / f"{arm}_greedy_{state}", "--duration", 900,
            "--request-timeout", 800, "--trace-gate", BASE / f"{arm}_boot/active")
    for state in ("cold", "warm"):
        run(f"{arm}_{state}", "scripts/bench_agent_mixed.py", "--manifest",
            manifest_path, "--output", BASE / f"{arm}_{state}",
            "--duration", 1200, "--request-timeout", 900,
            "--trace-gate", BASE / f"{arm}_boot/active")
    print(f"DONE {arm} unprofiled suite", flush=True)


if __name__ == "__main__":
    main()
