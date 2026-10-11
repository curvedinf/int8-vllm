#!/usr/bin/env python3
"""Execute the bounded campaign and restore production even on test failure.

Run as root with production and ingress stopped. Baseline boot is covered
by the restoration guard. All inference clients run as the service owner.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request

from run_mlp_car_experiment import CAMPAIGN, ROOT, client_env


def run(*args):
    print("RUN", *args, flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def ready(timeout=900):
    for _ in range(timeout):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8020/health", timeout=2):
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Timed out waiting for server health")


def stop(tag):
    shutil.copy2(ROOT / "logs/serve_recipe_qwen38/server.log",
                 CAMPAIGN / f"{tag}_server.log")
    run("scripts/run_mlp_car_experiment.py", "stop")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-already-running", action="store_true")
    args = parser.parse_args()
    result = {"status": "running"}
    try:
        if args.baseline_already_running:
            ready()
        else:
            run("scripts/run_mlp_car_experiment.py", "boot", "--arm", "baseline",
                "--tag", "baseline_boot")
        run("scripts/mlp_car_suite.py", "baseline")
        stop("baseline")
        run("scripts/run_mlp_car_experiment.py", "boot", "--arm", "candidate",
            "--tag", "candidate_boot")
        run("scripts/mlp_car_suite.py", "candidate")
        stop("candidate")
        for arm in ("baseline", "candidate"):
            run("scripts/run_mlp_car_experiment.py", "boot", "--arm", arm,
                "--tag", f"{arm}_profile_boot")
            run("scripts/mlp_car_suite.py", arm, "--profile")
            if arm == "baseline":
                run("scripts/mlp_car_suite.py", arm, "--oracle")
            stop(f"{arm}_profile")
        result["status"] = "completed"
    except Exception as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        subprocess.run(["systemctl", "stop", "vllm-mlp-car-experiment"], check=False)
        subprocess.run(["systemctl", "start", "vllm"], check=True)
        ready()
        request = urllib.request.Request(
            "http://127.0.0.1:8020/v1/completions",
            data=json.dumps({"model": "qwen3.8-27b-gptq8", "prompt":
                             "The sum of 2 and 2 is", "max_tokens": 24,
                             "temperature": 0}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + client_env()["VLLM_API_KEY"]})
        with urllib.request.urlopen(request, timeout=180) as response:
            smoke = json.load(response)
        result["restored_smoke"] = smoke["choices"][0]["text"]
        subprocess.run(
            ["systemctl", "unmask", "--runtime", "nginx.service"], check=True
        )
        subprocess.run(["systemctl", "start", "nginx"], check=True)
        result["production_restored"] = True
        if CAMPAIGN.is_dir():
            (CAMPAIGN / "campaign.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
