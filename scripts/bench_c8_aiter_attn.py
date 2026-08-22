#!/usr/bin/env python3
"""C8 benchmark harness for Qwen3.6-27B on MI100.

Starts a vLLM server with C8 concurrency settings, waits for readiness,
then runs `vllm bench serve` against it. Captures throughput and latency
metrics into a JSON summary.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.home() / "vllm-gfx908"
VENV = ROOT / ".venv"
MODEL_DIR = str(Path.home() / "models" / "Qwen3.6-27B-GPTQ-8bit-MTP2")
SERVED_MODEL_NAME = "qwen3.6-27b-gptq8"
API_KEY_FILE = "/etc/llama/llama-api.key"
HOST = "127.0.0.1"
PORT = 8020


def read_api_key():
    for env in ("VLLM_API_KEY", "LLAMA_API_KEY"):
        val = os.environ.get(env)
        if val:
            return val
    p = Path(API_KEY_FILE)
    try:
        if p.exists():
            return p.read_text().strip()
    except PermissionError:
        pass
    return ""


def run(cmd, cwd=None, env=None, timeout=None, check=True):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(
        cmd, cwd=cwd, env=env, timeout=timeout, check=check,
        text=True, capture_output=True,
    )


def wait_ready(base_url, timeout=900):
    for i in range(timeout):
        try:
            subprocess.run(
                ["curl", "-fsS", "--max-time", "2", f"{base_url}/v1/models"],
                check=True, text=True, capture_output=True,
            )
            return True
        except subprocess.CalledProcessError:
            time.sleep(1)
    return False


def parse_bench_output(stdout: str) -> dict:
    """Extract key metrics from vllm bench serve stdout."""
    metrics = {}
    # Look for lines like "Mean TTFT (ms): 123.45"
    for line in stdout.splitlines():
        m = re.match(r"\s*(Mean|Median|P90|P95|P99)\s+(TTFT|TPOT|ITL|Latency)\s*\(ms\):\s*([0-9.]+)", line)
        if m:
            metrics[f"{m.group(1).lower()}_{m.group(2).lower()}_ms"] = float(m.group(3))
        m = re.match(r"\s*(Request throughput|Output token throughput|Total token throughput)\s*\(tok/s\):\s*([0-9.]+)", line)
        if m:
            key = m.group(1).lower().replace(" ", "_")
            metrics[f"{key}_tok_per_s"] = float(m.group(2))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="experiment tag for output dir")
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--attention-backend", default="TRITON_ATTN")
    parser.add_argument("--mtp", action="store_true", help="enable MTP speculative decoding")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--output-len", type=int, default=1000)
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--skip-server", action="store_true", help="assume server already running")
    parser.add_argument("--extra-vllm-args", default="", help="extra args passed to vllm serve")
    parser.add_argument("--compilation-config", default='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}')
    parser.add_argument("--endpoint", default="/v1/completions", help="benchmark endpoint")
    args = parser.parse_args()

    out_dir = ROOT / "logs" / "c8_optimization" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "summary.json"
    server_log = out_dir / "server.log"
    bench_log = out_dir / "bench.log"

    api_key = read_api_key()
    base_url = f"http://{HOST}:{PORT}"

    server_proc = None
    if not args.skip_server:
        # stop any existing server on same port
        run([ROOT / "scripts" / "serve_direwolf_qwen36.sh", "stop"], check=False)
        time.sleep(2)

        env = os.environ.copy()
        env.update({
            "PATH": f"{VENV}/bin:{env.get('PATH', '')}",
            "ROCM_PATH": "/opt/rocm",
            "HIP_PATH": "/opt/rocm",
            "GPU_ARCHS": "gfx908",
            "PYTORCH_ROCM_ARCH": "gfx908",
            "BUILD_TARGET": "rocm",
            "MAX_JOBS": "48",
            "LD_LIBRARY_PATH": f"/opt/rocm/lib:{env.get('LD_LIBRARY_PATH', '')}",
            "PYTHONPATH": f"{ROOT}/python_startup:{ROOT}:{Path.home() / 'aiter'}:{env.get('PYTHONPATH', '')}",
            "HF_HOME": str(Path.home() / ".cache" / "huggingface"),
            "OMP_NUM_THREADS": "48",
            "MKL_NUM_THREADS": "48",
            "OPENBLAS_NUM_THREADS": "48",
            "NUMEXPR_NUM_THREADS": "48",
            "HIP_VISIBLE_DEVICES": "0,1,2,3",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "ROCR_VISIBLE_DEVICES": "0,1,2,3",
            "HSA_ENABLE_IPC_MODE_LEGACY": "0",
            "VLLM_TARGET_DEVICE": "rocm",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1800",
            "VLLM_ROCM_USE_AITER": "1",
            "VLLM_ROCM_USE_AITER_LINEAR": "1",
            "VLLM_ROCM_USE_AITER_TRITON_GEMM": "1",
            "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION": "1",
            "NCCL_ALGO": "Ring",
            "NCCL_PROTO": "Simple",
            "NCCL_P2P_DISABLE": "0",
            "NCCL_DMABUF_ENABLE": "0",
            "VLLM_API_KEY": api_key,
        })

        vllm_args = [
            str(VENV / "bin" / "vllm"),
            "serve", MODEL_DIR,
            "--served-model-name", SERVED_MODEL_NAME,
            "--host", HOST,
            "--port", str(PORT),
            "--tensor-parallel-size", "4",
            "--dtype", "half",
            "--max-model-len", "65536",
            "--max-num-seqs", str(args.max_num_seqs),
            "--gpu-memory-utilization", "0.95",
            "--attention-backend", args.attention_backend,
            "--compilation-config", args.compilation_config,
            "--language-model-only",
            "--skip-mm-profiling",
            "--disable-custom-all-reduce",
            "--disable-log-stats",
            "--disable-uvicorn-access-log",
            "--kv-cache-dtype", args.kv_cache_dtype,
        ]
        if args.max_num_batched_tokens is not None:
            vllm_args.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])
        if args.mtp:
            vllm_args.extend([
                "--speculative-config",
                '{"method":"mtp","num_speculative_tokens":2,"rejection_sample_method":"standard"}',
            ])
        if args.extra_vllm_args:
            vllm_args.extend(args.extra_vllm_args.split())

        with open(server_log, "w") as f:
            server_proc = subprocess.Popen(
                ["taskset", "-c", "0-47"] + vllm_args,
                stdout=f, stderr=subprocess.STDOUT,
                env=env,
            )
        print(f"Server pid={server_proc.pid}, log={server_log}")
        if not wait_ready(base_url, timeout=900):
            print("Server failed to start", file=sys.stderr)
            if server_proc:
                server_proc.terminate()
            return 1

    # Run benchmark
    bench_cmd = [
        str(VENV / "bin" / "vllm"), "bench", "serve",
        "--base-url", base_url,
        "--model", SERVED_MODEL_NAME,
        "--tokenizer", MODEL_DIR,
        "--dataset-name", "random",
        "--num-prompts", str(args.num_prompts),
        "--max-concurrency", "8",
        "--random-input-len", str(args.input_len),
        "--random-output-len", str(args.output_len),
        "--endpoint", args.endpoint,
        "--no-stream",
    ]
    if args.endpoint == "/v1/completions":
        bench_cmd.append("--skip-chat-template")
    if api_key:
        bench_cmd.extend(["--api-key", api_key])

    print(f"Running benchmark: {' '.join(bench_cmd)}")
    bench_start = time.time()
    result = run(bench_cmd, timeout=args.timeout, check=False)
    bench_elapsed = time.time() - bench_start
    bench_log.write_text(result.stdout + "\n" + result.stderr)
    print(f"Benchmark finished in {bench_elapsed:.1f}s, log={bench_log}")

    metrics = parse_bench_output(result.stdout)
    summary = {
        "tag": args.tag,
        "args": vars(args),
        "returncode": result.returncode,
        "bench_elapsed_sec": bench_elapsed,
        "metrics": metrics,
        "raw_stdout": result.stdout,
        "raw_stderr": result.stderr,
    }
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_file}")

    if server_proc:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()

    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
