#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Check long-context retrieval, GPU prefix reuse and optional CPU restoration.

Run on an otherwise idle recipe server. CPU restoration requires the diagnostic
VLLM_SERVER_DEV_MODE=1 endpoint; do not enable that mode for production.
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from prometheus_client.parser import text_string_to_metric_families
from tokenizers import Tokenizer


def restore_counters(metrics: str) -> tuple[float, float]:
    hit_tokens = copied_bytes = 0.0
    for family in text_string_to_metric_families(metrics):
        for sample in family.samples:
            if sample.name == "vllm:external_prefix_cache_hits_total":
                hit_tokens += sample.value
            elif (
                sample.name == "vllm:kv_offload_total_bytes_total"
                and sample.labels.get("transfer_type") == "CPU_to_GPU"
            ):
                copied_bytes += sample.value
    return hit_tokens, copied_bytes


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-url", default="http://127.0.0.1:8020")
parser.add_argument("--api-key-file", type=Path)
parser.add_argument("--tokenizer", type=Path, required=True)
parser.add_argument("--model", default="nemotron-bolt")
parser.add_argument("--max-tokens", type=int, default=24)
parser.add_argument("--background-tokens", type=int, default=28000)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--reset-gpu", action="store_true")
args = parser.parse_args()
key = (
    args.api_key_file.read_text().strip()
    if args.api_key_file
    else os.environ.get("VLLM_API_KEY", "")
)
if not key:
    parser.error("Provide --api-key-file or VLLM_API_KEY")
url = args.base_url.rstrip("/")
hdr = {"Authorization": "Bearer " + key}
args.output_dir.mkdir(parents=True, exist_ok=True)
tokenizer = Tokenizer.from_file(str(args.tokenizer))
background = "\n".join(
    f"Entry {i}: The archive records an ordinary shipping crate with a blue "
    "label and no special handling requirements."
    for i in range(2000)
)
background = tokenizer.decode(
    tokenizer.encode(background, add_special_tokens=False).ids[:args.background_tokens]
)
marker = "The designated verification phrase for this archive is ORCHID CEDAR 4829."
prefix = (
    "Read this archive, then answer the question at the end.\n"
    + background[: len(background) // 2]
    + "\n"
    + marker
    + "\n"
    + background[len(background) // 2 :]
)
(args.output_dir / "bench-prefix.txt").write_text(
    prefix + "\nIgnore the archive for this question. "
)
body = {
    "model": args.model,
    "messages": [
        {
            "role": "user",
            "content": prefix
            + "\nQuestion: What is the designated verification phrase? "
            "Reply with that phrase only.",
        }
    ],
    "temperature": 0,
    "max_tokens": args.max_tokens,
    "chat_template_kwargs": {"enable_thinking": False},
}
(args.output_dir / "request.json").write_text(json.dumps(body, indent=2))
results = []
for name in ["cold", "repeat"] + (["cpu_restore"] if args.reset_gpu else []):
    if name == "cpu_restore":
        time.sleep(11)
        before = requests.get(url + "/metrics", headers=hdr, timeout=30)
        before.raise_for_status()
        (args.output_dir / "before-reset.metrics").write_text(before.text)
        reset = requests.post(url + "/reset_prefix_cache", headers=hdr, timeout=60)
        reset.raise_for_status()
        assert reset.json().get("success"), reset.text
    start = time.monotonic()
    response = requests.post(
        url + "/v1/chat/completions", headers=hdr, json=body, timeout=1800
    )
    response.raise_for_status()
    data = response.json()
    results.append(
        {"name": name, "seconds": time.monotonic() - start, "response": data}
    )
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    # Some templates emit a reasoning preamble even with thinking disabled.
    # Compare the answer after that preamble on cold and restored requests.
    content = data["choices"][0]["message"]["content"] or ""
    assert "<think>" not in content or "</think>" in content, "Reasoning was truncated"
    content = content.rsplit("</think>", 1)[-1].strip()
    if name == "cold":
        cold_answer = content
    assert "ORCHID CEDAR 4829" in content, content
    assert content == cold_answer
    print(name, results[-1]["seconds"], data["usage"], content, flush=True)
    if name == "cpu_restore":
        time.sleep(11)
        after = requests.get(url + "/metrics", headers=hdr, timeout=30)
        after.raise_for_status()
        (args.output_dir / "after-restore.metrics").write_text(after.text)
        old_hits, old_bytes = restore_counters(before.text)
        hits, copied = restore_counters(after.text)
        assert hits > old_hits and copied > old_bytes, (
            "Answer matched after GPU reset, but no CPU restore was measured: "
            f"{hits - old_hits:g} external hit tokens, {copied - old_bytes:g} bytes."
        )
        print(f"Restored {hits - old_hits:g} tokens, {copied - old_bytes:g} bytes.")
print("PASS: retrieval and repeated answer")
