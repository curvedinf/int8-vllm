#!/usr/bin/env python3
"""Capture and compare stagewise DFlash2 diagnostics.

The server-side capture reuses VLLM_QUANT_AUDIT.  ``capture`` sends a fixed
greedy corpus and stores both responses and counter deltas.  ``compare``
aligns rank-0 tensors from two legs and reports the first numerical boundary
where the candidate path diverges.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path

import torch


ROOT = Path(
    os.environ.get("DFLASH_AUDIT_DIR", "/home/curved/models/kld/dflash_audit")
)
PROMPTS = [
    "Implement an LRU cache in Python with O(1) get and put operations.",
    "Solve 3x^2 - 12x + 9 = 0 and show each step.",
    "Explain why TCP retransmission can cause head-of-line blocking.",
    "Write a Rust function that performs binary search with a comparator.",
    "A jar has 4 red and 6 blue balls. Find the probability of drawing 3 red balls without replacement.",
    "Debug this function and explain the bug: def add(a, b): return a - b",
]


def _get(path: str) -> str:
    req = urllib.request.Request(
        os.environ.get("DFLASH_URL", "http://127.0.0.1:8020") + path,
        headers={"Authorization": f"Bearer {os.environ['DFLASH_KEY']}"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode()


def _post(payload: dict) -> dict:
    req = urllib.request.Request(
        os.environ.get("DFLASH_URL", "http://127.0.0.1:8020")
        + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['DFLASH_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.loads(response.read())


def _metrics() -> dict[str, float]:
    values: dict[str, float] = {}
    for line in _get("/metrics").splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([^ {]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$", line)
        if match:
            values[match.group(1)] = values.get(match.group(1), 0.0) + float(
                match.group(2)
            )
    return values


def capture(tag: str) -> None:
    before = _metrics()
    responses = []
    for prompt in PROMPTS:
        result = _post(
            {
                "model": "qwen3.8-27b-gptq8",
                "prompt": prompt,
                "max_tokens": 192,
                "temperature": 0,
                "logprobs": 1,
            }
        )
        choice = result["choices"][0]
        responses.append(
            {
                "prompt": prompt,
                "text": choice.get("text", ""),
                "tokens": (choice.get("logprobs") or {}).get("tokens", []),
            }
        )
    after = _metrics()
    delta = {
        key: value - before.get(key, 0.0)
        for key, value in after.items()
        if "spec_decode" in key and value != before.get(key, 0.0)
    }
    steps = delta.get("vllm:spec_decode_num_drafts_total", 0.0)
    drafted = delta.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    derived = {
        "mean_acceptance_length": 1.0 + accepted / steps if steps else None,
        "draft_token_acceptance": accepted / drafted if drafted else None,
    }
    out = ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "capture.json").write_text(
        json.dumps(
            {
                "tag": tag,
                "responses": responses,
                "metrics": delta,
                "derived": derived,
            },
            indent=2,
        )
    )
    time.sleep(5)
    print(
        json.dumps(
            {
                "tag": tag,
                "requests": len(responses),
                "metrics": delta,
                "derived": derived,
            },
            indent=2,
        )
    )


def _load_leg(tag: str) -> dict[tuple[str, int, str], torch.Tensor]:
    result = {}
    pattern = re.compile(r"dflash_(.+)_(\d+)__(.+)\.pt$")
    for path in sorted((ROOT / tag / "rank0").glob("dflash_*.pt")):
        match = pattern.match(path.name)
        if match:
            result[(match.group(1), int(match.group(2)), match.group(3))] = torch.load(
                path, map_location="cpu", weights_only=True
            )
    return result


def _tensor_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape)}
    if not a.is_floating_point():
        return {
            "shape": list(a.shape),
            "exact": float((a == b).float().mean()),
            "mismatches": int((a != b).sum()),
        }
    af, bf = a.float().flatten(), b.float().flatten()
    diff = bf - af
    denom = max(float(torch.linalg.vector_norm(af)), 1e-30)
    an = float(torch.linalg.vector_norm(af))
    bn = float(torch.linalg.vector_norm(bf))
    cosine = float(torch.dot(af, bf) / max(an * bn, 1e-30))
    return {
        "shape": list(a.shape),
        "rel_l2": float(torch.linalg.vector_norm(diff)) / denom,
        "cosine": cosine,
        "max_abs": float(diff.abs().max()),
        "ref_absmax": float(af.abs().max()),
        "test_absmax": float(bf.abs().max()),
        "finite": bool(torch.isfinite(bf).all()),
    }


def compare(reference: str, test: str) -> None:
    ref, candidate = _load_leg(reference), _load_leg(test)
    rows = []
    for key in sorted(ref.keys() & candidate.keys()):
        stage, instance, name = key
        row = {"stage": stage, "instance": instance, "tensor": name}
        row.update(_tensor_stats(ref[key], candidate[key]))
        rows.append(row)
    summary = {
        "reference": reference,
        "test": test,
        "common_tensors": len(rows),
        "only_reference": len(ref.keys() - candidate.keys()),
        "only_test": len(candidate.keys() - ref.keys()),
        "rows": rows,
    }
    out = ROOT / f"compare_{test}_vs_{reference}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"{test} vs {reference}: {len(rows)} aligned tensors")
    for row in rows:
        if "rel_l2" in row:
            print(
                f"{row['instance']:>2} {row['stage']:<30} {row['tensor']:<10} "
                f"rel={row['rel_l2']:.6g} cos={row['cosine']:.8f} "
                f"max={row['max_abs']:.6g}"
            )
        elif "exact" in row:
            print(
                f"{row['instance']:>2} {row['stage']:<30} {row['tensor']:<10} "
                f"exact={row['exact']:.4f} mismatch={row['mismatches']}"
            )
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--tag", required=True)
    comp = sub.add_parser("compare")
    comp.add_argument("--reference", required=True)
    comp.add_argument("--test", required=True)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.tag)
    else:
        compare(args.reference, args.test)


if __name__ == "__main__":
    main()
