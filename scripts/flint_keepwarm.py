#!/usr/bin/env python3
"""Keep the XGMI fabric warm with one tiny request per interval.

The 2026-11-12 slow window started at the first request after a 55-min idle:
the kernel log shows XGMI PCS FCInitTimeoutErr plus ~450 uncorrectable
xgmi_wafl errors on all four GPUs at that instant (after 47 min of silence),
and the server stayed 2-6x slower until a restart. A periodic request keeps
the links out of the idle state and doubles as a latency canary.

  .venv/bin/python scripts/flint_keepwarm.py --interval 600 --duration 3600
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


async def once(client, url, hdr, messages):
    payload = {"model": "flint", "messages": messages, "max_tokens": 8,
               "temperature": 0.0, "stream": False}
    t0 = time.monotonic()
    r = await client.post(url + "/v1/chat/completions", headers=hdr,
                          json=payload, timeout=300)
    r.raise_for_status()
    return time.monotonic() - t0, r.json()["usage"]["completion_tokens"]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8020")
    ap.add_argument("--api-key", default="recipe-local")
    ap.add_argument("--interval", type=float, default=600)
    ap.add_argument("--duration", type=float, default=3600)
    ap.add_argument("--corpus", default=str(
        Path.home() / "ksdraft/runs/onpolicy_ready/train_tokens.pt"))
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(
        Path.home() / "models/Flint-GPTQ-INT8-W8A8-GS128-PTQR-r1s40-512k"))
    tokens = torch.load(args.corpus, map_location="cpu", weights_only=True)["tokens"]
    text = tok.decode(tokens[:20000].tolist())[:6000]
    messages = [{"role": "system",
                 "content": "You are a coding agent.\n\n" + text},
                {"role": "user", "content": "ping"}]
    hdr = {"Authorization": f"Bearer {args.api_key}"}
    end = time.monotonic() + args.duration
    async with httpx.AsyncClient() as client:
        while time.monotonic() < end:
            try:
                dt, n = await once(client, args.base_url, hdr, messages)
                print(json.dumps({"t": round(time.time(), 1),
                                  "latency_s": round(dt, 3), "tokens": n}),
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"t": round(time.time(), 1),
                                  "error": type(exc).__name__}), flush=True)
            await asyncio.sleep(args.interval)


if __name__ == "__main__":
    asyncio.run(main())
