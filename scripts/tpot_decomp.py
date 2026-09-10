#!/usr/bin/env python3
"""Streaming TPOT-vs-context decomposition on prod-shaped completions.

For each input length: one unseeded streaming request (prod params),
recording first-chunk latency (prefill+TTFT) and steady-state decode rate
from chunk timestamps. Separates prefill / per-step decode / acceptance
decay without any analysis-script heuristics — raw timestamps only.
"""
import sys, os, json, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_repro import build_prompt

API = "http://127.0.0.1:8020/v1/chat/completions"
KEY = os.environ.get("VLLM_API_KEY", "")


def leg(in_tokens, out_tokens=1200, tag=""):
    corpus = build_prompt(in_tokens, f"tpot-{tag}-{int(time.time())}")
    body = {
        "model": "qwen3.8-27b-gptq8",
        "messages": [{"role": "user", "content": (
            "You are given reference notes. Write a long, coherent "
            "chronological essay synthesizing them. Write as much as "
            "possible; do not stop early.\n\n" + corpus)}],
        "max_tokens": out_tokens,
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 0.0, "repetition_penalty": 1.0,
        "stream": True, "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    first = None
    marks = []  # (t, cum_tokens)
    n = 0
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "usage" in ev and ev.get("usage"):
                usage = ev["usage"]
            ch = (ev.get("choices") or [{}])[0]
            delta = ch.get("delta", {})
            c = delta.get("content") or delta.get("reasoning_content") or ""
            if c and not c.strip():
                c = c  # count whitespace chunks too (they are tokens)
            if c:
                if first is None:
                    first = time.time()
                n += 1
                marks.append((time.time(), n))
    t_end = time.time()
    ttft = (first - t0) if first else float("nan")
    # steady-state decode rate: slope over the last 80% of tokens
    if len(marks) >= 20:
        k = int(len(marks) * 0.2)
        (ta, na), (tb, nb) = marks[k], marks[-1]
        rate = (nb - na) / (tb - ta) if tb > ta else float("nan")
        # rate over first 200 tokens vs last 200 (acceptance decay check)
        early = None
        if len(marks) > 260:
            (t1, n1), (t2, n2) = marks[20], marks[220]
            early = (n2 - n1) / (t2 - t1) if t2 > t1 else float("nan")
            (t3, n3), (t4, n4) = marks[-220], marks[-20]
            late = (n4 - n3) / (t4 - t3) if t4 > t3 else float("nan")
        else:
            early = late = float("nan")
    else:
        rate = early = late = float("nan")
    total_out = (usage or {}).get("completion_tokens", n)
    print(f"in={in_tokens:6d} ttft={ttft:7.1f}s total={t_end-t0:7.1f}s "
          f"out={total_out} wall_rate={total_out/(t_end-t0):6.2f} tok/s "
          f"decode_rate={rate:6.2f} early={early:6.2f} late={late:6.2f}", flush=True)


if __name__ == "__main__":
    for in_tok in (2000, 8000, 20000):
        leg(in_tok, 1200, tag=str(in_tok))
