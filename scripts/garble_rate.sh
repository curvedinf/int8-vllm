#!/usr/bin/env bash
# Rate leg: N no-tools 40k probes, temp 1.0, count corrupt tails.
set -euo pipefail
N="${1:-10}"
export PYTHONPATH="${HOME}/vllm-gfx908:${HOME}/aiter"
P=$(pgrep -f "[.]venv/bin/vllm serve" | head -1)
export VLLM_API_KEY=$(tr '\0' '\n' < /proc/$P/environ | grep '^VLLM_API_KEY=' | cut -d= -f2)
cd "${HOME}/vllm-gfx908/scripts"
../.venv/bin/python - "$N" <<'PY'
import sys, json, time, os, urllib.request
sys.path.insert(0, '.')
from garble_docs_probe import build_corpus
from garble_repro2 import get_tok, MODEL, API, save
N = int(sys.argv[1])
tok = get_tok()
corpus = build_corpus(tok)
corrupt = 0
for k in range(N):
    body = {"model": MODEL, "messages": [{"role": "user", "content":
        "Summarize the documentation below as exhaustive release notes with headers "
        "and numbered lists, quoting key config names inline. Do not stop early.\n\n" + corpus}],
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 4096,
        "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY','')}"})
    pieces=[]; t0=time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: "): continue
                d0 = line[6:]
                if d0 == "[DONE]": break
                try: chunk=json.loads(d0)
                except Exception: continue
                d = chunk.get("choices",[{}])[0].get("delta",{}) or {}
                if d.get("content"): pieces.append(d["content"])
    except Exception as e:
        print(f"[r{k}] stream error {e}", flush=True); continue
    text="".join(pieces)
    if len(text) < 3000:
        print(f"[r{k}] short/BAILED chars={len(text)}", flush=True); continue
    save(f"RATE_r{k}", text)
    tail = text[-300:]
    deg = tail.count("**") + sum(1 for ln in tail.splitlines() if len(ln.strip())<6)
    is_c = deg > 25
    corrupt += is_c
    print(f"[r{k}] dur={time.time()-t0:.0f}s chars={len(text)} degen={deg} corrupt={'YES' if is_c else 'no'}", flush=True)
print(f"RATE: {corrupt} corrupt of {N}", flush=True)
PY
