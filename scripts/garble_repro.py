#!/usr/bin/env python3
"""Repro harness for long-output garble on the qwen38 recipe server.

Sends long-input or short-input requests with long greedy outputs, saves the
full text, and prints a coarse garble scan (non-word char ratio + n-gram
repetition per window) so the onset position is visible without eyeballing
4k tokens.
"""
import argparse
import json
import os
import random
import re
import time
import urllib.request

MODEL = "qwen3.8-27b-gptq8"
API = "http://127.0.0.1:8020/v1/chat/completions"
TOKENIZER_PATH = "/home/curved/models/Qwen3.8-27B-GPTQ-8bit-gs128"

WORDS = ("harbor lantern telescope ledger orchard compass fabric marble cinder "
         "viaduct pasture anvil thicket glacier meadow ratchet sprocket basin "
         "canyon spindle trellis ember quarry ferry mortar lattice kestrel "
         "pasture signal riddle bolster caravan nugget pillar wicket zenith "
         "anchor furrow hamlet prairie skirmish vigilo beacon cistern drumlin").split()

TOPICS = [
    "the coastal trade routes of the northern league",
    "early mechanical calculators and their workshops",
    "the irrigation councils of the river valleys",
    "migration of craftspeople across the high passes",
    "the guild archives and their indexing systems",
    "surveyors, maps, and the standardization of measure",
    "the postal relay networks of the interior provinces",
    "ore prospecting and the smelting cooperatives",
]


def make_corpus(target_tokens: int, nonce: str) -> str:
    rng = random.Random(hash(nonce) & 0xFFFF)
    parts = [f"Reference dossier {nonce}. Below are independent notes."]
    n = 60
    while True:
        topic = rng.choice(TOPICS)
        words = [rng.choice(WORDS) for _ in range(rng.randint(60, 110))]
        parts.append(f"Note {n}. On {topic}: " + " ".join(words) + ".")
        n += 1
        if n % 40 == 0 and est_tokens(parts) >= target_tokens:
            break
    return "\n".join(parts)


_cache = {}


def est_tokens(parts):
    tok = _cache.get("tok")
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        _cache["tok"] = tok
    return len(tok("\n".join(parts))["input_ids"])


def build_prompt(target_tokens: int, nonce: str):
    tok = _cache.get("tok")
    if tok is None:
        est_tokens(["x"])
    tok = _cache["tok"]
    corpus = make_corpus(target_tokens, nonce)
    ids = tok(corpus)["input_ids"]
    if len(ids) > target_tokens:
        corpus = tok.decode(ids[:target_tokens])
    return corpus


def request(in_tokens: int, out_tokens: int, nonce: str, seed: int):
    corpus = build_prompt(in_tokens, nonce)
    body = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": (
                "You are given reference notes. Write a long, coherent "
                "chronological essay synthesizing them. Write as much as "
                "possible; do not stop early.\n\n" + corpus),
        }],
        "temperature": 0.0,
        "max_tokens": out_tokens,
        "seed": seed,
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', '')}",
        })
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    dt = time.time() - t0
    text = out["choices"][0]["message"]["content"]
    usage = out.get("usage", {})
    return text, dt, usage


WORD_RE = re.compile(r"[A-Za-z0-9 ,.:;'\n\"()\-]")


def scan(text: str, window: int = 2000, label: str = ""):
    """Print per-window weird-char and repetition stats; return onset char idx."""
    onset = None
    for i in range(0, len(text), window):
        w = text[i:i + window]
        weird = 1.0 - (len(WORD_RE.findall(w)) / max(1, len(w)))
        toks = w.split()
        uniq = len(set(zip(toks, toks[1:]))) / max(1, len(toks) - 1)
        flag = ""
        if weird > 0.08 or uniq < 0.35:
            flag = "  <-- SUSPECT"
            onset = onset if onset is not None else i
        print(f"[{label}] char {i:>7}: weird={weird:.3f} uniq2={uniq:.2f}{flag}")
    return onset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-tokens", type=int, default=20000)
    ap.add_argument("--out-tokens", type=int, default=4096)
    ap.add_argument("--tag", default="leg")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    nonce = f"{args.tag}-{int(time.time())}"
    text, dt, usage = request(args.in_tokens, args.out_tokens, nonce, args.seed)
    path = f"/home/curved/vllm-gfx908/logs/garble/{args.tag}.txt"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"[{args.tag}] {dt:.1f}s usage={usage} -> {path} ({len(text)} chars)")
    onset = scan(text, label=args.tag)
    if onset is not None:
        print(f"[{args.tag}] GARBLE ONSET ~char {onset}")
    else:
        print(f"[{args.tag}] no garble detected by heuristic")


if __name__ == "__main__":
    main()
