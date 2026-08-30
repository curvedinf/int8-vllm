#!/usr/bin/env python3
"""Find the first token divergence between two captured reasoning streams."""
import sys
from transformers import AutoTokenizer

TOK = "/home/curved/models/Qwen3.8-27B-GPTQ-8bit-gs128"


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(TOK)
    a = tok(open(a_path).read())["input_ids"]
    b = tok(open(b_path).read())["input_ids"]
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    print(f"lens: {len(a)} vs {len(b)}; first divergence at reasoning token {i}")
    lo = max(0, i - 40)
    ctx_a = tok.decode(a[lo:i + 40])
    ctx_b = tok.decode(b[lo:i + 40])
    print(f"--- A ({a_path}) around divergence:\n{ctx_a[:1200]}")
    print(f"--- B ({b_path}) around divergence:\n{ctx_b[:1200]}")
    # Repetition profile of the tail of the longer stream
    long_ = a if len(a) >= len(b) else b
    tail = tok.decode(long_[max(0, len(long_) - 300):])
    uniq = len(set(zip(tail.split(), tail.split()[1:]))) / max(1, len(tail.split()) - 1)
    print(f"--- longer-stream tail uniq2={uniq:.2f}:\n{tail[:600]}")


if __name__ == "__main__":
    main()
