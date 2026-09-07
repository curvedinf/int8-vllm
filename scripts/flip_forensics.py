#!/usr/bin/env python3
"""Flip-round forensics on a captured death from the P-ring dumps.

Usage: python /tmp/flip_forensics.py <dump_dir>
Finds the last segment (the dead request), locates the permanent-death
round, and prints ±10 rounds of: n, p, top5, row_absmax, row_nan,
drafts, rejected, committed token ids. Also decodes the committed tokens
around the flip.
"""
import glob
import math
import pickle
import sys

d = sys.argv[1] if len(sys.argv) > 1 else 'logs/ptqr/p_ring_g1'
recs = []
for fp in sorted(glob.glob(f'{d}/*.dump')):
    with open(fp, 'rb') as f:
        while True:
            try:
                r = pickle.load(f)
            except EOFError:
                break
            except Exception:
                continue
            if isinstance(r, dict) and 'n' in r:
                recs.append(r)
segs, cur = [], [recs[0]]
for r in recs[1:]:
    if r['pos0'] < cur[-1]['pos0'] - 20:
        segs.append(cur)
        cur = [r]
    else:
        cur.append(r)
segs.append(cur)
dead = [s for s in segs if len(s) > 200 and
        sum(1 for r in s if r['n'] == 1) / len(s) > 0.6]
print(f"{len(segs)} segments; dead segments: {len(dead)}")
if not dead:
    sys.exit(0)
s = dead[-1]
base = s[0]['pos0']
flip = None
for j in range(5, len(s) - 20):
    if all(r['n'] == 1 for r in s[j:j + 20]):
        flip = j
        break
print(f"dead segment: {len(s)} rounds, pos0 {s[0]['pos0']}->{s[-1]['pos0']}, "
      f"flip at round {flip} (output token {s[flip]['pos0'] - base})")
if flip is None:
    flip = len(s) // 2
lo = max(0, flip - 8)
for r in s[lo:flip + 8]:
    p = r.get('p', [])
    print(f"pos0 {r['pos0']} n {r['n']:2} p0 {p[0] if p else -1:.4f} "
          f"top5 {r.get('top5')} absmax {r.get('row_absmax')} "
          f"nan {r.get('row_nan')} toks {r.get('tok', [])[:6]} "
          f"drafts {r.get('drafts', [])[:4]} rej {r.get('rejected', [])[:3]}")
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained('/home/curved/models/Qwen3.8-27B-GPTQ-8bit-gs128')
toks = []
for r in s[max(0, flip - 30):flip + 30]:
    toks.extend(r.get('tok', []))
print("=== decoded window around flip ===")
print(tk.decode(toks)[-1200:])
