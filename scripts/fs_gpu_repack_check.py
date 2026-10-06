#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte gate for the AIS rank-repack cache.

Verifies every manifest entry against the original checkpoint, modeling the
stock loader's duplicate-name semantics (shards walk in natural-sorted
order; later yields of a duplicated tensor name win in the params — PTQR
exports duplicate some names across shards, and the cache replays the same
sequence). Also sys.path-bootstraps the repo like the other fs_gpu scripts.

Usage:
  .venv/bin/python scripts/fs_gpu_repack_check.py [CACHE_DIR MODEL_DIR]
(defaults: the recipe target + draft checkpoints)
"""
import glob
import json
import mmap as mm
import os
import struct
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_REPO, "vllm")):
    sys.path.insert(0, _REPO)

_d = sys.argv[1:]
cache_dir = _d[0] if _d else "/home/curved/.cache/vllm/ais_repack/Qwen3.8-27B-PTQR-R10S60"
model_dir = _d[1] if len(_d) > 1 else "/home/curved/models/Qwen3.8-27B-PTQR-R10S60"

meta = json.load(open(f"{cache_dir}/rank0.meta.json"))
refs = {}
for f in sorted(glob.glob(f"{model_dir}/*.safetensors")):
    fh = open(f, "rb")
    (hl,) = struct.unpack("<Q", fh.read(8))
    hdr = json.loads(fh.read(hl))
    refs[f] = (8 + hl, hdr, mm.mmap(fh.fileno(), 0, access=mm.ACCESS_READ), fh)

# Yield-order source map: the stock iterator walks shards in natural-sorted
# order and yields every (name, tensor) — PTQR exports duplicate some names
# across shards, later yield wins in the params. Cache entries replay the
# same sequence, so each entry must match ITS successive source.
from vllm.model_executor.model_loader.weight_utils import _natural_sort_key  # noqa: E402

sources: dict[str, list[tuple[str, int]]] = {}
for f in sorted(refs.keys(), key=_natural_sort_key):
    db, hdr, m, fh = refs[f]
    for name, e in hdr.items():
        if name == "__metadata__":
            continue
        sources.setdefault(name, []).append((f, db + e["data_offsets"][0]))
counts: dict[str, int] = {}

binf = open(f"{cache_dir}/rank0.bin", "rb")
bad = 0
checked = 0
examples = []
for t in meta["tensors"]:
    name = t["name"]
    i = counts.get(name, 0)
    counts[name] = i + 1
    src = sources.get(name, [])
    if i >= len(src):
        examples.append(f"NO-SOURCE {name} #{i}")
        bad += 1
        continue
    f, dstart = src[i]
    m = refs[f][2]
    binf.seek(t["block_off"])
    block = binf.read(t["block_len"])
    off = 0
    ok = True
    for bo, n in t["runs"]:
        want = m[dstart + bo : dstart + bo + n]
        if block[off : off + n] != want:
            ok = False
            if len(examples) < 8:
                examples.append(
                    f"MISMATCH {name} #{i}: run @{bo}+{n} of {len(t['runs'])} runs"
                )
            break
        off += n
    checked += 1
    if not ok:
        bad += 1
print(f"checked {checked} tensors, bad {bad}")
print("\n".join(examples))
for f, (db, hdr, m, fh) in refs.items():
    m.close()
    fh.close()
sys.exit(1 if bad else 0)
