#!/usr/bin/env python3
"""Analyze VLLM_KV_BTSCHECK bt_*.jsonl records for forward-metadata
corruption: duplicate block ids (aliasing) and settled-prefix hash flips.

Usage: .venv/bin/python scripts/analyze_btcheck.py logs/garble/btcheck_legz
"""

import glob
import json
import sys
from collections import defaultdict


def main(d):
    files = sorted(glob.glob(f"{d}/bt_*.jsonl"))
    if not files:
        print("no bt records")
        return
    recs = []
    for fp in files:
        for line in open(fp):
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    recs.sort(key=lambda r: r["n"])
    print(f"{len(recs)} records, n range {recs[0]['n']}..{recs[-1]['n']}")

    dups = [(r["n"], r["inst"], row)
            for r in recs for row in r["bt"] if row[3] > 0]
    print(f"\nrecords with duplicate block ids in the live window: "
          f"{len(dups)}")
    for n, inst, row in dups[:20]:
        print(f"  n={n} inst={inst} settled={row[0]} seq={row[1]} "
              f"dup={row[3]} tail3={row[4]}")

    # settled-prefix hash must be stable per (inst,row,settled) during decode
    last = {}
    flips = []
    for r in recs:
        for ri, row in enumerate(r["bt"]):
            settled, seq, h, dup, tail = row
            if settled == 0:
                continue
            key = (r["inst"], ri)
            prev_settled, prev_h, prev_n = last.get(key, (None, None, None))
            if prev_h is not None and h != prev_h and settled >= prev_settled \
                    and seq >= 1 and r["n"] - prev_n <= 3:
                # grew monotonically but the already-settled prefix changed
                flips.append((r["n"], key, prev_settled, settled, prev_h, h))
            last[key] = (settled, h, r["n"])
    print(f"\nsettled-prefix hash flips (possible metadata corruption): "
          f"{len(flips)}")
    for f in flips[:20]:
        print("  ", f)

    # sanity: distribution of seq lens seen
    seqs = [row[1] for r in recs for row in r["bt"]]
    if seqs:
        print(f"\nrow seq_len: max={max(seqs)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/garble/btcheck_legz")
