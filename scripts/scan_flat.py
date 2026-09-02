#!/usr/bin/env python3
"""Scan p_ring dumps for streams with an early sustained flat-top1 regime
(top1 < 0.2) and report whether the committed text is still clean there.

Usage: scan_flat.py <dump> [<dump> ...]
"""
import pickle
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from garble_repro2 import get_tok  # noqa


def streams_from(dump):
    entries = []
    with open(dump, "rb") as fh:
        while True:
            try:
                entries.append(pickle.load(fh))
            except EOFError:
                break
    streams = defaultdict(list)
    gen = defaultdict(int)
    for e in entries:
        rs = e["rs"]
        if streams[(rs, gen[rs])] and e["pos0"] < streams[(rs, gen[rs])][-1]["pos0"]:
            gen[rs] += 1
        streams[(rs, gen[rs])].append(e)
    return streams


def main():
    tok = get_tok()
    for dump in sys.argv[1:]:
        print(f"=== {dump}")
        for (rs, g), rounds in sorted(streams_from(dump).items()):
            n = len(rounds)
            if n < 50:
                continue
            flat = [i for i, e in enumerate(rounds) if e["top1"] and e["top1"][-1] < 0.2]
            frac = len(flat) / n
            if frac < 0.15:
                continue
            # onset = first round index of a sustained flat run (>=20 of next 25)
            onset = None
            for i in flat:
                win = rounds[i : i + 25]
                if sum(1 for e in win if e["top1"] and e["top1"][-1] < 0.2) >= 20:
                    onset = i
                    break
            pos0 = rounds[onset]["pos0"] if onset is not None else -1
            # committed text around onset: tokens committed in rounds onset-5..onset+40
            lo, hi = max(0, (onset or 0) - 5), min(n, (onset or 0) + 40)
            toks = [t for e in rounds[lo:hi] for t in e["tok"]]
            txt = tok.decode(toks)[:400].replace("\n", " ")
            nan_rounds = sum(1 for e in rounds if e.get("row_nan", 0) > 0)
            print(f"  rs={rs} g={g} rounds={n} flat_frac={frac:.2f} "
                  f"onset_round={onset} onset_pos0={pos0} nan_rounds={nan_rounds}")
            print(f"    text@onset: {txt!r}")


if __name__ == "__main__":
    main()
