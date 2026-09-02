#!/usr/bin/env python3
"""One-shot post-leg analysis: scan the fresh bf16 p_ring dumps, pick the
most-degraded and cleanest streams, and print their flat stats. The same-boot
replays are then run by the caller.

Usage: post_leg.py <dump_dir>
"""
import glob
import os
import pickle
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_repro2 import get_tok  # noqa


def main():
    dump_dir = sys.argv[1]
    dumps = sorted(glob.glob(os.path.join(dump_dir, "p_ring_*.dump")),
                   key=os.path.getmtime)
    if not dumps:
        print("no dumps")
        return
    # newest boot = the 4 (or N) newest files with similar sizes
    latest = dumps[-4:] if len(dumps) >= 4 else dumps
    print("using dumps:", [os.path.basename(d) for d in latest])
    tok = get_tok()
    all_streams = {}
    for dump in latest:
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
            key = (rs, gen[rs])
            if streams[key] and e["pos0"] < streams[key][-1]["pos0"]:
                gen[rs] += 1
                key = (rs, gen[rs])
            streams[key].append(e)
        for k, v in streams.items():
            if len(v) > len(all_streams.get(k, [])):
                all_streams[k] = v
    print(f"{'rs':>3} {'gen':>3} {'rounds':>6} {'flat%':>6} {'onset_pos':>9}  text@onset")
    for (rs, g), rounds in sorted(all_streams.items()):
        if len(rounds) < 50:
            continue
        flat = [i for i, e in enumerate(rounds)
                if e["top1"] and e["top1"][-1] < 0.2]
        frac = len(flat) / len(rounds)
        onset = None
        for i in flat:
            win = rounds[i:i + 25]
            if sum(1 for e in win if e["top1"] and e["top1"][-1] < 0.2) >= 20:
                onset = i
                break
        onset_pos = rounds[onset]["pos0"] if onset is not None else -1
        toks = ([t for e in rounds[max(0, (onset or 0) - 2):(onset or 0) + 30]
                 for t in e["tok"]]) if onset is not None else []
        txt = tok.decode(toks)[:120].replace("\n", " ") if toks else ""
        print(f"{rs:>3} {g:>3} {len(rounds):>6} {frac:>6.2f} {onset_pos:>9}  {txt!r}")


if __name__ == "__main__":
    main()
