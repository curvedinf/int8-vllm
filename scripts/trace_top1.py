#!/usr/bin/env python3
"""Dump the top1 trajectory + text snippets for one p_ring stream."""
import pickle
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from garble_repro2 import get_tok  # noqa


def main():
    dump, rs_q, gen_q = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    n_tail = int(sys.argv[4]) if len(sys.argv) > 4 else 260
    tok = get_tok()
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
    rounds = streams[(rs_q, gen_q)]
    print(f"rounds: {len(rounds)} pos0 {rounds[0]['pos0']} .. {rounds[-1]['pos0']}")
    for i, e in enumerate(rounds[:n_tail]):
        if i % 10 == 0 or (e["top1"] and e["top1"][-1] < 0.3):
            t1 = e["top1"][-1] if e["top1"] else -1
            txt = tok.decode(e["tok"])[:60].replace("\n", " ")
            top5 = e.get("top5", [])
            top5s = " ".join(tok.decode([t])[:8] for t in top5[:5])
            print(f"  r{i:4d} pos0={e['pos0']} n={e['n']} top1={t1:.3f} "
                  f"top5=[{top5s}] text={txt!r}")


if __name__ == "__main__":
    main()
