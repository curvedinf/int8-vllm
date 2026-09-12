#!/usr/bin/env python3
"""Correlate per-round PRING verify data with the echo divergence curve.

Inputs: logs/garble/pring_ab/p_ring_*.dump (appended pickles, one record per
request per round) + logs/garble/lp_echo_<leg>.json (per-position live-vs-
echo logprob diffs). Joins output positions to rounds and splits by
committed-token role (accepted draft vs correction) and round acceptance n.

Usage: python scripts/pring_correlate.py --leg pring_leg2
"""
import argparse
import glob
import json
import pickle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--leg", required=True)
    p.add_argument("--dump", default="logs/garble/pring_ab")
    p.add_argument("--committed-pt", default=None,
                   help="optional <leg>_committed.pt to match request by tokens")
    args = p.parse_args()

    recs = []
    for path in sorted(glob.glob(f"{args.dump}/p_ring_*.dump")):
        with open(path, "rb") as f:
            while True:
                try:
                    recs.append(pickle.load(f))
                except EOFError:
                    break
    print(f"PRING records: {len(recs)}")

    echo = json.load(open(f"logs/garble/lp_echo_{args.leg}.json"))
    rows = echo["rows"]                     # (i, eng_lp, echo_lp, diff)
    diff_at = {r[0]: r[3] for r in rows}
    n_out = max(diff_at) + 1 if diff_at else 0
    print(f"echo rows: {len(rows)} positions")

    # Single-request legs: order records by pos0, build position->round map.
    recs.sort(key=lambda r: r["pos0"])
    # cumulative committed positions (pos0 is the CONTEXT position of the
    # first verify row; committed tokens follow it)
    pos_round = {}      # output index -> (round_idx, role, n)
    rounds = []
    covered = 0
    out = 0
    for ridx, r in enumerate(recs):
        n = r["n"]
        rounds.append(r)
        for j in range(n):
            role = "accepted" if j < n - 1 else "correction"
            if out < n_out + 4096:
                pos_round[out] = (ridx, role, n)
            out += 1
    print(f"rounds: {len(rounds)}, committed tokens: {out}")

    # join
    joined = []
    for i, d in diff_at.items():
        if i in pos_round:
            ridx, role, n = pos_round[i]
            joined.append((i, d, ridx, role, n))
    print(f"joined {len(joined)} of {len(diff_at)}")

    def agg(items, tag):
        if not items:
            print(f"{tag}: (none)")
            return
        mad = sum(abs(x[1]) for x in items) / len(items)
        big = sum(1 for x in items if abs(x[1]) > 0.3)
        print(f"{tag}: n={len(items):4d} mean|d|={mad:7.4f} "
              f"frac>0.3={big/len(items):5.3f}")

    print("\nby committed-token role:")
    agg([x for x in joined if x[3] == "accepted"], "  accepted drafts")
    agg([x for x in joined if x[3] == "correction"], "  corrections     ")

    print("\nby round acceptance n:")
    for n in sorted({x[4] for x in joined}):
        agg([x for x in joined if x[4] == n], f"  n={n}")

    print("\nby round index:")
    W = 20
    rmax = max(x[2] for x in joined)
    for w0 in range(0, rmax + 1, W):
        agg([x for x in joined if w0 <= x[2] < w0 + W], f"  rounds {w0:4d}-{w0+W-1:4d}")

    print("\nby role within early (output<100) vs late (>=800):")
    for lo, hi, tag in ((0, 100, "early 0-99"), (100, 800, "mid 100-799"),
                        (800, 10**9, "late 800+")):
        seg = [x for x in joined if lo <= x[0] < hi]
        acc = [x for x in seg if x[3] == "accepted"]
        cor = [x for x in seg if x[3] == "correction"]
        print(f"  {tag}:")
        agg(acc, "    accepted")
        agg(cor, "    correction")

    # Round-level dirt: fraction of a round's tokens >0.3
    print("\ndirtiest rounds (mean|d| over the round's tokens):")
    by_round = {}
    for i, d, ridx, role, n in joined:
        by_round.setdefault(ridx, []).append(abs(d))
    worst = sorted(by_round.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))[:15]
    for ridx, vals in worst:
        r = rounds[ridx]
        print(f"  round {ridx:4d}: mean|d|={sum(vals)/len(vals):7.4f} "
              f"k={len(vals)} n={r['n']} pos0={r['pos0']} "
              f"row_absmax={r.get('row_absmax')} row_nan={r.get('row_nan')}")

    with open(f"logs/garble/pring_corr_{args.leg}.json", "w") as f:
        json.dump({"joined": joined,
                   "rounds": [{k: r[k] for k in ("pos0", "n", "row_nan",
                                                 "row_absmax")}
                              for r in rounds]}, f)
    print(f"\nsaved logs/garble/pring_corr_{args.leg}.json")


if __name__ == "__main__":
    main()
