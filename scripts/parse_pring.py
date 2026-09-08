#!/usr/bin/env python3
"""Parse a VLLM_P_RING pickle-stream dump (per-round rejection-sampler records).

Each record: dict(rs, pos, n, tok, p, top1, drafts, rejected, row_nan,
row_absmax, row_inf, row_famax, top5). Under VLLM_ACCEPT1 the clamp happens
AFTER the ring append, so 'tok' shows the NATURAL acceptance; the committed
count is 1 per round.

Usage: python scripts/parse_pring.py <dump-file> [--limit N]
"""
import argparse
import io
import pickle
import sys


def load_stream(path):
    recs = []
    with open(path, "rb") as f:
        data = f.read()
    bio = io.BytesIO(data)
    while True:
        try:
            r = pickle.load(bio)
        except EOFError:
            break
        except Exception:
            break
        if isinstance(r, dict):
            recs.append(r)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    recs = load_stream(args.path)
    print(f"{len(recs)} records")
    rs_vals = sorted({r.get("rs") for r in recs})
    print(f"rs slots: {rs_vals}")
    for rs in rs_vals:
        sub = [r for r in recs if r.get("rs") == rs]
        if not sub:
            continue
        nat = [len([t for t in r.get("tok", []) if t is not None and t >= 0])
               for r in sub]
        import collections
        hist = collections.Counter(nat)
        print(f"rs={rs}: {len(sub)} rounds, natural-commit histogram "
              f"{dict(sorted(hist.items()))}")
        print(f"  first pos={sub[0].get('pos')} last pos={sub[-1].get('pos')}")
    for r in recs[:args.limit]:
        print({k: r.get(k) for k in ("rs", "pos", "n", "tok", "drafts",
                                     "rejected")})


if __name__ == "__main__":
    main()
