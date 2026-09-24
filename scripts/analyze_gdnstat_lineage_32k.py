#!/usr/bin/env python3
"""Check adjacent speculative GDN state-window norms in a live capture.

``mamba_get_block_table_tensor`` starts at ``(seq_len - 1) // block_size``.
The GDN kernel reads the previous accepted checkpoint at relative column
``ri`` and writes this round's accepted checkpoint at ``na - 1`` for the
recurrent state. The convolution kernel instead uses the base slot (relative
column 0) with ``ri`` as an offset inside that slot's rolling buffer.
Other convolution slots can contain uninitialized data. Captures taken before
the probe correction on 2026-09-24 have a ``kbase`` field with an extra +1.
"""

import argparse
import glob
import math
from collections import defaultdict
from pathlib import Path

import torch


def load_rows(directory: Path, prefix: str):
    groups = defaultdict(list)
    for filename in glob.glob(str(directory / f"{prefix}_*.pt")):
        path = Path(filename)
        parts = path.stem.split("_")
        pid = int(parts[1])
        groups[pid].append(path)
    result = {}
    for pid, paths in groups.items():
        rows = []
        for path in sorted(paths, key=lambda x: int(x.stem.rsplit("_", 1)[1])):
            rows.extend(torch.load(path, weights_only=False)["rounds"])
        result[pid] = rows
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("--min-nct", type=int, default=31900)
    ap.add_argument("--max-gap", type=int, default=8)
    ap.add_argument("--max-t", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.0005)
    args = ap.parse_args()

    pres = load_rows(args.directory, "gdnstat")
    posts = load_rows(args.directory, "gdnstatpost")
    for pid, pre_rows in sorted(pres.items()):
        post_by_key = {(r["n"], r["rs"]): r for r in posts.get(pid, [])}
        last_pre = {}
        last_post = {}
        checked = bad = skipped = 0
        crossing_checked = crossing_bad = 0
        outliers = []
        delta_hist = defaultdict(int)
        for pre in sorted(pre_rows, key=lambda r: r["n"]):
            rs = pre["rs"]
            cur_delta = pre.get("kbase", -999) - pre["col"]
            if pre["nct"] >= args.min_nct:
                delta_hist[cur_delta] += 1
            prev = last_pre.get(rs)
            post = last_post.get(rs)
            if (
                pre["nct"] >= args.min_nct
                and prev is not None
                and post is not None
                and prev["T"] <= args.max_t
                and pre["T"] <= args.max_t
                and 0 <= pre["nct"] - prev["nct"] <= args.max_gap
            ):
                for key, new_norms in pre["norms"].items():
                    if key.endswith("#st0"):
                        old_slot = new_slot = 0
                    else:
                        old_slot = post["na"] - 1
                        new_slot = pre["ri"]
                    old_norms = post["norms"].get(key)
                    if (
                        old_norms is None
                        or not 0 <= old_slot < len(old_norms)
                        or not 0 <= new_slot < len(new_norms)
                    ):
                        skipped += 1
                        continue
                    old, new = old_norms[old_slot], new_norms[new_slot]
                    if not math.isfinite(old) or not math.isfinite(new):
                        skipped += 1
                        continue
                    relative = abs(new - old) / max(abs(old), 1)
                    checked += 1
                    crossed = pre["col"] != prev["col"]
                    if crossed:
                        crossing_checked += 1
                    if relative > args.tol:
                        bad += 1
                        if crossed:
                            crossing_bad += 1
                        if len(outliers) < 30:
                            outliers.append({
                                "n": pre["n"], "rs": rs, "nct": pre["nct"],
                                "key": key, "old": old, "new": new,
                                "relative": round(relative, 6),
                                "old_slot": old_slot, "new_slot": new_slot,
                                "old_col": prev["col"], "new_col": pre["col"],
                                "old_na": post["na"], "new_ri": pre["ri"],
                                "crossed": crossed,
                            })
            last_pre[rs] = pre
            last_post[rs] = post_by_key.get((pre["n"], rs))
        print(f"pid={pid} pre_rows={len(pre_rows)} post_rows="
              f"{len(posts.get(pid, []))} checked={checked} "
              f"mismatches={bad} skipped={skipped} "
              f"crossing_checked={crossing_checked} "
              f"crossing_mismatches={crossing_bad} "
              f"kbase_minus_col={dict(sorted(delta_hist.items()))}")
        for item in outliers:
            print(item)


if __name__ == "__main__":
    main()
