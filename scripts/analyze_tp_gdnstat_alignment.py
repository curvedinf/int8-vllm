#!/usr/bin/env python3
"""Compare per-round speculative positions and acceptance across TP workers.

GDNSTAT request-state slots are local to each worker, so rows are aligned by
round number and batch order within the round. The capture must come from one
boot with all workers recording the same request schedule.
"""

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import torch


def load(directory: Path, prefix: str) -> dict[int, dict[int, list[dict]]]:
    groups: dict[int, list[Path]] = defaultdict(list)
    for name in glob.glob(str(directory / f"{prefix}_*.pt")):
        path = Path(name)
        groups[int(path.stem.split("_")[1])].append(path)
    result = {}
    for pid, paths in groups.items():
        rounds: dict[int, list[dict]] = defaultdict(list)
        for path in sorted(paths, key=lambda p: int(p.stem.rsplit("_", 1)[1])):
            for row in torch.load(path, weights_only=False)["rounds"]:
                rounds[row["n"]].append(row)
        result[pid] = rounds
    return result


def compare(directory: Path, prefix: str, fields: tuple[str, ...]) -> None:
    workers = load(directory, prefix)
    pids = sorted(workers)
    if len(pids) < 2:
        raise RuntimeError(f"{prefix}: expected at least two TP workers")
    common = set.intersection(*(set(worker) for worker in workers.values()))
    checked = mismatches = skipped = 0
    examples = []
    for n in sorted(common):
        batches = [workers[pid][n] for pid in pids]
        if len({len(batch) for batch in batches}) != 1:
            skipped += 1
            continue
        for row_idx in range(len(batches[0])):
            values = [tuple(batch[row_idx][field] for field in fields)
                      for batch in batches]
            checked += 1
            if len(set(values)) > 1:
                mismatches += 1
                if len(examples) < 8:
                    examples.append((n, row_idx, values))
    print(f"{prefix}: workers={pids} rounds={len(common)} rows={checked} "
          f"mismatches={mismatches} skipped_rounds={skipped}")
    for n, row_idx, values in examples:
        print(f"  n={n} row={row_idx} values={values}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    compare(args.directory, "gdnstat", ("nct", "T", "ri"))
    compare(args.directory, "gdnstatpost", ("na",))


if __name__ == "__main__":
    main()
