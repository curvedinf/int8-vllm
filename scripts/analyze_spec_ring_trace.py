#!/usr/bin/env python3
"""Match saved API token chunks and clean ranks to a VLLM_P_RING dump."""

import argparse
import glob
import json
import pickle
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", type=Path,
                    help="logs/garble/<tag>_decodeN without an extension")
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--ring-dir", type=Path, required=True)
    args = ap.parse_args()

    trace = json.loads(args.prefix.with_suffix(".token_ids.json").read_text())
    prompt_len = len(trace["prompt_token_ids"])
    chunks = [json.loads(line) for line in
              args.prefix.with_suffix(".token_chunks.jsonl").read_text().splitlines()]
    clean = json.loads(args.oracle.read_text())["rows"]

    dumps = sorted(glob.glob(str(args.ring_dir / "p_ring_*.dump")))
    if not dumps:
        ap.error(f"no p_ring dump in {args.ring_dir}")
    by_key = {}
    with open(dumps[0], "rb") as stream:
        while True:
            try:
                record = pickle.load(stream)
            except EOFError:
                break
            key = (record["pos0"], tuple(record["tok"]))
            by_key[key] = record

    matched = 0
    missing = 0
    output = []
    for chunk in chunks:
        start = chunk["start"]
        ids = chunk["ids"]
        key = (prompt_len + start - 1, tuple(ids))
        record = by_key.get(key)
        if record is None:
            missing += 1
            continue
        matched += 1
        for offset, token_id in enumerate(ids):
            i = start + offset
            if i >= len(clean):
                continue
            row = clean[i]
            if row["rank"] <= 20:
                continue
            output.append({
                "i": i,
                "id": token_id,
                "clean_rank": row["rank"],
                "clean_lp": row["lp"],
                "chunk_len": len(ids),
                "chunk_offset": offset,
                "spec_p": record["p"][offset],
                "spec_top1_p": record["top1"][offset],
                "spec_raw_p": record.get("raw_p") if offset == len(ids) - 1
                else None,
                "spec_top5": record.get("top5")
                if offset == len(ids) - 1 else None,
                "resample_row_nan": record.get("row_nan")
                if offset == len(ids) - 1 else None,
            })
    print(json.dumps({"chunks": len(chunks), "matched": matched,
                      "missing": missing, "rare": output}, indent=2))


if __name__ == "__main__":
    main()
