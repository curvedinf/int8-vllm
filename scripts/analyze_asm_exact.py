#!/usr/bin/env python3
"""Compare captured target input anchors with exact API prompt/output IDs.

The ASM ring records every scheduled input row. Draft lookahead tokens may be
rejected, so only row 0 of each decode round must equal the committed API
token at that absolute position.
"""

import argparse
import glob
import json
import pickle
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="mixed_phase_probe run tag")
    ap.add_argument("ring_dir", type=Path)
    args = ap.parse_args()

    traces = {}
    for path in sorted(Path("logs/garble").glob(f"{args.tag}_*.token_ids.json")):
        name = path.name.removeprefix(args.tag + "_").removesuffix(
            ".token_ids.json"
        )
        traces[name] = json.loads(path.read_text())
    if not traces:
        ap.error("no saved token ID traces for tag")

    files = sorted(glob.glob(str(args.ring_dir / "asm_ring_*.dump")))
    if not files:
        ap.error("no ASM ring dump")
    rounds = defaultdict(list)
    # TP workers see the same request/input IDs; one rank is sufficient.
    for path in files[:1]:
        with open(path, "rb") as stream:
            step = 0
            while True:
                try:
                    record = pickle.load(stream)
                except EOFError:
                    break
                off = 0
                for req_id, n in zip(record["req_ids"], record["nsched"]):
                    n = int(n)
                    if n > 0:
                        ids = record["ids"][off:off + n]
                        pos = record["pos"][off:off + n]
                        rounds[req_id].append((step, ids, pos))
                    off += n
                step += 1

    result = {}
    for req_id, req_rounds in rounds.items():
        # A first prefill chunk spans the unique "stream decodeN/prefillN"
        # prefix. Score prompt token IDs only, never speculative lookahead.
        scores = []
        for name, trace in traces.items():
            prompt = trace["prompt_token_ids"]
            same = checked = 0
            for _, ids, pos in req_rounds[:3]:
                for token_id, position in zip(ids, pos):
                    if 0 <= position < len(prompt):
                        same += token_id == prompt[position]
                        checked += 1
            scores.append((same, checked, name))
        scores.sort(reverse=True)
        same, checked, name = scores[0]
        trace = traces[name]
        prompt_len = len(trace["prompt_token_ids"])
        output = trace["output_token_ids"]
        anchor_checked = 0
        anchor_bad = []
        first_decode_step = None
        by_pos0 = {}
        for step, ids, pos in req_rounds:
            if pos:
                by_pos0[pos[0]] = (step, ids)
            if not pos or pos[0] < prompt_len:
                continue
            if first_decode_step is None:
                first_decode_step = step
            index = pos[0] - prompt_len
            if 0 <= index < len(output):
                anchor_checked += 1
                if ids[0] != output[index]:
                    anchor_bad.append({"step": step, "output_i": index,
                                       "fed": ids[0], "committed": output[index]})
        chunks_path = Path("logs/garble") / (
            f"{args.tag}_{name}.token_chunks.jsonl"
        )
        chunk_checked = accepted_checked = 0
        chunk_bad = []
        if chunks_path.exists():
            for line in chunks_path.read_text().splitlines():
                chunk = json.loads(line)
                start, emitted = chunk["start"], chunk["ids"]
                found = by_pos0.get(prompt_len + start - 1)
                if found is None:
                    continue
                step, fed = found
                expected_anchor = (
                    trace["prompt_token_ids"][-1]
                    if start == 0 else output[start - 1]
                )
                chunk_checked += 1
                if fed[0] != expected_anchor:
                    chunk_bad.append({"step": step, "output_i": start,
                                      "kind": "anchor", "fed": fed[0],
                                      "committed": expected_anchor})
                for offset, expected in enumerate(emitted[:-1], start=1):
                    if offset >= len(fed):
                        chunk_bad.append({"step": step, "output_i": start + offset - 1,
                                          "kind": "missing_input"})
                        continue
                    accepted_checked += 1
                    if fed[offset] != expected:
                        chunk_bad.append({"step": step, "output_i": start + offset - 1,
                                          "kind": "accepted_draft", "fed": fed[offset],
                                          "committed": expected})
        result[name] = {
            "req_id": req_id,
            "prompt_match": f"{same}/{checked}",
            "rounds_captured": len(req_rounds),
            "first_decode_step": first_decode_step,
            "anchor_checked": anchor_checked,
            "anchor_mismatch_count": len(anchor_bad),
            "first_anchor_mismatches": anchor_bad[:10],
            "chunks_matched_to_inputs": chunk_checked,
            "accepted_draft_tokens_checked": accepted_checked,
            "chunk_input_mismatch_count": len(chunk_bad),
            "first_chunk_input_mismatches": chunk_bad[:10],
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
