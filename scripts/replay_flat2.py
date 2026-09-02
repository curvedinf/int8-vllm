#!/usr/bin/env python3
"""Calibrated same-boot replay differential.

Empirically aligns the p_ring committed stream with the probe output file
(finds the token shift k by exact match), submits prompt + true output tokens
to the prefill path (prompt_logprobs=20), and compares, per committed token:

  in-engine verify p/top1 (ring)  vs  reference p/top1 (pl[Q], Q = true pos)

Verdict rule: at the corruption entry (in-engine top1 crash on a still-clean
committed prefix), reference top1 >> in-engine top1 => verify-path state
corruption convicted. Reference ALSO flat => model behavior.

Usage: replay_flat2.py <dump> <rs> <gen> <probe_file> [n_tokens]
"""
import json
import math
import os
import pickle
import statistics
import sys
import urllib.request
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from garble_docs_probe import build_corpus  # noqa
from garble_repro2 import get_tok  # noqa

API = "http://127.0.0.1:8020/v1/completions"


def main():
    dump, rs_q, gen_q, probe_file = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    n_tok = int(sys.argv[5]) if len(sys.argv) > 5 else 2200
    tok = get_tok()
    corpus = build_corpus(tok)
    msg = ("Summarize the documentation below as exhaustive release notes with "
           "headers and numbered lists, quoting key config names inline. "
           "Do not stop early.\n\n" + corpus)
    tmpl = tok.apply_chat_template([{"role": "user", "content": msg}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    prompt_ids = tok.encode(tmpl, add_special_tokens=False)
    P = len(prompt_ids)
    print(f"prompt ids: {P}")

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
        if e["pos0"] < P - 40:  # probe requests only
            continue
        rs = e["rs"]
        key = (rs, gen[rs])
        if streams[key] and e["pos0"] < streams[key][-1]["pos0"]:
            gen[rs] += 1
            key = (rs, gen[rs])
        streams[key].append(e)
    rounds = streams[(rs_q, gen_q)]
    committed = [t for e in rounds for t in e["tok"]]
    in_eng = {}  # committed idx j -> (p, top1)
    j = 0
    for e in rounds:
        for c, (t, p, t1) in enumerate(zip(e["tok"], e["p"], e["top1"])):
            in_eng[j] = (p, t1)
            j += 1
    print(f"rounds: {len(rounds)} committed: {len(committed)} "
          f"pos0 {rounds[0]['pos0']}..{rounds[-1]['pos0']}")

    # --- empirical token alignment with the probe file ---
    out_ids = tok.encode(open(probe_file, errors="replace").read(),
                         add_special_tokens=False)
    best_k, best_score = None, -1
    for k in range(0, 6):
        m = sum(1 for a, b in zip(committed, out_ids[k:]) if a == b)
        if m > best_score:
            best_k, best_score = k, m
    k = best_k
    print(f"alignment: k={k} exact-match={best_score}/{min(len(committed), len(out_ids)-k)}")
    # committed[j] == out_ids[j + k]; out_ids[i] sits at abs position P + i.
    # => committed[j] abs position Q = P + j + k; reference prediction = pl[Q].

    submit = prompt_ids + out_ids[: k + n_tok]
    body = {"model": "qwen3.8-27b-gptq8", "prompt": submit,
            "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 20}
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY','')}"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    pl = out["choices"][0].get("prompt_logprobs") or []
    print(f"submitted {len(submit)} tokens, pl entries: {len(pl)}")

    rows = []
    n_comp = min(n_tok, len(committed), len(out_ids) - k)
    for jj in range(n_comp):
        Q = P + jj + k
        if Q >= len(pl) or Q < 1:
            continue
        entry = pl[Q]
        if not entry:
            continue
        items = sorted(entry.items(), key=lambda kv: kv[1]["logprob"], reverse=True)
        ref_top1 = math.exp(items[0][1]["logprob"])
        ref_top5 = [int(i_) for i_, _ in items[:5]]
        m = entry.get(str(committed[jj]))
        ref_p = math.exp(m["logprob"]) if m else 0.0
        p_in, t1_in = in_eng[jj]
        # guaranteed alignment: the submitted token at Q IS committed[jj]
        aligned = committed[jj] == out_ids[jj + k]
        rows.append((jj, Q, committed[jj], p_in, t1_in, ref_p, ref_top1,
                     ref_top5, aligned))
    # dump per-row data for later slicing
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "logs", "garble",
        f"replay2_rs{rs_q}g{gen_q}.jsonl")
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps({"j": r[0], "Q": r[1], "tok": r[2],
                                "p_in": r[3], "t1_in": r[4], "ref_p": r[5],
                                "ref_t1": r[6], "ref_top5": r[7],
                                "aligned": r[8]}) + "\n")
    print(f"rows dumped to {out_path}")

    # calibration: overall agreement on this stream
    d = [abs(r[3] - r[5]) for r in rows]
    print(f"|p_in - p_ref|: median={statistics.median(d):.4f} "
          f"p90={sorted(d)[int(0.9*len(d))]:.4f} max={max(d):.4f} (n={len(rows)})")

    # GUARANTEED-ALIGNED rows only, split by region around the first
    # decode-time mamba boundary (1728 * 24 = 41472)
    B = 1728 * 24
    for tag, sel in [("pre-boundary", [r for r in rows if r[8] and r[1] < B]),
                     ("post-boundary", [r for r in rows if r[8] and r[1] >= B])]:
        if not sel:
            print(f"{tag}: no aligned rows"); continue
        dd = [abs(r[3] - r[5]) for r in sel]
        flat = [r for r in sel if r[4] < 0.5]
        convict = [r for r in flat if r[6] > 0.7]
        t1_agree = [r for r in sel if abs(r[4] - r[6]) < 0.25]
        print(f"{tag}: aligned rows={len(sel)} "
              f"|dp| med={statistics.median(dd):.4f} "
              f"top1-agree(<0.25)={len(t1_agree)/len(sel):.2f} "
              f"flat_in={len(flat)} convict={len(convict)}")
        for jj, Q, t, p_in, t1_in, ref_p, ref_t1, ref_top5, _ in convict[:12]:
            t5 = " ".join(repr(tok.decode([x]))[:12] for x in ref_top5)
            print(f"    j={jj} pos={Q}: committed={tok.decode([t])!r:>18} "
                  f"p_in={p_in:.3f} top1_in={t1_in:.3f} ref_p={ref_p:.3f} "
                  f"ref_top1={ref_t1:.3f} ref_top5=[{t5}]")
    # and the reverse: in-engine sharp, reference flat (would indicate the
    # ring mapping is broken) — restricted to guaranteed-aligned rows
    rev = [r for r in rows if r[8] and r[4] > 0.9 and r[6] < 0.3]
    print(f"reverse-mismatch (aligned, top1_in>0.9, ref_top1<0.3): "
          f"{len(rev)}/{sum(1 for r in rows if r[8])}")


if __name__ == "__main__":
    main()
