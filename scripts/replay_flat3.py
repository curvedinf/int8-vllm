#!/usr/bin/env python3
"""replay_flat3: verify-vs-prefill differential using the p_ring's OWN
committed tokens (no probe-file retokenization — works on corrupt streams).

Submits prompt + committed tokens to /v1/completions (prefill reference,
prompt_logprobs=20). For each committed token j at absolute position P+j,
reference prediction = pl[P+j][token]; in-engine = ring p/top1. The ring
misses the prefill-sampled first token, so committed[0] may sit at P or P+1;
both shifts are scored and the better one is used (self-calibrating on the
clean prefix).

Verdict: at the first degenerate position (in-engine top1>0.9 on a garbage
token or top1<0.3 flat), reference sharp-clean => verify path convicted;
reference agrees => model behavior given its own prefix.

Usage: replay_flat3.py <dump> <rs> <gen> <n_tokens>
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
    dump, rs_q, gen_q, n_tok = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
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
        if e["pos0"] < P - 40:
            continue
        rs = e["rs"]
        key = (rs, gen[rs])
        if streams[key] and e["pos0"] < streams[key][-1]["pos0"]:
            gen[rs] += 1
            key = (rs, gen[rs])
        streams[key].append(e)
    rounds = streams[(rs_q, gen_q)]
    committed = [t for e in rounds for t in e["tok"]][:n_tok]
    in_eng = []  # (p, top1) per committed token, in order
    for e in rounds:
        for p, t1 in zip(e["p"], e["top1"]):
            in_eng.append((p, t1))
    in_eng = in_eng[:n_tok]
    print(f"rounds {len(rounds)} committed {len(committed)}")

    submit = prompt_ids + committed
    body = {"model": "qwen3.8-27b-gptq8", "prompt": submit,
            "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 20}
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY','')}"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    pl = out["choices"][0].get("prompt_logprobs") or []
    print(f"pl entries: {len(pl)}")

    # self-calibrate the shift: try shift 0 and 1, score agreement on the
    # first 300 tokens (clean prefix region)
    def score(shift):
        agree = 0
        n = 0
        for j in range(min(300, len(committed))):
            Q = P + j + shift
            if Q >= len(pl) or not pl[Q]:
                continue
            entry = pl[Q]
            m = entry.get(str(committed[j]))
            if m is None:
                continue
            p_ref = math.exp(m["logprob"])
            p_in = in_eng[j][0]
            if abs(p_ref - p_in) < 0.25:
                agree += 1
            n += 1
        return agree / max(n, 1)
    s0, s1 = score(0), score(1)
    shift = 0 if s0 >= s1 else 1
    print(f"shift calibration: shift0={s0:.2f} shift1={s1:.2f} -> using shift={shift}")

    # full comparison
    rows = []
    for j in range(min(len(committed), len(in_eng))):
        Q = P + j + shift
        if Q >= len(pl) or not pl[Q]:
            continue
        entry = pl[Q]
        items = sorted(entry.items(), key=lambda kv: kv[1]["logprob"], reverse=True)
        ref_top1 = math.exp(items[0][1]["logprob"])
        ref_top5 = [int(i_) for i_, _ in items[:5]]
        m = entry.get(str(committed[j]))
        ref_p = math.exp(m["logprob"]) if m else 0.0
        p_in, t1_in = in_eng[j]
        rows.append((j, committed[j], p_in, t1_in, ref_p, ref_top1, ref_top5))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "logs", "garble",
                            f"replay3_rs{rs_q}g{gen_q}.jsonl")
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps({"j": r[0], "tok": r[1], "p_in": r[2],
                                "t1_in": r[3], "ref_p": r[4], "ref_t1": r[5],
                                "ref_top5": r[6]}) + "\n")
    print(f"rows dumped to {out_path}")
    d = [abs(r[2] - r[4]) for r in rows]
    print(f"|p_in - p_ref|: median={statistics.median(d):.4f} "
          f"p90={sorted(d)[int(0.9*len(d))]:.4f} max={max(d):.4f} (n={len(rows)})")
    # degenerate in-engine positions: top1_in>0.9 with garbage committed, or flat
    def is_garbage(t):
        s = tok.decode([t], errors="replace")
        return any(ord(c) > 0x2500 for c in s)
    degen = [r for r in rows if (r[3] > 0.9 and is_garbage(r[1])) or r[3] < 0.3]
    print(f"degenerate in-engine positions: {len(degen)}")
    convict = [r for r in degen if r[5] > 0.6 and not is_garbage(r[6][0])]
    print(f"CONVICT (in-engine degenerate, reference sharp+clean): {len(convict)}/{len(degen)}")
    for j, t, p_in, t1_in, ref_p, ref_t1, ref_top5 in convict[:20]:
        t5 = " ".join(repr(tok.decode([x]))[:12] for x in ref_top5)
        print(f"  j={j}: committed={tok.decode([t])!r:>16} p_in={p_in:.3f} top1_in={t1_in:.3f} "
              f"ref_p={ref_p:.3f} ref_top1={ref_t1:.3f} ref_top5=[{t5}]")
    agree_model = [r for r in degen if r[5] < 0.3 or (r[4] > 0.5)]
    print(f"reference ALSO degenerate/low (model behavior): {len(agree_model)}/{len(degen)}")
    print("\nALL degenerate-position detail:")
    for j, t, p_in, t1_in, ref_p, ref_t1, ref_top5 in degen:
        t5 = " ".join(repr(tok.decode([x]))[:12] for x in ref_top5)
        print(f"  j={j}: committed={tok.decode([t])!r:>16} p_in={p_in:.3f} top1_in={t1_in:.3f} "
              f"ref_p={ref_p:.3f} ref_top1={ref_t1:.3f} ref_top5=[{t5}]")
        ctx = tok.decode(committed[max(0, j-12):j+1])
        print(f"      context tail: {ctx[-120:]!r}")


if __name__ == "__main__":
    main()
