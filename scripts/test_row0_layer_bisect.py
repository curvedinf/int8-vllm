#!/usr/bin/env python3
"""Row-0 layer bisect: first divergent layer of a q_len=14 verify-window
emulation vs a q_len=1 decode at the same anchor position.

For each probe (a committed position j whose in-vivo round was convicted
failing, plus healthy controls), at anchor absolute position A (= ring
round pos0 - 1 in prompt+committed coordinates):

  Path A1 (decode, q_len=1): prefill seq[:A] in exact C-sized chunks (C | A),
    then a decode step forced to consume seq[A] (allowed_token_ids) -- a true
    q_len=1 decode forward over the full prefix.
  Path A2 (prefill tail, q_len=1): prefill seq[:A+1]; the final chunk is
    exactly 1 row = seq[A]. Also provides the reference top-5 at row A.
  Path B (verify window, q_len=14): prefill seq[:A] + 14 committed tokens;
    the final chunk is exactly 14 rows, row 0 = seq[A]. prompt_logprobs gives
    row-0's distribution via the pl entry at A+1.

Hidden states at row 0 are captured per decoder layer by TP-rank-0 forward
hooks (scripts/bisect_hooks/sitecustomize.py, env-gated). The per-layer
cos/rel-L2 between A1 and B (and A2 and B) localizes the first divergent
layer. Thresholds: rel-L2 > 5% or cos < 0.995.

Run:  .venv/bin/python scripts/test_row0_layer_bisect.py [--dry-run]
      [--limit-failing N] [--limit-healthy N] [--only j,j,...]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
AITER = os.path.join(os.path.dirname(REPO), "aiter")
HOOKS = os.path.join(HERE, "bisect_hooks")
CAPTURE_DIR = os.path.join(REPO, "logs", "garble", "row0_layer_bisect", "capture")
OUT_DIR = os.path.join(REPO, "logs", "garble", "row0_layer_bisect")
RING_DUMP = os.path.join(REPO, "logs", "garble", "p_ring_bf16", "p_ring_3464108.dump")
JSONL = os.path.join(REPO, "logs", "garble", "replay3_rs0g0.jsonl")
MODEL = "/home/curved/models/Qwen3.8-27B-GPTQ-8bit-gs128"
PROMPT_LEN = 40040  # ring arithmetic assumes this; asserted below

ENV = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_TARGET_DEVICE": "rocm",
    "ROCM_PATH": "/opt/rocm",
    "HIP_PATH": "/opt/rocm",
    "PYTORCH_ROCM_ARCH": "gfx908",
    "GPU_ARCHS": "gfx908",
    "HIP_VISIBLE_DEVICES": "0,1,2,3",
    "VLLM_ROCM_USE_AITER": "1",
    "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION": "1",
    "VLLM_ROCM_USE_AITER_TRITON_GEMM": "1",
    "VLLM_ROCM_USE_AITER_CUSTOM_AR": "0",
    "VLLM_GFX908_INT8_LM_HEAD": "1",
    "VLLM_GFX908_CK_W8A8": "1",
    "VLLM_GFX908_CK_FREE_GS128": "1",
    "VLLM_GFX908_INT8_EMBEDDING": "1",
    "VLLM_GFX908_ACT_QUANT": "round",
    "VLLM_DISABLED_KERNELS": "TritonW8A16LinearKernel",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "3600",
    "BISECT_CAPTURE_DIR": CAPTURE_DIR,
    "BISECT_CAPTURE_MAXN": "32",
    "NCCL_ALGO": "Ring",
    "NCCL_PROTO": "Simple",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def boot_env() -> None:
    """Ensure env (incl. sitecustomize injection) then re-exec once."""
    changed = False
    for k, v in ENV.items():
        if os.environ.get(k) != v:
            os.environ[k] = v
            changed = True
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld.startswith("/opt/rocm/lib"):
        os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/lib:" + ld
        changed = True
    pp = os.environ.get("PYTHONPATH", "")
    parts = [p for p in pp.split(os.pathsep) if p]
    want = [HOOKS, REPO, AITER]
    if parts[: len(want)] != want:
        rest = [p for p in parts if p not in want]
        os.environ["PYTHONPATH"] = os.pathsep.join(want + rest)
        changed = True
    if changed:
        sys.stderr.write("[bisect] re-exec with canonical env\n")
        os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:])


# ---------------------------------------------------------------------------
# stream reconstruction (verbatim scripts/replay_flat3.py recipe)
# ---------------------------------------------------------------------------


def build_prompt_ids():
    sys.path.insert(0, HERE)
    cache = os.path.join(OUT_DIR, "prompt_ids.json")
    if os.path.exists(cache):
        ids = json.load(open(cache))
        if len(ids) == PROMPT_LEN:
            return None, ids
    from garble_docs_probe import build_corpus
    from garble_repro2 import get_tok

    tok = get_tok()
    corpus = build_corpus(tok)
    msg = (
        "Summarize the documentation below as exhaustive release notes with "
        "headers and numbered lists, quoting key config names inline. "
        "Do not stop early.\n\n" + corpus
    )
    tmpl = tok.apply_chat_template(
        [{"role": "user", "content": msg}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tok.encode(tmpl, add_special_tokens=False)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(prompt_ids, f)
    return tok, prompt_ids


def load_stream():
    entries = []
    with open(RING_DUMP, "rb") as fh:
        while True:
            try:
                entries.append(pickle.load(fh))
            except EOFError:
                break
    streams = defaultdict(list)
    gen = defaultdict(int)
    for e in entries:
        if e["pos0"] < PROMPT_LEN - 40:
            continue
        rs = e["rs"]
        key = (rs, gen[rs])
        if streams[key] and e["pos0"] < streams[key][-1]["pos0"]:
            gen[rs] += 1
            key = (rs, gen[rs])
        streams[key].append(e)
    rounds = streams[(0, 0)]
    committed = [t for e in rounds for t in e["tok"]]
    return rounds, committed


def round_index(rounds):
    out, c = [], 0
    for e in rounds:
        out.append((c, e["pos0"], e["n"], e["p"][0], e["top1"][0]))
        c += len(e["tok"])
    return out


def divisors(x):
    ds, i = [], 1
    while i * i <= x:
        if x % i == 0:
            ds += [i, x // i]
        i += 1
    return sorted(set(ds))


def pick_C(A):
    """Chunk size C | A, big enough to be fast, small enough that the
    prompt_logprobs logits transient (C x vocab x 4B) fits free VRAM."""
    for lo, hi in ((1600, 4608), (1024, 4608), (512, 4608)):
        cands = [d for d in divisors(A) if lo <= d <= hi]
        if cands:
            return max(cands)
    return None


def select_probes(rounds, rows_by_j, j_max=700):
    ridx = round_index(rounds)
    j0map = {j0: (r, pos0, n, p0, t1) for r, (j0, pos0, n, p0, t1) in enumerate(ridx)}
    failing, healthy = [], []
    for j in sorted(rows_by_j):
        if j > j_max or j not in j0map:
            continue
        row = rows_by_j[j]
        r, pos0, n, p0, t1 = j0map[j]
        A = pos0 - 1
        C = pick_C(A)
        if C is None:
            continue
        entry = dict(j=j, round=r, n=n, pos0=pos0, A=A, C=C, p0=p0, t1_0=t1)
        entry.update(row)
        if abs(row["p_in"] - row["ref_p"]) > 0.5:
            failing.append(entry)
        elif abs(row["p_in"] - row["ref_p"]) < 0.05 and row["t1_in"] > 0.9:
            healthy.append(entry)
    return failing, healthy


# ---------------------------------------------------------------------------
# capture plumbing
# ---------------------------------------------------------------------------


def read_captures(torch, last_fwd):
    files = glob.glob(os.path.join(CAPTURE_DIR, "fwd_*.pt"))
    out = []
    for f in files:
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        finally:
            os.unlink(f)
        last_fwd = max(last_fwd, d["fwd"])
        out.append(d)
    out.sort(key=lambda d: d["fwd"])
    return out, last_fwd


def pick_target(caps, n_rows, pos0, first_id):
    return [
        c
        for c in caps
        if c["n"] == n_rows
        and c["pos"]
        and c["pos"][0] == pos0
        and c["ids"][0] == first_id
    ]


def row0(cap, name):
    t = None
    if name == "embed":
        t = cap.get("embed")
    elif name == "norm":
        t = cap.get("norm")
    else:
        t = cap.get("layers", {}).get(int(name))
    if t is None:
        return None
    return t[0].float()


def compare(torch, capA, capB, name):
    a, b = row0(capA, name), row0(capB, name)
    if a is None or b is None:
        return None
    na, nb = a.norm(), b.norm()
    cos = (torch.dot(a, b) / (na * nb)).item() if na > 0 and nb > 0 else 0.0
    rel = ((a - b).norm() / (na if na > 0 else 1)).item()
    return cos, rel


def top5_from_dict(d):
    items = sorted(d.items(), key=lambda kv: kv[1].logprob, reverse=True)[:5]
    return [(int(t), math.exp(lp.logprob)) for t, lp in items]


def pl_entry(pl, idx, tok):
    e = pl[idx] if idx < len(pl) else None
    if not e:
        return None, 0.0
    m = e.get(tok) if isinstance(e, dict) else None
    if m is None and isinstance(e, dict):
        m = e.get(str(tok))
    p = math.exp(m.logprob) if m is not None else 0.0
    return e, p


def layer_names(meta):
    return ["embed"] + [str(i) for i in range(meta["num_layers"])] + ["norm"]


def layer_type(meta, name):
    if not name.isdigit():
        return name
    return meta["layer_types"].get(name, meta["layer_types"].get(int(name), "?"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit-failing", type=int, default=8)
    ap.add_argument("--limit-healthy", type=int, default=8)
    ap.add_argument("--only", default=None, help="comma-separated j list")
    args = ap.parse_args()

    boot_env()
    t_start = time.time()

    tok, prompt_ids = build_prompt_ids()
    P = len(prompt_ids)
    print(f"[bisect] prompt ids: {P}", flush=True)
    assert P == PROMPT_LEN, f"prompt length {P} != {PROMPT_LEN}; ring arithmetic broken"

    rounds, committed = load_stream()
    ridx = round_index(rounds)
    print(f"[bisect] rounds {len(rounds)} committed {len(committed)}", flush=True)
    bad = [x for x in ridx if x[0] >= 1 and x[1] != P + 1 + (x[0] - 1)]
    assert not bad, f"ring pos0 misalignment: {bad[:3]}"

    rows = [json.loads(l) for l in open(JSONL)]
    rows_by_j = {r["j"]: r for r in rows}
    failing, healthy_pool = select_probes(rounds, rows_by_j)

    healthy = []
    for e in healthy_pool:
        if not healthy or e["j"] - healthy[-1]["j"] >= 40:
            healthy.append(e)
    probes = failing[: args.limit_failing] + healthy[: args.limit_healthy]
    if args.only:
        want = {int(x) for x in args.only.split(",")}
        probes = [p for p in failing + healthy if p["j"] in want]

    print(f"[bisect] {len(failing)} failing / {len(healthy)} healthy probes", flush=True)
    for p in probes:
        print(
            f"  j={p['j']:<4} {'FAIL' if abs(p['p_in']-p['ref_p'])>0.5 else 'ok  '} "
            f"round={p['round']:<4} n={p['n']:<3} A={p['A']} C={p['C']} "
            f"p_in={p['p_in']:.3f} ref_p={p['ref_p']:.3f} p0={p['p0']:.3f}",
            flush=True,
        )
    if args.dry_run:
        return

    os.makedirs(CAPTURE_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(CAPTURE_DIR, "fwd_*.pt")) + glob.glob(
        os.path.join(CAPTURE_DIR, "meta.json")
    ):
        os.unlink(f)

    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    t0 = time.time()
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=4,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.80,
        kv_cache_memory_bytes=2_000_000_000,
        max_model_len=65536,
        max_num_seqs=4,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        kv_cache_dtype="int8_per_token_head",
        mamba_ssm_cache_dtype="float32",
        language_model_only=True,
        skip_mm_profiling=True,
        disable_log_stats=True,
    )
    print(f"[bisect] engine loaded in {time.time()-t0:.0f}s", flush=True)

    meta_path = os.path.join(CAPTURE_DIR, "meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit("worker hooks did not install (no meta.json); aborting")
    meta = json.load(open(meta_path))
    print(f"[bisect] hooks installed: {meta['num_layers']} layers, "
          f"types e.g. {sorted(set(meta['layer_types'].values()))}", flush=True)

    # locate in-process scheduler for chunk-size control
    eng = llm.llm_engine
    core = getattr(eng, "engine_core", None)
    core = getattr(core, "engine_core", core)
    sched = getattr(core, "scheduler", None)
    assert sched is not None, "cannot reach scheduler; is V1 multiproc disabled?"
    sched_cfg = sched.scheduler_config

    def set_C(C):
        sched_cfg.max_num_batched_tokens = C
        sched_cfg.max_num_scheduled_tokens = C
        sched.max_num_scheduled_tokens = C

    caps, last_fwd = read_captures(torch, -1)
    print(f"[bisect] warmup forwards discarded: {[c['n'] for c in caps]}", flush=True)

    results = []
    seq = prompt_ids + committed
    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, f"results_{int(time.time())}.json")

    for pi, p in enumerate(probes):
        j, A, C = p["j"], p["A"], p["C"]
        anchor = int(seq[A])
        next_tok = int(seq[A + 1])
        window = [int(x) for x in seq[A : A + 14]]
        set_C(C)
        t_p = time.time()

        # --- Path A1: prefill seq[:A], decode forced to consume the anchor
        out1 = llm.generate(
            [TokensPrompt(prompt_token_ids=list(seq[:A]))],
            SamplingParams(
                max_tokens=2, temperature=0.0, logprobs=5, allowed_token_ids=[anchor]
            ),
            use_tqdm=False,
        )[0]
        caps1, last_fwd = read_captures(torch, last_fwd)
        hit = pick_target(caps1, 1, A, anchor)
        assert len(hit) == 1, (
            f"A1 j={j}: expected 1 decode fwd at pos {A} id {anchor}, got {len(hit)}; "
            f"ns={[c['n'] for c in caps1]}"
        )
        hA1 = hit[0]

        # --- Path A2: prefill seq[:A+1], tail chunk of exactly 1 row
        out2 = llm.generate(
            [TokensPrompt(prompt_token_ids=list(seq[: A + 1]))],
            SamplingParams(max_tokens=1, temperature=0.0, logprobs=5),
            use_tqdm=False,
        )[0]
        caps2, last_fwd = read_captures(torch, last_fwd)
        hit = pick_target(caps2, 1, A, anchor)
        assert len(hit) == 1, (
            f"A2 j={j}: expected 1 tail fwd at pos {A} id {anchor}, got {len(hit)}; "
            f"ns={[c['n'] for c in caps2]}"
        )
        hA2 = hit[0]

        # --- Path B: prefill seq[:A] + 14-row window, tail chunk of exactly 14
        outB = llm.generate(
            [TokensPrompt(prompt_token_ids=list(seq[:A]) + window)],
            SamplingParams(max_tokens=1, temperature=0.0, logprobs=5, prompt_logprobs=5),
            use_tqdm=False,
        )[0]
        capsB, last_fwd = read_captures(torch, last_fwd)
        hit = pick_target(capsB, 14, A, anchor)
        assert len(hit) == 1, (
            f"B j={j}: expected 1 window fwd at pos {A} id {anchor}, got {len(hit)}; "
            f"ns={[c['n'] for c in capsB]}"
        )
        hB = hit[0]

        # --- per-layer comparison at row 0
        per_layer, first_div_A1, first_div_A2 = {}, None, None
        for name in layer_names(meta):
            c1 = compare(torch, hA1, hB, name)
            c2 = compare(torch, hA2, hB, name)
            per_layer[name] = {
                "a1_cos": None if c1 is None else round(c1[0], 6),
                "a1_rel": None if c1 is None else round(c1[1], 6),
                "a2_cos": None if c2 is None else round(c2[0], 6),
                "a2_rel": None if c2 is None else round(c2[1], 6),
            }
            if c1 and (c1[0] < 0.995 or c1[1] > 0.05) and first_div_A1 is None:
                first_div_A1 = name
            if c2 and (c2[0] < 0.995 or c2[1] > 0.05) and first_div_A2 is None:
                first_div_A2 = name

        # --- logits at row A (A2 sampled top-5 vs B prompt_logprobs[A+1])
        a2_top5 = top5_from_dict(out2.outputs[0].logprobs[0])
        eB, b_actual = pl_entry(outB.prompt_logprobs, A + 1, next_tok)
        b_top5 = top5_from_dict(eB) if eB else []
        a2_actual = next((lp for t, lp in a2_top5 if t == next_tok), None)

        res = dict(
            j=j,
            kind="failing" if abs(p["p_in"] - p["ref_p"]) > 0.5 else "healthy",
            round=p["round"],
            in_vivo_n=p["n"],
            A=A,
            C=C,
            anchor=anchor,
            next_tok=next_tok,
            in_vivo_p0=p["p0"],
            in_vivo_t1=p["t1_0"],
            ref_p=p["ref_p"],
            ref_t1=p["ref_t1"],
            first_div_A1=first_div_A1,
            first_div_A2=first_div_A2,
            max_rel_A1=max(v["a1_rel"] for v in per_layer.values() if v["a1_rel"] is not None),
            max_rel_A2=max(v["a2_rel"] for v in per_layer.values() if v["a2_rel"] is not None),
            a2_top5=a2_top5,
            b_top5=b_top5,
            a2_actual_p=a2_actual,
            b_actual_p=b_actual,
            per_layer=per_layer,
            t_probe_s=round(time.time() - t_p, 1),
        )
        results.append(res)

        b_t1 = f"{b_top5[0][0]}:{b_top5[0][1]:.3f}" if b_top5 else "-:-"
        fd1 = f"{first_div_A1}({layer_type(meta, first_div_A1)})" if first_div_A1 else "-"
        print(
            f"[probe {pi+1}/{len(probes)}] j={j} {res['kind']:7s} A={A} C={C} "
            f"firstDiv(A1)={fd1} maxRelA1={res['max_rel_A1']:.4f} "
            f"top1 A2={a2_top5[0][0]}:{a2_top5[0][1]:.3f} B={b_t1} "
            f"({res['t_probe_s']}s)",
            flush=True,
        )
        with open(out_json, "w") as f:
            json.dump(results, f, indent=1)

    # ---------------- summary ----------------
    print("\n================ SUMMARY ================", flush=True)
    hdr = f"{'j':<5}{'kind':<9}{'firstDiv(A1)':<24}{'maxRel(A1)':<12}{'firstDiv(A2)':<24}{'maxRel(A2)':<12}logits@rowA"
    print(hdr, flush=True)
    for r in results:
        agree = bool(r["b_top5"]) and r["a2_top5"][0][0] == r["b_top5"][0][0]
        pb = r["b_top5"][0][1] if r["b_top5"] else 0.0
        print(
            f"{r['j']:<5}{r['kind']:<9}{r['first_div_A1'] or '-':<24}"
            f"{r['max_rel_A1']:<12.4f}{r['first_div_A2'] or '-':<24}"
            f"{r['max_rel_A2']:<12.4f}"
            f"{'top1match' if agree else 'top1DIFF'} pA2={r['a2_top5'][0][1]:.3f} pB={pb:.3f}",
            flush=True,
        )

    fail = [r for r in results if r["kind"] == "failing"]
    ok = [r for r in results if r["kind"] == "healthy"]

    # healthy-calibrated per-layer floor: the q_len=14-vs-1 kernel paths differ
    # numerically everywhere (smooth compounding), so anomaly = failing probe
    # far above the healthy band at the same layer.
    if ok:
        print("\n--- healthy-calibrated view (A1 vs B, rel-L2 per layer) ---", flush=True)
        names = layer_names(meta)
        floor = {}
        for name in names:
            vals = [r["per_layer"][name]["a1_rel"] for r in ok
                    if r["per_layer"][name]["a1_rel"] is not None]
            if vals:
                floor[name] = (sorted(vals)[len(vals) // 2], max(vals))
        print(f"{'layer':<7}{'healthy_med':>12}{'healthy_max':>12}", flush=True)
        for name in ["embed"] + [str(i) for i in range(0, 64, 4)] + ["norm"]:
            if name in floor:
                print(f"{name:<7}{floor[name][0]:>12.4f}{floor[name][1]:>12.4f}", flush=True)
        for r in fail:
            anom = None
            for name in names:
                v = r["per_layer"][name]["a1_rel"]
                if v is not None and name in floor and v > max(floor[name][1] * 1.5, floor[name][1] + 0.03):
                    anom = name
                    break
            pb = r["b_top5"][0][1] if r["b_top5"] else 0.0
            pa = r["a2_top5"][0][1]
            print(
                f"FAIL j={r['j']:<4} firstAnomalousLayer(vs healthy band)={anom} "
                f"maxRel={r['max_rel_A1']:.4f} top1p: A2={pa:.3f} B={pb:.3f} "
                f"{'FLAT/corrupt row0' if pb < 0.3 else 'sharp row0'}",
                flush=True,
            )

    if ok:
        print(
            f"\nhealthy noise floor: max rel-L2 (A1 vs B) across layers = "
            f"{max(r['max_rel_A1'] for r in ok):.5f}",
            flush=True,
        )
    if fail:
        print(
            f"failing probes: max rel-L2 (A1 vs B) = "
            f"{max(r['max_rel_A1'] for r in fail):.5f}",
            flush=True,
        )
    print(f"\nresults -> {out_json}", flush=True)
    print(f"[bisect] total wall {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
