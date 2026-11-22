#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Mine ksdraft training signals at full serving depth (up to 1,048,576).

Boots the deployed W4A8/int8-KV target offline (TP4, eager) with the
KSDraftMinerMixin injected into the workers, prefills each corpus pack as ONE
request (the engine's own chunk-carry computes every position's KV with the
true prefix), and assembles a ksdraft store-v2 directory:

    <out>/k_codes.bin  int8   (rows, taps, 1024)
    <out>/k_scales.bin fp16   (rows, taps, 8)
    <out>/v_codes.bin, v_scales.bin
    <out>/topk_ids.pt  int32  (rows, 32)   raw-logit top-32, global merge
    <out>/topk_probs.pt fp16  (rows, 32)   exp(logit - logsumexp(full vocab))
    <out>/boundaries.pt, meta.json

Usage:
  PYTHONPATH=$PWD:$PWD/../int8-aiter .venv/bin/python \
    scripts/mine_ksdraft_deep.py --corpus ../ksdraft/runs/inkling_corpus/million \
    --out ../ksdraft/runs/inkling_signals_million --packs 5 --mnbt 8192
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

TAPS = ((4, 0), (10, 2), (16, 2), (22, 0))
TOPK = 32
PACK = 1_048_576


def _env_defaults():
    os.environ.setdefault("ROCM_PATH", "/opt/rocm")
    os.environ.setdefault("HIP_PATH", "/opt/rocm")
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx908")
    os.environ.setdefault("GPU_ARCHS", "gfx908")
    os.environ.setdefault("VLLM_TARGET_DEVICE", "rocm")
    os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
    os.environ.setdefault("VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION", "1")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if "/opt/rocm/lib" not in ld:
        os.environ["LD_LIBRARY_PATH"] = f"/opt/rocm/lib:{ld}"


def build_llm(args):
    from vllm import LLM

    return LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="float16",
        kv_cache_dtype="int8_block_g128",
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.mnbt,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=True,
        worker_extension_cls=(
            "vllm.v1.worker.gpu.spec_decode.ksdraft.miner.KSDraftMinerMixin"
        ),
        disable_log_stats=True,
    )


def join_pack(pack_dir: Path, out_root: Path, rows: int, world: int,
              heads_total: int, meta_src: dict) -> None:
    """Assemble one pack's rank shards into the store layout."""
    out_root.mkdir(parents=True, exist_ok=True)

    def openw(name):
        return open(out_root / name, "wb")

    f_kc, f_ks = openw("k_codes.bin"), openw("k_scales.bin")
    f_vc, f_vs = openw("v_codes.bin"), openw("v_scales.bin")
    for tap in TAPS:
        # rank files: tap{li}_{pi}_rank{r}_{kc,ks,vc,vs}.bin
        shard = {}
        for r in range(world):
            base = pack_dir / f"tap{tap[0]}_{tap[1]}_rank{r}"
            shard[r] = {
                s: np.fromfile(str(base) + f"_{s}.bin",
                               dtype=np.int8 if s in ("kc", "vc") else np.float16)
                for s in ("kc", "ks", "vc", "vs")
            }
        # rows per rank: heads-per-rank x 128 codes / (heads, 1) scales
        heads = heads_total // world
        n = shard[0]["kc"].size // (heads * 128)
        assert n == rows, (n, rows)
        kc = np.concatenate(
            [shard[r]["kc"].reshape(n, heads, 128) for r in range(world)],
            axis=1).reshape(n, heads_total * 128)
        vc = np.concatenate(
            [shard[r]["vc"].reshape(n, heads, 128) for r in range(world)],
            axis=1).reshape(n, heads_total * 128)
        ks = np.concatenate(
            [shard[r]["ks"].reshape(n, heads) for r in range(world)],
            axis=1)                                     # (n, heads_total)
        vs = np.concatenate(
            [shard[r]["vs"].reshape(n, heads) for r in range(world)],
            axis=1)
        f_kc.write(kc.tobytes())
        f_vc.write(vc.tobytes())
        f_ks.write(ks.astype(np.float16).tobytes())
        f_vs.write(vs.astype(np.float16).tobytes())
    for f in (f_kc, f_ks, f_vc, f_vs):
        f.close()

    # ---- global top-32: merge per-rank candidates, exact probs via LSE sum
    ids_r, val_r, lse = [], [], None
    for r in range(world):
        base = pack_dir / f"topk_rank{r}"
        ids_r.append(np.fromfile(str(base) + "_ids.bin", dtype=np.int32)
                     .reshape(-1, TOPK))
        val_r.append(np.fromfile(str(base) + "_vals.bin", dtype=np.float32)
                     .reshape(-1, TOPK))
        lse_r = np.fromfile(str(base) + "_lse.bin", dtype=np.float32)
        # full-vocab logsumexp = log(sum_r exp(lse_r))
        lse = lse_r if lse is None else np.logaddexp(lse, lse_r)
    assert all(len(x) == rows for x in ids_r), [len(x) for x in ids_r]
    ids = np.concatenate(ids_r, axis=1)                  # (rows, world*32)
    vals = np.concatenate(val_r, axis=1)
    order = np.argsort(-vals, axis=1)[:, :TOPK]
    top_ids = np.take_along_axis(ids, order, axis=1).astype(np.int32)
    top_lg = np.take_along_axis(vals, order, axis=1)
    probs = np.exp(top_lg - lse[:, None]).astype(np.float16)
    torch.save(torch.from_numpy(top_ids.astype(np.int64)),
               out_root / "topk_ids.pt")
    torch.save(torch.from_numpy(probs), out_root / "topk_probs.pt")
    torch.save(torch.tensor([0], dtype=torch.long),
               out_root / "boundaries.pt")
    (out_root / "meta.json").write_text(json.dumps({
        "source": meta_src.get("source"), "corpus": meta_src.get("corpus"),
        "tokens": int(rows), "taps": [t[0] for t in TAPS], "topk": TOPK,
        "format": "ksdraft-kv-int8-v2",
        "kv_grid": "int8_block_g128 codes+scales (fork kernel semantics)",
        "codes_dtype": "int8", "scales_dtype": "fp16", "group": 128,
        "depth": "engine prefill at full pack length",
    }, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser(
        "~/models/Inkling-2-Mini-PTQR-W4A8-G128"))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--packs", type=int, default=5)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--mnbt", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=1_048_576)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--pack-len", type=int, default=PACK)
    args = ap.parse_args()
    _env_defaults()

    corpus = Path(args.corpus)
    tokens = torch.load(corpus / "tokens.pt", weights_only=True).long()
    meta_src = json.loads((corpus / "meta.json").read_text())

    llm = build_llm(args)
    from vllm import SamplingParams

    out = Path(args.out)
    packs_root = out / "_packs"
    packs_root.mkdir(parents=True, exist_ok=True)
    total = min(tokens.numel(), args.packs * args.pack_len)
    for i in range(args.packs):
        lo = i * args.pack_len
        hi = min(lo + args.pack_len, total)
        if hi - lo < 1024:
            break
        rows = hi - lo
        pack_dir = packs_root / f"pack{i}"
        pack_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        info = llm.collective_rpc("ksdraft_miner_setup",
                                  args=(str(pack_dir), list(TAPS)))
        sp = SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0,
                            top_k=-1, prompt_logprobs=1,
                            ignore_eos=True, detokenize=False)
        llm.generate([{"prompt_token_ids": tokens[lo:hi].tolist()}],
                     sp, use_tqdm=False)
        llm.collective_rpc("ksdraft_miner_finish")
        print(f"pack {i}: mined {rows:,} rows in {time.time() - t0:.0f}s "
              f"({info})", flush=True)
        join_pack(pack_dir, out / f"pack{i}", rows, args.tp, 8,
                  {"source": meta_src.get("source"), "corpus": str(corpus)})
        print(f"pack {i}: store written to {out / f'pack{i}'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
