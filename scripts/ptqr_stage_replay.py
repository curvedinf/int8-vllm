#!/usr/bin/env python3
"""Replay-bisect my dense DFlash2 draft forward against serving tensor dumps.

Feeds serving's dumped L0 layer_in into my layer-0 (LN + attention_conv.prepare)
and diffs against the dumped attn_conv_prep. Rows are the 14-token draft block.

KNOWN INSTRUMENT CAVEAT (2026-09-06): under VLLM_SPEC_DEBUG_TENSORS the dump
round's query_embed has NaN slots (rows 0-6 observed) — the synthetic dump-only
round is NOT the real decode input, and the NaN propagates via conv taps into
all later rows. Before this bisect can arbitrate, the dump must capture the
REAL decode round (or the dump round must fill unused slots with finite
values). The embed table itself is verified NaN-free (all 248320 rows).

Usage: .venv/bin/python scripts/ptqr_stage_replay.py [--dump-dir /tmp/spec_tensors]
"""
import argparse
import glob
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from ptqr_train_draft import DraftModel, load_draft  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

MODEL = "/home/curved/models/Qwen3.8-27B-PTQR-R10S60"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump-dir", default="/tmp/spec_tensors")
    p.add_argument("--layer", type=int, default=0)
    args = p.parse_args()

    embed = lm = None
    for fp in sorted(glob.glob(f"{MODEL}/model-*.safetensors")):
        sd = load_file(fp)
        if embed is None and "model.language_model.embed_tokens.weight" in sd:
            embed = sd["model.language_model.embed_tokens.weight"]
        if lm is None and "lm_head.weight" in sd:
            lm = sd["lm_head.weight"]
    draft = load_draft(DraftModel(embed, lm)).float()

    L = args.layer
    ins = {f.split("_")[-1][:-3]: f
           for f in glob.glob(f"{args.dump_dir}/L{L}_layer_in_*.pt")}
    preps = {f.split("_")[-1][:-3]: f
             for f in glob.glob(f"{args.dump_dir}/L{L}_attn_conv_prep_*.pt")}
    for mid in sorted(set(ins) & set(preps)):
        lay_in = torch.load(ins[mid])[None]
        prep_ref = torch.load(preps[mid])
        if torch.isnan(lay_in).any() or torch.isnan(prep_ref).any():
            nan_rows = torch.isnan(lay_in[0]).any(dim=1).nonzero().flatten().tolist()
            print(f"{mid}: SKIP — NaN in dump (layer_in rows {nan_rows[:8]})")
            continue
        lay = draft.layers[L]
        h = lay.input_layernorm(lay_in.permute(1, 0, 2))
        prep, _ = lay.attention_conv.prepare(h)
        mine, ref = prep[:, 0], prep_ref
        d = (mine - ref).abs()
        rel = d.mean() / ref.pow(2).mean().sqrt()
        print(f"{mid}: my rms {mine.pow(2).mean().sqrt():.3f} ref rms "
              f"{ref.pow(2).mean().sqrt():.3f} mean|d| {d.mean():.4f} rel {rel:.3f}")


if __name__ == "__main__":
    main()
