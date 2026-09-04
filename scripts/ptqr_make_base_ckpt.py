#!/usr/bin/env python3
"""Convert the multimodal BF16 reference safetensors to a stripped LM-only .pt.

The PTQR trainer (and any torch.load(mmap=True) consumer) needs a single
file-backed checkpoint of the language model + lm_head: the published reference
is the full multimodal model (model.language_model.*, model.visual.*, mtp.*).
The mmap'd .pt lets 4 TP ranks share page-cache pages instead of 4 anon
copies (61 GB host RAM cannot hold 4 eager loads — measured OOM-kill).

Output form: {"model_state_dict": {"layers.N...": ..., "embed_tokens...": ...,
"lm_head.weight": ...}} — the stripped Qwen3_5TextModel+head form.

Run (any venv with torch+safetensors):
  /home/curved/.venvs/sdgraft-train/bin/python scripts/ptqr_make_base_ckpt.py \
      --src /home/curved/models/Qwen3.8-27B-bf16-ref \
      --dst /home/curved/models/Qwen3.8-27B-bf16-ref-lm.pt
"""
import argparse
import gc
from pathlib import Path

import torch
from safetensors.torch import load_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/home/curved/models/Qwen3.8-27B-bf16-ref")
    p.add_argument("--dst",
                   default="/home/curved/models/Qwen3.8-27B-bf16-ref-lm.pt")
    args = p.parse_args()

    out: dict[str, torch.Tensor] = {}
    n_vis = 0
    for f in sorted(Path(args.src).glob("*.safetensors")):
        sd = load_file(str(f))
        for k, v in sd.items():
            if k.startswith("model.language_model."):
                out["model." + k[len("model.language_model."):]] = v
            elif k == "lm_head.weight":
                out[k] = v
            else:
                n_vis += 1
        del sd
        gc.collect()
        print(f"  {f.name}: cumulative {len(out)} LM tensors "
              f"({n_vis} skipped visual/mtp)", flush=True)

    assert "model.embed_tokens.weight" in out, "embedding missing"
    assert "lm_head.weight" in out, "lm_head missing"
    total = sum(v.numel() * v.element_size() for v in out.values())
    print(f"writing {len(out)} tensors, {total / 2**30:.1f} GiB -> {args.dst}",
          flush=True)
    torch.save({"model_state_dict": out}, args.dst)
    print("done", flush=True)


if __name__ == "__main__":
    main()
