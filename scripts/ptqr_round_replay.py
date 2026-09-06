#!/usr/bin/env python3
"""Replay a captured serving decode round through my full dense draft.

Feeds the serving query_embed (real draft input embeds) + ctx_states +
absolute positions through all 5 layers + aux head; prints top-1 tokens per
draft slot. STATUS: captured ctx rounds are DUMMY precompute calls (C=14,
cpos all 0) — the real ctx path runs inside a custom op (see
qwen3_dflash.py ~line 615: "_project_context_kv only runs without the custom
op"), so real-round ctx must come from reading the speculator feeding
convention or dumping inside the custom op wrapper.
"""
import glob
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import ptqr_train_draft as D  # noqa: E402
from ptqr_train_draft import DraftModel, load_draft, rope_cos_sin  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

MODEL = "/home/curved/models/Qwen3.8-27B-PTQR-R10S60"


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "/home/curved/models/Qwen3.8-27B-GPTQ-8bit-gs128")
    embed = lm = None
    for fp in sorted(glob.glob(f"{MODEL}/model-*.safetensors")):
        sd = load_file(fp)
        if embed is None and "model.language_model.embed_tokens.weight" in sd:
            embed = sd["model.language_model.embed_tokens.weight"]
        if lm is None and "lm_head.weight" in sd:
            lm = sd["lm_head.weight"]
    m = load_draft(DraftModel(embed, lm)).float().eval()
    for r in sys.argv[1:] or ["R10"]:
        qe = torch.load(f"/tmp/spec_tensors/{r}_query_embed.pt")
        qpos = torch.load(f"/tmp/spec_tensors/{r}_query_pos.pt")
        cpos = torch.load(f"/tmp/spec_tensors/{r}_ctx_pos.pt")
        ctx = torch.load(f"/tmp/spec_tensors/{r}_ctx_states.pt")
        if qe.isnan().any() or cpos.max() == 0:
            print(f"{r}: dummy ctx (C={ctx.shape[0]}, cpos max {int(cpos.max())}) — skip")
            continue
        with torch.no_grad():
            cn = m.hidden_norm(ctx)
            cos, sin = rope_cos_sin(
                int(max(cpos.max(), qpos.max())) + 1,
                D.CFG["hd"], D.CFG["rope"], qe.device, torch.float32)
            h, res, exits = qe[:, None].float(), None, []
            for layer in m.layers:
                h, res = layer(h, res, cn[:, None], cos, sin, D.CFG["window"],
                               qpos=qpos, cpos=cpos)
                exits.append(m.norm(h + res)[:, 0])
            aux = m.fc(torch.cat(exits, dim=-1))
            logits = F.linear(m.hidden_norm(aux), m.lm_head_weight)
        pred = logits.argmax(-1)
        print(f"{r}: " + repr("".join(tok.decode([t]) for t in pred[:8].tolist())))


if __name__ == "__main__":
    main()
