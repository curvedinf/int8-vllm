#!/usr/bin/env python3
"""Stage-2 replay: my dense draft attention vs serving R{id} captures.

Stage-1 (LN+conv prepare) is EXACT (corr 0.9999). Stage-2 (attention) shows
corr ~0.94 with ~12% rms gap. This script brackets the remaining semantic
difference: ctx visibility (causal vs full), rope theta, window.

Usage: .venv/bin/python scripts/ptqr_stage2_replay.py [round_id]
"""
import glob
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from ptqr_train_draft import (  # noqa: E402
    CFG, DraftModel, apply_rope, load_draft, rope_cos_sin)
from safetensors.torch import load_file  # noqa: E402

MODEL = "/home/curved/models/Qwen3.8-27B-PTQR-R10S60"


def load_weights():
    embed = lm = None
    for fp in sorted(glob.glob(f"{MODEL}/model-*.safetensors")):
        sd = load_file(fp)
        if embed is None and "model.language_model.embed_tokens.weight" in sd:
            embed = sd["model.language_model.embed_tokens.weight"]
        if lm is None and "lm_head.weight" in sd:
            lm = sd["lm_head.weight"]
    return load_draft(DraftModel(embed, lm)).float()


def run(draft, R, ctx_causal=True, window=True, theta=None):
    att = draft.layers[0].self_attn
    prep = torch.load(f"{R}L0_attn_conv_prep.pt")
    ctx_raw = torch.load(f"{R}ctx_states.pt")
    cpos = torch.load(f"{R}ctx_pos.pt"); qpos = torch.load(f"{R}query_pos.pt")
    ref = torch.load(f"{R}L0_attn_out.pt")
    ctx = draft.hidden_norm(ctx_raw)
    th = theta or CFG["rope"]
    maxp = int(max(cpos.max(), qpos.max())) + 1
    cos, sin = rope_cos_sin(maxp, CFG["hd"], th, prep.device, torch.float32)
    C, T = ctx.shape[0], prep.shape[0]
    q = att._heads(att.q_proj(prep[:, None]), att.heads)
    kq = att._heads(att.k_proj(prep[:, None]), att.kvh)
    vq = att._heads(att.v_proj(prep[:, None]), att.kvh)
    q, kq = att.q_norm(q), att.k_norm(kq)
    q = apply_rope(q, cos[qpos], sin[qpos])
    kq = apply_rope(kq, cos[qpos], sin[qpos])
    kc = att.k_norm(att._heads(att.k_proj(ctx[:, None]), att.kvh))
    vc = att._heads(att.v_proj(ctx[:, None]), att.kvh)
    kc = apply_rope(kc, cos[cpos], sin[cpos])
    k = torch.cat([kc, kq], dim=2); v = torch.cat([vc, vq], dim=2)
    qi = qpos[:, None]; ki = torch.cat([cpos, qpos])[None, :]
    if ctx_causal:
        allowed = ki <= qi
    else:
        # ctx fully visible; causal only among query tokens
        allowed = torch.cat([torch.ones(T, C, dtype=torch.bool),
                             ki[:, C:] <= qi], dim=1)
    if window and CFG["window"]:
        allowed = allowed & ((qi - ki) < CFG["window"])
    a = F.scaled_dot_product_attention(
        q, k, v, attn_mask=allowed[None, None].float(),
        scale=1.0 / math.sqrt(CFG["hd"]), enable_gqa=True)
    mine = att.o_proj(a.permute(2, 0, 1, 3).reshape(T, 1, -1))[:, 0]
    c = torch.corrcoef(torch.stack([mine.flatten(), ref.flatten()]))[0, 1]
    return (f"my rms {mine.pow(2).mean().sqrt():.3f} ref rms "
            f"{ref.pow(2).mean().sqrt():.3f} corr {c:.4f}")


def main():
    rid = sys.argv[1] if len(sys.argv) > 1 else None
    if rid is None:
        rs = sorted(glob.glob("/tmp/spec_tensors/R*_ctx_states.pt"))
        rid = rs[-1].split("/")[-1].split("_")[0] if rs else "R7"
    R = f"/tmp/spec_tensors/{rid}_"
    draft = load_weights()
    for lbl, kw in (("causal+win", {}),
                    ("FULLCTX+win", {"ctx_causal": False}),
                    ("causal+nowin", {"window": False}),
                    ("FULLCTX+nowin", {"ctx_causal": False, "window": False}),
                    ("theta1e6", {"theta": 1e6})):
        print(f"{lbl:14s}", run(draft, R, **kw))


if __name__ == "__main__":
    main()
