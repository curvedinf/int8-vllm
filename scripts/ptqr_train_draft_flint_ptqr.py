#!/usr/bin/env python3
"""Flint-DFlash2 draft PTQR training (port of ptqr_train_draft_ptqr.py).

Same program: student = dense draft with deployed quantization in the loop
(G128 PTQR linears, int8_block KV g128, per-token round act quant),
teacher = frozen bf16 copy, CE + candidate-KLD on cached target exit
states. Differences from the Qwen draft: hidden 2048 / inter 6144 /
vocab 150144 / rope 1e4 / mask 150019, exit layers [5,15,26,36,46]
(cache from scripts/ptqr_flint_draft_cache.py), embed/lm from the Flint
bf16 target checkpoint.

Every nn.Linear trains PTQR (fc included); the EXPORT decides which
surfaces deploy quantized — per the deployed-surface doctrine the ctx
projection (fc) exports its bf16 master (acceptance ladder bf16 71.2 >
W8A8 69.2), codebooks/base_kernel are parameters and never quantized.

Usage:
  PYTHONPATH=scripts:. .venv/bin/python scripts/ptqr_train_draft_flint_ptqr.py \
      --steps 60 --lr 1e-5 --out /home/curved/models/flint_ptqr_draft_r1
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import ptqr_train_draft as D  # noqa: E402
from ptqr_train_draft import CFG, DraftModel, load_draft  # noqa: E402
from ptqr_train_target import (  # noqa: E402
    PTQRLinear, act_quant_fake, kv_block_quant_fake,
    attach_weight_sgd_hooks, anneal_tau)

FLINT_DRAFT = "/home/curved/models/flint-dflash2-bf16"
FLINT_TARGET = "/home/curved/models/flint-bf16"

_dfl = json.load(open(f"{FLINT_DRAFT}/config.json"))["dflash_config"]
_tcfg = json.load(open(f"{FLINT_DRAFT}/config.json"))
# mutate the module-level CFG BEFORE any DraftModel construction (all CFG
# reads are runtime except RMSNorm's eps default — same value 1e-6 both)
CFG.update(hidden=_tcfg["hidden_size"], inter=_tcfg["intermediate_size"],
           heads=_tcfg["num_attention_heads"],
           kv_heads=_tcfg["num_key_value_heads"], hd=_tcfg["head_dim"],
           layers=_tcfg["num_hidden_layers"],
           window=_tcfg["sliding_window"], eps=_tcfg["rms_norm_eps"],
           vocab=_tcfg["vocab_size"], taps=_dfl["conv_kernel_size"],
           group=_dfl["conv_group_size"], ns=13,
           mask_token=_dfl["mask_token_id"],
           rope=_tcfg["rope_parameters"]["rope_theta"])
D.DRAFT_DIR = FLINT_DRAFT


def load_weights():
    """embed/lm from the Flint bf16 target (what the export carries; the
    deployed serving applies its own int8 gather / W8A8 on top)."""
    from safetensors import safe_open

    idx = json.load(open(f"{FLINT_TARGET}/model.safetensors.index.json")
                    )["weight_map"]
    out = {}
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        with safe_open(f"{FLINT_TARGET}/{idx[name]}", framework="pt",
                       device="cpu") as sf:
            out[name] = sf.get_tensor(name).to(torch.bfloat16)
    return out["model.embed_tokens.weight"], out["lm_head.weight"]


def quantize_model_(m, group=128, kv_group=128, tau=0.0):
    replaced = {}
    for name, mod in list(m.named_modules()):
        for cn, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear) and child.weight.dim() == 2:
                pq = PTQRLinear(child.weight, group=group)
                pq.tau = tau
                setattr(mod, cn, pq)
                replaced[f"{name}.{cn}" if name else cn] = pq
    m._ptqr_tau = tau
    m._kv_group = kv_group
    return replaced


def _apply_kv_quant(m):
    tau_holder = m
    g = m._kv_group
    for layer in m.layers:
        layer.self_attn._kv_fake = (True, g)
    D.KV_FAKE_QUANT = (True, g, lambda: getattr(m, "_ptqr_tau", 0.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--kv_group", type=int, default=128)
    p.add_argument("--probe_every", type=int, default=8)
    p.add_argument("--temp_start", type=float, default=0.001)
    p.add_argument("--temp_end", type=float, default=0.001)
    p.add_argument("--cache", default="/home/curved/models/flint_draft_cache")
    p.add_argument("--out", default="/home/curved/models/flint_ptqr_draft_r1")
    args = p.parse_args()
    dev = torch.device("cuda:0")

    embed, lm = load_weights()
    teacher = load_draft(DraftModel(embed, lm), FLINT_DRAFT) \
        .to(torch.bfloat16).to(dev).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)

    student = load_draft(DraftModel(embed, lm), FLINT_DRAFT).to(dev)
    replaced = quantize_model_(student, group=args.group,
                               kv_group=args.kv_group)
    import types as _types
    for pq in replaced.values():
        _f = pq.forward

        def _fwd(self, x, _f=_f):
            return _f(x.to(self.weight.dtype)).to(x.dtype)
        pq.forward = _types.MethodType(_fwd, pq)
    _apply_kv_quant(student)
    n_hook = attach_weight_sgd_hooks(replaced, args.lr, args.lr)
    print(f"student: {len(replaced)} PTQR linears, {n_hook} SGD hooks "
          f"(cfg hidden={CFG['hidden']} vocab={CFG['vocab']})", flush=True)

    cache = sorted(glob.glob(f"{args.cache}/seq*.pt"))
    print(f"cache: {len(cache)} sequences", flush=True)
    assert cache, "run scripts/ptqr_flint_draft_cache.py first"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    si = 0
    while step < args.steps:
        d = torch.load(cache[si % len(cache)], weights_only=True)
        si += 1
        tokens, states = d["tokens"].to(dev), d["states"].to(dev)
        C = states.shape[0]
        tau = anneal_tau(step, args.steps, args.temp_start, args.temp_end)
        for pq in replaced.values():
            pq.tau = tau
        student._ptqr_tau = tau

        positions = list(range(64, C - 1, args.probe_every))
        losses = []
        for pos in positions:
            anchor = tokens[pos]
            toks = torch.full((1 + CFG["ns"],), CFG["mask_token"],
                              dtype=torch.long, device=dev)
            toks[0] = anchor
            ctx = states[:pos + 1]                      # [C', 5, H] fp16
            ctx_t = [ctx[:, i, :].unsqueeze(1).to(torch.bfloat16)
                     for i in range(CFG["layers"])]
            ctx_s = [ctx[:, i, :].unsqueeze(1).to(torch.float32)
                     for i in range(CFG["layers"])]
            qpos = torch.arange(pos + 1, pos + 1 + len(toks), device=dev)
            cpos = torch.arange(pos + 1, device=dev)
            with torch.no_grad():
                _, _, t_info = teacher(toks, ctx_t)
            _, _, s_info = student(toks, ctx_s)
            truth = tokens[pos + 1]
            t_topi, t_sc = t_info[0]
            s_topi, s_sc = s_info[0]
            s_hit = (s_topi[0] == truth)
            if s_hit.any():
                j = s_hit.nonzero().flatten()[0]
                ce = -F.log_softmax(s_sc[0].float(), -1)[j]
            else:
                ce = s_sc[0].float().mean() * 0.0
            t_dense = torch.full((CFG["vocab"],), float("-inf"),
                                 device=dev, dtype=torch.float32)
            s_dense = torch.full((CFG["vocab"],), float("-inf"),
                                 device=dev, dtype=torch.float32)
            t_dense[t_topi[0]] = t_sc[0].float()
            s_dense[s_topi[0]] = s_sc[0].float()
            both = torch.isfinite(t_dense) & torch.isfinite(s_dense)
            if both.any():
                tp = F.softmax(t_dense[both], -1)
                slp = F.log_softmax(s_dense[both], -1)
                kl = F.kl_div(slp, tp, reduction="sum")
            else:
                kl = s_sc[0].float().sum() * 0.0
            losses.append(ce + 2.0 * kl)
            if len(losses) >= 8:
                break
        loss = torch.stack(losses).mean()
        loss.backward()
        for pq in replaced.values():
            if pq.weight.grad is not None:
                with torch.no_grad():
                    g = pq.weight.grad
                    gn = g.norm()
                    if gn > 1.0:
                        g = g / gn
                    pq.weight.data.add_(-args.lr * g.float().to(pq.weight.dtype))
                    pq.weight.grad = None
        if step % 5 == 0:
            print(f"step {step} tau {tau:.4f} loss {loss.item():.4f}",
                  flush=True)
        if step > 0 and step % 20 == 0:
            torch.save(student.state_dict(), out_dir / f"draft_step{step}.pt")
        step += 1
    torch.save(student.state_dict(), out_dir / "draft_final.pt")
    print("FLINT DRAFT TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
