#!/usr/bin/env python3
"""Draft PTQR training: INT8-everything DFlash2 draft, distilled vs its own
frozen bf16 forward (chain-level), on cached target exit states.

One GPU. Student = dense draft (correct serving semantics) with the deployed
quantization in the loop: G128 int8 weight groups (PTQRLinear), int8_block KV
g128 (ctx + block K), per-token round int8 activations. Teacher = identical
frozen unquantized copy. Loss = CE(chain token vs ground truth) +
KLD(student || teacher) over each layer's top-k candidate scores.

Usage:
  .venv-sdgraft/bin/python scripts/ptqr_train_draft_ptqr.py \
      --steps 60 --lr 1e-5 --out /home/curved/models/ptqr_draft_r1
"""
import argparse
import copy
import glob
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/curved/SDGraft")
import ptqr_train_draft as D  # noqa: E402
from ptqr_train_draft import (  # noqa: E402
    CFG, DraftModel, load_draft, rope_cos_sin)
from ptqr_train_target import (  # noqa: E402
    PTQRLinear, act_quant_fake, kv_block_quant_fake, attach_weight_sgd_hooks,
    anneal_tau)

MODEL = "/home/curved/models/Qwen3.8-27B-PTQR-R10S60"


def load_weights():
    from safetensors.torch import load_file
    embed = lm = None
    for fp in sorted(glob.glob(f"{MODEL}/model-*.safetensors")):
        sd = load_file(fp)
        if embed is None and "model.language_model.embed_tokens.weight" in sd:
            embed = sd["model.language_model.embed_tokens.weight"]
        if lm is None and "lm_head.weight" in sd:
            lm = sd["lm_head.weight"]
    return embed.to(torch.bfloat16), lm.to(torch.bfloat16)


def quantize_model_(m, group=128, kv_group=128, tau=0.0):
    """Install the deployed quantization into a DraftModel in place.

    - every nn.Linear with 2D weight -> PTQRLinear (per-channel lm_head stays)
    - attention K tensors (ctx + block) -> int8_block_g fake-quant
    - sublayer inputs -> per-token round int8 (via PTQRLinear.quant_input)
    Returns the replaced-module dict.
    """
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
    """Monkey-patch DraftAttention.forward to fake-quant K after projection."""
    import types
    tau_holder = m
    g = m._kv_group
    for layer in m.layers:
        att = layer.self_attn
        orig = att.forward

        def patched(self, q_tok, ctx_states, cos, sin, window,
                    qpos=None, cpos=None, _orig=orig, _g=g, _m=tau_holder):
            tau = getattr(_m, "_ptqr_tau", 0.0)
            if ctx_states is not None:
                # [C, B, H] -> quantize per (token) row's head split happens
                # inside _heads; quantize the PROJECTED per-head K instead.
                pass
            out = _orig(q_tok, ctx_states, cos, sin, window,
                        qpos=qpos, cpos=cpos)
            return out
        # K quant at the head level: wrap _heads results is complex; instead
        # quantize the k AFTER rope by wrapping apply via a hook on outputs
        # is also complex. Simplest faithful point: quantize ctx_states rows
        # is wrong (they are hidden states). We patch inside via a flag the
        # attention reads: set att._kv_fake = (tau, g) and let
        # DraftAttention.forward consult it (added to ptqr_train_draft).
        att._kv_fake = (True, g)
        att.forward = types.MethodType(patched, att)
    # enable the in-forward KV quant in the dense module
    D.KV_FAKE_QUANT = (True, g, lambda: getattr(m, "_ptqr_tau", 0.0))


def forward_chain(m, tokens, states, device):
    """Run the draft on one cached sequence; return per-layer scores.

    tokens: [T+1] ground truth (input = tokens[:-1] anchor block at probe
    positions); states: [C, 5, H] fp16 target exit states.
    Probes: at each pos, anchor=tokens[pos], ctx=states[:pos+1].
    Returns list over layers of (topi [P,K], scores [P,K]) plus truths [P].
    """
    raise NotImplementedError  # assembled in main via model forward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--kv_group", type=int, default=128)
    p.add_argument("--probe_every", type=int, default=8)
    p.add_argument("--batch_seqs", type=int, default=2)
    p.add_argument("--temp_start", type=float, default=0.001)
    p.add_argument("--temp_end", type=float, default=0.001)
    p.add_argument("--out", default="/home/curved/models/ptqr_draft_r1")
    args = p.parse_args()
    dev = torch.device("cuda:0")

    embed, lm = load_weights()
    teacher = load_draft(DraftModel(embed, lm)).float().to(dev).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)

    student = load_draft(DraftModel(embed, lm)).float().to(dev)
    replaced = quantize_model_(student, group=args.group,
                               kv_group=args.kv_group)
    _apply_kv_quant(student)
    n_hook = attach_weight_sgd_hooks(replaced, args.lr, args.lr)
    print(f"student: {len(replaced)} PTQR linears, {n_hook} SGD hooks", flush=True)

    cache = sorted(glob.glob("/home/curved/models/ptqr_draft_cache/seq*.pt"))
    print(f"cache: {len(cache)} sequences", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    si = 0
    opt_tau = (args.temp_start, args.temp_end)
    while step < args.steps:
        d = torch.load(cache[si % len(cache)], weights_only=True)
        si += 1
        tokens, states = d["tokens"].to(dev), d["states"].to(dev).float()
        C = states.shape[0]

        tau = anneal_tau(step, args.steps, opt_tau[0], opt_tau[1])
        for pq in replaced.values():
            pq.tau = tau
        student._ptqr_tau = tau

        # probes along the sequence
        positions = list(range(64, C - 1, args.probe_every))
        losses = []
        for pos in positions:
            anchor = tokens[pos]
            toks = torch.full((1 + CFG["ns"],), CFG["mask_token"],
                              dtype=torch.long, device=dev)
            toks[0] = anchor
            ctx = states[:pos + 1]                      # [C', 5, H]
            ctx_list = [ctx[:, i, :] for i in range(CFG["layers"])]
            qpos = torch.arange(pos + 1, pos + 1 + len(toks), device=dev)
            cpos = torch.arange(pos + 1, device=dev)
            with torch.no_grad():
                tl, _ = teacher(toks, ctx_list)
                t_top = tl[0, 0].topk(16).indices       # teacher slot-1 topk
            sl, _ = student(toks, ctx_list)
            truth = tokens[pos + 1]
            ce = F.cross_entropy(sl[0, 0][None], truth[None])
            # KLD over the teacher's top candidates
            sl_small = sl[0, 0][t_top]
            tl_small = tl[0, 0][t_top]
            kl = F.kl_div(F.log_softmax(sl_small, -1),
                          F.softmax(tl_small, -1), reduction="batchmean")
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
            print(f"step {step} tau {tau:.4f} loss {loss.item():.4f}", flush=True)
        if step > 0 and step % 20 == 0:
            sd = {k: v for k, v in student.state_dict().items()}
            torch.save(sd, out_dir / f"draft_step{step}.pt")
        step += 1
    torch.save(student.state_dict(), out_dir / "draft_final.pt")
    print("DRAFT TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
