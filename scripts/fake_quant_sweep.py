#!/usr/bin/env python3
"""Fake-quant precision sweep v2: teacher-forced KLD gate.

v1 lessons applied:
- REFERENCE IS QUANTIZED: ref = gs-32 RTN (what the prod W8A16 path computes
  weights-wise), so comparisons are quant-vs-quant, not quant-vs-bf16.
- TEACHER-FORCED: the reference generates ONE fixed continuation per prompt.
  Every variant scores THAT SAME token sequence (logprobs at each position,
  identical prefixes throughout). No rollout divergence contaminating KLD.
  Greedy agreement (free rollout) is kept as a separate, secondary metric.

Configs sweep weight group-size (32/128), activation quant (A8 vs A16),
lm_head int8, embedding int8. Only the winning combination gets a baked
GPTQ checkpoint (then re-gated on real kernels via kld_probe.py).

Run: HIP_VISIBLE_DEVICES=0,1,2,3 <quant-venv py> scripts/fake_quant_sweep.py
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from kld_probe import PROMPTS, MAX_TOKENS  # noqa: E402

MODEL_ID = "Qwen/Qwen3.8-27B"
TOP_K = 20
BATCH = 4  # v1 OOM'd at 16 on the GDN chunked kernel (1GB activations over 28GB weights)
OUT = Path.home() / "models" / "kld"


def rtn_quant_(w: torch.Tensor, gs: int) -> None:
    """In-place RTN int8 fake-quant of a 2D weight [N, K], groups along K."""
    N, K = w.shape
    gs_eff = gs if gs and K % gs == 0 else (K if not gs else None)
    if gs_eff is None:  # non-divisible: pad
        groups = (K + gs - 1) // gs
        Kp = groups * gs
    else:
        groups = K // gs_eff
        Kp = K
    wp = w.float()
    if Kp != K:
        wp = F.pad(wp, (0, Kp - K))
    g = wp.view(N, groups, -1)
    scale = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    q = (g / scale).round().clamp_(-128, 127)
    g.copy_(q * scale)
    w.data.copy_(wp[:, :K].to(w.dtype))


def fake_act_quant(x: torch.Tensor) -> torch.Tensor:
    """Dynamic per-token per-128-block int8 — exactly the A8W8 kernel's view."""
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1])
    K = x2.shape[-1]
    blocks = (K + 127) // 128
    Kp = blocks * 128
    xp = F.pad(x2.float(), (0, Kp - K)).view(-1, blocks, 128)
    scale = xp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    q = (xp / scale).round().clamp_(-128, 127)
    out = (q * scale).view(-1, Kp)[:, :K]
    return out.reshape(orig_shape).to(x.dtype)


class ActQuantWrapper(torch.nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    def forward(self, x):
        return self.lin(fake_act_quant(x))


class Int8Embedding(torch.nn.Module):
    def __init__(self, emb: torch.nn.Embedding):
        super().__init__()
        w = emb.weight.data.float()
        scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        self.register_buffer("q", (w / scale).round().clamp_(-128, 127).to(torch.int8))
        self.register_buffer("scale", scale.squeeze(-1).to(emb.weight.dtype))
        self.padding_idx = emb.padding_idx

    def forward(self, ids):
        return F.embedding(
            ids, self.q.to(self.scale.dtype) * self.scale.unsqueeze(-1),
            padding_idx=self.padding_idx)


def apply_config(model, ws: int, act: bool, lmhead: bool, embed: bool):
    """Apply fake-quant to a FRESHLY loaded model (mutates weights in place).
    Returns nothing; each config reloads the model afterwards."""
    targets, emb_mod, emb_parent, emb_child, head = [], None, None, None, None

    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            if mod in (input_emb, output_emb):
                continue
            targets.append((name, mod))

    for name, mod in targets:
        if ws > 0:
            rtn_quant_(mod.weight.data, ws)
        if act and mod.weight.shape[-1] % 128 == 0 and mod.weight.shape[-1] >= 128:
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            child = name.rsplit(".", 1)[-1]
            setattr(parent, child, ActQuantWrapper(mod))

    if lmhead and isinstance(output_emb, torch.nn.Linear):
        rtn_quant_(output_emb.weight.data, 128)

    if embed and isinstance(input_emb, torch.nn.Embedding):
        for name, mod in model.named_modules():
            for ch, cm in mod.named_children():
                if cm is input_emb:
                    emb_parent, emb_child = mod, ch
        setattr(emb_parent, emb_child, Int8Embedding(input_emb))


def encode_and_score(model, tok, device, sequences):
    """Teacher-forced: score fixed token sequences, return per-position
    top-20 (id, logprob) lists. sequences: list of full token id lists."""
    results = []
    pad_id = tok.pad_token_id or tok.eos_token_id
    maxlen = max(len(s) for s in sequences)
    for i in range(0, len(sequences), BATCH):
        chunk = sequences[i : i + BATCH]
        ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for j, s in enumerate(chunk):
            ids[j, : len(s)] = torch.tensor(s)
            mask[j, : len(s)] = 1
        with torch.no_grad():
            logits = model(input_ids=ids.to(device),
                           attention_mask=mask.to(device)).logits
        logp = F.log_softmax(logits.float(), dim=-1)  # [B, T, V]
        for j, s in enumerate(chunk):
            # score positions predicting s[t] from prefix s[:t], t>=1
            positions = []
            for t in range(1, len(s)):
                v, idx = logp[j, t - 1].topk(TOP_K)
                positions.append(
                    [(int(idx[k]), float(v[k])) for k in range(TOP_K)])
            results.append(positions)
        del logits, logp
        torch.cuda.empty_cache()
    return results


def kld_teacher_forced(base_dump, var_dump):
    """Mean KL(base||var) over all positions of identical token sequences."""
    kls = []
    for base_pos_list, var_pos_list in zip(base_dump, var_dump):
        for bd, vd in zip(base_pos_list, var_pos_list):
            b = torch.tensor([lp for _, lp in bd])
            v = torch.tensor([lp for _, lp in vd])
            p = torch.softmax(b, -1)
            q = torch.softmax(v, -1)
            kl = torch.sum(p * (torch.log(p + 1e-12) - torch.log(q + 1e-12)))
            kls.append(kl.item())
    return float(torch.tensor(kls).mean())


def greedy_agreement(base_texts, var_texts):
    from kld_probe import greedy_agreement as ga
    return ga(base_texts, var_texts)


CONFIGS = {
    "ref_gs32": dict(ws=32, act=False, lmhead=False, embed=False),
    "gs128": dict(ws=128, act=False, lmhead=False, embed=False),
    "gs128_act": dict(ws=128, act=True, lmhead=False, embed=False),
    "gs128_act_lmhead": dict(ws=128, act=True, lmhead=True, embed=False),
    "gs128_act_embed": dict(ws=128, act=True, lmhead=False, embed=True),
    "gs128_act_lmhead_embed": dict(ws=128, act=True, lmhead=True, embed=True),
}


def load_model():
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda:0"

    # ---- Pass 1: reference generates fixed sequences (greedy) ----
    seq_cache = OUT / "ref_sequences.json"
    if seq_cache.exists():
        fixed = json.loads(seq_cache.read_text())
        print(f"[ref] loaded cached sequences ({len(fixed)} prompts)")
    else:
        model = load_model()
        apply_config(model, **CONFIGS["ref_gs32"])  # reference IS gs32-quantized
        fixed = []
        for i in range(0, len(PROMPTS), BATCH):
            chunk = PROMPTS[i : i + BATCH]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      padding_side="left").to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_TOKENS,
                                     do_sample=False)
            plen = enc["input_ids"].shape[1]
            fixed.extend(out[j, plen:].tolist() for j in range(len(chunk)))
            del out, enc
            torch.cuda.empty_cache()
        seq_cache.write_text(json.dumps(fixed))
        sequences = [tok(PROMPTS[i])["input_ids"] + fixed[i]
                     for i in range(len(PROMPTS))]
        (OUT / "ref_full_sequences.json").write_text(json.dumps(sequences))
        print(f"[ref] generated + cached {len(fixed)} sequences")

    sequences = [tok(PROMPTS[i])["input_ids"] + fixed[i]
                 for i in range(len(PROMPTS))]

    results_path = OUT / "sweep_results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for cfg_name in args.configs:
        cache = OUT / f"sim_{cfg_name}.json"
        if cache.exists():
            print(f"[skip] {cfg_name} cached")
            continue
        model = load_model()
        apply_config(model, **CONFIGS[cfg_name])
        dump = encode_and_score(model, tok, device, sequences)
        # secondary metric: free greedy rollout agreement
        texts = []
        for i in range(0, len(PROMPTS), BATCH):
            chunk = PROMPTS[i : i + BATCH]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      padding_side="left").to(device)
            with torch.no_grad():
                gout = model.generate(**enc, max_new_tokens=MAX_TOKENS,
                                      do_sample=False)
            plen = enc["input_ids"].shape[1]
            texts.extend(tok.decode(gout[j, plen:], skip_special_tokens=True)
                         for j in range(len(chunk)))
            del gout, enc
            torch.cuda.empty_cache()
        (OUT / f"sim_{cfg_name}_texts.json").write_text(json.dumps(texts))

        if cfg_name == "ref_gs32":
            (OUT / "ref_dump.json").write_text(json.dumps(dump))
            results[cfg_name] = {"kld": 0.0, "agreement": 1.0}
        else:
            ref_dump = json.loads((OUT / "ref_dump.json").read_text())
            k = kld_teacher_forced(ref_dump, dump)
            ref_texts = json.loads((OUT / "ref_texts.json").read_text()) \
                if (OUT / "ref_texts.json").exists() else None
            a = greedy_agreement(ref_texts, texts) if ref_texts else None
            results[cfg_name] = {
                "kld": round(k, 6),
                "agreement": (round(a, 4) if a is not None else None)}
        if cfg_name == "ref_gs32":
            (OUT / "ref_texts.json").write_text(
                (OUT / f"sim_{cfg_name}_texts.json").read_text())
        cache.write_text(json.dumps({"metrics": results[cfg_name]}))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[done] {cfg_name}: {results[cfg_name]}")
        results_path.write_text(json.dumps(results, indent=1))

    print("\n=== SWEEP (teacher-forced; gate kld<=0.02 & agree>=85%) ===")
    for name, m in results.items():
        if name == "ref_gs32":
            continue
        v = "PASS" if (m["kld"] <= 0.02 and (m["agreement"] or 0) >= 0.85) else \
            ("MARGINAL" if m["kld"] <= 0.05 else "FAIL")
        a = f"{m['agreement']:.1%}" if m.get("agreement") is not None else "n/a"
        print(f"{name:28s} kld={m['kld']:.5f} agree={a} -> {v}")


if __name__ == "__main__":
    main()
