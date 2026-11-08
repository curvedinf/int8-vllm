#!/usr/bin/env python3
"""Export Flint PTQR rank-shard checkpoints to the GPTQ GS128 format vLLM loads.

Flint port of ptqr_export_gptq.py (same gptqmodel byte layout — quant_pack is
imported verbatim). Stitch geometry mirrors shard_flint_tp:

  layers.N.attn.qkv          per-rank fused [q|k|v] rows  -> separate
                             self_attn.{q,k,v}_proj (row-parallel dim0)
  layers.N.attn.o            column-sharded               -> o_proj (dim1)
  layers.N.mlp.gate_up       per-rank fused [gate|up]     -> mlp.{gate,up}_proj
  layers.N.mlp.down          column-sharded               -> mlp.down_proj
  layers.N.mlp.shared.*      same, under shared_expert.*
  layers.N.mlp.experts.E     WHOLE expert on rank E%world -> experts.E.*
  layers.N.mlp.router        replicated dense (no scale)  -> mlp.gate bf16
  embed / lm_head            vocab-sharded dim0           -> bf16 (the serving
                             loader applies its own int8 conversions)
  norms / qk_norm / sink     replicated params            -> bf16

Usage (any torch venv):
  python scripts/ptqr_export_flint.py \
    --ckpt_dir /home/curved/models/flint-ptqr-r1 --step 60 --world 4 \
    --src /home/curved/models/flint-bf16 \
    --out /home/curved/models/Flint-GPTQ-INT8-W8A8-GS128-PTQR
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from ptqr_export_gptq import load_rank_shards, quant_pack


def split_qkv(per, q_r, kv_r):
    """per: list of [q_r + 2*kv_r, in] rank slices -> (q_full, k_full, v_full)."""
    qs = [p[:q_r] for p in per]
    ks = [p[q_r:q_r + kv_r] for p in per]
    vs = [p[q_r + kv_r:] for p in per]
    return (torch.cat(qs, 0), torch.cat(ks, 0), torch.cat(vs, 0))


def split_gated(per):
    """per: list of [2*half, in] fused [gate|up] slices -> (gate, up) fulls."""
    h = per[0].shape[0] // 2
    gate = torch.cat([p[:h] for p in per], 0)
    up = torch.cat([p[h:] for p in per], 0)
    return gate, up


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--src", default="/home/curved/models/flint-bf16",
                   help="bf16 checkpoint dir (config/tokenizer + verify ref)")
    p.add_argument("--out", required=True)
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--verify", action="store_true",
                   help="report stitched-master vs bf16-source max|diff|")
    args = p.parse_args()

    shards = load_rank_shards(Path(args.ckpt_dir), args.step, args.world)
    cfg = json.loads((Path(args.src) / "config.json").read_text())
    L = cfg["num_hidden_layers"]
    V, H = cfg["vocab_size"], cfg["hidden_size"]
    hd = cfg["head_dim"]
    q_full = cfg["num_attention_heads"] * hd
    kv_full = cfg["num_key_value_heads"] * hd
    q_r, kv_r = q_full // args.world, kv_full // args.world
    dense_layers = set(cfg.get("mlp_only_layers") or [])

    def rank_of(name):
        for r, s in enumerate(shards):
            if name in s:
                return r
        raise KeyError(name)

    # Build the export plan: list of (shard_base_name, hf_base_name, kind)
    # kind: qkv | gated | col | row | whole | param
    plan = []
    for n in range(L):
        a = f"layers.{n}.attn"
        plan.append((f"{a}.qkv", f"model.layers.{n}.self_attn", "qkv"))
        plan.append((f"{a}.o", f"model.layers.{n}.self_attn.o_proj", "col"))
        plan.append((f"layers.{n}.norm1.weight",
                     f"model.layers.{n}.input_layernorm.weight", "param"))
        plan.append((f"layers.{n}.norm2.weight",
                     f"model.layers.{n}.post_attention_layernorm.weight",
                     "param"))
        plan.append((f"{a}.q_norm.weight",
                     f"model.layers.{n}.self_attn.q_norm.weight", "param"))
        plan.append((f"{a}.k_norm.weight",
                     f"model.layers.{n}.self_attn.k_norm.weight", "param"))
        plan.append((f"{a}.sink",
                     f"model.layers.{n}.self_attn.softmax_offset", "param"))
        if n in dense_layers:
            plan.append((f"layers.{n}.mlp.gate_up",
                         f"model.layers.{n}.mlp", "gated"))
            plan.append((f"layers.{n}.mlp.down",
                         f"model.layers.{n}.mlp.down_proj", "col"))
        else:
            plan.append((f"layers.{n}.mlp.router.weight",
                         f"model.layers.{n}.mlp.gate.weight", "param"))
            plan.append((f"layers.{n}.mlp.shared.gate_up",
                         f"model.layers.{n}.mlp.shared_expert", "gated"))
            plan.append((f"layers.{n}.mlp.shared.down",
                         f"model.layers.{n}.mlp.shared_expert.down_proj",
                         "col"))
            for e in range(cfg["num_experts"]):
                r = e % args.world
                loc = e // args.world
                b = f"layers.{n}.mlp.experts.{loc}"
                hb = f"model.layers.{n}.mlp.experts.{e}"
                plan.append((f"{b}.gate_up", hb, "gated1"))  # whole, single rank
                plan.append((f"{b}.down", f"{hb}.down_proj", "whole"))
    plan.append(("embed.weight", "model.embed_tokens.weight", "row"))
    plan.append(("lm_head.weight", "lm_head.weight", "row"))
    plan.append(("final_norm.weight", "model.norm.weight", "param"))

    def get_w(shard_name):
        s = shards[rank_of(shard_name)]
        k = shard_name if shard_name in s else shard_name + ".weight"
        return s[k]

    def has_scale(shard_name):
        base = shard_name[:-len(".weight")] if shard_name.endswith(".weight") \
            else shard_name
        return any(base + ".scale" in s for s in shards)

    def get_scale(shard_name):
        base = shard_name[:-len(".weight")] if shard_name.endswith(".weight") \
            else shard_name
        return shards[rank_of(base + ".scale")][base + ".scale"]

    def emit(row):
        """Yield (hf_key, tensor-or-tuple) entries for one plan row."""
        sname, hf, kind = row
        if kind == "param":
            yield hf, get_w(sname).to(torch.bfloat16)
            return
        if kind == "qkv":
            if not has_scale(sname):   # router-like fallback; not expected
                yield hf + ".weight", get_w(sname).to(torch.bfloat16)
                return
            per = [s[sname + ".weight"] for s in shards]
            sc = [s[sname + ".scale"] for s in shards]
            q, k, v = split_qkv(per, q_r, kv_r)
            sq, sk, sv = split_qkv(sc, q_r, kv_r)
            for full, sfull, tag in ((q, sq, "q_proj"), (k, sk, "k_proj"),
                                     (v, sv, "v_proj")):
                qw, qz, s16 = quant_pack(full.contiguous(),
                                         sfull.contiguous(), args.group)
                yield hf + f".{tag}", (qw, qz, s16)
            return
        if kind == "gated":
            per = [s[sname + ".weight"] for s in shards]
            sc = [s[sname + ".scale"] for s in shards]
            gate, up = split_gated(per)
            gs, us = split_gated(sc)
            for w, s_, tag in ((gate, gs, "gate_proj"), (up, us, "up_proj")):
                qw, qz, s16 = quant_pack(w.contiguous(), s_.contiguous(),
                                         args.group)
                yield hf + f".{tag}", (qw, qz, s16)
            return
        if kind == "gated1":  # whole expert fused gate_up on one rank
            w = get_w(sname + ".weight")
            sc = get_scale(sname)
            gate, up = w[: w.shape[0] // 2], w[w.shape[0] // 2:]
            gs, us = sc[: sc.shape[0] // 2], sc[sc.shape[0] // 2:]
            for wt, st, tag in ((gate, gs, "gate_proj"), (up, us, "up_proj")):
                qw, qz, s16 = quant_pack(wt.contiguous(), st.contiguous(),
                                         args.group)
                yield hf + f".{tag}", (qw, qz, s16)
            return
        if kind == "col":
            w = torch.cat([s[sname + ".weight"] for s in shards], 1)
            sc = torch.cat([s[sname + ".scale"] for s in shards], 1)
            qw, qz, s16 = quant_pack(w.contiguous(), sc.contiguous(), args.group)
            yield hf, (qw, qz, s16)
            return
        if kind == "whole":
            r = rank_of(sname + ".weight")
            w = shards[r][sname + ".weight"]
            sc = shards[r][sname + ".scale"]
            qw, qz, s16 = quant_pack(w.contiguous(), sc.contiguous(), args.group)
            yield hf, (qw, qz, s16)
            return
        if kind == "row":
            yield hf, torch.cat([s[sname] for s in shards], 0) \
                .to(torch.bfloat16).contiguous()
            return
        raise ValueError(kind)

    # file plan: ~4.5 GiB chunks
    def entry_bytes(row):
        name, kind = row[0], row[2]
        if kind in ("param", "row"):
            return get_w(name).numel() * 2
        if kind == "qkv":
            return (q_full + 2 * kv_full) * H * 4
        w = get_w(name + ".weight") if not name.endswith(".weight") \
            else get_w(name)
        return w.numel() * (4 if kind in ("gated", "gated1", "col", "whole")
                            else w.element_size())

    file_plan, cur, cur_b = [], [], 0
    for row in plan:
        b = entry_bytes(row)
        if cur_b + b > 4.5e9 and cur:
            file_plan.append(cur)
            cur, cur_b = [], 0
        cur.append(row)
        cur_b += b
    if cur:
        file_plan.append(cur)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    n_quant = 0
    n_files = len(file_plan)
    for i, rows in enumerate(file_plan):
        chunk = {}
        for row in rows:
            for key, val in emit(row):
                if isinstance(val, tuple):
                    chunk[key + ".qweight"] = val[0]
                    chunk[key + ".qzeros"] = val[1]
                    chunk[key + ".scales"] = val[2]
                    n_quant += 1
                else:
                    chunk[key] = val
        fn = f"model-{i+1:05d}-of-{n_files:05d}.safetensors"
        save_file(chunk, str(out_dir / fn))
        for k in chunk:
            index["weight_map"][k] = fn
        index["metadata"]["total_size"] += sum(
            t.numel() * t.element_size() for t in chunk.values())
        del chunk
        print(f"  wrote {fn}", flush=True)

    qcfg = {
        "bits": 8, "group_size": args.group, "desc_act": False, "sym": True,
        "lm_head": False, "method": "gptq", "quant_method": "gptq",
        "checkpoint_format": "gptq", "pack_dtype": "int32",
        "meta": {"quantizer": ["ptqr-retrain:scripts/ptqr_export_flint.py"]},
    }
    (out_dir / "quantize_config.json").write_text(json.dumps(qcfg, indent=2))
    for fname in ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "generation_config.json", "chat_template.jinja"):
        src = Path(args.src) / fname
        if not src.exists():
            continue
        if fname == "config.json":
            # vLLM 0.33 auto-detects the quant method from config.json's
            # `quantization_config` (quantize_config.json alone is not
            # enough): inject the GPTQ descriptor like gptqmodel writes it.
            cfg = json.loads(src.read_text())
            cfg["quantization_config"] = {
                "quant_method": "gptq", "bits": 8,
                "group_size": args.group, "desc_act": False, "sym": True,
                "checkpoint_format": "gptq", "lm_head": False,
                "format": "gptq",
            }
            (out_dir / fname).write_text(json.dumps(cfg, indent=2))
        else:
            (out_dir / fname).write_bytes(src.read_bytes())
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    print(f"{n_quant} tensors packed to GPTQ GS128; index written", flush=True)

    if args.verify:
        import glob

        import safetensors.torch as st
        ref = {}
        for f in sorted(glob.glob(str(Path(args.src) / "*.safetensors"))):
            ref.update(st.load_file(f))
        worst = []
        for row in plan:
            name, kind = row[0], row[2]
            if kind in ("param", "row", "whole", "gated1"):
                continue
            if kind == "qkv":
                per = [s[name + ".weight"] for s in shards]
                q, k, v = split_qkv(per, q_r, kv_r)
                for tag, full in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
                    r = ref.get(f"{row[1]}.{tag}.weight")
                    if r is not None:
                        worst.append((
                            (full.float() - r.float()).abs().max().item(),
                            f"{row[1]}.{tag}"))
                continue
            if kind == "gated":
                per = [s[name + ".weight"] for s in shards]
                gate, up = split_gated(per)
                for tag, full in (("gate_proj", gate), ("up_proj", up)):
                    r = ref.get(f"{row[1]}.{tag}.weight")
                    if r is not None:
                        worst.append((
                            (full.float() - r.float()).abs().max().item(),
                            f"{row[1]}.{tag}"))
                continue
            full = torch.cat([s[name + ".weight"] for s in shards], 1)
            r = ref.get(row[1] + ".weight")
            if r is not None:
                worst.append(((full.float() - r.float()).abs().max().item(),
                              row[1]))
        worst.sort(reverse=True)
        print("verify (stitched master vs bf16 source, top-8 max|diff|):")
        for d, n_ in worst[:8]:
            print(f"  {d:.6f}  {n_}")


if __name__ == "__main__":
    main()
