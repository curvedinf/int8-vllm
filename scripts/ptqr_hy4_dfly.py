#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Train HY4's DFly2 W8A8 companion from real target latent KV and hidden states."""

import argparse
import copy
import json
import os
import random
from contextlib import ExitStack
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from scripts.hy4_gpu_checkpoint import GPUCheckpoint
from scripts.hy4_hub_source import HubDirectSource
from scripts.ptqr_hy4_vision import GroupedPTQR
from scripts.ptqr_train_target import _weight_integer, anneal_tau
from vllm.fs_gpu.checkpoint import DirectCheckpoint
from vllm.fs_gpu.direct import DirectStorage
from vllm.fs_gpu.region import RegisteredFile
from vllm.fs_gpu.safetensor_writer import DirectTensorWriter

PREFIX = "model.dfly_layers."


def key(name):
    return name.replace(".", "__")


def grouped_kv(x, group):
    from vllm.models.hy_v4_flash.mla_gfx908 import quantize_vector

    codes, scale = quantize_vector(x, group)
    y = (
        (codes.reshape(*scale.shape, group).float() * scale.unsqueeze(-1))
        .reshape_as(x)
        .to(x.dtype)
    )
    return x + (y - x).detach()


def vocab_grid(weights, *, embedding):
    from vllm.model_executor.layers.vocab_parallel_embedding import _quantize_vocab_rows

    codes, scales = _quantize_vocab_rows(
        weights,
        torch.float16 if embedding else torch.float32,
        min_amax=1e-8 if embedding else 1e-12,
        min_code=-128 if embedding else -127,
    )
    result = torch.empty_like(weights)
    for i in range(0, len(weights), 2048):
        result[i : i + 2048] = (codes[i : i + 2048].float() * scales[i : i + 2048]).to(
            result.dtype
        )
    return result


def head_input(x):
    scale = (x.float().abs().amax(-1, keepdim=True) / 127.0).clamp_min(1e-12)
    value = ((x.float() / scale).round().clamp(-127, 127) * scale).to(x.dtype)
    return x + (value - x).detach()


class Draft(nn.Module):
    def __init__(self, source, config):
        super().__init__()
        self.cfg = config["text_config"]
        self.layers = config["dfly_target_kv_layers"]
        self.mask = config["dfly_mask_token_id"]
        self.eps = self.cfg["rms_norm_eps"]
        self.matrices = nn.ModuleDict()
        self.vectors = nn.ParameterDict()
        self.quantized = False
        names = [n for n in source.index if n.startswith(PREFIX)]
        if hasattr(source, "file_order"):
            names.sort(key=source.file_order)
        for raw in names:
            name = raw.removeprefix(PREFIX)
            v = source.read(raw)
            # Fixed-width serving skips the original adaptive confidence head.
            # Preserve its source dtype and values for other DFly2 consumers.
            if name.startswith("confidence_head."):
                self.vectors[key(name)] = nn.Parameter(v, requires_grad=False)
                continue
            v = v.half()
            if v.ndim == 2 and name.endswith(".weight"):
                with torch.device("meta"):
                    mod = nn.Linear(v.shape[1], v.shape[0], bias=False)
                mod.weight = nn.Parameter(v, requires_grad=False)
                self.matrices[key(name)] = mod
            else:
                self.vectors[key(name)] = nn.Parameter(v, requires_grad=False)
        self.ffn = self.matrices[key("0.mlp.up_proj.weight")].out_features

    def norm(self, x, name):
        weight = self.vectors[key(name + ".weight")]
        return F.rms_norm(x.float(), (x.shape[-1],), weight.float(), self.eps).to(
            x.dtype
        )

    def mm(self, name, x):
        return self.matrices[key(name + ".weight")](x)

    def rope(self, x, positions):
        # Text positions occupy all three mRoPE planes equally. The selected
        # sparse target layers and all DFly layers use the same theta.
        cfg = self.cfg
        rhs = cfg.get("rope_parameters", {}).get("deepseek_sparse_attention", {})
        base = rhs.get("rope_theta", cfg.get("rope_theta", 10000000.0))
        if isinstance(base, dict):
            base = base.get("deepseek_sparse_attention", 10000000.0)
        dim = x.shape[-1]
        freq = torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim
        angles = positions.float()[:, None] * torch.pow(float(base), -freq)[None, :]
        cos, sin = angles.cos(), angles.sin()
        while cos.ndim < x.ndim:
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        a, b = x.float()[..., 0::2], x.float()[..., 1::2]
        return (
            torch.stack((a * cos - b * sin, b * cos + a * sin), dim=-1)
            .flatten(-2)
            .to(x.dtype)
        )

    def forward(self, embedding, anchor_id, hidden, context, anchor, width):
        cfg = self.cfg
        nh = cfg["num_attention_heads"]
        qn, qr, vw = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
        latent = cfg["kv_lora_rank"]
        ids = torch.full((width,), self.mask, device=embedding.device, dtype=torch.long)
        ids[0] = anchor_id
        e = embedding[ids]
        fused = self.mm(
            "mtp_fusion.eh_proj",
            torch.cat(
                (
                    self.norm(e[:1], "mtp_fusion.enorm"),
                    self.norm(hidden.view(1, -1), "mtp_fusion.hnorm"),
                ),
                -1,
            ),
        )
        x = torch.cat((e[:1] + self.vectors[key("mtp_fusion.gate")] * fused, e[1:]), 0)
        positions = torch.arange(anchor, anchor + width, device=x.device)
        keys = torch.arange(anchor + width, device=x.device)
        for i, target in enumerate(self.layers):
            layer = f"{i}."
            att = layer + "self_attn."
            normalized = self.norm(x, layer + "input_layernorm")
            qa = self.norm(self.mm(att + "q_a_proj", normalized), att + "q_a_layernorm")
            q = self.mm(att + "q_b_proj", qa).reshape(width, nh, qn + qr)
            qplain, qrot = q.split([qn, qr], -1)
            qrot = self.rope(qrot, positions)
            k = self.mm(att + "kv_a_proj_with_mqa", normalized)
            kplain, krot = k.split((latent, qr), -1)
            row = torch.cat(
                (self.norm(kplain, att + "kv_a_layernorm"), self.rope(krot, positions)),
                -1,
            )
            joined = torch.cat((context[target][:anchor], row), 0)
            kl, kr = joined.split((latent, qr), -1)
            if self.quantized:
                # The deployed MLA absorbs the dequantized kv_b_proj matrices into
                # Q and the attention output. Applying this projection to an identity
                # yields the identical weight grid and uses the existing PTQR STE.
                if not hasattr(self, "basis"):
                    self.register_buffer(
                        "basis",
                        torch.eye(latent, dtype=x.dtype, device=x.device),
                        persistent=False,
                    )
                weight = self.mm(att + "kv_b_proj", self.basis).T.reshape(
                    nh, qn + vw, latent
                )
                wkey, wvalue = weight.split((qn, vw), dim=1)
                qlatent = grouped_kv(torch.einsum("thd,hdl->thl", qplain, wkey), 128)
                qrot = grouped_kv(qrot, 64)
                kl, kr = grouped_kv(kl, 128), grouped_kv(kr, 64)
                scores = torch.einsum("thl,sl->hts", qlatent.float(), kl.float())
            else:
                expanded = self.mm(att + "kv_b_proj", kl).reshape(
                    anchor + width, nh, qn + vw
                )
                kp, v = expanded.split((qn, vw), -1)
                scores = torch.einsum("thd,shd->hts", qplain.float(), kp.float())
            scores = (
                scores + torch.einsum("thd,sd->hts", qrot.float(), kr.float())
            ) * (qn + qr) ** -0.5
            scores = scores.masked_fill(
                keys[None, None, :] > positions[None, :, None], float("-inf")
            )
            sink = (
                self.vectors[key(att + "learnable_sink_param")]
                .view(nh, 1, 1)
                .expand(nh, width, 1)
            )
            prob = torch.cat((scores, sink.float()), -1).softmax(-1)[..., :-1]
            if self.quantized:
                mixed = torch.einsum("hts,sl->thl", prob.to(x.dtype), kl)
                result = torch.einsum("thl,hvl->thv", mixed, wvalue).reshape(width, -1)
            else:
                result = (
                    torch.einsum("hts,shd->thd", prob, v.float())
                    .reshape(width, -1)
                    .to(x.dtype)
                )
            result = result * torch.sigmoid(self.mm(att + "linear_gate", normalized))
            x = x + self.norm(self.mm(att + "o_proj", result), att + "post_proj_norm")
            ff = self.norm(x, layer + "post_attention_layernorm")
            middle = F.silu(self.mm(layer + "mlp.gate_proj", ff)) * self.mm(
                layer + "mlp.up_proj", ff
            )
            x = x + self.mm(layer + "mlp.down_proj", middle)
        return self.norm(x, "final_norm")

    def corrected(self, z, previous):
        mixed = torch.cat(
            (
                self.norm(z, "hidden_correction.hidden_norm"),
                self.norm(previous, "hidden_correction.embed_norm"),
            ),
            -1,
        )
        update = F.silu(self.mm("hidden_correction.gate_proj", mixed)) * self.mm(
            "hidden_correction.up_proj", mixed
        )
        return z + self.mm("hidden_correction.down_proj", update)


def read_examples(path, io, layers):
    def read(file):
        with RegisteredFile(str(file), direct=True) as f:
            header = io.safetensor_header(f)
            result = {}
            for spec in header:
                v = torch.empty(spec.shape, dtype=spec.dtype, device=io.pool.device)
                io.read(f, header.data_begin + spec.start, v)
                result[spec.name] = v
            return result

    main = read(path / "target-hidden.safetensors")
    caches = {layer: read(path / f"target-kv-{layer}.safetensors") for layer in layers}
    seqs = []
    for i in range(len(main) // 2):
        seqs.append(
            (
                main[f"ids-{i}"],
                main[f"hidden-{i}"],
                {layer: cache[f"seq-{i}"] for layer, cache in caches.items()},
            )
        )
    return seqs


@torch.no_grad()
def targets(teacher, embed, head, seq, anchor, ns, generator):
    ids, hidden, cache = seq
    z = teacher(embed, ids[anchor], hidden[anchor - 1], cache, anchor, ns + 1)
    prev = ids[anchor].view(1)
    previous, expected = [], []
    for s in range(ns):
        previous.append(prev)
        hc = teacher.corrected(z[s + 1 : s + 2], embed[prev])
        logits = F.linear(hc, head).float().flatten()
        expected.append(logits)
        vals, ind = logits.topk(20)
        probs = vals.softmax(-1)
        probs = probs.masked_fill(probs.cumsum(-1) - probs >= 0.95, 0.0)
        prev = ind[torch.multinomial(probs, 1, generator=generator)]
    return torch.cat(previous), torch.stack(expected)


def loss_at(student, embed, head, seq, anchor, previous, teacher_logits):
    ids, hidden, cache = seq
    z = student(
        embed, ids[anchor], hidden[anchor - 1], cache, anchor, len(previous) + 1
    )
    logits = F.linear(
        head_input(student.corrected(z[1:], embed[previous])), head
    ).float()
    loss = F.kl_div(
        logits.log_softmax(-1), teacher_logits.softmax(-1), reduction="batchmean"
    )
    correct = logits.argmax(-1) == teacher_logits.argmax(-1)
    return loss, correct


@torch.no_grad()
def agreement_with_target(model, embed, projection, original_head, seqs, held, width):
    """Compare draft and target at the same real prefixes and previous tokens."""
    temperatures = (0.5, 1.0, 2.0)
    agreement = []
    distances = {t: [] for t in temperatures}
    for index, anchor, _, _ in held:
        ids, hidden, cache = seqs[index]
        positions = slice(anchor, anchor + width)
        states = model(embed, ids[anchor], hidden[anchor - 1], cache, anchor, width + 1)
        prediction = model.corrected(states[1:], embed[ids[positions]])
        if model.quantized:
            prediction = head_input(prediction)
        draft = F.linear(prediction, projection).float()
        target = F.linear(hidden[positions], original_head).float()
        agreement.append((draft.argmax(-1) == target.argmax(-1)).float())
        for t in temperatures:
            log_p = (target / t).log_softmax(-1)
            log_q = (draft / t).log_softmax(-1)
            p = log_p.exp()
            kl = (p * (log_p - log_q)).sum(-1)
            tv = (p - log_q.exp()).abs().sum(-1) / 2
            distances[t].append(torch.stack((kl, tv)))
    result = {"top1_per_position": torch.stack(agreement).mean(0).tolist()}
    for t, rows in distances.items():
        kl, tv = torch.stack(rows).mean(0).tolist()
        result[str(t)] = {"kl_target_to_draft": kl, "total_variation": tv}
    return result


def train(teacher, student, qs, embed, head, seqs, args):
    rng = random.Random(args.seed)
    generator = torch.Generator(device=embed.device).manual_seed(args.seed)
    idx = list(range(len(seqs)))
    rng.shuffle(idx)
    split = min(args.validation_sequences, len(idx) // 3)
    if split < 1:
        raise ValueError("Need independent draft training and validation sequences")
    val, train = idx[:split], idx[split:]

    def example(i):
        seq = seqs[i]
        anchor = rng.randrange(8, len(seq[0]) - args.ns - 1)
        return (
            i,
            anchor,
            *targets(teacher, embed, head, seq, anchor, args.ns, generator),
        )

    held = [example(i) for i in val]
    student_embed = vocab_grid(embed, embedding=True)
    student_head = vocab_grid(head, embedding=False)

    @torch.no_grad()
    def score():
        results = [
            loss_at(student, student_embed, student_head, seqs[i], a, prev, expect)
            for i, a, prev, expect in held
        ]
        return {
            "kl": float(torch.stack([x[0] for x in results]).mean()),
            "top1": float(torch.cat([x[1] for x in results]).float().mean()),
        }

    base = score()
    print("Held-out DFly before PTQR", base, flush=True)
    opt = torch.optim.AdamW(
        [q.scale for q in qs.values()], lr=args.scale_lr, weight_decay=0
    )
    best = None
    history = []
    for step in range(1, args.steps + 1):
        i, anchor, prev, expect = example(train[(step - 1) % len(train)])
        for q in qs.values():
            q.tau = (
                anneal_tau(step - 1, args.steps, args.dither, args.dither)
                if args.dither
                else 0.0
            )
        loss, _ = loss_at(
            student, student_embed, student_head, seqs[i], anchor, prev, expect
        )
        if not bool(loss.isfinite()):
            raise ValueError(f"Nonfinite draft loss at step {step}")
        loss.backward()
        with torch.no_grad():
            for q in qs.values():
                if q.weight.grad is None or q.scale.grad is None:
                    raise RuntimeError("Missing DFly PTQR gradient")
                grad = q.weight.grad
                q.weight.add_(grad / grad.float().norm().clamp_min(1.0), alpha=-args.lr)
                q.weight.grad = None
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for q in qs.values():
                q.scale.clamp_min_(2**-24)
                q.tau = 0.0
        if step % args.eval_every == 0 or step == args.steps:
            result = {"step": step, **score()}
            history.append(result)
            print("Held-out DFly", result, flush=True)
            if best is None or result["kl"] < best[0]["kl"]:
                best = (
                    result,
                    {
                        k: (q.weight.detach().clone(), q.scale.detach().clone())
                        for k, q in qs.items()
                    },
                )
    with torch.no_grad():
        for k, (v, s) in best[1].items():
            qs[k].weight.copy_(v)
            qs[k].scale.copy_(s)
    return {
        "baseline": base,
        "selected": best[0],
        "history": history,
        "train_sequences": len(train),
        "validation_sequences": split,
        "num_speculative_tokens": args.ns,
        "seed": args.seed,
        "heldout_target_agreement": {
            "anchors": [a for _, a, _, _ in held],
            "original_draft": agreement_with_target(
                teacher, embed, head, head, seqs, held, args.ns
            ),
            "ptqr_draft": agreement_with_target(
                student, student_embed, student_head, head, seqs, held, args.ns
            ),
        },
    }


@torch.no_grad()
def save_target_reference(path, seqs, head):
    # Emit only token identifiers and probabilities; the teacher weights and
    # hidden states stay on the GPU. This is a calibration-prefix diagnostic.
    with path.open("w") as out:
        for i, (ids, hidden, _) in enumerate(seqs):
            position = min(len(ids) - 1, 32 + (i * 17) % 126)
            logits = (
                F.linear(hidden[position : position + 1], head).float().log_softmax(-1)
            )
            values, tokens = logits[0].topk(20)
            out.write(
                json.dumps(
                    {
                        "input": ids[: position + 1].tolist(),
                        "top": tokens.tolist(),
                        "logprobs": values.tolist(),
                    }
                )
                + "\n"
            )


def serving_name(name):
    name = name.removesuffix(".weight")
    for a in ("gate_proj", "up_proj"):
        if name.endswith((".mlp." + a, ".hidden_correction." + a)):
            return name.removesuffix(a) + "gate_up_proj"
    for a in ("q_a_proj", "kv_a_proj_with_mqa"):
        if name.endswith(".self_attn." + a):
            return name.removesuffix(a) + "fused_qkv_a_proj"
    return name


@torch.no_grad()
def export(output, io, teacher, qs, writer_type):
    output.mkdir(parents=True, exist_ok=True)
    meta, index = {}, {}
    groups = {}
    for raw, v in list(teacher.vectors.items()) + list(teacher.matrices.items()):
        name = raw.replace("__", ".")
        groups.setdefault(name.split(".", 1)[0], []).append((PREFIX + name, raw, v))
    for group, weights in groups.items():
        spec = {}
        for name, raw, v in weights:
            if raw in qs:
                q = qs[raw]
                spec[name] = (q.weight.shape, torch.int8)
                spec[name.removesuffix(".weight") + ".weight_scale"] = (
                    q.scale.shape,
                    torch.float16,
                )
                meta[serving_name(name)] = 8
            else:
                spec[name] = (v.shape, v.dtype)
        filename = f"dfly-{group}.safetensors"
        with writer_type(output / filename, spec, io) as writer:
            for name, raw, v in weights:
                if raw not in qs:
                    writer.append(name, v)
                    continue
                q = qs[raw]
                writer.append(
                    name, _weight_integer(q.weight, q.scale, 128).to(torch.int8)
                )
                writer.append(
                    name.removesuffix(".weight") + ".weight_scale", q.scale.half()
                )
        index.update(dict.fromkeys(spec, filename))
    for fname, content in [
        ("model.safetensors.index.json", {"weight_map": index}),
        (
            "dfly.quant.json",
            {"quant_method": "hy4_neuqi", "group_size": 128, "quantized_modules": meta},
        ),
    ]:
        (output / fname).write_text(json.dumps(content, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("examples", type=Path)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--hub-repo")
    p.add_argument("--revision")
    p.add_argument("--workspace", type=Path)
    p.add_argument(
        "--target-reference",
        type=Path,
        help="Write original-teacher calibration-prefix logprobs",
    )
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--validation-sequences", type=int, default=12)
    p.add_argument("--ns", type=int, default=7)
    p.add_argument("--seed", type=int, default=4017)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--scale-lr", type=float, default=1e-7)
    p.add_argument("--dither", type=float, default=0.001)
    args = p.parse_args()
    if min(args.steps, args.eval_every, args.validation_sequences, args.ns) < 1:
        p.error("Training and evaluation counts must be positive")
    if args.hub_repo and not (args.revision and args.workspace):
        p.error("Hub acquisition requires a pinned revision and native workspace")
    os.environ.setdefault("HIPFILE_ALLOW_COMPAT_MODE", "false")
    os.environ.setdefault("HIPFILE_FORCE_COMPAT_MODE", "false")
    torch.manual_seed(args.seed)
    d = torch.device("cuda", args.device)
    torch.cuda.set_device(d)
    cfg = json.loads((args.source / "config.json").read_text())
    with ExitStack() as stack:
        io = stack.enter_context(DirectStorage(d))
        source = stack.enter_context(
            HubDirectSource(
                args.source, io, args.workspace, args.hub_repo, args.revision
            )
            if args.hub_repo
            else DirectCheckpoint(args.source, io)
        )
        teacher = Draft(source, cfg)
        embed = source.read("model.embed_tokens.weight").half()
        head = source.read("lm_head.weight").half()
        student = copy.deepcopy(teacher)
        student.quantized = True
        qs = {}
        for name, mod in list(student.matrices.items()):
            new = GroupedPTQR(mod)
            student.matrices[name] = new
            qs[name] = new.linear
        seqs = read_examples(args.examples, io, teacher.layers)
        if args.target_reference:
            save_target_reference(args.target_reference, seqs, head)
        results = train(teacher, student, qs, embed, head, seqs, args)
        bank = GPUCheckpoint(d, [d.index], args.workspace) if args.workspace else None
        export(
            args.output, io, teacher, qs, bank.writer if bank else DirectTensorWriter
        )
        if bank:
            bank.export()
        results.update(
            dfly_intermediate_size=teacher.ffn,
            steps=args.steps,
            lr=args.lr,
            scale_lr=args.scale_lr,
            dither=args.dither,
            zero_dither_final_fraction=0.1,
            source_revision=args.revision,
        )
        (args.output / "dfly-training.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
