# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Train stock DSpark on target trajectories and export native G128 W8A8 weights."""

import argparse
import json
import random
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.nn import functional as F

from scripts.hy4_gpu_checkpoint import GPUCheckpoint
from scripts.ptqr_train_target import (
    PTQRLinear,
    _weight_integer,
    anneal_tau,
    kv_block_quant_fake,
)
from vllm.fs_gpu.direct import DirectStorage
from vllm.fs_gpu.region import RegisteredFile


def norm(x, weight, eps):
    v = x.float()
    return (
        v * v.square().mean(-1, keepdim=True).add(eps).rsqrt() * weight.float()
    ).bfloat16()


def rotate(x, positions, theta):
    d = x.shape[-1]
    freq = theta ** (-torch.arange(0, d, 2, device=x.device).float() / d)
    angle = (positions[:, None] * freq)[:, None, :]
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat(
        (x1 * angle.cos() - x2 * angle.sin(), x2 * angle.cos() + x1 * angle.sin()), -1
    ).bfloat16()


class Draft(nn.Module):
    def __init__(self, weights, cfg, head, quantized):
        super().__init__()
        self.raw = weights
        self.cfg = cfg
        self.head = head
        self.quantized = quantized
        self.projections = nn.ModuleDict()
        for name, w in weights.items():
            if (
                not name.startswith("layers.")
                or not name.endswith(".weight")
                or w.ndim != 2
            ):
                continue
            if quantized:
                layer = PTQRLinear(w.float(), group=128, act_group=128)
                layer.scale.data.clamp_min_(2**-24)
                self.projections[name.removesuffix(".weight").replace(".", "__")] = (
                    layer
                )
        self.alpha = weights["fuse_weight"].float().softmax(-1).bfloat16()
        self.rnn = weights["markov_head.rnn_proj.weight"].float()
        self.rnn_bias = weights["markov_head.rnn_proj.bias"].float()
        w = weights["markov_head.hidden_out.weight"].double()
        b = weights["markov_head.hidden_out.bias"].double()
        self.fold = torch.cat([(h.double() @ w).bfloat16() for h in head.split(2048)])
        self.fold_bias = torch.cat(
            [(h.double() @ b).bfloat16() for h in head.split(2048)]
        )

    def linear(self, name, x):
        if self.quantized:
            return self.projections[name.replace(".", "__")](x.float()).bfloat16()
        return F.linear(x, self.raw[name + ".weight"])

    def mix(self, taps, row, name):
        x = (taps.float() * self.alpha[row].float()[None, :, None]).sum(1)
        return norm(x, self.raw[name + ".weight"], self.cfg["rms_norm_eps"])

    def forward(self, features, ids, anchor):
        cfg, w = self.cfg, self.raw
        eps, d = cfg["rms_norm_eps"], cfg["head_dim"]
        nq, nk, block = (
            cfg["num_attention_heads"],
            cfg["num_key_value_heads"],
            cfg["block_size"],
        )
        h = torch.cat(
            (features[anchor : anchor + 1, -1], w["mask_embed"].expand(block - 1, -1)),
            0,
        )
        pos = torch.arange(anchor + block, device=h.device)
        step = None
        for i in range(cfg["num_hidden_layers"]):
            p = f"layers.{i}."
            local = i in cfg["local_layer_ids"]
            start = max(0, anchor - cfg["sliding_window_size"]) if local else 0
            ctx = self.mix(features[start:anchor, :4], i, p + "dflare_norm")
            total = h if step is None else h.float() + step.float()
            x = norm(total, w[p + "input_layernorm.weight"], eps)
            h = total.bfloat16()
            a = p + "self_attn."
            q = self.linear(a + "q_proj", x).view(block, nq, d)
            k = self.linear(a + "k_proj", x).view(block, nk, d)
            v = self.linear(a + "v_proj", x).view(block, nk, d)
            ck = self.linear(a + "k_proj_ctx", ctx).view(-1, nk, d)
            cv = self.linear(a + "v_proj_ctx", ctx).view(-1, nk, d)
            q = rotate(
                norm(q, w[a + "q_norm.weight"], eps), pos[anchor:], cfg["rope_theta"]
            )
            k = rotate(
                norm(k, w[a + "k_norm.weight"], eps), pos[anchor:], cfg["rope_theta"]
            )
            ck = rotate(
                norm(ck, w[a + "k_norm_ctx.weight"], eps),
                pos[start:anchor],
                cfg["rope_theta"],
            )
            kv = [
                kv_block_quant_fake(value[None], d)[0].repeat_interleave(nq // nk, 1)
                for value in (torch.cat((ck, k)), torch.cat((cv, v)))
            ]
            probs = (
                torch.einsum("qhd,khd->hqk", q.float(), kv[0].float()) * d**-0.5
            ).softmax(-1)
            result = torch.einsum("hqk,khd->qhd", probs.bfloat16(), kv[1]).reshape(
                block, -1
            )
            total = self.linear(a + "o_proj", result).float() + h.float()
            x = norm(total, w[p + "post_attention_layernorm.weight"], eps)
            h = total.bfloat16()
            m = p + "mlp."
            gate = self.linear(m + "gate_proj", x)
            up = self.linear(m + "up_proj", x)
            step = self.linear(
                m + "down_proj", (F.silu(gate.float()) * up.float()).bfloat16()
            )
        hidden = norm(step.float() + h.float(), w["norm.weight"], eps)[1:]
        ctx = self.mix(features[anchor - 1 : anchor, :4], -1, "markov_head.dflare_norm")
        rank = cfg["markov_rank"]
        bias = F.linear(
            torch.cat((hidden, ctx.expand_as(hidden)), -1).float(),
            self.rnn[:, 2 * rank :],
            self.rnn_bias,
        )
        logits = F.linear(hidden, self.head)
        state = hidden.new_zeros(1, rank)
        scores = []
        for j in range(block - 1):
            prev = w["markov_head.markov_w1.weight"][ids[anchor + j : anchor + j + 1]]
            raw = F.linear(
                torch.cat((state, prev), -1).float(), self.rnn[:, : 2 * rank], bias[j]
            )
            gate, cand, out = raw.bfloat16().chunk(3, -1)
            gate = gate.sigmoid()
            state = gate * state + (1 - gate) * cand.tanh()
            correction = F.linear(out.tanh(), self.fold, self.fold_bias)
            z = logits[j : j + 1] + correction
            scores.append((z / cfg["softcap"]).tanh() * cfg["softcap"])
        return torch.cat(scores).float()


def read_stock(source, workspace, device):
    with TemporaryDirectory(dir=workspace) as tmp:
        path = Path(tmp) / "mtp.safetensors"
        shutil.copyfile(source / path.name, path)
        with (
            DirectStorage(device) as io,
            RegisteredFile(str(path), direct=True) as file,
        ):
            header = io.safetensor_header(file)
            weights = {}
            for spec in header:
                value = torch.empty(spec.shape, dtype=spec.dtype, device=device)
                io.read(file, header.data_begin + spec.start, value)
                weights[spec.name] = value
            torch.cuda.synchronize()
    return weights


def train(args):
    device = torch.device("cuda", 0)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    raw_config = json.loads((args.source / "config.json").read_text())
    cfg = dict(
        raw_config["mtp_config"],
        softcap=raw_config["text_config"]["final_logit_softcapping"],
    )
    weights = read_stock(args.source, args.workspace, device)
    vocab = raw_config["text_config"]["unpadded_vocab_size"]
    head = torch.cat(
        [
            torch.load(p, map_location=device, weights_only=True)
            for p in sorted(args.head.glob("head[0-9].pt"))
        ]
    )[:vocab].contiguous()
    if head.shape != (vocab, cfg["hidden_size"]):
        raise ValueError(
            "Supply the full dequantized target vocabulary head in TP rank order"
        )
    teacher = Draft(weights, cfg, head, False)
    student = Draft(weights, cfg, head, True)
    params = list(student.projections.values())
    optimizer = torch.optim.AdamW(
        [
            dict(params=[m.weight for m in params], lr=args.lr),
            dict(params=[m.scale for m in params], lr=args.scale_lr),
        ],
        weight_decay=0,
    )
    train_files = sorted(args.samples.glob("train-*.pt"))
    held = sorted(args.samples.glob("validation-*.pt"))
    if not train_files or not held:
        raise ValueError("Need independent target training and validation trajectories")

    def example(path, offset, quality=None):
        item = torch.load(path, map_location=device, weights_only=True)
        tokens, features = item["tokens"], item["features"]
        anchor = min(item["prompt_length"] + offset, len(tokens) - cfg["block_size"])
        with torch.no_grad():
            stock = teacher(features, tokens, anchor).log_softmax(-1)
            start = anchor + 1 - item["prompt_length"]
            rows = slice(start, start + cfg["block_size"] - 1)
            keys = item["target_top_ids"][rows]
            lp = item["target_logprobs"][rows]

            def with_tail(values):
                tail = (1 - values.exp().sum(-1, keepdim=True)).clamp_min(1e-10).log()
                return torch.cat((values, tail), dim=-1)

            target = with_tail(lp)
        pred = student(features, tokens, anchor).log_softmax(-1)
        kl = F.kl_div(pred, stock, reduction="batchmean", log_target=True)
        target_kl = F.kl_div(
            with_tail(pred.gather(1, keys)),
            target,
            reduction="batchmean",
            log_target=True,
        )
        ce = F.nll_loss(pred, tokens[anchor + 1 : anchor + cfg["block_size"]])
        agreement = (pred.argmax(-1) == stock.argmax(-1)).float().mean()
        if quality is not None:
            # The target exposes 64 exact probabilities. At T=1 the remaining
            # vocabulary can be scored as one bucket. At other temperatures
            # only the distribution conditional on the stored ids is known.
            for name, logp in (("stock", stock), ("ptqr", pred)):
                top = logp.gather(1, keys)
                for temp in (0.7, 1.0, 1.3):
                    if temp == 1:
                        reference, evaluated = target, with_tail(top)
                        support = "top64_and_tail"
                    else:
                        reference = (lp / temp).log_softmax(-1)
                        evaluated = (top / temp).log_softmax(-1)
                        support = "conditional_top64"
                    key = f"{name}_t{temp}_{support}"
                    quality.setdefault(key, []).append(
                        torch.stack(
                            (
                                F.kl_div(
                                    evaluated,
                                    reference,
                                    reduction="batchmean",
                                    log_target=True,
                                ),
                                (reference.exp() - evaluated.exp()).abs().sum(-1).mean()
                                / 2,
                            )
                        )
                    )
                quality.setdefault(f"{name}_target_argmax_by_position", []).append(
                    (logp.argmax(-1) == keys[:, 0]).float()
                )
        return (
            kl,
            target_kl,
            ce,
            agreement,
            (pred.argmax(-1) == keys[:, 0]).float().mean(),
        )

    @torch.no_grad()
    def evaluate():
        quality = {}
        results = [example(path, 8 + 5 * i, quality) for i, path in enumerate(held)]
        metrics = dict(
            zip(
                (
                    "kl_stock",
                    "kl_target_top64_tail",
                    "next_token_ce",
                    "top1_stock",
                    "top1_target",
                ),
                map(float, torch.stack([torch.stack(r) for r in results]).mean(0)),
            )
        )
        metrics["target_fidelity"] = {}
        for key, values in quality.items():
            mean = torch.stack(values).mean(0).tolist()
            metrics["target_fidelity"][key] = (
                mean if key.endswith("_position") else dict(zip(("kl", "tv"), mean))
            )
        return metrics

    base = evaluate()
    print("Initial held-out", base, flush=True)
    best_loss = float("inf")
    best = None
    history = []
    for step in range(args.steps):
        if step % len(train_files) == 0:
            rng.shuffle(train_files)
        for m in params:
            m.tau = anneal_tau(step, args.steps, args.dither, args.dither)
        kl, target_kl, ce, _, _ = example(
            train_files[step % len(train_files)], rng.randrange(0, 55)
        )
        loss = kl + 0.1 * target_kl + 0.02 * ce
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite draft loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for m in params:
                m.scale.clamp_min_(2**-24)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            for m in params:
                m.tau = 0
            row = dict(step=step + 1, **evaluate())
            history.append(row)
            print("Held-out", row, flush=True)
            if row["kl_stock"] < best_loss:
                best_loss = row["kl_stock"]
                best = {}
                for key, m in student.projections.items():
                    name = key.replace("__", ".") + ".weight"
                    best[name] = _weight_integer(
                        m.weight.detach(), m.scale.detach(), 128
                    ).to(torch.int8)
                    best[name + "_scale"] = m.scale.detach().half().clone()
    if not best:
        raise ValueError("No finite trained DSpark candidate")
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        out = Path(tmp) / "draft"
        out.mkdir()
        payload = weights | best
        bank = GPUCheckpoint(device, [0], args.workspace)
        with bank.writer(
            out / "mtp.safetensors",
            {k: (v.shape, v.dtype) for k, v in payload.items()},
            None,
        ) as writer:
            for k, v in payload.items():
                writer.append(k, v)
        bank.export()
        modules = set()
        for name in best:
            if not name.endswith(".weight"):
                continue
            stem = name.removesuffix(".weight")
            for orig, packed in (
                ("q_proj", "qkv_proj"),
                ("k_proj", "qkv_proj"),
                ("v_proj", "qkv_proj"),
                ("k_proj_ctx", "kv_proj_ctx"),
                ("v_proj_ctx", "kv_proj_ctx"),
                ("gate_proj", "gate_up_proj"),
                ("up_proj", "gate_up_proj"),
            ):
                if stem.endswith("." + orig):
                    stem = stem.removesuffix(orig) + packed
                    break
            modules.add(stem)
        raw_config["mtp_config"]["ptqr_w8a8_modules"] = sorted(modules)
        (out / "config.json").write_text(json.dumps(raw_config, indent=2) + "\n")
        (out / "ptqr-training.json").write_text(
            json.dumps(
                dict(
                    seed=args.seed,
                    initial=base,
                    history=history,
                    selected_kl=best_loss,
                    selected_step=min(history, key=lambda row: row["kl_stock"])["step"],
                    steps=args.steps,
                    weight_lr=args.lr,
                    scale_lr=args.scale_lr,
                    dither=args.dither,
                    group_size=128,
                    train_sequences=len(train_files),
                    held_out_sequences=len(held),
                ),
                indent=2,
            )
            + "\n"
        )
        if args.output.exists():
            raise FileExistsError(args.output)
        out.rename(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "samples", "head", "workspace", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=15)
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--scale-lr", type=float, default=1e-7)
    parser.add_argument("--dither", type=float, default=1.05)
    train(parser.parse_args())
