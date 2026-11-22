# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect Inkling PTQR inputs from disjoint training and validation documents."""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from scripts.inkling2_teacher import load_teacher


def quantized_projection(name, module):
    if not isinstance(module, nn.Linear):
        return False
    if name.startswith("model.vision_tower."):
        return module.in_features % 128 == 0 and module.out_features >= 128
    return name == "lm_head" or name.endswith(
        (
            "q_proj.linear",
            "k_proj.linear",
            "v_proj.linear",
            "o_proj.linear",
            "o_proj.gate_proj",
        )
    )


class InputSamples:
    """Uniform sample of token rows across documents and all visits to tied layers."""

    def __init__(self, seed, dense_rows, routed_rows):
        self.rng = torch.Generator().manual_seed(seed)
        self.dense_rows = dense_rows
        self.routed_rows = routed_rows
        self.rows = {}

    def observe(self, name, inputs, *, experts=False):
        x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
        limit = self.routed_rows if experts else self.dense_rows
        fields = [x]
        if experts:
            fields.append(inputs[1].reshape(len(x), -1))
        if name not in self.rows:
            self.rows[name] = (
                torch.full((limit,), -1.0, dtype=torch.float64),
                *(torch.empty((limit, *v.shape[1:]), dtype=v.dtype) for v in fields),
            )
        priorities, *buffers = self.rows[name]
        ranks = torch.rand(len(x), generator=self.rng, dtype=torch.float64)
        selected = torch.cat((priorities, ranks)).topk(limit).indices
        fresh = selected[selected >= limit] - limit
        if not fresh.numel():
            return
        # Overwrite evicted slots in place. Once full, a reservoir often accepts
        # just one or two rows from a document.
        replace = torch.ones(limit, dtype=torch.bool)
        replace[selected[selected < limit]] = False
        slots = replace.nonzero().flatten()
        priorities[slots] = ranks[fresh]
        for dest, values in zip(buffers, fields):
            captured = values.index_select(0, fresh.to(values.device)).cpu()
            dest.index_copy_(0, slots, captured)


@torch.inference_mode()
def collect(model, processor, corpus, images, split, args):
    samples = InputSamples(args.seed, args.dense_rows, args.routed_rows)
    handles = []
    expected = set()
    for name, module in model.named_modules():
        experts = name.endswith(".mlp.experts")
        if name == "lm_head":
            final_norm = model.model.language_model.norm

            def record_head(mod, values, output):
                samples.observe("lm_head", (output.to(model.lm_head.weight.dtype),))

            handles.append(final_norm.register_forward_hook(record_head))
            expected.add(name)
            continue
        if experts or quantized_projection(name, module):

            def record(mod, values, *, key=name, routed=experts):
                samples.observe(key, values, experts=routed)

            handles.append(module.register_forward_pre_hook(record))
            expected.add(name)

    try:
        for i, text in enumerate(corpus, 1):
            batch = processor.tokenizer(
                text,
                max_length=args.length,
                truncation=True,
                return_tensors="pt",
            )
            model(
                input_ids=batch.input_ids.to("cuda:0"),
                use_cache=False,
                logits_to_keep=1,
            )
            if i == 1 or i % 16 == 0:
                print(f"{split}: sampled text {i}/{len(corpus)}", flush=True)
        for i, path in enumerate(images, 1):
            with Image.open(path) as im:
                image = im.convert("RGB")
                image.thumbnail((640, 640))
                batch = processor(
                    text=processor.image_token + " Describe this image.",
                    images=image,
                    return_tensors="pt",
                ).to("cuda:0")
            model(**batch, use_cache=False, logits_to_keep=1)
            if i == 1 or i % 16 == 0:
                print(f"{split}: sampled image {i}/{len(images)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    missing = expected - samples.rows.keys()
    if missing:
        raise ValueError(f"Missing {split} activations for {sorted(missing)}")
    directory = args.samples / split
    directory.mkdir(parents=True, exist_ok=True)
    for name, (priorities, *tensors) in samples.rows.items():
        filled = priorities >= 0
        torch.save([t[filled] for t in tensors], directory / (name + ".pt"))
    print(
        f"{split}: {len(samples.rows)} modules, "
        f"{len(corpus)} texts, {len(images)} images",
        flush=True,
    )


def add_sampling_options(parser):
    parser.add_argument("--train-texts", type=int, default=96)
    parser.add_argument("--eval-texts", type=int, default=24)
    parser.add_argument("--train-images", type=int, default=48)
    parser.add_argument("--eval-images", type=int, default=12)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--dense-rows", type=int, default=768)
    parser.add_argument("--routed-rows", type=int, default=8192)


def collect_splits(model, args):
    from transformers import AutoProcessor

    corpus = list(
        dict.fromkeys(
            json.loads(line)["text"] for line in args.corpus.read_text().splitlines()
        )
    )
    images = sorted(
        p
        for p in args.images.rglob("*")
        if p.suffix.lower() in (".jpg", ".png", ".jpeg")
    )
    if (
        len(corpus) < args.train_texts + args.eval_texts
        or len(images) < args.train_images + args.eval_images
    ):
        raise ValueError("Insufficient distinct examples for the requested splits")
    processor = AutoProcessor.from_pretrained(args.source, local_files_only=True)
    for split, texts, paths in (
        ("train", corpus[: args.train_texts], images[: args.train_images]),
        (
            "eval",
            corpus[args.train_texts : args.train_texts + args.eval_texts],
            images[args.train_images : args.train_images + args.eval_images],
        ),
    ):
        collect(model, processor, texts, paths, split, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "corpus", "images", "samples"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--seed", type=int, default=721)
    add_sampling_options(parser)
    args = parser.parse_args()
    os.environ.setdefault("HIPFILE_ALLOW_COMPAT_MODE", "false")
    os.environ.setdefault("HIPFILE_FORCE_COMPAT_MODE", "false")
    collect_splits(load_teacher(args.source), args)


if __name__ == "__main__":
    main()
