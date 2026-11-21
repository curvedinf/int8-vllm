# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibrate and pack complete HY4 Engram tables from pinned Hub source ranges."""

import argparse
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.hy4_archive import archive_gpu_bytes
from scripts.hy4_hub_source import HubDirectSource
from scripts.quantize_hy4_flash import calibration_tokens
from vllm.fs_gpu.direct import DirectStorage
from vllm.fs_gpu.region import ALIGN
from vllm.fs_gpu.safetensor_writer import DirectTensorWriter
from vllm.models.hy_v4_flash.engram_q4 import (
    ROWS_PER_PAGE,
    WIDTH,
    encode_rows,
    n_pages,
    page_rows,
    table_path,
)


def corpus_identity(corpus, samples, length):
    return hashlib.sha256(corpus.read_bytes()).hexdigest() + f":{samples}:{length}"


def calibration_rows(source, config, layer_id, corpus):
    from transformers.models.hy_v4_flash.modeling_hy_v4_flash import (
        HYV4FlashEngramLayer,
        HYV4FlashTextModel,
    )

    with torch.device("meta"):
        model = HYV4FlashEngramLayer(config, layer_id)
    model.lookup_table = source.read(
        f"model.layers.{layer_id - 1}.engram_layer.lookup_table"
    )
    model.head_vocab_sizes = torch.tensor(
        model.config_head_vocab_sizes, device=source.device
    )
    model.hash_multipliers = model.draw_hash_multipliers(
        int(model.lookup_table.max()) + 1
    ).to(source.device)
    offsets = torch.tensor(
        [0] + model.config_head_vocab_sizes[:-1], device=source.device
    ).cumsum(0)
    indices = []
    for ids, _ in corpus:
        ngrams = HYV4FlashTextModel.build_ngram_tokens(
            SimpleNamespace(config=config), ids, None
        )
        indices.append((model.hash_ngrams(ngrams) + offsets).flatten())
    return torch.cat(indices).unique(sorted=True)


def pack(source, name, selected, output, subset, io, workspace, identity, batch_pages):
    spec = source.spec(name)
    nrows, width = spec.shape
    if width != WIDTH or spec.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"Unexpected Engram table shape for {name}")
    pages = torch.empty(
        (n_pages(nrows), ALIGN), dtype=torch.uint8, device=source.device
    )
    sample = torch.empty((len(selected), width), dtype=spec.dtype, device=source.device)
    boundaries = torch.arange(
        0, nrows, batch_pages * ROWS_PER_PAGE, device=source.device
    )
    fences = torch.searchsorted(selected, boundaries).tolist() + [len(selected)]
    nsteps = len(fences) - 1
    print(
        f"{name}: {nrows:,} source rows, {len(selected):,} distinct calibration rows",
        flush=True,
    )
    for step, lo in enumerate(range(0, len(pages), batch_pages)):
        hi = min(lo + batch_pages, len(pages))
        start = lo * ROWS_PER_PAGE
        count = min(nrows - start, (hi - lo) * ROWS_PER_PAGE)
        raw = source.read(name, start=start * width, shape=(count, width))
        a, b = fences[step : step + 2]
        if a < b:
            sample[a:b] = raw[selected[a:b] - start]
        packed = encode_rows(raw)
        dest = page_rows(pages[lo:hi].zero_())
        full, tail = divmod(count, ROWS_PER_PAGE)
        if full:
            dest[:full].copy_(
                packed[: full * ROWS_PER_PAGE].view(full, ROWS_PER_PAGE, -1)
            )
        if tail:
            dest[full, :tail].copy_(packed[-tail:])
        if (step + 1) % max(1, nsteps // 10) == 0:
            print(f"{name}: {step + 1}/{nsteps} packed blocks", flush=True)
    with DirectTensorWriter(
        subset,
        {
            "row_ids": (selected.shape, selected.dtype),
            "source_rows": (sample.shape, sample.dtype),
        },
        io,
        metadata={"source": name, "corpus": identity},
    ) as writer:
        writer.append("row_ids", selected)
        writer.append("source_rows", sample)
    # A complete packed table is resident in VRAM before serialization begins.
    archive_gpu_bytes(pages, output, workspace, io)
    print(
        f"Archived complete GPU-packed table {output.name} ({pages.numel():,} bytes)",
        flush=True,
    )


@torch.inference_mode()
def main():
    from transformers import AutoTokenizer
    from transformers.models.hy_v4_flash.configuration_hy_v4_flash import (
        HYV4FlashConfig,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--length", type=int, default=2304)
    p.add_argument("--hub-repo", default="tencent/Hy4-flash")
    p.add_argument("--revision", required=True)
    p.add_argument("--layer", type=int, action="append")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--batch-pages", type=int, default=512)
    p.add_argument("--window-mib", type=int, default=64)
    args = p.parse_args()
    if min(args.batch_pages, args.samples, args.length) < 1:
        p.error("Counts must be positive")
    os.environ.setdefault("HIPFILE_ALLOW_COMPAT_MODE", "false")
    os.environ.setdefault("HIPFILE_FORCE_COMPAT_MODE", "false")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    config = HYV4FlashConfig.from_pretrained(args.source).text_config
    ids = args.layer or config.engram_layer_ids
    if set(ids) - set(config.engram_layer_ids):
        p.error("Unexpected Engram layer ID")
    corpus = calibration_tokens(
        AutoTokenizer.from_pretrained(args.source),
        args.corpus,
        args.samples,
        args.length,
        device,
    )
    identity = corpus_identity(args.corpus, args.samples, args.length)
    args.output.mkdir(exist_ok=True, parents=True)
    with (
        DirectStorage(device) as io,
        HubDirectSource(
            args.source,
            io,
            args.workspace,
            args.hub_repo,
            args.revision,
            args.window_mib,
        ) as source,
    ):
        for layer in ids:
            name = (
                f"model.layers.{layer - 1}.engram_layer."
                "multi_head_embedding.embedding.weight"
            )
            target = table_path(args.output, layer)
            subset = args.workspace / f"calibration-engram-{layer}.safetensors"
            selected = calibration_rows(source, config, layer, corpus)
            pack(
                source,
                name,
                selected,
                target,
                subset,
                io,
                args.workspace,
                identity,
                args.batch_pages,
            )


if __name__ == "__main__":
    main()
