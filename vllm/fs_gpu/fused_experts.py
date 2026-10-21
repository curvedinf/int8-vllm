# SPDX-License-Identifier: Apache-2.0
"""Fused per-layer routed-expert cache for Bolt's packed IQ2 weights.

The per-rank repack cache stores each expert's TP-narrowed byte slice as a
compact block. Those bytes are already exactly the destination bytes of the
fused ``w13_weight``/``w2_weight`` parameters, so we can concatenate the
experts per layer once and load each layer with two contiguous copies instead
of 27,648 per-expert weight_loader calls.

Layout (per rank):
    rankN.experts.bin   [layer0 w13][layer0 w2][layer1 w13]...
    rankN.experts.json  source fingerprint + per-layer offsets/lengths
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any

import numpy as np
import torch

from vllm.fs_gpu.region import GpuStagingPool, RegisteredFile
from vllm.fs_gpu.repack import RepackCache

FUSED_VERSION = 1
# bcache's sequential_cutoff is 4 MiB; larger reads bypass the SSD cache.
_READ_CHUNK = 2 << 20
_EXPERT_RE = re.compile(
    r"\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight$"
)


def fused_paths(root: str, rank: int) -> tuple[str, str]:
    return (
        os.path.join(root, f"rank{rank}.experts.bin"),
        os.path.join(root, f"rank{rank}.experts.json"),
    )


def _expert_destination(
    spec: Any, compact: bytes, key: str, tp_size: int, rank: int
) -> tuple[bytes, int]:
    """Reconstruct the exact TP-narrowed destination from a compact block.

    The repack runs are the source tensor ranges the consumer reads; most
    experts' compact bytes are already the destination, but a small number of
    plans carry merged/overlapping runs. Scattering into the full checkpoint
    shape and narrowing is the same operation the model's weight_loader does,
    so this is correct for both cases.
    """
    full = np.zeros(math.prod(spec.shape), dtype=np.uint8)
    compact_arr = np.frombuffer(compact, dtype=np.uint8)
    cum = 0
    for bo, n in spec.runs:
        bo = int(bo)
        n = int(n)
        full[bo : bo + n] = compact_arr[cum : cum + n]
        cum += n
    full = full.reshape(spec.shape)
    if key == "w13":
        full_i = spec.shape[0]
        local_i = full_i // tp_size
        dest = full[rank * local_i : (rank + 1) * local_i]
    else:
        full_i = spec.shape[1] * 256
        local_i = full_i // tp_size
        # Match PackedG128MoEMethod.weight_loader: the first block comes from
        # the absolute offset, not from the modulo-256 tail.
        begin = rank * local_i
        start = begin % 256
        first = begin // 256
        blocks = (start + local_i + 255) // 256
        dest = full[:, first : first + blocks]
    dest = np.ascontiguousarray(dest)
    return dest.tobytes(), dest.size


def build_fused_cache(
    cache: RepackCache, files: list[str], *, progress: bool = True
) -> dict[str, Any]:
    """Concatenate the per-expert compact blocks into fused layer blocks."""
    if cache.tensors is None:
        raise ValueError("Repack cache must be validated before building fused")
    sources = RepackCache.fingerprint(files)
    groups: dict[int, dict[str, dict[int, Any]]] = {}
    for spec in cache.tensors:
        m = _EXPERT_RE.search(spec.name)
        if m is None:
            continue
        layer, expert, kind = int(m.group(1)), int(m.group(2)), m.group(3)
        key = "w13" if kind == "up_proj" else "w2"
        groups.setdefault(layer, {"w13": {}, "w2": {}})[key][expert] = spec

    bin_path, meta_path = fused_paths(cache.root, cache.rank)
    tmp_bin = f"{bin_path}.tmp-{os.getpid()}"
    src_fd = os.open(cache.bin_path, os.O_RDONLY)
    out_fd = os.open(tmp_bin, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
    layers: list[dict[str, Any]] = []
    off = 0
    try:
        for layer in sorted(groups):
            entry: dict[str, Any] = {"layer": layer}
            for key in ("w13", "w2"):
                specs = groups[layer][key]
                if not specs:
                    raise ValueError(f"layer {layer} missing {key} experts")
                ids = sorted(specs)
                if ids != list(range(len(ids))):
                    raise ValueError(f"layer {layer} {key} expert ids not dense")
                width = None
                start = off
                # The compact blocks of a layer are laid out in expert order;
                # read the whole span once instead of one small pread per
                # expert (27,648 of them made a rebuild take ~4 min/rank).
                span_off = min(s.block_off for s in specs.values())
                span_end = max(s.block_off + s.block_len for s in specs.values())
                span = os.pread(src_fd, span_end - span_off, span_off)
                if len(span) != span_end - span_off:
                    raise OSError(f"short repack span read for layer {layer} {key}")
                for expert in ids:
                    spec = specs[expert]
                    rel = spec.block_off - span_off
                    compact = span[rel : rel + spec.block_len]
                    if len(compact) != spec.block_len:
                        raise OSError(f"short repack read for {spec.name}")
                    dest, dest_len = _expert_destination(
                        spec, compact, key, cache.tp_size, cache.rank
                    )
                    if width is None:
                        width = dest_len
                    elif dest_len != width:
                        raise ValueError(
                            f"layer {layer} {key} narrow widths differ"
                        )
                    view = memoryview(dest)
                    done = 0
                    while done < dest_len:
                        end = min(done + _READ_CHUNK, dest_len)
                        written = os.write(out_fd, view[done:end])
                        if written == 0:
                            raise OSError("fused cache write made no progress")
                        done += written
                    off += dest_len
                entry[f"{key}_off"] = start
                entry[f"{key}_len"] = off - start
                entry[f"{key}_expert"] = width
                entry["experts"] = len(ids)
            layers.append(entry)
        os.fsync(out_fd)
    finally:
        os.close(src_fd)
        os.close(out_fd)
    os.replace(tmp_bin, bin_path)

    meta = {
        "version": FUSED_VERSION,
        "tp_size": cache.tp_size,
        "rank": cache.rank,
        "sources": sources,
        "layers": layers,
    }
    tmp_meta = f"{meta_path}.tmp-{os.getpid()}"
    with open(tmp_meta, "w") as f:
        json.dump(meta, f)
    os.replace(tmp_meta, meta_path)
    if progress:
        print(
            f"fused rank{cache.rank}: {len(layers)} layers, "
            f"{off / 2**30:.2f} GiB",
            flush=True,
        )
    return meta


class FusedExpertCache:
    """Validated fused expert file for one rank."""

    def __init__(self, root: str, tp_size: int, rank: int):
        self.root = root
        self.tp_size = tp_size
        self.rank = rank
        self.bin_path, self.meta_path = fused_paths(root, rank)
        self.layers: dict[int, dict[str, Any]] = {}

    def validate(self, sources: dict[str, list[int]]) -> bool:
        try:
            with open(self.meta_path) as f:
                meta = json.load(f)
            if (
                meta.get("version") != FUSED_VERSION
                or meta.get("tp_size") != self.tp_size
                or meta.get("rank") != self.rank
                or meta.get("sources") != sources
            ):
                return False
            size = os.path.getsize(self.bin_path)
            layers = {}
            for entry in meta.get("layers", []):
                if (
                    entry["w13_off"] + entry["w13_len"] > size
                    or entry["w2_off"] + entry["w2_len"] > size
                ):
                    return False
                layers[int(entry["layer"])] = entry
            self.layers = layers
            return bool(layers)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False

    def load_into(self, model: torch.nn.Module, device: str = "cuda") -> int:
        """Copy each fused layer block into its RoutedExperts parameters."""
        if not self.layers:
            return 0
        max_len = max(
            max(e["w13_len"], e["w2_len"]) for e in self.layers.values()
        )
        pool = GpuStagingPool(max(max_len, 1 << 20), device=device)
        loaded = 0
        started = time.perf_counter()
        try:
            with RegisteredFile(self.bin_path) as rf:
                for name, module in model.named_modules():
                    match = re.search(r"\.layers\.(\d+)\.", name)
                    if match is None or not hasattr(module, "w13_weight"):
                        continue
                    entry = self.layers.get(int(match.group(1)))
                    if entry is None:
                        continue
                    for key, param_name in (("w13", "w13_weight"), ("w2", "w2_weight")):
                        param = getattr(module, param_name)
                        n = entry[f"{key}_len"]
                        if param.numel() * param.element_size() != n:
                            raise ValueError(
                                f"{name}.{param_name} size mismatch with fused cache"
                            )
                        # bcache skips reads over sequential_cutoff (4 MiB) and
                        # serves them from the HDD; issue 2 MiB reads instead.
                        pos = 0
                        while pos < n:
                            cnt = min(_READ_CHUNK, n - pos)
                            rf.ais.read(
                                rf.fh,
                                pool.base + pos,
                                cnt,
                                entry[f"{key}_off"] + pos,
                                0,
                            )
                            pos += cnt
                        block = (
                            pool.view(0, n)
                            .view(param.dtype)
                            .view(param.shape)
                        )
                        param.data.copy_(block)
                    # The next AIS read overwrites this staging pool; wait for
                    # both copies before issuing it.
                    torch.cuda.current_stream().synchronize()
                    loaded += 1
            torch.cuda.current_stream().synchronize()
        finally:
            pool.close()
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "fused expert cache: %d layers in %.2fs",
            loaded,
            time.perf_counter() - started,
        )
        if loaded != len(self.layers):
            raise RuntimeError(
                f"fused expert cache loaded {loaded}/{len(self.layers)} layers"
            )
        return loaded
