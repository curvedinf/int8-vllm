# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache TP-local packed-model parameters before kernel-specific GPU packing."""

import hashlib
import json
import time
from pathlib import Path

import torch

from vllm import envs
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import get_tp_group
from vllm.fs_gpu.direct import DirectStorage
from vllm.fs_gpu.region import RegisteredFile
from vllm.fs_gpu.repack import RepackCache, default_repack_root
from vllm.fs_gpu.safetensor_writer import DirectTensorWriter
from vllm.fs_gpu.safetensors import SafetensorFile
from vllm.logger import init_logger

logger = init_logger(__name__)
_TOKEN_MAP = "__hy4_token_map__"


class HY4RankCache:
    def __init__(self, model, directory, recipe="hy4"):
        self.group = get_tp_group()
        self.params = dict(model.named_parameters())
        self.hash_state = (
            model.language_model.model.engram_hash if recipe == "hy4" else None
        )
        self.device = next(iter(self.params.values())).device
        index = directory / "model.safetensors.index.json"
        names = set(json.loads(index.read_text())["weight_map"].values())
        sources = RepackCache.fingerprint(
            [str(directory / name) for name in names]
            + [str(index), str(directory / "config.json")]
        )
        # The cache holds the result of weight_loader, before kernel packing.
        # Bump the mapping version if the loader's TP partitioning changes.
        signature = [
            1,
            str(directory.resolve()),
            self.group.world_size,
            self.group.rank_in_group,
            sources,
        ]
        self.digest = hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode()
        ).hexdigest()
        root = Path(default_repack_root(str(directory)))
        name = f"{recipe}-tp{self.group.world_size}-rank{self.group.rank_in_group}"
        if recipe == "inkling2":
            # Joint Q/gate projections and text-only serving have distinct
            # parameter layouts even when they read the same checkpoint.
            layout = [(key, list(p.shape), str(p.dtype)) for key, p in self.params.items()]
            name += "-" + hashlib.sha256(json.dumps(layout).encode()).hexdigest()[:12]
        self.path = root / (name + ".safetensors")

    @classmethod
    def for_model(cls, model, model_config):
        config = get_current_vllm_config_or_none()
        recipes = {
            "HYV4FlashVLForConditionalGeneration": "hy4",
            "Inkling2LoopForConditionalGeneration": "inkling2",
        }
        recipe = recipes.get(type(model).__name__)
        if (
            config is None
            or recipe is None
            or not config.additional_config.get(
                f"{recipe}_rank_cache", recipe == "inkling2"
            )
        ):
            return None
        parallel = config.parallel_config
        if (
            parallel.tensor_parallel_size <= 1
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.enable_expert_parallel
            or model_config.get_multimodal_config().mm_encoder_tp_mode != "data"
        ):
            return None
        directory = Path(model_config.model_weights or model_config.model)
        if not directory.is_dir():
            return None
        return cls(model, directory, recipe)

    def _all(self, ready):
        flag = torch.tensor(int(ready), dtype=torch.int32, device="cpu")
        torch.distributed.all_reduce(
            flag, op=torch.distributed.ReduceOp.MIN, group=self.group.cpu_group
        )
        return bool(flag.item())

    def _validate(self):
        if not self.path.is_file():
            return None
        header = SafetensorFile(str(self.path))
        if header.metadata.get("source") != self.digest:
            return None
        expected = set(self.params)
        if self.hash_state is not None:
            expected.add(_TOKEN_MAP)
        if set(header.tensors) != expected:
            return None
        end = 0
        for spec in sorted(header, key=lambda item: item.start):
            if spec.start != end:
                return None
            if spec.name == _TOKEN_MAP:
                if (
                    spec.dtype != torch.int32
                    or len(spec.shape) != 1
                    or not 0 < spec.shape[0] <= self.hash_state.layout.vocab_size
                ):
                    return None
            else:
                param = self.params[spec.name]
                if (
                    spec.shape != tuple(param.shape)
                    or spec.dtype != param.dtype
                    or spec.nbytes != param.nbytes
                ):
                    return None
            end = spec.end
        if self.path.stat().st_size != header.data_begin + end:
            return None
        return header

    def restore(self):
        started = time.monotonic()
        try:
            header = self._validate()
        except (OSError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Ignoring unreadable TP-local weight cache %s: %s", self.path, exc
            )
            header = None
        # The original loader uses TP collectives. All ranks must make the
        # same decision when a cache file is absent, stale or unreadable.
        if not self._all(header is not None):
            return False
        # Coalesce small tensors regardless of the underlying IO size. Layered
        # SSD caches can bypass their fast device for large sequential reads.
        window_bytes = 16 << 20
        transfer_bytes = min(window_bytes, envs.VLLM_AIS_REPACK_WINDOW_MB << 20)
        try:
            with (
                DirectStorage(
                    self.device, transfer_bytes, require_fastpath=False
                ) as io,
                RegisteredFile(str(self.path), direct=True) as file,
            ):
                # Coalesce consecutive scales, biases and norms on the disk.
                # An individual expert tensor can be larger than this GPU window.
                window = torch.empty(
                    window_bytes, dtype=torch.uint8, device=self.device
                )
                window_start = window_end = 0
                for spec in sorted(header, key=lambda value: value.start):
                    if spec.name == _TOKEN_MAP:
                        lookup = torch.empty(
                            spec.shape, dtype=spec.dtype, device=self.device
                        )
                        dest = lookup
                    else:
                        dest = self.params[spec.name].data
                    target = (
                        dest
                        if dest.is_contiguous()
                        else torch.empty_like(
                            dest, memory_format=torch.contiguous_format
                        )
                    )
                    if spec.nbytes >= window_bytes:
                        io.read(file, header.data_begin + spec.start, target)
                    else:
                        if not (window_start <= spec.start and spec.end <= window_end):
                            window_start = spec.start
                            window_end = min(
                                window_start + window_bytes,
                                file.size - header.data_begin,
                            )
                            io.read(
                                file,
                                header.data_begin + window_start,
                                window[: window_end - window_start],
                            )
                        target.reshape(-1).view(torch.uint8).copy_(
                            window[spec.start - window_start : spec.end - window_start]
                        )
                    if target is not dest:
                        dest.copy_(target)
            read = True
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Could not restore TP-local weight cache %s: %s", self.path, exc
            )
            read = False
        if not self._all(read):
            return False
        if self.hash_state is not None:
            self.hash_state.load_lookup(lookup)
        logger.info(
            "TP-local weight cache restored %d parameters in %.2f seconds "
            "from %s (%d MiB per read)",
            len(self.params),
            time.monotonic() - started,
            self.path,
            transfer_bytes >> 20,
        )
        return True

    def save(self):
        tensors = {name: value.detach() for name, value in self.params.items()}
        if self.hash_state is not None:
            count = len(self.hash_state.layout.lookup_table)
            tensors[_TOKEN_MAP] = self.hash_state.token_map[:count]
        declaration = {name: (tuple(t.shape), t.dtype) for name, t in tensors.items()}
        started = time.monotonic()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with (
                DirectStorage(self.device, 16 << 20, require_fastpath=False) as io,
                DirectTensorWriter(
                    self.path,
                    declaration,
                    io,
                    metadata={"format": "pt", "source": self.digest},
                    buffer_bytes=io.window,
                ) as writer,
            ):
                for name, value in tensors.items():
                    writer.append(name, value.contiguous())
            logger.info(
                "Wrote TP-local weight cache in %.2f seconds to %s",
                time.monotonic() - started,
                self.path,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Could not save optional TP-local weight cache %s: %s", self.path, exc
            )
