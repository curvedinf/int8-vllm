# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash frozen n-gram embeddings with TP head sharding and V2 lookback."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.common.engram import (
    DEAD_ID,
    NgramHashState,
    _hash_ids_kernel,
)
from vllm.models.qwen4_exp.common.ngram_embedding import (
    _lookup_ple_embedding_from_pinned_kernel,
)
from vllm.triton_utils import triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


class FlashEngramLayout:
    def __init__(self, config):
        self.layer_ids = tuple(config.engram_layer_ids or ())
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_heads = config.engram_n_head_per_ngram
        self.head_dim = config.engram_n_embed_per_ngram // self.n_heads
        self.pad_token_id = config.engram_pad_id
        self.vocab_size = config.vocab_size
        self.seed = config.engram_seed
        self.lookup_table = None
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("engram_pad_id is outside the vocabulary")
        groups = config.engram_layer_head_vocab_sizes
        self.primes = tuple(
            groups.get(str(layer), groups.get(layer)) for layer in self.layer_ids
        )
        flat = [[p for order in group for p in order] for group in self.primes]
        self.offsets = np.array(
            [np.cumsum([0, *sizes[:-1]]) for sizes in flat], dtype=np.int64
        )
        if any(sum(sizes) > np.iinfo(np.int32).max for sizes in flat):
            raise ValueError("Engram flattened bucket offsets must fit int32")

    def load_lookup(self, value):
        if (
            value.dtype not in (torch.int32, torch.int64)
            or value.ndim != 1
            or not 0 < value.numel() <= self.vocab_size
        ):
            raise ValueError("Invalid frozen Engram token map")
        lookup = value.detach().cpu().numpy()
        if np.any(lookup < 0) or np.any(lookup > np.iinfo(np.int32).max):
            raise ValueError("Invalid frozen Engram token map")
        if self.lookup_table is not None:
            if not np.array_equal(self.lookup_table, lookup):
                raise ValueError("Engram layers must share the same token map")
            return False
        self.compressed_vocab_size = int(lookup.max()) + 1
        self.lookup_table = lookup.copy()
        self.token_map = lookup.tolist()
        # Padded vocabulary entries have no normalized ID in Tencent's table.
        self.token_map.extend(
            [self.token_map[self.pad_token_id]]
            * (self.vocab_size - len(self.token_map))
        )
        bound = max(1, (np.iinfo(np.int64).max // self.compressed_vocab_size) // 2)
        self.multipliers = np.stack(
            [
                np.random.default_rng(self.seed + 10007 * layer).integers(
                    0, bound, size=self.max_ngram_size, dtype=np.int64
                )
                * 2
                + 1
                for layer in self.layer_ids
            ]
        )

        return True


def head_range(sizes, tp, rank):
    if not 1 <= tp <= len(sizes) or not 0 <= rank < tp:
        raise ValueError("Engram requires at least one complete hash head per TP rank")
    prefix = np.cumsum([0, *sizes]).tolist()
    boundaries = [0]
    for r in range(1, tp):
        target = prefix[-1] * r / tp
        candidates = range(boundaries[-1] + 1, len(sizes) - (tp - r) + 1)
        boundaries.append(min(candidates, key=lambda i: abs(prefix[i] - target)))
    boundaries.append(len(sizes))
    return boundaries[rank], boundaries[rank + 1]


class FlashNgramHashState(NgramHashState):
    def __init__(self, config, *, dummy=False):
        nn.Module.__init__(self)
        layout = FlashEngramLayout(config)
        self.layout: FlashEngramLayout = layout  # type: ignore[assignment]
        self.lookback_depth = self.layout.max_ngram_size - 1
        self.block_size = 1
        self.use_slot_cache = False
        self._cache = None
        self.pad_id = 0
        self.visual_token_ids: tuple[int, ...] = ()
        self.register_buffer(
            "token_map",
            torch.zeros(config.vocab_size, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "multipliers",
            torch.zeros(
                len(self.layout.layer_ids),
                self.layout.max_ngram_size,
                dtype=torch.int64,
            ),
            persistent=False,
        )
        for name, values in (
            ("primes", self.layout.primes),
            ("offsets", self.layout.offsets),
        ):
            self.register_buffer(
                name, torch.tensor(values, dtype=torch.int64), persistent=False
            )
        if dummy:
            self.load_lookup(torch.arange(config.vocab_size, device="cpu"))

    def load_lookup(self, value):
        from vllm.fs_gpu.recorder import record_full_read

        if record_full_read(value):
            return
        if self.layout.load_lookup(value):
            self.token_map.copy_(
                torch.as_tensor(
                    self.layout.token_map,
                    dtype=torch.int32,
                    device=self.token_map.device,
                )
            )
            self.multipliers.copy_(
                torch.as_tensor(self.layout.multipliers, device=self.multipliers.device)
            )
            self.pad_id = self.layout.token_map[self.layout.pad_token_id]

    @eager_break_during_capture
    def prepare_hashes_with_output(self, input_ids, lookback, layer_name, output):
        metadata = get_forward_context().attn_metadata
        output.copy_(
            self.prepare_hashes(
                input_ids,
                lookback,
                metadata[layer_name] if isinstance(metadata, dict) else None,
            )
        )

    def prepare_hashes_into(self, input_ids, lookback, metadata, output):
        """Fill a possibly strided layer-major view in one V2 hash launch."""
        layers, tokens, columns = output.shape
        if (
            not tokens
            or not metadata.num_actual_tokens
            or not metadata.seq_lens.numel()
        ):
            return output.fill_(-1)
        if self.layout.lookup_table is None:
            raise ValueError("Missing checkpoint Engram token map")
        if lookback is None or lookback.shape[1] != self.lookback_depth:
            raise ValueError("Flash Engram requires V2 lookback_token_ids")
        heads = self.primes.shape[-1]
        if output.stride(1) != columns or output.stride(2) != 1:
            raise ValueError("Flash hash rows must be contiguous")
        starts = metadata.query_start_loc
        _hash_ids_kernel[(triton.cdiv(tokens, 32), layers)](
            input_ids,
            self.token_map,
            None,
            None,
            None,
            starts,
            self.multipliers,
            self.primes,
            self.offsets,
            None,
            lookback,
            None,
            output,
            tokens,
            0,
            self.pad_id,
            input_stride=input_ids.stride(0),
            mask_stride=0,
            position_stride=0,
            table_stride=0,
            table_col_stride=0,
            query_stride=starts.stride(0),
            num_query_rows=metadata.seq_lens.numel(),
            num_table_rows=0,
            max_blocks=0,
            cache_block_size=1,
            MAX_NGRAM=self.layout.max_ngram_size,
            num_heads=heads,
            BLOCK_T=32,
            BLOCK_H=triton.next_power_of_2(heads),
            dead_id=DEAD_ID,
            lookback_depth=self.lookback_depth,
            lookback_row_stride=lookback.stride(0),
            lookback_col_stride=lookback.stride(1),
            lookback_mask_row_stride=0,
            lookback_mask_col_stride=0,
            seq_lens=metadata.seq_lens,
            actual_tokens=metadata.num_actual_tokens,
            layer_stride=output.stride(0),
            VOCAB=self.token_map.numel(),
            SUPPRESSED=(self.layout.pad_token_id, *self.visual_token_ids),
            num_warps=4,
        )
        return output

    def prepare_hashes(self, input_ids, lookback, metadata):
        tokens = input_ids.numel()
        if (
            metadata is None
            or tokens == 0
            or metadata.num_actual_tokens == 0
            or metadata.seq_lens.numel() == 0
        ):
            return input_ids.new_full(
                (len(self.layout.layer_ids), tokens, self.offsets.shape[1]),
                -1,
                dtype=torch.int32,
            )
        if self.layout.lookup_table is None:
            raise ValueError("Missing checkpoint Engram token map")
        if lookback is None or lookback.shape[1] != self.lookback_depth:
            raise ValueError("Flash Engram requires V2 lookback_token_ids")
        starts = metadata.query_start_loc
        rows = torch.arange(tokens, device=input_ids.device)
        requests = torch.searchsorted(starts[1:], rows, right=True)
        requests = requests.clamp_max(starts.numel() - 2).long()
        logical = (
            metadata.seq_lens[requests]
            - (starts[requests + 1] - starts[requests])
            + rows
            - starts[requests]
        ).long()
        dead = (input_ids < 0) | (input_ids >= self.token_map.numel())
        safe_ids = input_ids.clamp(0, self.token_map.numel() - 1)
        lookback_dead = lookback >= self.token_map.numel()
        safe_lookback = lookback.clamp(-1, self.token_map.numel() - 1)
        hashes = super().forward(
            safe_ids,
            logical,
            starts,
            dead,
            safe_lookback,
            lookback_dead,
            slot_mapping=None,
            block_table=None,
        )
        offsets = self.offsets.unsqueeze(0)
        primes = self.primes.flatten(1).unsqueeze(0)
        hashes = torch.where(hashes < offsets, hashes + primes, hashes).int()
        no_hash = dead | (rows >= metadata.num_actual_tokens) | (rows >= starts[-1])
        no_hash |= input_ids == self.layout.pad_token_id
        for token_id in self.visual_token_ids:
            no_hash |= input_ids == token_id
        hashes.masked_fill_(no_hash[:, None, None], -1)
        return hashes.permute(1, 0, 2).contiguous()


class FlashEmbedding(nn.Module):
    def __init__(self, sizes, dim, dtype, cpu_offload):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.head_start, self.head_end = head_range(
            sizes, self.tp_size, get_tensor_model_parallel_rank()
        )
        self.vocab_start = sum(sizes[: self.head_start])
        self.vocab_end = sum(sizes[: self.head_end])
        self.num_embeddings = sum(sizes)
        self.n_hash_cols = len(sizes)
        self.dim = dim
        self.cpu_offload = cpu_offload
        if cpu_offload and not is_uva_available():
            raise RuntimeError("Flash Engram CPU offload requires UVA")
        rows = self.vocab_end - self.vocab_start
        storage = {"device": "cpu", "pin_memory": True} if cpu_offload else {}
        self.embedding = nn.Module()
        self.embedding.weight = nn.Parameter(
            torch.empty(
                (0,) if dtype is None else (rows, dim),
                dtype=dtype or torch.bfloat16,
                **storage,
            ),
            requires_grad=False,
        )
        self.fp8 = dtype == torch.float8_e4m3fn
        self.embedding.register_parameter("weight_scale_inv", None)
        if self.fp8:
            self.embedding.weight_scale_inv = nn.Parameter(
                torch.empty(rows, dtype=torch.bfloat16, **storage), requires_grad=False
            )
        for parameter in self.embedding.parameters():
            set_weight_attrs(parameter, {"weight_loader": self._load_shard})
        self._uva_weight = None
        self._uva_scale = None
        logger.info(
            "Flash Engram TP table: %d rows x %d, dtype=%s, cpu_offload=%s",
            rows,
            dim,
            dtype,
            cpu_offload,
        )

    def load_weight(self, leaf, value):
        if leaf == "weight_scale_inv" and self.embedding.weight_scale_inv is None:
            self.embedding.weight_scale_inv = nn.Parameter(
                self.embedding.weight.new_empty(0, dtype=torch.bfloat16),
                requires_grad=False,
            )
            set_weight_attrs(
                self.embedding.weight_scale_inv, {"weight_loader": self._load_shard}
            )
        self._load_shard(getattr(self.embedding, leaf), value)

    def _load_shard(self, parameter, value):
        is_weight = parameter is self.embedding.weight
        expected = (
            (self.num_embeddings, self.dim) if is_weight else (self.num_embeddings,)
        )
        dtypes = (
            (torch.bfloat16, torch.float8_e4m3fn) if is_weight else (torch.bfloat16,)
        )
        if tuple(value.shape) != expected or value.dtype not in dtypes:
            raise ValueError(
                f"Invalid Engram table: {tuple(value.shape)}/{value.dtype}"
            )
        if parameter.numel() and parameter.dtype != value.dtype:
            raise ValueError("Engram table dtype changed during weight reload")
        if is_weight:
            if (
                value.dtype != torch.float8_e4m3fn
                and self.embedding.weight_scale_inv is not None
            ):
                raise ValueError("BF16 Engram table must not have FP8 row scales")
        elif self.embedding.weight.numel() and not self.fp8:
            raise ValueError("BF16 Engram table must not have FP8 row scales")
        if parameter.numel() == 0:
            parameter.data = torch.empty(
                (self.vocab_end - self.vocab_start, *expected[1:]),
                dtype=value.dtype,
                device=parameter.device,
                pin_memory=self.cpu_offload,
            )
        default_weight_loader(parameter, value[self.vocab_start : self.vocab_end])
        if is_weight:
            self.fp8 = value.dtype == torch.float8_e4m3fn
        if is_weight and self.fp8 and self.embedding.weight_scale_inv is None:
            self.embedding.weight_scale_inv = nn.Parameter(
                parameter.new_empty(0, dtype=torch.bfloat16), requires_grad=False
            )
            set_weight_attrs(
                self.embedding.weight_scale_inv, {"weight_loader": self._load_shard}
            )
        self._uva_weight = None
        self._uva_scale = None

    def _lookup(self, parameter, indices, out, *, scale=False):
        view_name = "_uva_scale" if scale else "_uva_weight"
        view = getattr(self, view_name)
        if view is None:
            view = (
                get_accelerator_view_from_cpu_tensor(parameter)
                if self.cpu_offload
                else parameter
            )
            setattr(self, view_name, view)
        row_bytes = parameter.element_size() * (1 if scale else self.dim)
        _lookup_ple_embedding_from_pinned_kernel[(indices.numel(),)](
            view.view(torch.uint8),
            indices,
            out.view(torch.uint8),
            row_bytes,
            self.vocab_start,
            self.vocab_end,
            BLOCK_D=triton.next_power_of_2(row_bytes),
        )

    def forward(self, hashes):
        indices = hashes[:, self.head_start : self.head_end].long().contiguous()
        shape = (*indices.shape, self.dim)
        rows = torch.empty(
            shape, dtype=self.embedding.weight.dtype, device=hashes.device
        )
        if indices.numel() == 0:
            return rows.float()
        self._lookup(self.embedding.weight, indices, rows)
        if self.fp8:
            scales = torch.empty(
                indices.shape, dtype=torch.bfloat16, device=hashes.device
            )
            self._lookup(self.embedding.weight_scale_inv, indices, scales, scale=True)
            return rows.float() * scales.float().unsqueeze(-1)
        return rows.float()


class FlashPackedEmbedding(nn.Module):
    fp8 = False

    def __init__(
        self,
        sizes,
        dim,
        directory,
        layer_id,
        storage,
        row_threshold,
        cache_rows,
        prefetch_pages,
    ):
        super().__init__()
        from vllm.models.hy_v4_flash.engram_q4 import (
            WIDTH,
            PackedEngramTable,
            table_path,
        )

        if dim != WIDTH:
            raise ValueError(f"Packed Engram requires head_dim={WIDTH}")
        self.tp_size = get_tensor_model_parallel_world_size()
        self.head_start, self.head_end = head_range(
            sizes, self.tp_size, get_tensor_model_parallel_rank()
        )
        self.n_hash_cols = len(sizes)
        self.table = PackedEngramTable(
            table_path(Path(directory), layer_id),
            sum(sizes),
            sum(sizes[: self.head_start]),
            sum(sizes[: self.head_end]),
            storage.packed_table,
            storage.page_cache_mib,
            torch.device("cuda", torch.cuda.current_device()),
            row_threshold=row_threshold,
            cache_rows=cache_rows,
            prefetch_pages=prefetch_pages,
        )

    def load_weight(self, leaf, value):
        raise ValueError("Packed Engram checkpoint also contains an unpacked embedding")

    def forward(self, hashes):
        return self.table.lookup(hashes[:, self.head_start : self.head_end]).float()


class FlashEngram(nn.Module):
    def __init__(self, vllm_config, layout, layer_id, prefix):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        index = layout.layer_ids.index(layer_id)
        sizes = [p for order in layout.primes[index] for p in order]
        storage = vllm_config.engram_config
        if storage.embedding_across_dp or storage.dp_shared_memory:
            raise ValueError(
                "Flash Engram requires private DP replicas with TP-local tables"
            )
        dtype = None
        if vllm_config.load_config.load_format == "dummy":
            dtype = (
                torch.float8_e4m3fn if config.engram_fp8_embedding else torch.bfloat16
            )
        if storage.packed_table:
            directory = (
                storage.table_directory or vllm_config.model_config.model_weights
            )
            self.multi_head_embedding = FlashPackedEmbedding(
                sizes,
                layout.head_dim,
                directory,
                layer_id,
                storage,
                int(vllm_config.additional_config.get("hy4_engram_row_threshold", 256)),
                bool(vllm_config.additional_config.get("hy4_engram_cache_rows", False)),
                int(vllm_config.additional_config.get("hy4_engram_prefetch_pages", 0)),
            )
        else:
            self.multi_head_embedding = FlashEmbedding(
                sizes, layout.head_dim, dtype, storage.cpu_offload
            )
        self.hidden_size, self.hc_mult = config.hidden_size, config.hc_mult
        self.pack_projections = bool(
            storage.packed_table
            and vllm_config.additional_config.get("hy4_engram_fused_proj", True)
        )
        self.register_buffer("projection_weight", None, persistent=False)
        self.register_buffer("projection_bias", None, persistent=False)
        local = (
            self.multi_head_embedding.head_end - self.multi_head_embedding.head_start
        )
        self.fly_projection = bool(
            self.pack_projections
            and local == 4
            and layout.head_dim == 96
            and self.hidden_size * (self.hc_mult + 1) % 128 == 0
            and vllm_config.additional_config.get("hy4_flydsl_engram", False)
        )
        self.full_projection_min = int(
            vllm_config.additional_config.get("hy4_engram_full_proj_min", 0)
        )
        if self.full_projection_min < 0:
            raise ValueError("hy4_engram_full_proj_min cannot be negative")
        # Every rank must enter the same initialization collectives.
        self.replicate_prefill = (
            self.full_projection_min > 0
            and self.pack_projections
            and self.multi_head_embedding.tp_size == 4
            and len(sizes) == 16
            and all(head_range(sizes, 4, r) == (4 * r, 4 * r + 4) for r in range(4))
        )
        self.register_buffer("full_projection_weight", None, persistent=False)
        self.register_buffer("full_projection_bias", None, persistent=False)
        self.fused_residual = bool(
            self.pack_projections
            and self.hidden_size == 3072
            and self.hc_mult == 4
            and vllm_config.additional_config.get("hy4_engram_fused_residual", False)
        )
        for name in ("key_norm_weight", "query_norm_weight", "residual_scale"):
            self.register_buffer(name, None, persistent=False)
        self.register_buffer("wave_weight", None, persistent=False)
        self.register_buffer("wave_gamma", None, persistent=False)
        self.lookup_per_head_norms = nn.ModuleList(
            [nn.RMSNorm(layout.head_dim) for _ in range(local)]
        )
        self.value_proj = nn.Linear(local * layout.head_dim, self.hidden_size)
        self.key_projs = nn.ModuleList(
            [
                nn.Linear(local * layout.head_dim, self.hidden_size)
                for _ in range(self.hc_mult)
            ]
        )
        self.norm_key = nn.ModuleList(
            [nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)]
        )
        self.norm_query = nn.ModuleList(
            [nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)]
        )
        self.unified_pre_gamma = nn.Parameter(torch.empty(self.hidden_size))
        self.unified_per_hc_scales = nn.ParameterList(
            [nn.Parameter(torch.empty(self.hidden_size)) for _ in range(self.hc_mult)]
        )
        self.total_width = len(sizes) * layout.head_dim
        self.start = self.multi_head_embedding.head_start * layout.head_dim
        self.stop = self.multi_head_embedding.head_end * layout.head_dim
        self.rank = get_tensor_model_parallel_rank()
        for projection in [self.value_proj, *self.key_projs]:
            set_weight_attrs(
                projection.weight, {"weight_loader": self._load_projection}
            )
            set_weight_attrs(projection.bias, {"weight_loader": self._load_bias})

    def _load_projection(self, parameter, value):
        if value.shape != (self.hidden_size, self.total_width):
            raise ValueError("Flash Engram projection weight shape mismatch")
        default_weight_loader(parameter, value[:, self.start : self.stop])

    def _load_bias(self, parameter, value):
        default_weight_loader(
            parameter, value if self.rank == 0 else torch.zeros_like(value)
        )

    def prepare_projections(self):
        if not self.pack_projections:
            return
        projections = [*self.key_projs, self.value_proj]
        self.projection_weight = torch.cat([op.weight.detach() for op in projections])
        self.projection_bias = torch.cat([op.bias.detach() for op in projections])
        # The original names and loaders remain usable for a subsequent reload.
        for i, op in enumerate(projections):
            segment = slice(i * self.hidden_size, (i + 1) * self.hidden_size)
            op.weight.data = self.projection_weight[segment]
            op.bias.data = self.projection_bias[segment]

        if not (self.fly_projection or self.fused_residual or self.replicate_prefill):
            return
        from vllm.models.hy_v4_flash.rocm_attention import is_mi100

        if self.projection_weight.dtype != torch.float16 or not is_mi100():
            return
        self.wave_gamma = torch.stack(
            [norm.weight.detach() for norm in self.lookup_per_head_norms]
        )
        if self.fly_projection:
            from aiter.ops.flydsl.kernels.gfx908_engram_projection import (
                pack_projection,
            )

            self.wave_weight = pack_projection(self.projection_weight)
        if self.fused_residual:
            self.key_norm_weight = torch.stack(
                [n.weight.detach() for n in self.norm_key]
            )
            self.query_norm_weight = torch.stack(
                [n.weight.detach() for n in self.norm_query]
            )
            self.residual_scale = torch.stack(
                [s.detach() for s in self.unified_per_hc_scales]
            )
        if self.replicate_prefill:
            self.full_projection_weight = tensor_model_parallel_all_gather(
                self.projection_weight, dim=-1
            )
            self.full_projection_bias = tensor_model_parallel_all_reduce(
                self.projection_bias.clone()
            )

    def _project_rows(self, rows, hashes, dtype, full):
        if (
            not full
            and self.fly_projection
            and self.wave_weight is not None
            and dtype == torch.float16
            and rows.dtype in (torch.float16, torch.float32)
            and rows.shape[0] <= 96
        ):
            return torch.ops.aiter.gfx908_engram_projection(
                rows, hashes, self.wave_gamma, self.wave_weight, self.projection_bias
            )
        gamma = self.wave_gamma
        if gamma is None:
            gamma = torch.stack([norm.weight for norm in self.lookup_per_head_norms])
        gamma = gamma.float()
        x = rows.float().masked_fill(hashes.unsqueeze(-1) < 0, 0)
        eps = torch.finfo(torch.float32).eps
        x = (
            (x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * gamma)
            .to(dtype)
            .flatten(1)
        )
        if full:
            # At large M, gather 1,536 normalized features instead of reducing
            # 15,360 projected features. Each GPU then computes the whole result.
            x = tensor_model_parallel_all_gather(x, dim=-1)
            return torch.nn.functional.linear(
                x, self.full_projection_weight, self.full_projection_bias
            )
        if self.projection_weight is not None:
            return torch.nn.functional.linear(
                x, self.projection_weight, self.projection_bias
            )
        return [op(x) for op in (*self.key_projs, self.value_proj)]

    def forward(self, hidden, hashes, rows=None):
        local = self.multi_head_embedding
        rows = local(hashes) if rows is None else rows
        local_hashes = hashes[:, local.head_start : local.head_end]
        full = (
            self.full_projection_weight is not None
            and self.full_projection_min > 0
            and hidden.dtype == torch.float16
            and rows.shape[0] >= self.full_projection_min
        )
        out = self._project_rows(rows, local_hashes, hidden.dtype, full)
        if self.projection_weight is not None:
            if local.tp_size > 1 and not full:
                out = tensor_model_parallel_all_reduce(out)
            *keys, value = out.split(self.hidden_size, dim=-1)
            if (
                self.fused_residual
                and self.key_norm_weight is not None
                and hidden.dtype == torch.float16
            ):
                from vllm.models.hy_v4_flash.engram_kernel import mix_engram_residual

                return mix_engram_residual(
                    hidden,
                    out,
                    value,
                    hashes,
                    self.key_norm_weight,
                    self.query_norm_weight,
                    self.unified_pre_gamma,
                    self.residual_scale,
                )
        else:
            if local.tp_size > 1:
                out = [tensor_model_parallel_all_reduce(t) for t in out]
            *keys, value = out
        eps = torch.finfo(torch.float32).eps
        gates = []
        for channel, key in enumerate(keys):
            gate = (
                self.norm_key[channel](key)
                * self.norm_query[channel](hidden[:, channel])
            ).sum(-1)
            gate = gate * self.hidden_size**-0.5
            gates.append((gate.abs().clamp_min(1e-6).sqrt() * gate.sign()).sigmoid())
        gate = torch.stack(gates, dim=1)
        value = value.float()
        signal = (
            self.unified_pre_gamma.float()
            * value
            * (value.square().mean(-1, keepdim=True) + eps).rsqrt()
        )
        scales = torch.stack(list(self.unified_per_hc_scales)).float()
        result = ((gate.float().unsqueeze(-1) + scales) * signal.unsqueeze(1)).to(
            hidden.dtype
        )
        return hidden + result.masked_fill((hashes < 0).all(-1)[:, None, None], 0)
