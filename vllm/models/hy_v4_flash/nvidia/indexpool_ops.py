# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coarse IndexPool mathematics, independent of cache allocation."""

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope
from vllm.triton_utils import tl, triton

INDEX_DIM = 128
FP8_MAX = 448.0


def fwht_quantize(
    vectors: torch.Tensor, int_group: int = 0, *, fused=True
) -> tuple[torch.Tensor, torch.Tensor]:
    if vectors.ndim != 2 or vectors.shape[1] != INDEX_DIM:
        raise ValueError("IndexPool quantization requires [rows, 128] vectors")
    if int_group and fused and vectors.is_cuda:
        from vllm.models.hy_v4_flash.indexpool_gfx908 import quantize_index_vectors

        return quantize_index_vectors(vectors, int_group)
    x = vectors.float()
    stride = 1
    while stride < INDEX_DIM:
        pairs = x.reshape(-1, 2, stride)
        left, right = pairs[:, 0], pairs[:, 1]
        x = torch.stack((left + right, left - right), dim=1).reshape(vectors.shape)
        stride *= 2
    x = x * INDEX_DIM**-0.5
    if int_group:
        if int_group not in (4, 8, 16, 32, 64, 128):
            raise ValueError("Invalid signed INT8 IndexPool group size")
        grouped = x.reshape(-1, INDEX_DIM // int_group, int_group)
        scales = (grouped.abs().amax(-1) / 127).clamp_min(2**-24).half()
        quant = (grouped / scales.float().unsqueeze(-1)).round().clamp_(-127, 127)
        return quant.to(torch.int8).reshape_as(vectors), scales
    scale = torch.exp2(
        torch.ceil(torch.log2(x.abs().amax(-1).clamp_min(1e-4) / FP8_MAX))
    )
    quant = (x / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quant, scale


def gated_pool(values, gates, ape, norm_weight, eps):
    probabilities = (gates.float() + ape.float()).softmax(1)
    pooled = (values.float() * probabilities).sum(1)
    return F.rms_norm(pooled, (INDEX_DIM,), norm_weight.float(), eps)


def rotate_indexer(x, positions, rope_dim, parameters):
    if rope_dim == 0:
        return x
    if parameters.get("rope_type", "default") != "default":
        raise ValueError("Flash IndexPool currently requires default RoPE")
    theta = parameters.get("rope_theta", 10000.0)
    inverse = theta ** (
        -torch.arange(0, rope_dim, 2, device=x.device, dtype=torch.float32) / rope_dim
    )
    if positions.ndim == 1:
        frequency = positions.float()[:, None] * inverse
    else:
        section = parameters.get("mrope_section")
        if positions.shape[0] != 3 or section is None or sum(section) * 2 != rope_dim:
            raise ValueError("IndexPool requires valid three-axis MRoPE sections")
        planes = positions.float()[:, :, None] * inverse
        if parameters.get("mrope_interleaved", True):
            frequency = apply_interleaved_rope(planes, section)
        else:
            parts, offset = [], 0
            for axis, width in enumerate(section):
                parts.append(planes[axis, :, offset : offset + width])
                offset += width
            frequency = torch.cat(parts, -1)
    nope, pe = x.split((x.shape[-1] - rope_dim, rope_dim), dim=-1)
    pe = (
        pe.reshape(*pe.shape[:-1], rope_dim // 2, 2).transpose(-2, -1).reshape(pe.shape)
    )
    frequency = torch.cat((frequency, frequency), -1)
    if pe.ndim == 3:
        frequency = frequency[:, None]
    left, right = pe.float().chunk(2, dim=-1)
    rotated = (
        pe.float() * frequency.cos() + torch.cat((-right, left), -1) * frequency.sin()
    )
    return torch.cat((nope, rotated.to(x.dtype)), dim=-1)


def expand_groups(group_ids, positions, ratio, active=None):
    offsets = torch.arange(ratio, device=group_ids.device)
    upper = positions // ratio
    live = positions >= 0
    if active is not None:
        live = live & active
    history = group_ids[..., None] * ratio + offsets
    valid = (group_ids >= 0) & (group_ids < upper[:, None]) & live[:, None]
    history = torch.where(valid[..., None], history, -1)
    tail = upper[:, None] * ratio + offsets
    tail = torch.where(live[:, None] & (tail <= positions[:, None]), tail, -1)
    candidates = torch.cat((history.flatten(1), tail), -1).int()
    valid = candidates >= 0
    destinations = torch.where(
        valid,
        valid.long().cumsum(-1) - 1,
        valid.sum(-1, keepdim=True) + (~valid).long().cumsum(-1) - 1,
    )
    return torch.empty_like(candidates).scatter(-1, destinations, candidates)


def select_groups(logits, upper, budget):
    columns = torch.arange(logits.shape[1], device=logits.device)
    empty = -torch.inf if logits.is_floating_point() else torch.iinfo(logits.dtype).min
    scores = logits.masked_fill(columns[None] >= upper[:, None], empty)
    count = min(budget, logits.shape[1])
    values, indices = scores.topk(count, dim=-1)
    sentinel = torch.iinfo(torch.int32).max
    indices = torch.where(values != empty, indices, sentinel).int()
    indices = indices.sort(dim=-1).values
    indices.masked_fill_(indices == sentinel, -1)
    return F.pad(indices, (0, budget - count), value=-1)


@triton.jit
def _scatter_rows(
    Dst,
    Src,
    Offsets,
    SRC_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offset = tl.load(Offsets + row)
    if offset >= 0:
        columns = tl.arange(0, BLOCK)
        value = tl.load(Src + row * SRC_STRIDE + columns, columns < WIDTH, 0)
        tl.store(Dst + offset + columns, value, columns < WIDTH)


def write_pool(cache, table, owners, groups, active, quant, scales):
    page_size = cache.shape[1]
    columns = (groups // page_size).clamp(0, table.shape[1] - 1)
    base = table[owners, columns].long() * cache.stride(0)
    row = groups % page_size
    if groups.numel() == 0:
        return
    row_bytes = scales.numel() // groups.numel() * scales.element_size()
    _scatter_rows[(groups.numel(),)](
        cache,
        quant.view(torch.uint8),
        torch.where(active, base + row * INDEX_DIM, -1),
        quant.stride(0),
        INDEX_DIM,
        INDEX_DIM,
    )
    _scatter_rows[(groups.numel(),)](
        cache,
        scales.contiguous().view(torch.uint8),
        torch.where(active, base + page_size * INDEX_DIM + row * row_bytes, -1),
        row_bytes,
        row_bytes,
        row_bytes,
    )
