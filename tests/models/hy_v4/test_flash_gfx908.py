# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.hy_v4_flash.rocm_attention import is_mi100

pytestmark = pytest.mark.skipif(not is_mi100(), reason="Requires MI100")


@pytest.mark.parametrize(
    ("group", "heads", "window", "page_size"),
    [(32, 8, 0, 16), (128, 16, 17, 16), (128, 8, 0, 32), (128, 16, 17, 32)],
)
def test_packed_mla_handles_sink_padding_and_noncontiguous_pages(
    group, heads, window, page_size
):
    from vllm.models.hy_v4_flash.mla_gfx908 import (
        cache_row_bytes,
        causal_mla,
        paged_mla,
        quantize_mla,
        write_mla_cache,
    )

    generator = torch.Generator(device="cuda").manual_seed(901)
    rope_group = min(64, group)
    lengths = torch.tensor([12, 7, 35], dtype=torch.int32, device="cuda")
    starts = torch.tensor([0, 3, 3, 8], dtype=torch.int32, device="cuda")
    table = torch.tensor(
        [[5, 4, 1], [3, 0, 2], [6, -1, 8]], dtype=torch.int32, device="cuda"
    )
    owners = torch.tensor([0] * 3 + [2] * 5, dtype=torch.int32, device="cuda")
    positions = torch.tensor([9, 10, 11, 30, 31, 32, 33, 34], device="cuda")
    logical = torch.arange(35, device="cuda").expand(8, -1)
    keep = logical <= positions[:, None]
    if window:
        keep &= logical > positions[:, None] - window
    ids = torch.where(keep, logical, -1).int().contiguous()
    cache = torch.zeros(
        9,
        page_size,
        cache_row_bytes(group, rope_group),
        dtype=torch.uint8,
        device="cuda",
    )
    keys = (
        torch.randn(
            9 * page_size, 576, dtype=torch.float16, device="cuda", generator=generator
        )
        * 0.4
    )
    queries = (
        torch.randn(
            8, heads, 576, dtype=torch.float16, device="cuda", generator=generator
        )
        * 0.5
    )
    sink = torch.linspace(-3, 4, heads, device="cuda")
    write_mla_cache(
        keys[:, :512],
        keys[:, 512:],
        cache,
        torch.arange(len(keys), device="cuda"),
        group,
        rope_group,
    )
    q, scales = quantize_mla(queries[..., :512], queries[..., 512:], group, rope_group)

    def dequantize(x, size):
        grouped = x.float().unflatten(-1, (-1, size))
        step = (
            (grouped.abs().amax(-1, keepdim=True) / 127)
            .clamp_min(2**-24)
            .half()
            .float()
        )
        return ((grouped / step).round().clamp(-127, 127) * step).flatten(-2)

    def unpack(x):
        return torch.cat(
            (dequantize(x[..., :512], group), dequantize(x[..., 512:], rope_group)), -1
        )

    def reference(indices):
        live = (indices >= 0) & (indices < table.shape[1] * page_size)
        safe = torch.where(live, indices, 0)
        physical = table[owners[:, None], safe // page_size]
        live &= physical >= 0
        k = unpack(keys)[physical.clamp_min(0).long() * page_size + safe % page_size]
        scores = torch.einsum("mhd,mkd->mhk", unpack(queries), k) / 16
        scores.masked_fill_(~live[:, None], -torch.inf)
        prob = torch.cat((scores, sink.expand(8, heads)[..., None]), -1).softmax(-1)[
            ..., :-1
        ]
        return torch.einsum("mhk,mkd->mhd", prob, k[..., :512])

    common = (q, scales, cache)
    causal = causal_mla(
        *common,
        starts,
        lengths,
        table,
        sink,
        1 / 16,
        group,
        rope_group,
        5,
        window,
        query_tile=2,
    )
    if page_size == 32:
        args = (
            *common,
            starts,
            lengths,
            table,
            sink,
            1 / 16,
            group,
            rope_group,
            5,
            window,
        )
        wide = causal_mla(*args, query_tile=2, value_tile=256)
        paired = causal_mla(*args, query_tile=2, value_tile=256, page_pair=True)
        torch.testing.assert_close(paired, wide, atol=0, rtol=0)
        torch.testing.assert_close(
            paired.float(), reference(ids), atol=0.0003, rtol=0.002
        )
    torch.testing.assert_close(causal.float(), reference(ids), atol=0.0003, rtol=0.002)
    if window:
        # Slice through a request boundary. Offsets are global in the engine's
        # query metadata while pointers into the query batch are local.
        first = positions - window + 1
        columns = torch.arange(window, device=keys.device)
        explicit = (first[:, None] + columns).to(torch.int32)
        offset = 2
        for score_tile in (0, 64):
            args = (
                q[offset:],
                scales[offset:],
                cache,
                explicit[offset:],
                owners[offset:],
                table,
                sink,
                1 / 16,
                group,
                rope_group,
            )
            expected = paged_mla(*args, score_tile=score_tile, splits=2)
            explicit_copy = torch.full_like(explicit[offset:], -1)
            implicit = paged_mla(
                *args[:3],
                explicit_copy,
                *args[4:],
                score_tile=score_tile,
                splits=2,
                window=window,
                starts=starts,
                lengths=lengths,
                row_offset=offset,
            )
            torch.testing.assert_close(implicit, expected, atol=0, rtol=0)
    # The gather path also receives empty rows and invalid entries from the
    # drafter. Only the virtual sink contributes to a fully masked row.
    ids[0] = -1
    ids[1, 0] = 100
    expected = reference(ids)
    fused = None
    for score_tile in (0, 64):
        got = paged_mla(
            *common,
            ids,
            owners,
            table,
            sink,
            1 / 16,
            group,
            rope_group,
            splits=2,
            score_tile=score_tile,
        )
        torch.testing.assert_close(got.float(), expected, atol=0.0003, rtol=0.002)
        if fused is None:
            fused = got
        else:
            torch.testing.assert_close(got, fused, atol=0, rtol=0)


def test_neuqi_fixed_step_and_packed_constant_rows():
    from vllm.models.hy_v4_flash.neuqi import _best_offset, quantize_rows

    gen = torch.Generator(device="cuda").manual_seed(2026)
    x = torch.randn(4, 128, generator=gen, device="cuda", dtype=torch.float64) * 4.1
    importance = torch.rand(x.shape, generator=gen, device=x.device, dtype=x.dtype) ** 3
    importance /= importance.sum(-1, keepdim=True)
    # Independent exhaustive oracle over every rounding discontinuity of INT4.
    events = (
        (x[..., None] - torch.arange(15, device=x.device) - 0.5)
        .flatten(1)
        .sort()
        .values
    )
    events = torch.cat((events[:, :1] - 16, events, events[:, -1:] + 16), dim=-1)
    left, right = events[:, :-1], events[:, 1:]
    codes = (x[:, None] - ((left + right) / 2)[..., None]).round().clamp(0, 15)
    centers = (importance[:, None] * (x[:, None] - codes)).sum(-1).clamp(left, right)
    best = (
        (importance[:, None] * (x[:, None] - centers[..., None] - codes).square())
        .sum(-1)
        .amin(-1)
    )
    center = _best_offset(x, importance)
    found = (
        importance * (x - center - (x - center).round().clamp(0, 15)).square()
    ).sum(-1)
    torch.testing.assert_close(found, best, atol=2e-8, rtol=0)

    weights = torch.stack(
        (torch.full_like(x[0], 0.03125), torch.zeros_like(x[0]))
    ).half()
    data = quantize_rows(weights, torch.zeros_like(weights[0]))
    unpacked = torch.stack((data.codes & 15, data.codes >> 4), dim=-1).flatten(-2)
    restored = (unpacked - data.zeros) * data.scales
    torch.testing.assert_close(restored, weights, atol=0, rtol=0)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("layout", [2, 3])
def test_packed_w4_w8_share_the_signed_activation_contract(bits, layout):
    from aiter.ops.gemm_packed_native import affine_g128, pack_affine_weights
    from aiter.ops.triton.gemm.basic.gemm_packed_g128 import quantize

    gen = torch.Generator(device="cuda").manual_seed(1739)
    m, n, k = 19, (144 if layout == 2 else 192), 384
    x = torch.randn(m, k, generator=gen, device="cuda", dtype=torch.float16)
    q, xs, _ = quantize(x, with_sums=True)
    q[0, 0] = -128
    sums = q.reshape(m, k // 128, 128).sum(-1, dtype=torch.int32)
    raw = torch.randint(
        0 if bits == 4 else -128,
        16 if bits == 4 else 128,
        (n, k),
        generator=gen,
        device=x.device,
        dtype=torch.int16,
    )
    ws = (
        torch.rand(n, k // 128, generator=gen, device=x.device, dtype=torch.float16)
        / 10
    )
    zero = (
        torch.rand(n, k // 128, generator=gen, device=x.device, dtype=torch.float16)
        * 8.99
        if bits == 4
        else None
    )
    w = (raw[:, ::2] | raw[:, 1::2] << 4).byte() if bits == 4 else raw.to(torch.int8)
    expected_w = (
        (raw.reshape(n, -1, 128).float() - (zero[..., None] if zero is not None else 0))
        * ws[..., None]
    ).reshape(n, k)
    expected_x = (q.reshape(m, -1, 128).float() * xs[..., None]).reshape(m, k)
    expected = expected_x @ expected_w.T
    w, ws, zero = pack_affine_weights(
        w, ws, zero, vector=layout == 2, overlap=layout == 3
    )
    actual = affine_g128(
        q,
        xs,
        sums,
        w,
        ws,
        zero,
        split=3,
        wave_format=layout,
        packet_load=True,
        triton_tiles=(16, 64, 4),
    )
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=2e-4)
    if layout == 3:
        from aiter.ops.flydsl.kernels.gfx908_g128 import gfx908_g128_overlap

        result = gfx908_g128_overlap(q, xs, sums if bits == 4 else None, w, ws, zero, 3)
        torch.testing.assert_close(result, expected, atol=3e-4, rtol=2e-4)
        # The packed fallback also serves narrow tiles, including partial token
        # blocks. The two paths consume the very same weight allocation.
        narrow = affine_g128(
            q,
            xs,
            sums,
            w,
            ws,
            zero,
            wave_format=layout,
            packet_load=True,
            triton_tiles=(16, 32, 2),
        )
        torch.testing.assert_close(narrow, expected, atol=3e-4, rtol=2e-4)


@pytest.mark.parametrize("group", [4, 32, 128])
def test_indexpool_transform_and_sorted_tail(group):
    from vllm.models.hy_v4_flash.indexpool_gfx908 import expand_sorted_groups
    from vllm.models.hy_v4_flash.nvidia.indexpool_ops import (
        expand_groups,
        fwht_quantize,
    )

    gen = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn(17, 256, generator=gen, device="cuda", dtype=torch.float16)[:, ::2]
    x[0].zero_()
    actual = fwht_quantize(x, group)
    expected = fwht_quantize(x, group, fused=False)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    groups = torch.tensor(
        [[1, 2, -1], [0, -1, -1], [-1, -1, -1]], dtype=torch.int32, device=x.device
    )
    pos = torch.tensor([15, 5, -1], device=x.device)
    live = torch.tensor([True, False, True], device=x.device)
    expected_ids = expand_groups(groups, pos, 4, live)
    actual_ids = expand_sorted_groups(
        groups, pos, 4, live, torch.empty_like(expected_ids)
    )
    torch.testing.assert_close(actual_ids, expected_ids, atol=0, rtol=0)


@pytest.mark.parametrize("width", [2, 64, 2048, 8192, 32768])
def test_indexpool_selection_preserves_ids_and_ties(width):
    from vllm.models.hy_v4_flash.indexpool_gfx908 import select_and_expand
    from vllm.models.hy_v4_flash.nvidia.indexpool_ops import expand_groups
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import hy4_int8_paged_mqa_logits

    budget, ratio, heads, page = 31, 4, 16, 64
    q = torch.zeros((4, heads, 128), device="cuda", dtype=torch.int8)
    q[:, :, 0] = 1
    scale = torch.ones(4, heads, 1, device=q.device, dtype=torch.float16)
    count = (width + page - 1) // page
    cache = torch.zeros(count, page, 130, device=q.device, dtype=torch.uint8)
    flat = cache.flatten(1)
    key = flat[:, : page * 128].view(torch.int8).view(count, page, 128)
    key[:, :, 0] = (
        torch.arange(count * page, device=q.device).view(count, page) % 13 - 6
    )
    flat[:, page * 128 :].view(torch.float16).fill_(1)
    weights = torch.ones(4, heads, device=q.device)
    weights[2].neg_()
    upper = torch.tensor([0, 1, width // 2, width], device=q.device, dtype=torch.int32)
    positions = upper * ratio + 1
    table = (
        torch.arange(count, device=q.device, dtype=torch.int32).expand(4, -1).clone()
    )
    table[2, 0] = -1
    live = torch.tensor([False, True, True, True], device=q.device)
    owner = torch.arange(4, device=q.device, dtype=torch.int32)
    inputs = (q, scale, cache, weights, upper, table, owner, width, 128)
    original = hy4_int8_paged_mqa_logits(*inputs, mask_unused=True)
    ranks = hy4_int8_paged_mqa_logits(*inputs, rank=True)
    k = min(budget, width)
    # Equal scores keep the earliest logical positions, including at the
    # selection boundary. An invalid physical page must never be read.
    ordered = torch.argsort(original, dim=-1, descending=True, stable=True)[:, :k]
    groups = torch.where(original.gather(1, ordered).isfinite(), ordered, -1)
    groups = torch.where(groups < 0, width, groups).sort(1).values
    groups = torch.nn.functional.pad(
        groups.masked_fill(groups == width, -1).int(), (0, budget - k), value=-1
    )
    expected = expand_groups(groups, positions, ratio, live)
    choices = [original, ranks]
    if width <= 16384:
        # Uninitialized padding must never enter the per-row graph selection.
        scratch = torch.full_like(original, float("nan"))
        choices.append(
            hy4_int8_paged_mqa_logits(*inputs, out=scratch, keep_all_at_most=budget)
        )
    for scores in choices:
        got = select_and_expand(
            scores, positions, ratio, live, torch.empty_like(expected)
        )
        torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_disk_engram_selected_rows_match_page_lookup(tmp_path):
    from vllm.models.hy_v4_flash.engram_q4 import (
        ALIGN,
        ROWS_PER_PAGE,
        PackedEngramTable,
        decode_rows,
        encode_rows,
        page_rows,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=device).manual_seed(67)
    pages = torch.zeros(3, ALIGN, device=device, dtype=torch.uint8)
    raw = torch.randn(
        3 * ROWS_PER_PAGE, 96, generator=gen, dtype=torch.float16, device=device
    )
    encoded = encode_rows(raw)
    page_rows(pages).copy_(encoded.view(3, ROWS_PER_PAGE, -1))
    file = tmp_path / "table.q4_0"
    file.write_bytes(pages.cpu().numpy().tobytes())

    low, high = 37, 180
    ids = torch.tensor(
        [[low - 1, low, 74, 75, 74], [high - 1, high, low + 128, -1, 0]],
        dtype=torch.long,
        device=device,
    )
    expected = decode_rows(encoded[ids.clamp(0, high)])
    expected[(ids < low) | (ids >= high)] = 0
    for row_threshold, cache_rows in [(0, False), (1, False), (1, True)]:
        table = PackedEngramTable(
            file,
            len(encoded),
            low,
            high,
            "bcache",
            1,
            device,
            row_threshold=row_threshold,
            cache_rows=cache_rows,
            prefetch_pages=1,
        )
        try:
            for batch in (ids, ids[1:], ids):
                reference = expected[1:] if batch.shape[0] == 1 else expected
                torch.testing.assert_close(
                    table.lookup(batch), reference, atol=0, rtol=0
                )
        finally:
            table.close()


def test_fused_engram_hash_keeps_chunk_boundaries_and_padding():
    from types import SimpleNamespace

    from vllm.models.hy_v4_flash.nvidia.engram import FlashNgramHashState

    config = SimpleNamespace(
        engram_layer_ids=[2, 12],
        engram_max_ngram_size=3,
        engram_n_head_per_ngram=2,
        engram_n_embed_per_ngram=192,
        engram_pad_id=0,
        vocab_size=128,
        engram_seed=0,
        engram_layer_head_vocab_sizes={
            "2": [[17, 19], [23, 29]],
            "12": [[31, 37], [41, 43]],
        },
    )
    with torch.device("cuda"):
        state = FlashNgramHashState(config)
        state.load_lookup(torch.arange(128) % 97)
        state.visual_token_ids = (118, 119)
        ids = torch.tensor(
            [27, 118, 5, -4, 0, 128, 22, 6, 119, 7, 3, 11], dtype=torch.int32
        )
        lookback = torch.tensor([[2, 3], [-1, -1], [129, 23]], dtype=torch.int32)
        meta = SimpleNamespace(
            num_actual_tokens=10,
            seq_lens=torch.tensor([35, 0, 10], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4, 4, 10], dtype=torch.int32),
        )
        storage = torch.full((2, 16, 4), -77, dtype=torch.int32)
        result = state.prepare_hashes_into(ids, lookback, meta, storage[:, :12])
        expected = state.prepare_hashes(ids, lookback, meta)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert (storage[:, 12:] == -77).all()


@pytest.mark.parametrize(
    ("rows", "groups", "splits"), [(12, (64, 32), 2), (96, (128, 64), 1)]
)
def test_fly_sparse_mla_replays_native_probabilities(rows, groups, splits):
    pytest.importorskip("flydsl")
    from vllm.models.hy_v4_flash.mla_gfx908 import (
        cache_row_bytes,
        paged_mla,
        quantize_mla,
        write_mla_cache,
    )

    group, rope_group = groups
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(312)
    width, page, heads = 1024, 64, 8
    pages = width // page
    cache = torch.empty(
        pages,
        page,
        cache_row_bytes(group, rope_group),
        device=device,
        dtype=torch.uint8,
    )
    latent = torch.randn(width, 576, generator=gen, device=device, dtype=torch.float16)
    write_mla_cache(
        latent[:, :512],
        latent[:, 512:],
        cache,
        torch.arange(width, device=device),
        *groups,
    )
    x = torch.randn(rows, heads, 576, generator=gen, device=device, dtype=torch.float16)
    q, scales = quantize_mla(x[..., :512], x[..., 512:], *groups)
    ids = torch.arange(width, device=device, dtype=torch.int32).repeat(rows, 1)
    ids[::2, ::5] = -1
    ids[0] = -1
    table = torch.randperm(pages, device=device, generator=gen, dtype=torch.int32)[None]
    table[0, -2] = -1
    owner = torch.zeros(rows, dtype=torch.int32, device=device)
    sink = torch.linspace(-2, 3, heads, device=device)
    args = (q, scales, cache, ids, owner, table, sink, 1 / 16, group, rope_group)
    kwargs = dict(score_tile=64, save_addresses=True, splits=splits, warps=4)
    ref = paged_mla(*args, **kwargs)
    got = paged_mla(*args, **kwargs, fly_scores=True, fly_values=True)
    torch.testing.assert_close(got, ref, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("rows", "dtype", "index_dtype"),
    [
        (5, torch.float32, torch.int64),
        (17, torch.float16, torch.int32),
        (17, torch.float16, torch.int64),
        (81, torch.float16, torch.int32),
    ],
)
def test_engram_projection_and_residual_preserve_mask_and_strides(
    rows, dtype, index_dtype
):
    pytest.importorskip("flydsl")
    from aiter.ops.flydsl.kernels.gfx908_engram_projection import (
        gfx908_engram_projection,
        pack_projection,
    )

    from vllm.models.hy_v4_flash.engram_kernel import mix_engram_residual

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(819)

    def rand(*shape, kind=torch.float16):
        return torch.randn(shape, generator=generator, device=device, dtype=kind)

    raw = rand(rows, 6, 96, kind=dtype)[:, :4]
    hashes = torch.randint(
        -3, 7, (rows, 16), generator=generator, device=device, dtype=index_dtype
    )
    hashes[0].fill_(-1)
    local = hashes[:, 8:12]
    gamma = rand(4, 96)
    weight = rand(128, 384) * 0.04
    bias = rand(128) * 0.02
    eps = torch.finfo(torch.float32).eps
    x = raw.float().masked_fill(local[..., None] < 0, 0)
    x = (x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * gamma.float()).half()
    expected = torch.nn.functional.linear(x.flatten(1), weight, bias)
    actual = gfx908_engram_projection(raw, local, gamma, pack_projection(weight), bias)
    torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
    assert torch.equal(actual[0], bias)
    dim = 3072
    hidden = rand(rows, 4, dim)
    projections = rand(rows, 5 * dim)
    *keys, value = projections.split(dim, dim=-1)
    key_norm, query_norm = rand(4, dim), rand(4, dim)
    value_norm, scales = rand(dim), rand(4, dim)
    gates = []
    for stream, key in enumerate(keys):
        k = torch.nn.functional.rms_norm(key, (dim,), key_norm[stream])
        q = torch.nn.functional.rms_norm(hidden[:, stream], (dim,), query_norm[stream])
        product = (k * q).sum(-1) * dim**-0.5
        gates.append(
            (product.abs().clamp_min(1.0e-6).sqrt() * product.sign()).sigmoid()
        )
    signal = (
        value_norm.float()
        * value.float()
        * (value.float().square().mean(-1, keepdim=True) + eps).rsqrt()
    )
    residual = (
        (torch.stack(gates, 1).float().unsqueeze(-1) + scales.float()) * signal[:, None]
    ).half()
    expected = hidden + residual.masked_fill((hashes < 0).all(-1)[:, None, None], 0)
    result = mix_engram_residual(
        hidden, projections, value, hashes, key_norm, query_norm, value_norm, scales
    )
    assert torch.equal(result[0], hidden[0])
    torch.testing.assert_close(result, expected, rtol=0.003, atol=0.008)


def test_g128_flydsl_reuses_correct_scale_precision():
    pytest.importorskip("flydsl")
    from aiter.ops.flydsl.kernels.gfx908_g128 import gfx908_g128_overlap
    from aiter.ops.gemm_packed_native import pack_affine_weights

    gen = torch.Generator(device="cuda").manual_seed(479)
    m, n, k = 17, 64, 256
    x = torch.randint(-16, 16, (m, k), device="cuda", dtype=torch.int8, generator=gen)
    w = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8, generator=gen)
    xs = torch.rand(m, 2, generator=gen, device="cuda") * 0.4
    scales = torch.rand(n, 2, generator=gen, device="cuda") * 0.7
    for dtype in (torch.float16, torch.float32, torch.float16):
        ws = scales.to(dtype)
        packed, group_scales, _ = pack_affine_weights(w, ws, overlap=True)
        result = gfx908_g128_overlap(x, xs, None, packed, group_scales, None, 1)
        expected = torch.zeros_like(result)
        for group in range(2):
            section = slice(group * 128, (group + 1) * 128)
            dots = x[:, section].float() @ w[:, section].float().T
            expected += (dots * xs[:, group, None]) * ws[:, group].float()
        torch.testing.assert_close(result, expected, atol=0.004, rtol=1e-5)


def test_ihc_partial_handles_final_wave_and_preserves_residual():
    pytest.importorskip("flydsl")
    from aiter.ops.flydsl.kernels.gfx908_ihc import pack_ihc_gate

    from vllm.models.hy_v4.amd.triton_ihc import (
        triton_ihc_post_pre_rms_norm,
        triton_ihc_pre,
    )

    generator = torch.Generator(device="cuda").manual_seed(807)

    def rand(*shape, dtype=torch.float16):
        return torch.randn(shape, device="cuda", dtype=dtype, generator=generator)

    rows = 137
    residual = rand(rows, 4, 3072)
    x = rand(rows, 3072)
    gate = rand(rows, 4, dtype=torch.float32).sigmoid()
    weight = rand(8, 12288, dtype=torch.float32) * 0.013
    scaling = rand(2, dtype=torch.float32) * 0.15
    bias = rand(8, dtype=torch.float32)
    norm = rand(3072) * 0.05 + 1
    packed = pack_ihc_gate(weight)
    for fn, args in (
        (triton_ihc_pre, (residual, weight, scaling, bias)),
        (
            triton_ihc_post_pre_rms_norm,
            (x, residual, gate, weight, scaling, bias, norm),
        ),
    ):
        expected = fn(*args, 2.0, 1e-6, 1e-5)
        actual = fn(*args, 2.0, 1e-6, 1e-5, packed)
        torch.testing.assert_close(actual[1], expected[1], atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(actual[0], expected[0], atol=0.004, rtol=0.001)
        if len(actual) == 3:
            assert torch.equal(actual[2], expected[2])


@pytest.mark.parametrize("packed_projection", [False, True])
def test_single_group_sigmoid_router_preserves_experts_and_probabilities(
    packed_projection,
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        GroupedTopKRouter,
    )
    from vllm.models.hy_v4_flash.nvidia.moe import HYV4FlashMoE

    generator = torch.Generator(device="cuda").manual_seed(603)
    logits = torch.randn(137, 512, generator=generator, device="cuda")
    bias = torch.linspace(-0.05, 0.09, 512, device="cuda")
    hidden = torch.empty(137, 3072, device="cuda", dtype=torch.float16)
    router = GroupedTopKRouter(
        top_k=16,
        global_num_experts=512,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=3.998,
        e_score_correction_bias=bias,
    )
    gate = SimpleNamespace(
        weight=torch.empty((512, 3072), device="meta"), prepare_gfx908_overlap=Mock()
    )
    moe = SimpleNamespace(gate=gate, experts=SimpleNamespace(router=router))
    splits = {137: 8} if packed_projection else None
    HYV4FlashMoE.prepare_gfx908_router(moe, splits)
    assert router.single_group_sigmoid
    if packed_projection:
        gate.prepare_gfx908_overlap.assert_called_once_with(splits)
    else:
        gate.prepare_gfx908_overlap.assert_not_called()
    weights, ids = router._compute_routing(hidden, logits, torch.int32)
    for _ in range(3):
        repeat_weights, repeat_ids = router._compute_routing(
            hidden, logits, torch.int32
        )
        assert torch.equal(repeat_ids, ids)
        assert torch.equal(repeat_weights, weights)
    scores = logits.sigmoid()
    expected_ids = (scores + bias).topk(16, dim=-1).indices.sort(dim=-1).values
    actual = ids.long().sort(dim=-1)
    assert torch.equal(actual.values, expected_ids)
    expected = scores.gather(1, expected_ids)
    expected = expected / expected.sum(dim=-1, keepdim=True) * 3.998
    torch.testing.assert_close(weights.gather(1, actual.indices), expected)


@pytest.mark.parametrize("rows", [1, 7, 12])
def test_dfly_pair_quantizes_strided_hidden_and_embedding_with_independent_weights(
    rows,
):
    from aiter.ops.flydsl.kernels.gfx908_g128_dual_rms import gfx908_g128_dual_rms
    from aiter.ops.triton.gemm.basic.gemm_packed_g128 import quantize

    from vllm._custom_ops import rms_norm

    generator = torch.Generator(device="cuda").manual_seed(417)

    def sample(*shape):
        return torch.randn(
            shape, generator=generator, device="cuda", dtype=torch.float16
        )

    hidden = sample(rows, 8, 3073)[:, 2, 1:]
    embed = sample(rows, 3072)
    hidden[0].zero_()
    gamma, beta = sample(3072), sample(3072)
    normalized = []
    for value, weight in ((hidden, gamma), (embed, beta)):
        out = torch.empty(value.shape, device=value.device, dtype=value.dtype)
        rms_norm(out, value, weight, 1e-5)
        normalized.append(out)
    expected_q, expected_scale = quantize(torch.cat(normalized, dim=-1))
    q, scales = gfx908_g128_dual_rms(hidden, embed, gamma, beta, 1e-5)
    delta = (q.int() - expected_q.int()).abs()
    assert delta.max().item() <= 1
    assert delta.count_nonzero().item() <= max(1, q.numel() // 500)
    assert torch.count_nonzero(q[0, :3072]).item() == 0
    torch.testing.assert_close(scales, expected_scale, atol=1e-30, rtol=0.001)


def test_indexpool_exchanges_group_starts_with_masked_tails():
    from vllm.models.hy_v4_flash.indexpool_gfx908 import (
        _pack_verify_groups,
        _unpack_verify_groups,
    )

    ratio, groups, per_rank = 4, 512, 16
    rows = 4 * per_rank
    device = "cuda"
    positions = torch.arange(rows, device=device, dtype=torch.int32) + 260001
    positions[::11] = -1
    starts = torch.arange(groups, device=device, dtype=torch.int32).repeat(rows, 1) * 16
    starts[:, -1] = positions // ratio * ratio
    starts[:, 3::13] = -1
    offsets = torch.arange(ratio, device=device, dtype=torch.int32)
    token = starts[:, :, None] + offsets
    selected = torch.where(
        (starts[:, :, None] >= 0) & (token <= positions[:, None, None]), token, -1
    ).reshape(rows, groups * ratio)

    packed = torch.zeros(rows, groups, dtype=torch.float32, device=device)
    staging = torch.empty_like(packed)
    for offset in range(0, rows, per_rank):
        _pack_verify_groups[(rows,)](
            selected[offset : offset + per_rank],
            staging,
            per_rank,
            offset,
            groups,
            ratio,
        )
        packed.add_(staging)
    actual = torch.empty_like(selected)
    _unpack_verify_groups[(rows,)](packed, positions, actual, groups, ratio)
    torch.testing.assert_close(actual, selected, rtol=0, atol=0)


def test_router_four_split_fp32_order():
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale

    from vllm.model_executor.kernels.linear.mixed_precision.aiter_w8a16 import (
        _get_aiter_w8a8_config,
    )
    from vllm.model_executor.layers.quantization.utils.packed_g128_int8 import quantize

    generator = torch.Generator(device="cuda").manual_seed(173)
    x, xs = quantize(torch.randn(50, 3072, device="cuda", generator=generator))
    w, ws = quantize(torch.randn(512, 3072, device="cuda", generator=generator))
    config = _get_aiter_w8a8_config(50, 512, 3072, 128)
    partials = gemm_a8w8_blockscale(
        x, w, xs, ws, dtype=torch.float32, config=config.copy(), skip_reduce=True
    )
    a, b, c, d = partials.unbind()
    expected = (a + c) + (b + d)
    assert not torch.equal(expected, (a + b) + (c + d))
    actual = gemm_a8w8_blockscale(
        x, w, xs, ws, dtype=torch.float32, config=config, order_four_splits=True
    )
    assert torch.equal(actual, expected)


def test_expert_swiglu_g128_rounding_at_fp16_midpoints():
    from vllm.model_executor.layers.quantization.utils.packed_g128_int8 import quantize

    # Each midpoint determines the maximum and therefore the scale of its group.
    x = torch.zeros((2, 256), device="cuda", dtype=torch.float32)
    x[:, 0] = torch.tensor([-2.724609375, -0.7138671875], device="cuda")
    x[:, 128:] = 1
    codes, scales, sums = quantize(x, swiglu=True, clamp_limit=10.0, with_sums=True)
    activations = torch.tensor([-0.167724609375, -0.2347412109375], device="cuda")
    assert torch.equal(scales[:, 0], activations.abs() / 127)
    assert torch.all(codes[:, 0] == -127) and torch.count_nonzero(codes[:, 1:]) == 0
    assert torch.all(sums == -127)
