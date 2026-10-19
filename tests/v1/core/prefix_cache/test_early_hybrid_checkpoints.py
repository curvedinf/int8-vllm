# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short shared instructions followed by unrelated documents on hybrid drafts."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.v1.core.test_prefix_caching import (
    _make_hybrid_kv_cache_config,
    make_kv_cache_manager,
    make_request,
)
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import MambaSpec, SlidingWindowSpec

pytestmark = pytest.mark.cpu_test


def make_manager(checkpoints=(8, 16, 24), block_size=64, unit=4, window=20):
    init_none_hash(sha256)
    config = _make_hybrid_kv_cache_config(
        block_size, 512, ["mamba_align", "full", "sliding_window"]
    )
    config.prefix_cache_checkpoint_tokens = checkpoints
    for group in config.kv_cache_groups:
        if isinstance(group.kv_cache_spec, MambaSpec):
            group.kv_cache_spec = replace(group.kv_cache_spec, num_speculative_blocks=3)
        if isinstance(group.kv_cache_spec, SlidingWindowSpec):
            group.kv_cache_spec = replace(group.kv_cache_spec, sliding_window=window)
            group.is_eagle_group = True
    return make_kv_cache_manager(
        config,
        max_model_len=4096,
        enable_caching=True,
        hash_block_size=unit,
        use_eagle=True,
        retention_interval=0,
    )


def prefill(manager, name, tokens, states):
    """Model worker copies and exact recurrent state positions at chunk ends."""
    unit = manager.block_pool.hash_block_size
    size = manager.coordinator.scheduler_block_size
    request = make_request(name, tokens, unit, sha256)
    manager.new_step_starts()
    cached, hit, shared = manager.get_computed_blocks(request)
    request.shared_prefix_boundary = shared
    stub = SimpleNamespace(
        block_size=size,
        hash_block_size=unit,
        use_eagle=True,
        max_num_scheduled_tokens=2 * size,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=manager.coordinator.enable_partial_hash_hits,
        prefix_checkpoint_tokens=manager.kv_cache_config.prefix_cache_checkpoint_tokens,
    )
    first = True
    while request.num_computed_tokens < len(tokens):
        manager.new_step_starts()
        start = request.num_computed_tokens + (hit if first else 0)
        # The splitter sees the effective local-hit start, as the scheduler does.
        request.num_computed_tokens = start
        count = Scheduler._mamba_block_aligned_split(
            stub, request, min(2 * size, len(tokens) - start)
        )
        assert count > 0
        if first:
            request.num_computed_tokens = 0
            allocated = manager.allocate_slots(
                request, count, hit, cached, num_lookahead_tokens=3
            )
        else:
            allocated = manager.allocate_slots(request, count, num_lookahead_tokens=3)
        assert allocated is not None
        copies, retained = manager.take_kv_cache_block_copies()
        for copy in copies:
            if copy.src_block_id in states:
                states[copy.dst_block_id] = states[copy.src_block_id]
        mamba = manager.coordinator.single_type_managers[0]
        running = mamba.req_to_blocks[name][cdiv(start + count, size) - 1]
        # If a restored partial state is in this page it must be exact.
        if first and start % size:
            assert states[running.block_id] == start
        states[running.block_id] = start + count
        manager.block_pool.free_blocks(retained)
        request.num_computed_tokens = start + count
        first = False
    manager.free(request)
    manager.new_step_starts()
    return hit


@pytest.mark.parametrize("prompt_length", [43, 155, 283])
def test_short_prefix_survives_document_growth_and_tail_promotion(prompt_length):
    manager = make_manager()
    states = {}
    shared = list(range(29))
    for i in range(4):
        tokens = shared + [100 + i] * (prompt_length - len(shared))
        hit = prefill(manager, str(i), tokens, states)
        assert hit == (0 if i == 0 else 24)
    # Every retained recurrent checkpoint must still describe its own state,
    # even though subsequent chunks kept updating the producer's running slot.
    for block in manager.block_pool.blocks:
        if block.block_hash is not None and block.block_id in states:
            assert states[block.block_id] == block.block_hash_num_tokens


def test_short_prefix_is_a_miss_without_checkpoints():
    manager = make_manager(checkpoints=())
    states = {}
    for i in range(3):
        assert prefill(manager, str(i), list(range(29)) + [100 + i] * 150, states) == 0


def test_changed_instruction_cannot_hit_early_checkpoint():
    manager = make_manager()
    states = {}
    prefill(manager, "a", list(range(29)) + [100] * 150, states)
    assert prefill(manager, "b", [999] + list(range(1, 29)) + [101] * 150, states) == 0


def test_evicted_draft_prefix_invalidates_shared_checkpoint():
    manager = make_manager()
    states = {}
    tokens = list(range(29)) + [100] * 150
    prefill(manager, "a", tokens, states)
    request = make_request("probe", tokens, 4, sha256)
    draft = manager.block_pool.get_cached_block(request.block_hashes[6], [2])
    assert draft is not None
    manager.block_pool.evict_blocks({draft[0].block_id})
    assert prefill(manager, "b", list(range(29)) + [101] * 150, states) == 0


def test_repeated_long_prompt_preserves_replay_hit():
    manager = make_manager()
    states = {}
    tokens = list(range(283))
    assert prefill(manager, "a", tokens, states) == 0
    assert prefill(manager, "b", tokens, states) >= 256


@pytest.mark.parametrize(
    "shared_length, expected_hit", [(73, 64), (141, 128), (247, 232)]
)
def test_bolt_geometry_reuses_short_instructions(shared_length, expected_hit):
    manager = make_manager(
        checkpoints=(64, 128, 232), block_size=4256, unit=8, window=1024
    )
    # The test uses a smaller layer count, but the deployed page/hash/window
    # geometry and observed instruction/document lengths.
    states = {}
    shared = list(range(shared_length))
    assert prefill(manager, "a", shared + [1000] * 4732, states) == 0
    assert prefill(manager, "b", shared + [1001] * 4743, states) == expected_hit


@pytest.mark.parametrize("checkpoints", [(0,), (3,), (64,), (60,)])
def test_invalid_checkpoint_geometry_is_rejected(checkpoints):
    with pytest.raises(ValueError, match="Early prefix checkpoints"):
        make_manager(checkpoints)
