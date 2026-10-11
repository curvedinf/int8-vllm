# SPDX-License-Identifier: Apache-2.0
"""Bounded warm-follow-up bypass must not starve cold FCFS requests."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, PriorityRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def request(name, tokens=42000, **kwargs):
    fields = dict(request_id=name, num_tokens=tokens, num_computed_tokens=0,
                  num_preemptions=0, num_stale_output_tokens=0, mm_features=[],
                  lora_request=None, status=RequestStatus.WAITING)
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def setup_queue():
    head, warm, warm2 = request("cold"), request("warm"), request("warm2")
    queue = FCFSRequestQueue([head, warm, warm2])
    stub = SimpleNamespace(prefix_fair_admission=True, block_size=1664,
                           _prefix_bypassed_heads=set(),
                           waiting=queue, skipped_waiting=FCFSRequestQueue(),
                           kv_cache_manager=SimpleNamespace(
                               peek_cached_tokens=lambda r:
                               39936 if r.request_id.startswith("warm") else 0))
    return stub, queue, head, warm, warm2


def test_regular_cached_turn_can_bypass_single_deferred_cold_head():
    stub, queue, head, warm, warm2 = setup_queue()
    assert queue.pop_request() is head
    stub.skipped_waiting.add_request(head)
    Scheduler._prioritize_cached_waiter(stub, stub.skipped_waiting, 1664)
    assert list(stub.skipped_waiting) == [warm, head]
    assert list(queue) == [warm2]
    stub.skipped_waiting.pop_request()
    Scheduler._prioritize_cached_waiter(stub, stub.skipped_waiting, 1664)
    assert stub.skipped_waiting.peek_request() is head


def test_scan_bound_is_shared_between_both_queues():
    stub, queue, head, warm, warm2 = setup_queue()
    queue.remove_request(warm)
    queue.remove_request(warm2)
    for i in range(16):
        queue.add_request(request(f"uncached-{i}"))
    stub.skipped_waiting.add_request(warm)
    Scheduler._prioritize_cached_waiter(stub, queue, 1664)
    assert queue.peek_request() is head
    assert stub.skipped_waiting.peek_request() is warm


def test_one_bypass_per_cold_head():
    stub, queue, head, warm, warm2 = setup_queue()
    Scheduler._prioritize_cached_waiter(stub, queue, 1664)
    assert list(queue) == [warm, head, warm2]
    assert queue.pop_request() is warm
    # Even a stream of warm arrivals cannot bypass this cold head again.
    Scheduler._prioritize_cached_waiter(stub, queue, 1664)
    assert queue.pop_request() is head
    assert queue.pop_request() is warm2


def test_temporarily_blocked_head_cannot_receive_repeated_bypasses():
    stub, queue, head, warm, warm2 = setup_queue()
    Scheduler._prioritize_cached_waiter(stub, queue, 1664)
    queue.pop_request()
    queue.pop_request()
    other = request("other")
    queue.prepend_request(other)
    Scheduler._prioritize_cached_waiter(stub, queue, 1664)
    assert queue.pop_request() is warm2
    queue.prepend_request(head)
    queue.add_request(request("warm3"))
    Scheduler._prioritize_cached_waiter(stub, queue, 1664)
    assert queue.peek_request() is head


@pytest.mark.parametrize("field,value", [
    ("status", RequestStatus.WAITING_FOR_REMOTE_KVS),
    ("num_computed_tokens", 1), ("num_preemptions", 1),
    ("num_stale_output_tokens", 1), ("mm_features", [object()]),
    ("lora_request", object()),
])
def test_ineligible_followup_is_not_promoted(field, value):
    stub, queue, head, warm, warm2 = setup_queue()
    queue.remove_request(warm2)
    setattr(warm, field, value)
    Scheduler._prioritize_cached_waiter(stub, queue, 1664)
    assert list(queue) == [head, warm]


def test_zero_budget_disabled_short_head_and_priority_policy_unchanged():
    stub, queue, head, warm, warm2 = setup_queue()
    for enabled, budget in [(False, 1664), (True, 0)]:
        stub.prefix_fair_admission = enabled
        Scheduler._prioritize_cached_waiter(stub, queue, budget)
        assert queue.peek_request() is head
    stub.prefix_fair_admission = True
    head.num_tokens = 100
    Scheduler._prioritize_cached_waiter(stub, queue, 1664)
    assert queue.peek_request() is head
    priority = PriorityRequestQueue()
    Scheduler._prioritize_cached_waiter(stub, priority, 1664)


def test_cache_peek_obeys_read_disable_and_does_not_emit_events():
    r = request("r")
    r.block_hashes = ["hash"]
    coordinator = Mock()
    coordinator.find_longest_cache_hit.return_value = ([], 39936, 0)
    manager = SimpleNamespace(coordinator=coordinator,
                              prefix_cache_lookup_enabled=lambda r: False)
    assert KVCacheManager.peek_cached_tokens(manager, r) == 0
    coordinator.find_longest_cache_hit.assert_not_called()
    manager.prefix_cache_lookup_enabled = lambda r: True
    assert KVCacheManager.peek_cached_tokens(manager, r) == 39936
    coordinator.find_longest_cache_hit.assert_called_once_with(["hash"], 41999)
    assert coordinator.method_calls == [
        ("find_longest_cache_hit", (["hash"], 41999), {})]
