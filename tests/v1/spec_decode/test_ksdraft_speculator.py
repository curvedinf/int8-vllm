# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from vllm.v1.worker.gpu.spec_decode.ksdraft.speculator import KSDraftSpeculator


def _spec(window=4, steps=3, reqs=2):
    s = object.__new__(KSDraftSpeculator)
    s.device = torch.device("cpu")
    s.window = window
    s._win = torch.arange(window)
    s.num_speculative_steps = steps
    s.draft_tokens = torch.zeros(reqs, steps, dtype=torch.int64)
    s.draft_logits = None
    s.draft_token_confidence_probs = torch.zeros_like(s.draft_tokens,
                                                      dtype=torch.float32)
    s._kv_ok = False
    s.use_block = False
    s.adapt = False
    s._is_src = True
    s._tp = SimpleNamespace(world_size=1, rank_in_group=0,
                            broadcast=lambda *a, **k: None)
    return s


def test_propose_gathers_trailing_window_and_publishes():
    spec = _spec()
    seen = {}

    class _Proposer:
        def propose(self, toks, n, temperature=None):
            seen["toks"] = toks.clone()
            seen["n"] = n
            return torch.full((toks.shape[0], n), 7, dtype=torch.int64)

    spec.proposer = _Proposer()
    spec.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.arange(20).reshape(2, 10)),
        total_len=SimpleNamespace(gpu=torch.tensor([10, 5])),
    )
    batch = SimpleNamespace(num_reqs=2, idx_mapping=torch.tensor([0, 1]))

    out = spec.propose(
        batch, None, None, None, None, None, None, None, None,
        torch.zeros(2), torch.zeros(2, dtype=torch.int64),
    )
    # request 0: last 4 of 10 -> 6..9; request 1 (row starts at 10): 11..14
    assert seen["toks"].tolist() == [[6, 7, 8, 9], [11, 12, 13, 14]]
    assert seen["n"] == 3
    assert out.tolist() == [[7, 7, 7], [7, 7, 7]]
    assert spec.draft_tokens[:2].tolist() == out.tolist()


def test_propose_short_request_does_not_read_before_start():
    spec = _spec(window=4, steps=2, reqs=1)
    seen = {}

    class _Proposer:
        def propose(self, toks, n, temperature=None):
            seen["toks"] = toks.clone()
            return torch.zeros(toks.shape[0], n, dtype=torch.int64)

    spec.proposer = _Proposer()
    spec.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.arange(3).reshape(1, 3)),
        total_len=SimpleNamespace(gpu=torch.tensor([2])),
    )
    batch = SimpleNamespace(num_reqs=1, idx_mapping=torch.tensor([0]))
    spec.propose(batch, None, None, None, None, None, None, None, None,
                 torch.zeros(1), torch.zeros(1, dtype=torch.int64))
    # two real tokens, right-aligned; the window clamps at the first token
    assert seen["toks"].tolist() == [[0, 0, 0, 1]]


def test_kv_proposes_after_last_sample_and_masks_chunked_prefill():
    spec = _spec(steps=5)
    spec._kv_ok = True
    spec._block_len = 4
    spec.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.arange(20).reshape(2, 10)),
        total_len=SimpleNamespace(gpu=torch.tensor([10, 9])),
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([3, 8])),
    )
    batch = SimpleNamespace(num_reqs=2, idx_mapping=torch.tensor([1, 0]))

    def read_kv(num_reqs, end):
        assert end.tolist() == [7, 2]
        return object()

    def propose(toks, kv, n, first_token):
        assert toks.tolist() == [[14, 15, 16, 17], [0, 0, 1, 2]]
        assert first_token.tolist() == [18, 9]
        assert n == 4
        return first_token[:, None] + torch.arange(n)

    spec._read_kv_streams = read_kv
    spec.proposer = SimpleNamespace(propose_block_with_kv=propose)
    out = spec.propose(batch, None, None, None, None, torch.tensor([3, 0]),
                       None, None, None, torch.ones(2), None,
                       num_speculative_tokens=4)
    assert out.tolist() == [[19, 20, 21, -1, -1], [-1] * 5]


def test_kv_streams_follow_each_tap_group_and_refresh_after_cache_rebind():
    from vllm.v1.worker.gpu.spec_decode.ksdraft.speculator import (
        KS_TAPS,
        _inline_scale_views,
    )

    spec = _spec(window=2, reqs=1)
    spec.kv_window = 2
    spec._tap_views = None
    spec._tap_ptrs = ()
    layers = [None] * 23
    groups = [[], []]
    for tap, (index, pass_id) in enumerate(KS_TAPS):
        prefix = f"layer.{index}.pass.{pass_id}"
        group = (tap + 1) % 2
        groups[group].append(prefix)
        cache = torch.zeros((3, 2, 2, 264), dtype=torch.int8)
        k, ks, v, vs = _inline_scale_views(cache, 128, 128)
        for block in range(3):
            for off in range(2):
                k[block, :, off] = tap * 10 + block * 2 + off
        v.copy_(-k)
        ks.fill_(tap + 0.5)
        vs.fill_(tap + 1.5)
        obj = SimpleNamespace(prefix=prefix, kv_cache=cache)
        layers[index] = SimpleNamespace(attn=SimpleNamespace(
            head_dim=128, caches=[None] * pass_id + [obj]))
    spec._target_model = SimpleNamespace(layers=layers)
    spec._tp.gather = lambda tensor, **kwargs: tensor
    spec.set_attn(kv_cache_config=SimpleNamespace(kv_cache_groups=[
        SimpleNamespace(layer_names=names) for names in groups]),
        block_tables=SimpleNamespace(input_block_tables=[
            torch.tensor([[1, 2]]), torch.tensor([[2, 1]])]))

    k, ks, v, vs = spec._read_kv_streams(1, torch.tensor([2]))
    assert k[0, :, :, 0].tolist() == [[5, 5, 2], [13, 13, 14],
                                        [25, 25, 22], [33, 33, 34]]
    assert torch.equal(v, -k)
    assert ks[0, :, 0, 0].tolist() == [0.5, 1.5, 2.5, 3.5]
    assert torch.equal(vs, ks + 1)
    obj.kv_cache = torch.zeros_like(cache)
    k, _, _, _ = spec._read_kv_streams(1, torch.tensor([2]))
    assert k[0, -1].count_nonzero() == 0
