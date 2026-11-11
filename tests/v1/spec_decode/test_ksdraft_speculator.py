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
