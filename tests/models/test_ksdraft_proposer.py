# SPDX-License-Identifier: Apache-2.0
import tempfile

import torch

from vllm.models.ksdraft.proposer import KSDraftProposer


def _export(tmp_path):
    from ksdraft.config import KSDraftConfig
    from ksdraft.export import export_model
    from ksdraft.model import KSDraftModel

    torch.manual_seed(0)
    cfg = KSDraftConfig(vocab_size=256, layers=2, d_model=96, head_size=32,
                        rosa_streams=32, ks3_charts=64, block_len=8)
    model = KSDraftModel(cfg, teacher_dim=0).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.05)
    d = tempfile.mkdtemp(dir=str(tmp_path))
    export_model(model, d)
    return d


def test_propose_block_greedy_and_sampled(tmp_path):
    prop = KSDraftProposer(_export(tmp_path), device="cpu", int8=False, window=8)
    tok = torch.randint(0, 256, (3, 8))
    greedy = prop.propose_block(tok, 4, temperature=torch.zeros(3))
    assert greedy.shape == (3, 4) and greedy.dtype == torch.int64
    # deterministic for greedy
    assert torch.equal(greedy, prop.propose_block(tok, 4,
                                                  temperature=torch.zeros(3)))
    samp = prop.propose_block(tok, 3, temperature=torch.ones(3))
    assert samp.shape == (3, 3)
    assert int(samp.max()) < 256 and int(samp.min()) >= 0


def test_propose_ar_still_works(tmp_path):
    prop = KSDraftProposer(_export(tmp_path), device="cpu", int8=False, window=8)
    tok = torch.randint(0, 256, (2, 8))
    out = prop.propose(tok, 2, temperature=torch.zeros(2))
    assert out.shape == (2, 2)


def test_adaptive_draft_len_policy():
    from vllm.models.ksdraft.proposer import adaptive_draft_len

    high = torch.full((2, 4), 0.9)
    assert adaptive_draft_len(high, 4, tau=0.3, load=2) == 4
    # load regime: full at <=4, half at <=8, one above 16
    assert adaptive_draft_len(high, 4, load=6) == 2
    assert adaptive_draft_len(high, 4, load=10) == 2
    assert adaptive_draft_len(high, 4, load=17) == 1
    # depth-rising: second offset below tau stops the block at 1
    mid = torch.tensor([[0.9, 0.2, 0.1, 0.05],
                        [0.8, 0.1, 0.1, 0.1]])
    assert adaptive_draft_len(mid, 4, tau=0.3, load=2) == 1
    # at least one token is always proposed
    low = torch.full((1, 4), 0.01)
    assert adaptive_draft_len(low, 4, tau=0.3, load=1) == 1


def test_propose_block_with_conf(tmp_path):
    prop = KSDraftProposer(_export(tmp_path), device="cpu", int8=False, window=8)
    tok = torch.randint(0, 256, (3, 8))
    toks, conf = prop.propose_block_with_conf(tok, 4)
    assert toks.shape == (3, 4)
    assert conf.shape == (3, 4)
    assert float(conf.min()) >= 0.0 and float(conf.max()) <= 1.0
