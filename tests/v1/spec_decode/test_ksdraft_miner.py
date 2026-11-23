# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.v1.worker.gpu.spec_decode.ksdraft.miner import gather_cache_kv


def test_gather_cache_kv_reads_cache_rows_and_zeroes_padding():
    torch.manual_seed(0)
    nb, bs, heads, d = 2, 4, 8, 128
    kv = torch.zeros(nb, bs, heads, 2 * (d + 2), dtype=torch.int8)

    k_codes = torch.randint(-127, 127, (heads, d), dtype=torch.int8)
    v_codes = torch.randint(-127, 127, (heads, d), dtype=torch.int8)
    k_scale = torch.full((heads,), 0.5, dtype=torch.float16)
    v_scale = torch.full((heads,), 0.25, dtype=torch.float16)

    def put(block: int, pos: int) -> None:
        kv[block, pos, :, :d] = k_codes
        kv[block, pos, :, d + 2:2 * d + 2] = v_codes
        kv[block, pos, :, d:d + 2] = k_scale.view(torch.int8).reshape(heads, 2)
        kv[block, pos, :, 2 * d + 2:2 * d + 4] = \
            v_scale.view(torch.int8).reshape(heads, 2)

    put(1, 1)   # slot 1 * bs + 1 = 5
    put(0, 2)   # slot 2
    slots = torch.tensor([5, 2, -1])

    kc, ks, vc, vs = gather_cache_kv(kv, slots, d)

    assert kc.shape == (3, heads, d) and ks.shape == (3, heads)
    assert torch.equal(kc[0], k_codes) and torch.equal(kc[1], k_codes)
    assert torch.equal(vc[0], v_codes) and torch.equal(vc[1], v_codes)
    assert torch.equal(ks[0], k_scale) and torch.equal(vs[0], v_scale)
    assert not kc[2].any() and not vc[2].any()
    assert not ks[2].any() and not vs[2].any()
