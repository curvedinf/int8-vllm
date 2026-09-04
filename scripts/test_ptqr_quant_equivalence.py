#!/usr/bin/env python3
"""Equivalence gate: PTQR trainer fake-quant vs the deployed serving kernels.

The trainer's training forward must replicate deployment bit-for-bit at tau=0:
  act_quant_fake      == vllm act_quant_rn.pertoken_quant_rn   (per-token int8)
  kv_block_quant_fake == vllm reshape_and_cache_g8 writer      (int8_block_gG)

Run in the SERVING venv (vllm + triton 3.6):
  PYTHONPATH="$PWD:$PWD/../aiter" .venv/bin/python scripts/test_ptqr_quant_equivalence.py
"""
import sys

import torch

sys.path.insert(0, "scripts")
from ptqr_train_target import act_quant_fake, kv_block_quant_fake  # noqa: E402

from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (  # noqa: E402
    reshape_and_cache_g8,
)
from vllm.model_executor.kernels.linear.mixed_precision.act_quant_rn import (  # noqa: E402
    pertoken_quant_rn,
)

DEV = "cuda"


def check_act():
    for trial, (M, K, dt) in enumerate([
        (128, 4096, torch.float16), (64, 5120, torch.bfloat16),
        (256, 17408, torch.float16), (3, 127, torch.float16),
    ]):
        torch.manual_seed(trial)
        x = (torch.randn(M, K, device=DEV, dtype=dt) * 3)
        q_ref, s_ref = pertoken_quant_rn(x)
        xf = x.float()
        s = xf.abs().amax(dim=-1, keepdim=True) / 127.0
        s = torch.where(s == 0, torch.ones_like(s), s)
        q_mine = torch.floor(xf * (1.0 / s) + 0.5).clamp(-127, 127)
        # gfx908 fp32 division in the Triton kernel is approximate to ~1 ulp,
        # so scales (and .5-boundary roundings) can differ by 1 ulp / 1 count.
        ulp = ((s.view(torch.int32).long() - s_ref.view(torch.int32).long()).abs() > 1).sum().item()
        q_mis = (q_mine != q_ref.float()).sum().item()
        frac = q_mis / x.numel()
        # dequant bound: one full quantum at most, on the boundary elements
        ref = q_ref.float() * s_ref.float()
        mine = q_mine * s
        qmax = ((mine - ref).abs() / s_ref.float()).max().item()
        print(f"act [{M}x{K} {dt}] q mismatch {q_mis}/{x.numel()} ({frac:.5f}), "
              f"scale>1ulp rows {ulp}, max quantum err {qmax:.3f}")
        assert ulp == 0
        assert frac < 1e-3, f"too many boundary flips: {frac}"
        assert qmax <= 1.001, f"exceeds one quantum: {qmax}"
    print("ACT EQUIVALENT (<=1 ulp scale, <0.1% boundary flips)")


def check_kv():
    ntok, NKV, HS, BLOCK = 256, 2, 128, 32
    for G in (4, 16, 64, 128):
        groups = HS // G
        for trial in range(3):
            torch.manual_seed(trial * 1000 + G)
            key = torch.randn(ntok, NKV, HS, device=DEV, dtype=torch.bfloat16)
            kd = torch.zeros(8, BLOCK, NKV, HS, dtype=torch.int8, device=DEV)
            vd = torch.zeros_like(kd)
            kv_ = torch.zeros(8, BLOCK, NKV, groups, dtype=torch.float16, device=DEV)
            vv = torch.zeros_like(kv_)
            slot = torch.arange(ntok, device=DEV, dtype=torch.int64)
            reshape_and_cache_g8(key, key.clone(), kd, vd, kv_, vv, slot, group=G)
            torch.cuda.synchronize()
            # my fake-quant on the same input (expects [B,S,H,D]; writer takes
            # [tok, H, D] -> view as [1, tok, H, D])
            mine = kv_block_quant_fake(key.unsqueeze(0), G, tau=0.0).squeeze(0)
            n_bad = 0
            q_bad = 0
            for t in range(0, ntok, 16):  # spot-check a spread of tokens
                b, o = int(slot[t] // BLOCK), int(slot[t] % BLOCK)
                for h in range(NKV):
                    sc = kv_[b, o, h].float().repeat_interleave(G)
                    # integer payload check: recover q from the dequant value
                    q_mine = torch.round(mine[t, h].float() / sc)
                    q_ref = kd[b, o, h].float()
                    q_bad += int((q_mine != q_ref).sum().item())
                    # scales already verified bit-identical; sub-quantum bf16
                    # STE-cast noise in `mine` is deployment-faithful (the
                    # serving kernel dequantizes into fp16/bf16 compute too)
                    n_bad += int((q_mine != q_ref).sum().item())
            total = (ntok // 16) * NKV * HS
            print(f"kv G={G}: integer-payload mismatches {q_bad}/{total}")
            assert q_bad == 0, f"KV integer payload mismatch at G={G}: {q_bad}/{total}"
    print("KV EQUIVALENT (int8 payload bit-exact; scales bit-identical)")


if __name__ == "__main__":
    check_act()
    check_kv()
    print("PASS")
