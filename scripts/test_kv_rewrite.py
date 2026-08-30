#!/usr/bin/env python3
"""Rewrite test for the int8 per-token-head KV cache-write kernel.

The spec-decode rejection path rewrites cache slots that a previous verify
already wrote (rejected draft positions). This test writes the SAME slots
repeatedly with different values via the REAL Triton kernel
(triton_reshape_and_cache_flash_per_token_head_quant) and checks payload AND
inline scales always reflect the LAST write. It also simulates the DFlash
step shape: write 14 slots (a draft block), accept k, rewrite the 14-k
rejected tail next round with new values.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_per_token_head_quant,
)
from vllm.v1.kv_cache_interface import KVQuantMode

DEV = "cuda"
DT = torch.float16
NKV = 2
HS = 128
BS = 32          # block size (slots per block)
SLOTS = BS * 8   # a few blocks
QUANT_MAX = 127.0

torch.manual_seed(0)

cache_k = torch.zeros(SLOTS // BS, BS, NKV, HS, dtype=torch.int8, device=DEV)
cache_v = torch.zeros(SLOTS // BS, BS, NKV, HS, dtype=torch.int8, device=DEV)
# Scale caches carved like the impl: float32 views over the padded halves.
ks = torch.full((SLOTS // BS, BS, NKV), float("nan"), dtype=torch.float32, device=DEV)
vs = torch.full((SLOTS // BS, BS, NKV), float("nan"), dtype=torch.float32, device=DEV)


def write(slots, k, v):
    triton_reshape_and_cache_flash_per_token_head_quant(
        k, v, cache_k, cache_v, ks, vs, slots, KVQuantMode.INT8_PER_TOKEN_HEAD
    )


def read(slots):
    ks_q = torch.stack([cache_k[s // BS, s % BS] for s in slots.tolist()])
    vs_q = torch.stack([cache_v[s // BS, s % BS] for s in slots.tolist()])
    ks_s = torch.stack([ks[s // BS, s % BS] for s in slots.tolist()])
    vs_s = torch.stack([vs[s // BS, s % BS] for s in slots.tolist()])
    k = ks_q.to(torch.float32) * ks_s.unsqueeze(-1)
    v = vs_q.to(torch.float32) * vs_s.unsqueeze(-1)
    return k, v


def quant_ref(x):
    """per-(token, head) symmetric int8 quantization reference."""
    scale = x.abs().amax(dim=-1) / QUANT_MAX
    scale = torch.clamp(scale, min=1e-12)
    q = torch.clamp(torch.round(x / scale.unsqueeze(-1)), -127, 127).to(torch.int8)
    return q, scale


def check(name, slots, k_expect, v_expect, tol=None):
    k, v = read(slots)
    # Reference quant of the expected values
    kq, ksc = quant_ref(k_expect)
    vq, vsc = quant_ref(v_expect)
    k_ref = kq.to(torch.float32) * ksc.unsqueeze(-1)
    v_ref = vq.to(torch.float32) * vsc.unsqueeze(-1)
    # Tolerance scales with the per-head quant step (kernel rounds
    # half-to-even; torch.round is half-away — one quant step of slack).
    step = k_expect.abs().amax(dim=-1).max().item() / QUANT_MAX
    tol = tol or max(0.05, 1.1 * step)
    dk = (k - k_ref).abs().max().item()
    dv = (v - v_ref).abs().max().item()
    finite = bool(torch.isfinite(k).all() and torch.isfinite(v).all())
    ok = dk < tol and dv < tol and finite
    print(f"{name}: max_k_err={dk:.4f} max_v_err={dv:.4f} finite={finite} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


ok = True

# --- Test 1: simple write once
slots = torch.arange(0, BS, device=DEV, dtype=torch.int64)
k1 = torch.randn(BS, NKV, HS, dtype=DT, device=DEV)
v1 = torch.randn(BS, NKV, HS, dtype=DT, device=DEV)
write(slots, k1, v1)
ok &= check("write-once", slots, k1.float(), v1.float())

# --- Test 2: rewrite the SAME slots with different values (rejection path)
k2 = torch.randn(BS, NKV, HS, dtype=DT, device=DEV) * 3
v2 = torch.randn(BS, NKV, HS, dtype=DT, device=DEV) * 0.1
write(slots, k2, v2)
ok &= check("rewrite-same-slots", slots, k2.float(), v2.float())

# --- Test 3: many rewrites (stress ordering/scale staleness)
for it in range(20):
    ki = torch.randn(BS, NKV, HS, dtype=DT, device=DEV)
    vi = torch.randn(BS, NKV, HS, dtype=DT, device=DEV)
    write(slots, ki, vi)
ok &= check("rewrite-x20", slots, ki.float(), vi.float())

# --- Test 4: DFlash shape — 14-slot block, accept k, rewrite the tail
base_slot = 64
tail_slots = torch.arange(base_slot, base_slot + 14, device=DEV, dtype=torch.int64)
kexp = torch.randn(14, NKV, HS, dtype=DT, device=DEV)
vexp = torch.randn(14, NKV, HS, dtype=DT, device=DEV)
write(tail_slots, kexp, vexp)
for accept in (0, 3, 7, 11):  # rewrite rejected tail with new values
    n_rej = 14 - accept
    rej = tail_slots[accept:]
    kr = torch.randn(n_rej, NKV, HS, dtype=DT, device=DEV)
    vr = torch.randn(n_rej, NKV, HS, dtype=DT, device=DEV)
    write(rej, kr, vr)
    kexp[accept:] = kr
    vexp[accept:] = vr
    ok &= check(f"dflash-accept{accept}", tail_slots, kexp.float(), vexp.float())

# --- Test 5: same-step double write (draft writes, then target verify writes
# the same slots — both models write the identical positions per step)
slots5 = torch.arange(128, 128 + 14, device=DEV, dtype=torch.int64)
k_draft = torch.randn(14, NKV, HS, dtype=DT, device=DEV)
v_draft = torch.randn(14, NKV, HS, dtype=DT, device=DEV)
k_tgt = torch.randn(14, NKV, HS, dtype=DT, device=DEV)
v_tgt = torch.randn(14, NKV, HS, dtype=DT, device=DEV)
write(slots5, k_draft, v_draft)
write(slots5, k_tgt, v_tgt)   # target must win (writes second)
ok &= check("double-write-target-wins", slots5, k_tgt.float(), v_tgt.float())

print("OVERALL:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
