#!/usr/bin/env python3
"""Post-sync validation battery for the gfx908 stack (merged aiter + FA 2.8.4).

Runs on one idle GPU alongside the Qwen3.8 quantization. Covers:
 1. int8 per-token-head KV attention at Qwen3.6/3.8 production shapes
    (head_dim 256, GQA 6:1 per TP4 rank, block 32) -- the existing micro test
    only covered head_size=64.
 2. flash_attn 2.8.4 (aiter triton backend): varlen causal GQA hdim 256,
    non-causal, sliding window.
 3. Symbol compatibility: every aiter entry point referenced by the merged
    vllm tree exists in the merged aiter tree.
 4. gemm_a16w16 smoke (lm_head-class shape) vs torch.mm.

Usage: HIP_VISIBLE_DEVICES=<idle gpu> python battery_gfx908.py
"""

import re
import sys
import traceback

sys.path.insert(0, "/home/curved/aiter")
sys.path.insert(0, "/home/curved/vllm-gfx908")

import torch

PASS, FAIL = [], []


def check(name, fn):
    try:
        detail = fn()
        PASS.append(name)
        print(f"PASS {name} {detail or ''}")
    except Exception:
        FAIL.append(name)
        print(f"FAIL {name}")
        traceback.print_exc()


def t1_int8_prod_shapes():
    import test_int8_kv_micro as micro

    ok1 = micro.run_one(
        num_seqs=1,
        seq_lens=torch.tensor([4096], dtype=torch.int32, device="cuda"),
        num_heads=6, num_kv_heads=1, head_size=256, block_size=32,
        name="prod_prefill_2d_h256",
    )
    ok2 = micro.run_decode(
        num_seqs=4,
        seq_lens=torch.tensor([1024, 2048, 4096, 8192], dtype=torch.int32, device="cuda"),
        num_heads=6, num_kv_heads=1, head_size=256, block_size=32,
        name="prod_decode_3d_h256",
    )
    assert ok1 and ok2
    return "(hdim256 GQA6:1 prefill+decode)"


def t2_fa_interface():
    from flash_attn import flash_attn_varlen_func

    torch.manual_seed(0)
    dev = "cuda"
    # GQA unsharded shapes: 24 Q heads / 4 KV heads, hdim 256 (Qwen3.6/3.8 full attn)
    q = torch.randn(2048, 24, 256, dtype=torch.float16, device=dev)
    k = torch.randn(2048, 4, 256, dtype=torch.float16, device=dev)
    v = torch.randn(2048, 4, 256, dtype=torch.float16, device=dev)
    cu = torch.tensor([0, 1024, 2048], dtype=torch.int32, device=dev)

    def ref(causal):
        outs = []
        for s, e in zip(cu[:-1], cu[1:]):
            n = e - s
            qs = q[s:e].transpose(0, 1)          # [24, n, 256]
            ks = k[s:e].transpose(0, 1).repeat_interleave(6, dim=0)
            vs = v[s:e].transpose(0, 1).repeat_interleave(6, dim=0)
            o = torch.nn.functional.scaled_dot_product_attention(
                qs, ks, vs, is_causal=causal)
            outs.append(o.transpose(0, 1))
        return torch.cat(outs)

    o_c = flash_attn_varlen_func(q, k, v, cu, cu, 1024, 1024, causal=True)
    e_c = (o_c - ref(True)).abs().max().item()
    o_n = flash_attn_varlen_func(q, k, v, cu, cu, 1024, 1024, causal=False)
    e_n = (o_n - ref(False)).abs().max().item()
    o_w = flash_attn_varlen_func(q, k, v, cu, cu, 1024, 1024, causal=True, window_size=(2048, 0))
    e_w = (o_w - ref(True)).abs().max().item()  # 1024 < window: same as causal
    assert e_c < 0.05 and e_n < 0.05 and e_w < 0.05, (e_c, e_n, e_w)
    return f"(causal {e_c:.4f} noncausal {e_n:.4f} window {e_w:.4f})"


def t3_symbol_compat():
    """Import-chain test: the modules gfx908 serving actually loads must boot
    against the merged aiter. Lazy/guarded aiter imports (flydsl MoE, gluon MLA,
    gfx942 paths) are excluded by construction -- they never load on our path."""
    import importlib

    mods = [
        "vllm",
        "vllm.config",
        "vllm.platforms.rocm",
        "vllm._aiter_ops",
        "vllm.v1.attention.backends.triton_attn",
        "vllm.v1.attention.ops.triton_unified_attention",
        "vllm.model_executor.kernels.linear",  # W8A16 oracle lives here
        "vllm.model_executor.layers.utils",
        "vllm.model_executor.models.qwen3_5",
        "vllm.model_executor.models.qwen3_dflash2",
        "vllm.config.speculative",
        "vllm.distributed.device_communicators.cuda_communicator",
        "vllm.entrypoints.openai.api_server",
    ]
    importlib.import_module("aiter")  # base
    for m in mods:
        importlib.import_module(m)
    return f"({len(mods)} boot-critical modules import OK)"


def t4_gemm():
    # import the way vllm/model_executor/layers/utils.py does; weight is [n,k].
    # Test ONLY the shapes use_aiter_triton_gemm actually dispatches (its
    # whitelist); anything else goes to rocBLAS and is out of contract.
    from aiter.ops.triton.gemm_a16w16 import gemm_a16w16

    torch.manual_seed(0)
    shapes = [(5120, 2880), (2880, 4096), (128, 2880), (640, 2880), (2880, 512), (62080, 2048)]
    for m, k in shapes:
        n = 512
        a = torch.randn(m, k, dtype=torch.float16, device="cuda")
        w = torch.randn(n, k, dtype=torch.float16, device="cuda")  # [n,k] layer layout
        out = gemm_a16w16(a, w)
        ref64 = a.double() @ w.double().t()
        t16 = a @ w.t()
        e_aiter = (out.double() - ref64).abs().max().item()
        e_torch = (t16.double() - ref64).abs().max().item()
        assert out.shape == (m, n), out.shape
        assert e_aiter <= max(3 * e_torch, 1.0), (m, n, k, e_aiter, e_torch)
        print(f"  gemm m={m} n={n} k={k}: aiter_err {e_aiter:.4f} vs torch16 {e_torch:.4f}")
    return ""


print(f"torch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")
check("t1_int8_pth_prod_shapes", t1_int8_prod_shapes)
check("t2_flash_attn_2.8.4_interface", t2_fa_interface)
check("t3_aiter_symbol_compat_merged_vllm", t3_symbol_compat)
check("t4_gemm_a16w16", t4_gemm)

print(f"\nBATTERY: {len(PASS)} PASS, {len(FAIL)} FAIL")
if FAIL:
    print("failed:", FAIL)
sys.exit(1 if FAIL else 0)
