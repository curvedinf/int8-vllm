#!/usr/bin/env python3
"""PTQR retraining of the DFlash2 draft for the all-int8 stack (Phase 2).

STATUS: skeleton — model implementation + loading written; training loop
reuses ptqr_train_target machinery. NOT yet smoke-tested (see ledger).

Draft architecture (from vllm/model_executor/models/qwen3_dflash2.py and the
bf16 checkpoint at <models>/dflash2-bf16-with-tokenizer, 81 tensors):
  * 5 decoder layers, sliding-window attention (window from config), head_dim
    128, hidden 5120 — q/k/v/o + gate/up/down are the W8A8 GEMM surfaces.
  * Each layer wraps attn and MLP with a DFlashGroupedConv (taps=2,
    group_size=16, block_size=1+NS): kernel_projection Linear + base_kernel
    [2, taps, hidden] parameter (bf16 conv surface — kernel_projection gets
    the deployed per-channel W8A8 contract, see process_weights_after_loading).
  * Head: hidden_norm (RMSNorm) + fc Linear (per-channel int8 at serve).
  * candidate_selector: predecessor/successor codebooks (vocab x rank, keep
    bf16 — measured float exception) + hidden_projection Linear (G128 W8A8).
  * Draft CONSUMES target hidden states at target layers [5,19,33,47,61];
    each exit predicts the target's next token.

Training plan (per goal docs):
  Stage A: precompute target hidden states at the 5 exit layers for the
    training corpus using the FROZEN bf16 target (reuse
    ptqr_train_target.build_lm_model --no_ptqr; hook the 5 layers; save
    [N, L_exit, hidden] fp16 shards to disk).
  Stage B (this script's loop): student = PTQR-quantized draft (PTQRLinear on
    all Linear surfaces above, deployed contracts), teacher = frozen bf16
    draft, both fed the SAME cached target hidden states; loss =
    sum over exits of KL(student || teacher) + CE vs the true next token.
  Gate: greedy acceptance at 40k ctx >= 3.67/14 (PRING probe) — the trainer's
    own KL is only a progress signal.

Usage (stage B):
  python scripts/ptqr_train_draft.py --states <cache.pt> --steps 60 --lr 1e-5
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from ptqr_train_target import PTQRLinear, fake_quant_embedding_  # noqa: E402

DRAFT_CKPT = "/home/curved/models/dflash2-bf16-with-tokenizer/model.safetensors"


# ---------------------------------------------------------------------------
# DFlash grouped conv (pure-torch port of qwen3_dflash2._grouped_conv; the
# training path uses dense ops — deploy uses the same math).
# ---------------------------------------------------------------------------

class DFlashGroupedConv(nn.Module):
    def __init__(self, hidden_size: int, taps: int, group_size: int,
                 block_size: int):
        super().__init__()
        self.block_size, self.taps, self.group_size = block_size, taps, group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.zeros(2, taps, hidden_size), requires_grad=False)
        self.kernel_projection = nn.Linear(hidden_size, 2 * taps * self.num_groups,
                                           bias=False)

    def _convolve(self, hidden_states: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        # hidden_states: [T, B, H]; delta: [T, B, 2*taps*groups]
        T, B, H = hidden_states.shape
        blocks = hidden_states.reshape(T, B, self.num_groups, self.group_size)
        coeff = self.base_kernel.view(1, 2, self.taps, self.num_groups,
                                      self.group_size) \
            + delta.reshape(T, B, 2, self.taps, self.num_groups, 1)
        which = delta  # placeholder; see prepare/finish split below
        raise NotImplementedError("see prepare/finish split in vllm impl; "
                                  "port before first run")

    def forward(self, x):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Draft model (sliding-attention decoder + conv wrappers + head + selector)
# ---------------------------------------------------------------------------

class DraftLayer(nn.Module):
    """One DFlash2 draft decoder layer (attn & MLP wrapped by grouped conv)."""

    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        # TODO: sliding-window attention — port from transformers Qwen3 with
        # window = cfg.sliding_window; q/k/v/o as PTQRLinears after load.
        raise NotImplementedError


class DraftModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError

    @classmethod
    def from_checkpoint(cls, path: str = DRAFT_CKPT) -> "DraftModel":
        from safetensors.torch import load_file
        sd = load_file(path)
        # TODO: build model from config.json (window, layer count, selector
        # rank/top_k), load_state_dict(strict=True).
        raise NotImplementedError


def replace_draft_linears(model: nn.Module, group: int = 128) -> dict[str, PTQRLinear]:
    """PTQR-wrap every Linear with its deployed contract:
    G128 W8A8 for q/k/v/o, gate/up/down, kernel_projection, selector
    hidden_projection; per-channel fp32 for fc (CK head path). Codebooks and
    base_kernel stay bf16 (not Linears).
    """
    replaced = {}
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{child_name}" if name else child_name
            is_head = child_name == "fc"
            pq = PTQRLinear(child.weight,
                            group=child.weight.shape[1] if is_head else group,
                            scale_fp16=not is_head)
            setattr(module, child_name, pq)
            replaced[full] = pq
    return replaced


# ---------------------------------------------------------------------------
# SERVING SEMANTICS (extracted from qwen3_dflash.py, verified 2026-09-06):
# The draft forward that training must replicate per exit layer i (0..4):
#   1. QUERY: embed(draft tokens) [* input_embedding_scale] -> [T_q, B, H]
#      (draft tokens = the committed anchor + NS mask tokens; mask_embedding
#      is a parameter, mask_token_id=248070).
#   2. CONTEXT: target hidden states at exit layer (target layers 5/19/33/47/61)
#      are projected per draft layer to K/V (fused _project_context_kv over
#      all 5 layers at serve), K gets the draft's per-layer k_norm, then RoPE
#      — written to that layer's KV cache (precompute_and_store_context_kv).
#   3. Draft layer i = standard Qwen3 decoder step on the query attending to
#      the context KV: input_layernorm -> attention_conv.prepare -> self_attn
#      -> attention_conv.finish -> post_attention_layernorm -> mlp_conv.prepare
#      -> mlp -> mlp_conv.finish (conv wraps attn AND mlp; prepare returns
#      (mixed, coeffs[1]); finish convolves with coeffs[1]).
#   4. norm(hidden, residual) -> hidden_norm -> fc -> per-exit logits
#      (candidate top-k via lm_head + selector for the beam; TRAINING loss
#      can use dense fc logits KL vs teacher + CE on true next token).
# Training loop therefore needs, per sequence: target exit-layer hidden states
# (stage-A cache) + the draft token ids/mask embedding. The conv math itself
# is the small _grouped_conv above (taps=2, group=16, block=1+NS).
# ---------------------------------------------------------------------------

def precompute_target_states(out_path: str, n_seqs: int = 256,
                             seq_len: int = 1024, exits=(5, 19, 33, 47, 61)):
    """Cache [n_seqs, 5, seq_len, hidden] target states + next tokens.

    Reuses the trainer's target build; hooks the 5 exit layers; runs under
    no_grad on the bf16 reference. Roughly 256*5*1024*5120*2B = 13 GiB fp16.
    """
    raise NotImplementedError("port from ptqr_train_target control harness")


# ---------------------------------------------------------------------------
# Stage B loop (student vs teacher on cached states)
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--states", required=True, help="stage-A cache .pt")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-5)
    args = p.parse_args()
    raise NotImplementedError("loop: KL(student||teacher) per exit + CE; "
                              "per-tensor SR SGD from ptqr_train_target")


if __name__ == "__main__":
    main()
