#!/usr/bin/env python3
"""Serving-vs-teacher hidden differential: run the SAME first-1024-token
corpus window through upstream vLLM Flint (TP4, bf16) with
return_hidden_states, dump layer 0/1/2/48 hiddens for comparison with
the PTQR teacher's sweep_states (same window, /tmp/teacher_states.pt).
"""
import torch
from vllm import LLM, SamplingParams

tok = torch.load("/home/curved/SDGraft/data/flint_longctx/train_tokens.pt",
                 weights_only=True)
ids = tok[:1024].long().tolist()

llm = LLM(model="/home/curved/models/flint-bf16", tensor_parallel_size=4,
          dtype="bfloat16", max_model_len=4096,
          gpu_memory_utilization=0.92, kv_cache_dtype="auto")
sp = SamplingParams(temperature=0.0, max_tokens=1,
                    prompt_logprobs=0, return_hidden_states=True)
out = llm.generate(prompt_token_ids=[ids], sampling_params=sp)
o = out[0]
hs = o.hidden_states  # list over steps; step0 = prompt
h0 = hs[0]            # list over layers [L+1][num_prompt_tokens, H]
print("layers:", len(h0), "tokens:", h0[0].shape[0], flush=True)
sel = {}
for li in (0, 1, 2, 3, len(h0) - 1):
    t = h0[li]
    if isinstance(t, torch.Tensor):
        sel[li] = t.float().cpu()
    else:
        sel[li] = t[0].float().cpu() if isinstance(t, (list, tuple)) else t
    print(f"layer {li}: rms "
          f"{sel[li].pow(2).mean().sqrt().item():.4f}", flush=True)
torch.save(sel, "/tmp/serving_states.pt")
lp = o.prompt_logprobs
vals = [next(iter(v.values())).logprob for v in lp[1:] if v]
print("SERVING NLL (same window):", -sum(vals) / len(vals), flush=True)
