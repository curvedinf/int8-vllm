#!/usr/bin/env python3
"""Ground-truth gate v2: upstream vLLM Flint (TP4, bf16) — chat-templated
generation + prompt logprobs (token NLL of the serving model on natural
text). Compares against the PTQR teacher's NLL on the same tokens."""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/curved/models/flint-bf16")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    msgs = [{"role": "user", "content": "What is the capital of France? "
             "Answer with just the city name."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print("PROMPT:", repr(prompt), flush=True)

    llm = LLM(model=args.model, tensor_parallel_size=4,
              dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.92,
              kv_cache_dtype="auto")
    sp = SamplingParams(temperature=0.0, max_tokens=24, prompt_logprobs=0)
    out = llm.generate([prompt], sp)
    o = out[0]
    print("GENERATION:", repr(o.outputs[0].text), flush=True)
    lps = o.prompt_logprobs
    vals = [next(iter(v.values())).logprob for v in lps[1:] if v]
    if vals:
        import math
        nll = -sum(vals) / len(vals)
        print(f"SERVING prompt NLL over {len(vals)} tokens: {nll:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
