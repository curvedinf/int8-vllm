#!/usr/bin/env python3
"""Ground-truth gate v3: upstream vLLM Flint in-process (V1
multiprocessing off so monkeypatches reach the model), chat-templated
generation + prompt NLL. --no_sinks neutralizes the learned attention
sinks (fills them with -1e9) to bisect the fork-UA sink path.
"""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/curved/models/flint-bf16")
    p.add_argument("--no_sinks", action="store_true")
    args = p.parse_args()

    import torch

    if args.no_sinks:
        import vllm.models.flint.model as fm

        _orig = fm.make_attention_sink

        def _dead_sink(n, dtype, *a, **k):
            return torch.full((n,), -1e9, dtype=dtype)

        fm.make_attention_sink = _dead_sink
        print("SINKS NEUTRALIZED (-1e9)", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    msgs = [{"role": "user", "content": "What is the capital of France? "
             "Answer with just the city name."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)

    llm = LLM(model=args.model, tensor_parallel_size=4,
              dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.92,
              kv_cache_dtype="auto")

    if args.no_sinks:
        import torch.nn as nn

        def find_sinks(obj, depth=0, seen=None):
            seen = seen or set()
            if id(obj) in seen or depth > 8:
                return []
            seen.add(id(obj))
            found = []
            mods = []
            if isinstance(obj, nn.Module) and hasattr(obj, "softmax_offset"):
                found.append(obj)
            for ch in getattr(obj, "children", lambda: [])():
                mods.append(ch)
            for a in dir(obj):
                if a.startswith("_"):
                    continue
                try:
                    v = getattr(obj, a)
                except Exception:
                    continue
                if isinstance(v, nn.Module):
                    mods.append(v)
            for m in mods:
                found += find_sinks(m, depth + 1, seen)
            return found

        mods = find_sinks(llm.llm_engine)
        n = 0
        for mod in mods:
            with torch.no_grad():
                mod.softmax_offset.fill_(-1e9)
            n += 1
        print(f"SINKS NEUTRALIZED POST-LOAD: {n} layers", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=24, prompt_logprobs=0)
    out = llm.generate([prompt], sp)
    o = out[0]
    print("GENERATION:", repr(o.outputs[0].text), flush=True)
    vals = [next(iter(v.values())).logprob
            for v in o.prompt_logprobs[1:] if v]
    if vals:
        print(f"SERVING prompt NLL over {len(vals)} tokens: "
              f"{-sum(vals) / len(vals):.4f}", flush=True)


if __name__ == "__main__":
    main()
