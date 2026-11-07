#!/usr/bin/env python3
"""Ground-truth gate: upstream vLLM Flint (TP4, bf16) on the real checkpoint.

Greedy generation — coherent text proves the merged serving path end-to-end
and indicts the PTQR trainer-side reimplementations (teacher NLL 13.8,
~uniform); coherent-model + broken-trainer localizes the remaining
semantic divergence to FlintRef/StreamedFlintTeacher.
"""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/curved/models/flint-bf16")
    p.add_argument("--prompt", default="The capital of France is")
    args = p.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, tensor_parallel_size=4,
              dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.92,
              kv_cache_dtype="auto")
    out = llm.generate([args.prompt],
                       SamplingParams(temperature=0.0, max_tokens=24))
    print("GENERATION:", repr(out[0].outputs[0].text), flush=True)


if __name__ == "__main__":
    main()
