# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture on-policy target features for stock Inkling DSpark PTQR."""

import argparse
import json
import os
from pathlib import Path

import torch


def enable_capture(worker, directory):
    from vllm.distributed import get_tensor_model_parallel_rank

    rank = get_tensor_model_parallel_rank()
    target = worker.get_model().get_language_model()
    head = target.lm_head.quant_method.dequantize_weight(target.lm_head)
    torch.save(head.to(torch.bfloat16).cpu(), directory / f"head{rank}.pt")
    if rank != 0:
        return
    holder = target.model
    holder._set_aux_hidden_state_layers((5, 23, 41, 48))
    holder.ptqr_rows = []

    def capture(module, args, output):
        hidden, taps = output
        positions, anchor_embeddings = args
        module.ptqr_rows.append(
            (
                positions.detach().cpu(),
                torch.stack([*taps, anchor_embeddings], dim=1).to(torch.bfloat16).cpu(),
            )
        )
        return hidden

    holder.register_forward_hook(capture)


def save_sample(worker, dest, prompt_len, tokens, top_ids, top_lp):
    from vllm.distributed import get_tensor_model_parallel_rank

    if get_tensor_model_parallel_rank() != 0:
        return
    rows = worker.get_model().get_language_model().model.ptqr_rows
    positions, features = map(torch.cat, zip(*rows))
    if (
        not torch.equal(positions, torch.arange(len(tokens) - 1))
        or not features.isfinite().all()
    ):
        raise ValueError("Incomplete target state sequence")
    tmp = dest.with_suffix(".tmp")
    torch.save(
        dict(
            tokens=tokens,
            prompt_length=prompt_len,
            features=features,
            target_top_ids=top_ids,
            target_logprobs=top_lp,
        ),
        tmp,
    )
    tmp.replace(dest)
    rows.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "corpus", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--train", type=int, default=32)
    parser.add_argument("--validation", type=int, default=8)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=80)
    parser.add_argument(
        "--additional-config",
        default="{}",
        help=(
            "Serving additional-config overlay, so the capture runs under the "
            "deployed numerics"
        ),
    )
    parser.add_argument("--expert-rank-counts", type=int, nargs="+")
    args = parser.parse_args()
    # Dispatch the capture functions directly from the local driver to its TP
    # workers. No callable needs to traverse the engine's frontend socket.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from tokenizers import Tokenizer

    from vllm import LLM, SamplingParams

    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    corpus = [json.loads(line)["text"] for line in args.corpus.read_text().splitlines()]
    count = args.train + args.validation
    if count > len(corpus):
        raise ValueError("Each capture needs a distinct input")
    args.output.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=4,
        max_num_seqs=12,
        max_model_len=args.context + 512,
        max_num_batched_tokens=1024,
        enforce_eager=True,
        enable_prefix_caching=False,
        load_format="ais",
        kv_cache_dtype="int8_block_g128",
        block_size=16,
        max_logprobs=64,
        dtype="float16",
        tokenizer_mode="inkling",
        limit_mm_per_prompt={"image": 0},
        additional_config={
            "inkling2_rank_cache": False,
            **json.loads(args.additional_config),
        },
        **(
            {"expert_rank_counts": args.expert_rank_counts}
            if args.expert_rank_counts
            else {}
        ),
    )
    llm.collective_rpc(enable_capture, args=(args.output.resolve(),))
    for i in range(count):
        split = "train" if i < args.train else "validation"
        dest = args.output / f"{split}-{i:03d}.pt"
        if dest.exists():
            continue
        text = "\n\n".join(corpus[i::count])
        ids = tokenizer.encode(text, add_special_tokens=False).ids[: args.context - 128]
        msg = (
            "Explain the main ideas in this material and give one concrete example.\n\n"
            + tokenizer.decode(ids)
        )
        reply = llm.chat(
            [{"role": "user", "content": msg}],
            sampling_params=SamplingParams(
                temperature=1,
                seed=17011 + i,
                max_tokens=args.output_tokens,
                ignore_eos=True,
                logprobs=64,
            ),
            use_tqdm=False,
        )[0]
        prompt = list(reply.prompt_token_ids)
        tokens = torch.tensor(prompt + list(reply.outputs[0].token_ids))

        actual = reply.outputs[0].logprobs
        top = [
            sorted(row.items(), key=lambda item: -item[1].logprob)[:64]
            for row in actual
        ]
        top_ids = torch.tensor([[idx for idx, _ in row] for row in top])
        top_lp = torch.tensor([[lp.logprob for _, lp in row] for row in top])
        prompt_len = len(prompt)

        llm.collective_rpc(
            save_sample, args=(dest, prompt_len, tokens, top_ids, top_lp)
        )
        print(f"Saved {split} sequence {i} with {len(tokens)} tokens", flush=True)


if __name__ == "__main__":
    main()
