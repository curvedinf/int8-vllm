# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare teacher and server distributions on identical token IDs."""

import argparse
import json
import math
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def capture(teacher, processor, destination, images=()):
    import torch

    from scripts.inkling2_teacher import SOURCE_REVISION
    from scripts.kld_probe_v2 import PROMPTS

    rows = []
    vocab = teacher.config.text_config.unpadded_vocab_size
    device = teacher.get_input_embeddings().weight.device

    @torch.inference_mode()
    def record(batch, messages=None):
        logits = teacher(**batch, use_cache=False, logits_to_keep=1).logits
        scores, tokens = logits[0, -1, :vocab].float().log_softmax(-1).topk(20)
        if not scores.isfinite().all():
            raise ValueError("Nonfinite reference probabilities")
        row = {
            "ids": batch["input_ids"][0].tolist(),
            "top": tokens.tolist(),
            "logp": scores.tolist(),
        }
        if messages is not None:
            row["messages"] = messages
        rows.append(row)

    for prompt in PROMPTS:
        ids = processor.tokenizer.encode(prompt, add_special_tokens=False)
        record({"input_ids": torch.tensor([ids], device=device)})
    if images:
        import base64
        import io

        from PIL import Image

        for path in images:
            with Image.open(path) as source_image:
                image = source_image.convert("RGB")
                image.thumbnail((640, 640))
            payload = io.BytesIO()
            image.save(payload, format="PNG")
            url = (
                "data:image/png;base64," + base64.b64encode(payload.getvalue()).decode()
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {
                            "type": "text",
                            "text": "Describe what you see in this image.",
                        },
                    ],
                },
                {"role": "assistant", "content": "The image shows"},
            ]
            # Use the release's own template and image processor. Continue a
            # visible answer so the prediction measures image content.
            text = processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
                reasoning_effort="none",
            )
            batch = processor(text=text, images=image, return_tensors="pt").to(device)
            record(batch, messages)
    destination.write_text(
        json.dumps({"revision": SOURCE_REVISION, "rows": rows}) + "\n"
    )


def bucket_kl(p_logs, q_logs):
    """KL on the teacher's selected IDs and one bucket for all other IDs."""
    p, q = ([math.exp(value) for value in logs] for logs in (p_logs, q_logs))
    for values in (p, q):
        total = math.fsum(values)
        if total > 1.00001:
            raise ValueError(f"Token probabilities sum to {total}")
        values.append(max(0.0, 1 - total))
    return math.fsum(a * math.log(a / max(b, 1e-300)) for a, b in zip(p, q) if a)


def compare(reference, url, model, modality_filter="both", details=None):
    import collections
    import statistics

    rows = json.loads(reference.read_text())["rows"]
    scores = collections.defaultdict(list)
    matches = collections.Counter()
    positions = []
    for index, row in enumerate(rows):
        image = "messages" in row
        modality = "image" if image else "text"
        if modality_filter not in ("both", modality):
            continue
        requested = row["top"]
        body = {
            "model": model,
            "prompt": row["ids"],
            "add_special_tokens": False,
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": len(requested),
            "logprob_token_ids": requested,
            "return_tokens_as_token_ids": True,
        }
        endpoint = "/v1/completions"
        if image:
            endpoint = "/v1/chat/completions"
            del body["prompt"]
            del body["add_special_tokens"]
            body.update(
                messages=row["messages"],
                reasoning_effort="none",
                continue_final_message=True,
                add_generation_prompt=False,
                return_token_ids=True,
                logprobs=True,
                top_logprobs=len(requested),
            )
        headers = {"Content-Type": "application/json"}
        if key := os.getenv("VLLM_API_KEY"):
            headers["Authorization"] = "Bearer " + key
        request = Request(
            url.rstrip("/") + endpoint, json.dumps(body).encode(), headers
        )
        try:
            with urlopen(request, timeout=300) as stream:
                response = json.load(stream)
        except HTTPError as exc:
            reason = exc.read(2048).decode(errors="replace")
            raise RuntimeError(
                f"{modality} position {index}: HTTP {exc.code}: {reason}"
            ) from exc
        if response["usage"]["prompt_tokens"] != len(row["ids"]):
            raise ValueError("The serving tokenizer changed the prompt length")
        lp = response["choices"][0]["logprobs"]
        if image:
            if response.get("prompt_token_ids") != row["ids"]:
                raise ValueError("Vision prompt tokens differ from the teacher")
            token = lp["content"][0]
            greedy = token["token"]
            values = {
                entry["token"]: entry["logprob"] for entry in token["top_logprobs"]
            }
        else:
            greedy, values = lp["tokens"][0], lp["top_logprobs"][0]
        matches[modality] += greedy == f"token_id:{requested[0]}"
        kl = bucket_kl(row["logp"], [values[f"token_id:{i}"] for i in requested])
        scores[modality].append(kl)
        if details:
            positions.append(
                dict(
                    index=index,
                    modality=modality,
                    kl=kl,
                    teacher_top1=requested[0],
                    served_token=greedy,
                )
            )
    if details:
        details.write_text(json.dumps(positions, indent=2) + "\n")
    return {
        modality: {
            "positions": len(values),
            "greedy_matches": matches[modality],
            "mean_top20_plus_tail_kl": statistics.mean(values),
            "median_top20_plus_tail_kl": statistics.median(values),
            "max_top20_plus_tail_kl": max(values),
        }
        for modality, values in scores.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser("capture")
    reference.add_argument("source", type=Path)
    reference.add_argument("artifact", type=Path)
    reference.add_argument("--image", action="append", type=Path, default=[])
    serving = commands.add_parser("compare")
    serving.add_argument("artifact", type=Path)
    serving.add_argument("--url", default="http://127.0.0.1:8020")
    serving.add_argument("--model", default="inkling-2-mini")
    serving.add_argument(
        "--modality", choices=("text", "image", "both"), default="both"
    )
    serving.add_argument("--details", type=Path)
    args = parser.parse_args()
    if args.command == "capture":
        from transformers import AutoProcessor

        from scripts.inkling2_teacher import load_teacher

        processor = AutoProcessor.from_pretrained(args.source, local_files_only=True)
        capture(load_teacher(args.source), processor, args.artifact, args.image)
    else:
        result = compare(
            args.artifact, args.url, args.model, args.modality, args.details
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
