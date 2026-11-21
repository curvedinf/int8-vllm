#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check HY4 probabilities, chat, vision and tool calls on a live server."""

import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
import math
import os
import random
import re
import statistics
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path

import requests

from scripts.kld_probe_v2 import PROMPTS, SEED
from scripts.ua_live_soak import check_reply


def response(args, endpoint, body):
    headers = {}
    if key := os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + key
    result = requests.post(
        args.url + endpoint,
        json={"model": args.model, **body},
        headers=headers,
        timeout=args.timeout,
    )
    if not result.ok:
        raise RuntimeError(f"HTTP {result.status_code}: {result.text[:1000]}")
    return result.json()


def request(args, endpoint, body):
    return response(args, endpoint, body)["choices"][0]


def chat(args, messages, **kwargs):
    return request(
        args,
        "/v1/chat/completions",
        {
            "messages": messages,
            "temperature": 0,
            "seed": SEED,
            "max_tokens": 128,
            "chat_template_kwargs": {"reasoning_effort": "no_think"},
            **kwargs,
        },
    )


def strict_format(index, reply):
    """Score the explicit formatting requirements in the fixed 52-prompt corpus."""
    answer = reply.strip()
    exact = {
        24: "APPLE-42",
        27: "canberra",
        28: "banana",
        29: "<answer>42</answer>",
        30: "12",
    }
    if index in exact:
        return (answer.casefold() if index in (27, 28) else answer) == exact[index]
    if index == 25:
        try:
            return json.loads(answer) == {"a": 5, "b": "cat"}
        except ValueError:
            return False
    if index == 26:
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        return len(lines) == 3 and all(
            re.fullmatch(rf"{i}[.)]\s+[A-Za-z][A-Za-z -]*[.]?", line)
            for i, line in enumerate(lines, 1)
        )
    if index == 31:
        return len(re.findall(r"[^.!?]+[.!?]+", answer)) == 2 and bool(
            re.search(r"[.!?]$", answer)
        )
    if index == 32:
        return answer.casefold().rstrip("!. ") in {"bonjour", "bon matin"}
    if index == 33:
        return re.fullmatch(r"1\s*,\s*2\s*,\s*3\s*,\s*4\s*,\s*5", answer) is not None
    return None


def corpus(args, entry):
    i, prompt = entry
    result = chat(
        args,
        [{"role": "user", "content": prompt}],
        max_tokens=args.chat_tokens,
        logprobs=True,
        top_logprobs=3,
        return_token_ids=True,
    )
    reply = result["message"].get("content") or ""
    positions = (result.get("logprobs") or {}).get("content") or []
    finite = bool(positions) and all(
        math.isfinite(v["logprob"])
        and all(math.isfinite(p["logprob"]) for p in v.get("top_logprobs", []))
        for v in positions
    )
    issue = check_reply(reply)
    strict = strict_format(i, reply)
    if strict is not None:
        issue = None if strict else "requested output format or answer did not match"
    elif i == 19 and reply.strip() == "11":
        issue = None
    if result["message"].get("reasoning") or re.search(
        r"<(?:think|｜reasoning_mode)", reply
    ):
        issue = "reasoning appeared in a no_think answer"
    ids = result.get("token_ids") or []
    return {
        "index": i,
        "pass": not issue and finite,
        "issue": issue,
        "finite_logprobs": finite,
        "strict_format": strict,
        "finish": result["finish_reason"],
        "content": reply,
        "ids_hash": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
    }


def multiple_choice(args, entry):
    index, row = entry
    letters = "ABCD"
    if len(row["choices"]) != len(letters) or row["answer"] not in tuple(letters):
        raise ValueError(f"Invalid multiple-choice question {index}")
    options = "\n".join(
        f"{letter}. {option}"
        for letter, option in zip(letters, row["choices"], strict=True)
    )
    prompt = f"{row['question']}\n\n{options}\n\nAnswer with just the letter."
    result = chat(args, [{"role": "user", "content": prompt}], max_tokens=8)
    content = (result["message"].get("content") or "").strip()
    match = re.fullmatch(r"([A-D])[.)]?", content, flags=re.IGNORECASE)
    answer = match.group(1).upper() if match else None
    return {
        "index": index,
        "subject": row.get("subject"),
        "expected": row["answer"],
        "answer": answer,
        "content": content,
        "correct": answer == row["answer"],
    }


def teacher_row(args, row):
    picked = request(
        args,
        "/v1/completions",
        {
            "prompt": row["input"],
            "cache_salt": uuid.uuid4().hex,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": len(row["top"]),
            "logprob_token_ids": row["top"],
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "seed": SEED,
        },
    )
    returned = picked["logprobs"]["top_logprobs"][0]
    selected = {int(k.removeprefix("token_id:")): v for k, v in returned.items()}
    logq = [selected[ix] for ix in row["top"]]
    logp = row["logprobs"]
    pp = [math.exp(x) for x in logp]
    other_p = max(0.0, 1.0 - sum(pp))
    other_q = max(1e-20, 1.0 - sum(map(math.exp, logq)))
    kl = sum(p * (a - b) for p, a, b in zip(pp, logp, logq))
    if other_p:
        kl += other_p * math.log(other_p / other_q)
    generated = picked["token_ids"][0]
    return {
        "tokens": len(row["input"]),
        "teacher_top1": row["top"][0],
        "target_top1": generated,
        "top1_agrees": generated == row["top"][0],
        "kl_top20_and_tail": max(0.0, kl),
    }


def media_message(data, mime, question, modality="image"):
    uri = "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")
    field = modality + "_url"
    return [
        {
            "role": "user",
            "content": [
                {"type": field, field: {"url": uri}},
                {"type": "text", "text": question},
            ],
        }
    ]


def modalities(args):
    from PIL import Image

    cases = []
    for color in ("red", "blue"):
        f = io.BytesIO()
        Image.new("RGB", (224, 224), color=color).save(f, "PNG")
        msgs = media_message(
            f.getvalue(),
            "image/png",
            "What is the color filling this image? Answer with one color word.",
        )
        text = chat(args, msgs, max_tokens=32)["message"].get("content") or ""
        cases.append(
            {
                "case": color,
                "pass": bool(re.search(r"\b" + color + r"\b", text, re.I)),
                "content": text,
            }
        )
    if args.cats:
        msgs = media_message(
            args.cats.read_bytes(),
            "image/jpeg",
            "What animals can you see on the furniture?",
        )
        text = chat(args, msgs, max_tokens=64)["message"].get("content") or ""
        cases.append(
            {
                "case": "COCO val2017 image 39769",
                "pass": bool(re.search(r"cat|kitten", text, re.I)),
                "content": text,
            }
        )
    if args.red_video:
        msgs = media_message(
            args.red_video.read_bytes(),
            "video/x-msvideo",
            "What color fills this video? Answer with one color word.",
            "video",
        )
        text = chat(args, msgs, max_tokens=32)["message"].get("content") or ""
        cases.append(
            {
                "case": "four-frame red video",
                "pass": bool(re.search(r"\bred\b", text, re.I)),
                "content": text,
            }
        )
    tool = {
        "type": "function",
        "function": {
            "name": "lookup_temperature",
            "description": "Get the temperature in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    result = chat(
        args,
        [{"role": "user", "content": "Use lookup_temperature for Paris."}],
        tools=[tool],
        tool_choice="auto",
        max_tokens=96,
    )
    calls = result["message"].get("tool_calls") or []
    good = False
    if len(calls) == 1 and calls[0]["function"]["name"] == "lookup_temperature":
        with suppress(KeyError, ValueError):
            good = (
                "paris" in json.loads(calls[0]["function"]["arguments"])["city"].lower()
            )
    cases.append({"case": "tool name and arguments", "pass": good, "calls": calls})
    return cases


def reasoning_modes(args):
    question = [
        {"role": "user", "content": "What is seven plus five? Answer in digits."}
    ]
    cases = {}
    for name, options in (
        ("default", {}),
        ("no_think", {"reasoning_effort": "no_think"}),
    ):
        choice = chat(args, question, chat_template_kwargs=options, max_tokens=512)
        message = choice["message"]
        answer = (message.get("content") or "").strip()
        thinking = bool((message.get("reasoning") or "").strip())
        cases[name] = {
            "pass": (
                bool(re.search(r"\b12\b", answer))
                and "<think" not in answer
                and "</think" not in answer
                and thinking == (name == "default")
                and choice["finish_reason"] == "stop"
            ),
            "content": answer,
            "has_reasoning": thinking,
            "finish": choice["finish_reason"],
        }
    return cases


def exact_context_prompt(args, passage, length, variant=0):
    """Build an exact-length token prompt around the server's own chat template."""
    slot = "HY4ARCHIVEFILL49517"
    tag = hashlib.sha256(f"{SEED}:{length}:{variant}".encode()).hexdigest()[:7].upper()
    answer = f"CEDAR-{tag}"
    template = response(
        args,
        "/tokenize",
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"Review these source excerpts.\n\n{slot}\n\n"
                        f"For this review, the release codename is {answer}.\n\n"
                        f"{slot}\n\nEnd of the source excerpts."
                    ),
                },
                {
                    "role": "user",
                    "content": "What release codename was assigned to this review? "
                    "Copy it exactly. Reply with the codename only.",
                },
            ],
            "chat_template_kwargs": {"reasoning_effort": "no_think"},
        },
    )
    ids = template["tokens"]

    def raw(text):
        return response(
            args, "/tokenize", {"prompt": text, "add_special_tokens": False}
        )["tokens"]

    marker = raw(slot)
    left, right = [i for i in range(len(ids)) if ids[i : i + len(marker)] == marker]
    fill = length - (len(ids) - 2 * len(marker))
    if fill < 1:
        raise ValueError(f"Context length {length} is shorter than the query template")
    # The chunks contain real repository text. Shuffle them on every pass
    # so one cacheable repeated document cannot stand in for the context.
    source = raw(passage)
    chunks = [source[i : i + 512] for i in range(0, len(source), 512)]
    if not chunks:
        raise ValueError("The context manifest has an empty shared passage")
    rng = random.Random(SEED + length + 9973 * variant)
    filler = []
    while len(filler) < fill:
        rng.shuffle(chunks)
        for chunk in chunks:
            filler.extend(chunk)
            if len(filler) >= fill:
                break
    first = fill // 5
    prompt = (
        ids[:left]
        + filler[:first]
        + ids[left + len(marker) : right]
        + filler[first:fill]
        + ids[right + len(marker) :]
    )
    assert len(prompt) == length
    return prompt, answer, left + first, template["max_model_len"]


def exact_completion(args, prompt, answer, salt, *, fill=0):
    budget = fill or 32
    body = {
        "prompt": prompt,
        "add_special_tokens": False,
        "temperature": 0,
        "seed": SEED,
        "max_tokens": budget,
        "cache_salt": salt,
        "return_token_ids": True,
    }
    if fill:
        body.update(min_tokens=fill, ignore_eos=True)
    start = time.monotonic()
    data = response(args, "/v1/completions", body)
    choice = data["choices"][0]
    usage = data["usage"]
    if usage["prompt_tokens"] != len(prompt) or choice["prompt_token_ids"] != prompt:
        raise RuntimeError("Server consumed a different prompt from the submitted IDs")
    details = usage.get("prompt_tokens_details") or {}
    content = choice["text"].strip()
    return {
        "content": content,
        "correct": content.startswith(answer) if fill else content == answer,
        "prompt": usage["prompt_tokens"],
        "output": usage["completion_tokens"],
        "cached": details.get("cached_tokens"),
        "ids": choice["token_ids"],
        "seconds": time.monotonic() - start,
    }


def pool_counters(args):
    key = os.environ.get("VLLM_API_KEY")
    headers = {"Authorization": "Bearer " + key} if key else {}
    res = requests.get(args.url + "/metrics", headers=headers, timeout=10)
    res.raise_for_status()
    values = {}
    for name in (
        "num_requests_running",
        "num_preemptions_total",
        "prefix_cache_hits_total",
    ):
        values[name] = sum(
            float(line.split()[-1])
            for line in res.text.splitlines()
            if line.startswith(f"vllm:{name}{{")
        )
    return values


def concurrent_context_recall(args, passage, length):
    count = args.context_batch
    templates = [exact_context_prompt(args, passage, length, i) for i in range(count)]
    salts = [uuid.uuid4().hex for _ in templates]
    budget = 64
    if length + budget > templates[0][3]:
        raise ValueError("The concurrent context probe needs room for 64 output tokens")
    # Prime the independent prefixes so the first client cannot finish decoding
    # while the last client's long prefill still waits to be scheduled.
    cold = [
        exact_completion(args, prompt, answer, salt)
        for (prompt, answer, _, _), salt in zip(templates, salts, strict=True)
    ]
    barrier = threading.Barrier(count)

    def decode(i):
        prompt, answer, _, _ = templates[i]
        barrier.wait(timeout=60)
        return exact_completion(args, prompt, answer, salts[i], fill=budget)

    before = pool_counters(args)
    peak = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        jobs = [pool.submit(decode, i) for i in range(count)]
        while not all(job.done() for job in jobs):
            peak = max(peak, pool_counters(args)["num_requests_running"])
            time.sleep(0.15)
        warm = [job.result() for job in jobs]
    after = pool_counters(args)
    preemptions = after["num_preemptions_total"] - before["num_preemptions_total"]
    hits = int(after["prefix_cache_hits_total"] - before["prefix_cache_hits_total"])
    return {
        "prompt_tokens": length,
        "max_model_len": templates[0][3],
        "requests": count,
        "peak_running": int(peak),
        "preemptions": int(preemptions),
        "cached_tokens_in_burst": hits,
        "aggregate_input_tokens": length * count,
        "runs": cold,
        "concurrent_decodes": warm,
        "pass": (
            peak == count
            and preemptions == 0
            and all(r["correct"] for r in cold)
            and all(r["correct"] and r["output"] == budget for r in warm)
        ),
    }


def exact_context_recall(args):
    passage = json.loads(args.context_manifest.read_text())["shared"]
    rows = []
    for length in args.context_tokens:
        if args.context_batch > 1:
            rows.append(concurrent_context_recall(args, passage, length))
            continue
        prompt, answer, position, limit = exact_context_prompt(args, passage, length)
        if length + 32 > limit:
            raise ValueError(f"{length} prompt tokens plus 32 output exceed {limit}")
        salt = uuid.uuid4().hex
        pair = [exact_completion(args, prompt, answer, salt) for _ in range(2)]
        row = {
            "prompt_tokens": length,
            "max_model_len": limit,
            "memo_start_token": position,
            "expected": answer,
            "runs": pair,
            "identical_ids": bool(pair[0]["ids"]) and pair[0]["ids"] == pair[1]["ids"],
            "pass": all(r["correct"] for r in pair),
        }
        if length + 32 == limit:
            boundary = exact_completion(args, prompt, answer, salt, fill=32)
            row["at_limit"] = boundary
            row["pass"] &= (
                boundary["correct"] and boundary["prompt"] + boundary["output"] == limit
            )
        rows.append(row)
    return rows


def context_recall(args):
    """Check retrieval beyond the sliding window and across prefix-cache reuse."""
    corpus = json.loads(args.context_manifest.read_text())
    passage = corpus["shared"]
    runs = []
    for repetitions, answer in ((1, "PEBBLE-7419"), (2, "HEMLock-3816")):
        text = "\n\n".join([passage] * repetitions)
        offset = len(text) // 5
        text = (
            text[:offset]
            + "\n\nFor this review, the release codename is "
            + answer
            + ".\n\n"
            + text[offset:]
        )
        body = {
            "messages": [
                {"role": "system", "content": text},
                {
                    "role": "user",
                    "content": "What release codename was assigned to this review? "
                    "Copy it exactly. Reply with the codename only.",
                },
            ],
            "temperature": 0,
            "seed": SEED,
            "max_tokens": 32,
            "cache_salt": uuid.uuid4().hex,
            "return_token_ids": True,
            "chat_template_kwargs": {"reasoning_effort": "no_think"},
        }
        pair = []
        for _ in range(2):
            data = response(args, "/v1/chat/completions", body)
            choice = data["choices"][0]
            details = data["usage"].get("prompt_tokens_details") or {}
            content = choice["message"].get("content") or ""
            pair.append(
                {
                    "content": content,
                    "correct": content.strip() == answer,
                    "ids": choice.get("token_ids"),
                    "prompt": data["usage"]["prompt_tokens"],
                    "cached": details.get("cached_tokens"),
                }
            )
        runs.append(
            {
                "prefix_repetitions": repetitions,
                "runs": pair,
                "identical_ids": bool(pair[0]["ids"])
                and pair[0]["ids"] == pair[1]["ids"],
                "pass": all(x["correct"] for x in pair),
            }
        )
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8020")
    p.add_argument("--model", default="hy4-flash")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument(
        "--multiple-choice",
        type=Path,
        help="JSONL with question, four choices and answer A–D",
    )
    p.add_argument("--context-manifest", type=Path)
    p.add_argument(
        "--context-batch",
        type=int,
        default=1,
        help="Prime this many independent contexts and verify concurrent decode",
    )
    p.add_argument(
        "--context-tokens",
        type=int,
        nargs="+",
        metavar="N",
        help="Run cold and repeated exact-length recall at these input lengths",
    )
    p.add_argument("--cats", type=Path)
    p.add_argument("--red-video", type=Path, help="Optional four-frame red AVI control")
    p.add_argument("--chat-tokens", type=int, default=160)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument(
        "--only",
        choices=(
            "all",
            "chat",
            "reference",
            "modalities",
            "context",
            "reasoning",
            "multiple_choice",
        ),
        default="all",
    )
    args = p.parse_args()
    if args.context_batch < 1 or (args.context_batch > 1 and not args.context_tokens):
        p.error("Set --context-tokens and a positive --context-batch")
    if args.context_tokens and (
        not args.context_manifest or args.only not in ("context", "all")
    ):
        p.error(
            "--context-tokens requires --context-manifest and --only context or all"
        )
    if args.only == "context" and not args.context_manifest:
        p.error("Long-context checks require --context-manifest")
    if args.only == "reference" and not args.reference:
        p.error("Reference comparison requires --reference")
    if args.only == "multiple_choice" and not args.multiple_choice:
        p.error("Multiple-choice evaluation requires --multiple-choice")
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        if args.only in ("all", "chat"):
            rows = list(pool.map(lambda p: corpus(args, p), enumerate(PROMPTS)))
            (args.output / "chat.json").write_text(json.dumps(rows, indent=2))
            results["chat"] = {
                "passed": sum(x["pass"] for x in rows),
                "total": len(rows),
                "failed": [x["index"] for x in rows if not x["pass"]],
                "strict_format_passed": sum(x["strict_format"] is True for x in rows),
                "strict_format_total": sum(
                    x["strict_format"] is not None for x in rows
                ),
            }
        if args.multiple_choice and args.only in ("all", "multiple_choice"):
            data = args.multiple_choice.read_bytes()
            questions = [json.loads(line) for line in data.splitlines() if line.strip()]
            if not questions:
                raise ValueError("Empty multiple-choice corpus")
            rows = list(
                pool.map(lambda item: multiple_choice(args, item), enumerate(questions))
            )
            (args.output / "multiple-choice.json").write_text(
                json.dumps(rows, indent=2)
            )
            results["multiple_choice"] = {
                "correct": sum(row["correct"] for row in rows),
                "total": len(rows),
                "unparseable": sum(row["answer"] is None for row in rows),
                "corpus_sha256": hashlib.sha256(data).hexdigest(),
            }
        if args.reference and args.only in ("all", "reference"):
            source = [
                json.loads(s) for s in args.reference.read_text().splitlines() if s
            ]
            rows = list(pool.map(lambda x: teacher_row(args, x), source))
            (args.output / "reference.json").write_text(json.dumps(rows, indent=2))
            errors = [x["kl_top20_and_tail"] for x in rows]
            results["teacher"] = {
                "prefixes": len(rows),
                "top1_agrees": sum(x["top1_agrees"] for x in rows),
                "mean_kl_top20_and_tail": statistics.mean(errors),
                "median_kl_top20_and_tail": statistics.median(errors),
                "p95_kl_top20_and_tail": sorted(errors)[
                    round(0.95 * (len(errors) - 1))
                ],
                "max_kl_top20_and_tail": max(errors),
            }
    if args.only in ("all", "reasoning"):
        cases = reasoning_modes(args)
        (args.output / "reasoning.json").write_text(json.dumps(cases, indent=2))
        results["reasoning"] = {name: row["pass"] for name, row in cases.items()}
    if args.only in ("all", "modalities"):
        cases = modalities(args)
        (args.output / "modalities.json").write_text(json.dumps(cases, indent=2))
        results["modalities"] = {x["case"]: x["pass"] for x in cases}
    if args.context_manifest and args.only in ("all", "context"):
        rows = (
            exact_context_recall(args) if args.context_tokens else context_recall(args)
        )
        (args.output / "context.json").write_text(json.dumps(rows, indent=2))
        results["context"] = {
            str(x.get("prompt_tokens") or x["prefix_repetitions"]): {
                "recall": x["pass"],
                **(
                    {"identical_ids": x["identical_ids"]}
                    if "identical_ids" in x
                    else {}
                ),
            }
            for x in rows
        }
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    failed_context = any(
        not all(row.values()) for row in results.get("context", {}).values()
    )
    if (
        results.get("chat", {}).get("failed")
        or failed_context
        or not all(results.get("modalities", {}).values())
        or not all(results.get("reasoning", {}).values())
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
