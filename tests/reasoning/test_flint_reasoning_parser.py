# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Flint reasoning parser."""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning.flint_reasoning_parser import FlintReasoningParser
from vllm.tokenizers import get_tokenizer

MODEL = "RecursiveSI/flint"


@pytest.fixture(scope="module")
def flint_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def parser(flint_tokenizer):
    return FlintReasoningParser(flint_tokenizer)


def test_extract_reasoning(parser):
    reasoning, content = parser.extract_reasoning(
        "<think>checking the file</think>the answer is 42", None
    )
    assert reasoning == "checking the file"
    assert content == "the answer is 42"


def test_extract_without_reasoning(parser):
    assert parser.extract_reasoning("plain answer", None) == (None, "plain answer")
    assert parser.extract_reasoning("</think>answer", None) == (None, "answer")


@pytest.mark.parametrize("thinking", [True, False])
def test_reasoning_with_client_thinking_flag(flint_tokenizer, thinking):
    parser = FlintReasoningParser(
        flint_tokenizer, chat_template_kwargs={"enable_thinking": thinking}
    )
    text = "<think>example <tool_call>lookup</tool_call></think>42"
    assert parser.extract_reasoning(text, None) == (
        "example <tool_call>lookup</tool_call>",
        "42",
    )
    assert parser.extract_reasoning("<think>unfinished", None) == ("unfinished", None)

    tokens = flint_tokenizer.encode(text, add_special_tokens=False)
    reasoning, content = [], []
    previous = ""
    for i, token in enumerate(tokens):
        delta = flint_tokenizer.decode([token], skip_special_tokens=False)
        msg = parser.extract_reasoning_streaming(
            previous, previous + delta, delta, tokens[:i], tokens[: i + 1], [token]
        )
        if msg:
            reasoning.append(msg.reasoning or "")
            content.append(msg.content or "")
        previous += delta
    assert "".join(reasoning) == "example <tool_call>lookup</tool_call>"
    assert "".join(content) == "42"


def test_reasoning_end_detection(parser, flint_tokenizer):
    start = flint_tokenizer.convert_tokens_to_ids("<think>")
    end = flint_tokenizer.convert_tokens_to_ids("</think>")
    assert not parser.is_reasoning_end([start, 1, 2, 3])
    assert parser.is_reasoning_end([start, 1, end])
    assert parser.is_reasoning_end_streaming([start, 1, end], [end])
    assert not parser.is_reasoning_end_streaming([start, 1], [1])


@pytest.mark.parametrize("chunk_size", [1, 7, 128])
@pytest.mark.parametrize("with_reasoning", [True, False])
def test_served_stream_with_tools(flint_tokenizer, chunk_size, with_reasoning):
    parser_cls = ParserManager.get_parser("flint", "flint", enable_auto_tools=True)
    req = ChatCompletionRequest(
        model="flint",
        messages=[],
        tool_choice="auto",
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        chat_template_kwargs={"enable_thinking": False},
    )
    parser = parser_cls(
        flint_tokenizer, req.tools, chat_template_kwargs=req.chat_template_kwargs
    )
    example = "example <tool_call>fake</tool_call>"
    prefix = f"<think>{example}</think>" if with_reasoning else ""
    text = prefix + "<tool_call>lookup</tool_call>"
    tokens = flint_tokenizer.encode(text, add_special_tokens=False)
    prompt = flint_tokenizer.encode("<|assistant|>", add_special_tokens=False)
    deltas = []
    for i in range(0, len(tokens), chunk_size):
        ids = tokens[i : i + chunk_size]
        delta = parser.parse_delta(
            flint_tokenizer.decode(ids, skip_special_tokens=False),
            ids,
            req,
            prompt_token_ids=prompt if i == 0 else None,
            finished=i + len(ids) == len(tokens),
        )
        if delta:
            deltas.append(delta)
    assert "".join(d.reasoning or "" for d in deltas) == (
        example if with_reasoning else ""
    )
    assert not "".join(d.content or "" for d in deltas).strip()
    assert [
        t.function.name
        for d in deltas
        for t in (d.tool_calls or [])
        if t.function and t.function.name
    ] == ["lookup"]
