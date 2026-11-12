# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Tests for the Flint tool-call parser and its structural-tag grammar."""

import pytest
from xgrammar import Grammar

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.flint_tool_parser import FlintToolParser
from vllm.tool_parsers.structural_tag_registry import (
    get_model_structural_tag,
    limit_to_single_tool_call,
)
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel

MODEL = "RecursiveSI/flint"


@pytest.fixture(scope="module")
def flint_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def parser(flint_tokenizer):
    return FlintToolParser(flint_tokenizer)


@pytest.fixture
def sample_tools():
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                strict=True,
            ),
        )
    ]


def test_extract_tool_call(parser):
    model_output = (
        "Let me check."
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Paris</arg_value></tool_call>"
    )
    result = parser.extract_tool_calls(
        model_output, ChatCompletionRequest(messages=[], model="flint")
    )
    assert result.tools_called
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_weather"
    assert result.tool_calls[0].function.arguments == '{"city": "Paris"}'
    assert result.content == "Let me check."


def test_extract_tool_call_with_trailing_space(parser):
    model_output = (
        "<tool_call>get_weather <arg_key>city</arg_key>"
        "<arg_value>Paris</arg_value></tool_call>"
    )
    result = parser.extract_tool_calls(
        model_output, ChatCompletionRequest(messages=[], model="flint")
    )
    assert result.tools_called
    assert result.tool_calls[0].function.name == "get_weather"


def test_extract_text_only(parser):
    result = parser.extract_tool_calls(
        "no call here", ChatCompletionRequest(messages=[], model="flint")
    )
    assert not result.tools_called
    assert result.content == "no call here"


def test_grammar_auto_and_forced(sample_tools):
    tag = get_model_structural_tag(
        "flint",
        sample_tools,
        "auto",
        reasoning=False,
        strict_level=ToolStrictLevel.FUNCTION,
    )
    assert tag.format.type == "triggered_tags"
    assert "<function_calls>" in tag.format.excludes
    assert "<tool_call>" not in tag.format.excludes
    # End-of-turn markers must stay unmasked or the model cannot stop after a
    # call and repeats it until max_tokens.
    assert "<|observation|>" not in tag.format.excludes
    assert "<|end_of_text|>" not in tag.format.excludes
    Grammar.from_structural_tag(tag)

    named = ChatCompletionNamedToolChoiceParam(
        type="function", function={"name": "get_weather"}
    )
    forced = get_model_structural_tag(
        "flint",
        sample_tools,
        named,
        reasoning=False,
        strict_level=ToolStrictLevel.FUNCTION,
    )
    assert forced.format.type == "tags_with_separator"
    limit_to_single_tool_call(forced)
    assert forced.format.stop_after_first
    Grammar.from_structural_tag(forced)


def test_grammar_reasoning_prefix(sample_tools):
    tag = get_model_structural_tag(
        "flint",
        sample_tools,
        "auto",
        reasoning=True,
        strict_level=ToolStrictLevel.FUNCTION,
    )
    assert [e.type for e in tag.format.elements] == ["tag", "triggered_tags"]
    Grammar.from_structural_tag(tag)
