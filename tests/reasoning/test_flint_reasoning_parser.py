# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Flint reasoning parser."""

import pytest

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
    # A turn with no ``</think>`` is still reasoning: the engine keeps the text
    # in the reasoning field and returns no content, the same contract the
    # qwen3 parser this fork serves has.
    reasoning, content = parser.extract_reasoning("plain answer", None)
    assert reasoning == "plain answer"
    assert content is None


def test_reasoning_end_detection(parser, flint_tokenizer):
    start = flint_tokenizer.convert_tokens_to_ids("<think>")
    end = flint_tokenizer.convert_tokens_to_ids("</think>")
    assert not parser.is_reasoning_end([start, 1, 2, 3])
    assert parser.is_reasoning_end([start, 1, end])
    assert parser.is_reasoning_end_streaming([start, 1, end], [end])
    assert not parser.is_reasoning_end_streaming([start, 1], [1])
