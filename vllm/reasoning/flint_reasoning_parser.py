# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parse Flint's optional, explicitly delimited reasoning blocks."""

from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
from vllm.parser.flint import FlintReasoningEngine


class FlintReasoningParser(ParserEngineReasoningAdapter):
    """Leave direct answers in content and split explicit reasoning from answers."""

    _parser_engine_cls = FlintReasoningEngine
    may_open_reasoning_in_output = True
