# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parse Flint's optional, explicitly delimited reasoning blocks."""

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)


class FlintReasoningEngine(ParserEngine):
    def __init__(self, tokenizer, **kwargs):
        # The Flint template has no thinking switch. Only generated markers
        # open/close reasoning, even if a client passes Qwen template kwargs.
        markers = {"THINK_START": "<think>", "THINK_END": "</think>"}
        super().__init__(
            tokenizer,
            parser_engine_config=ParserEngineConfig(
                name="flint",
                terminals=markers,
                token_id_terminals=markers,
                initial_state=ParserState.CONTENT,
                transitions={
                    (ParserState.CONTENT, "THINK_START"): Transition(
                        ParserState.REASONING, (EventType.REASONING_START,)
                    ),
                    (ParserState.CONTENT, "THINK_END"): Transition(
                        ParserState.CONTENT, ()
                    ),
                    (ParserState.REASONING, "THINK_END"): Transition(
                        ParserState.CONTENT, (EventType.REASONING_END,)
                    ),
                },
                turn_boundary_tokens=frozenset(
                    {
                        "<|assistant|>",
                        "<|user|>",
                        "<|system|>",
                        "<|observation|>",
                        "<|developer|>",
                        "<|end_of_text|>",
                    }
                ),
            ),
            **kwargs,
        )

    @property
    def reasoning_ended(self) -> bool:
        # A direct answer or tool call has no closing reasoning marker.
        return super().reasoning_ended or self._content_has_nonws

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # A speculative delta can contain the close and the first tool tokens.
        end = self.find_reasoning_end(input_ids)
        if end is not None and end.offset < len(input_ids):
            return input_ids[end.offset + 1 :]
        return input_ids
