# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flint reasoning parser: the checkpoint's ``<think>…</think>`` blocks.

Flint decides per turn whether to reason and wraps that reasoning in a
``<think>…</think>`` block before any content or tool call; both markers are
dedicated vocabulary tokens and the chat template treats them as protocol
(``flint_text`` strips user-supplied copies). The block structure is the same
one this fork serves for the ``qwen3`` parser, so extraction and the
structured-output reasoning gate are inherited unchanged; this entry gives the
recipe the name that matches the model.
"""

from vllm.parser.engine.registered_adapters import Qwen3ParserReasoningAdapter


class FlintReasoningParser(Qwen3ParserReasoningAdapter):  # type: ignore[valid-type, misc]
    """``<think>`` reasoning blocks for Flint."""
