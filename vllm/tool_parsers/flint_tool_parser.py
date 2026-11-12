# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flint tool-call parser: the GLM-4.7 XML call format plus Flint's grammar floor.

Flint's chat template writes calls as
``<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>``,
which is byte-identical to the GLM-4.7 format the ``glm47`` parser handles, so
parsing is inherited unchanged. What this registration adds is the name and the
grammar defaults from Recursive's serving kit (``--tool-call-parser flint``):

* every request with tools and ``tool_choice`` in {auto, required, named} is
  constrained to the ``flint`` structural tag (see
  ``vllm/tool_parsers/structural_tag_registry.py``), which keeps protocol
  markers out of free text and admits only well-formed calls, and
* ``tool_choice`` semantics match the kit: any number of calls for ``auto``,
  at least one for ``required``, exactly one for a named function, and one
  call total when the request sets ``parallel_tool_calls: false``.
"""

from vllm.parser.engine.registered_adapters import Glm47MoeParserToolAdapter
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel


class FlintToolParser(Glm47MoeParserToolAdapter):  # type: ignore[valid-type, misc]
    """Flint's ``<tool_call>`` XML format, constrained by the ``flint`` grammar."""

    supports_required_and_named = False
    structural_tag_model = "flint"
    # Constrain auto tool calls with the shallow tag by default, as the kit does.
    default_tool_strict_level = ToolStrictLevel.FUNCTION
